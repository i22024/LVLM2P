import numpy
import torch
import torch.nn.functional as F

from torch_ac.algos.base import BaseAlgo
from gemini_policy_2turn import query_probability_from_vlm_with_images, construct_vlm_with_key, construct_gemini_keys
from image_preprocess import create_single_image_without_wall
import time
import multiprocessing
from multiprocessing import get_context
from multiprocessing import Pool


def get_probability_from_vlm(processed_image,target_object,n_workers,rank=0):
    construct_gemini_keys(num_rollout_workers=n_workers)
    gemini_flash, gemini_pro, gemini_key = construct_vlm_with_key( rank=rank)
    error = None
    vlm_flag = False
    attempt_cnt = 0
    save_idx=rank
    while not vlm_flag:  
        prob_dist, _, error = query_probability_from_vlm_with_images(gemini_flash,gemini_pro, image=processed_image,  target_object=target_object,save_idx=save_idx)
        prob_dist.extend([0.0,0.0,0.0])
        if error is None:
            vlm_flag = True

        elif error.grpc_status_code.name in ["DEADLINE_EXCEEDED", "RESOURCE_EXHAUSTED", "INTERNAL", "UNKNOWN", "UNAVAILABLE"]:
            # Reference error: https://cloud.google.com/apis/design/errors#handling_errors
            print(f"[INFO] Got '{error.grpc_status_code.name}' from Gemini, retrying ({attempt_cnt}) ... (rank={0}, {gemini_key})")
            attempt_cnt+=1
            time.sleep(4)
        else:
            print(f"[INFO] Got unexpected Error: {error}")
            attempt_cnt += 1 
        
        if attempt_cnt >= 450:
            breakpoint()

    return prob_dist


class PPOAlgo(BaseAlgo):
    """The Proximal Policy Optimization algorithm
    ([Schulman et al., 2015](https://arxiv.org/abs/1707.06347))."""

    def __init__(self, envs, acmodel, device=None, num_frames_per_proc=None, discount=0.99, lr=0.001, gae_lambda=0.95,
                 entropy_coef=0.01, value_loss_coef=0.5, max_grad_norm=0.5, recurrence=4,
                 adam_eps=1e-8, clip_eps=0.2, epochs=4, batch_size=256, preprocess_obss=None,n_workers=12,action_option=None, env_name=None,
                 reshape_reward=None):
        num_frames_per_proc = num_frames_per_proc or 128

        super().__init__(envs, acmodel, device, num_frames_per_proc, discount, lr, gae_lambda, entropy_coef,
                         value_loss_coef, max_grad_norm, recurrence, preprocess_obss, reshape_reward,n_workers,action_option,env_name)

        self.clip_eps = clip_eps
        self.epochs = epochs
        self.batch_size = batch_size
        self.action_option=action_option
        self.env_name=env_name

        assert self.batch_size % self.recurrence == 0

        self.optimizer = torch.optim.Adam(self.acmodel.parameters(), lr, eps=adam_eps)
        self.batch_num = 0

    def update_parameters(self, exps,rgb_images,missions):
        # Collect experiences
        for _ in range(self.epochs):
            # Initialize log values
            log_entropies = []
            log_values = []
            log_policy_losses = []
            log_value_losses = []
            log_KL_divs=[]
            log_grad_norms = []

            for inds in self._get_batches_starting_indexes():
                # Initialize batch values
                batch_entropy = 0
                batch_value = 0
                batch_policy_loss = 0
                batch_value_loss = 0
                batch_loss = 0
                batch_KL_div=0

                # Initialize memory
                if self.acmodel.recurrent:
                    memory = exps.memory[inds]

                for i in range(self.recurrence):
                    # Create a sub-batch of experience
                    sb = exps[inds + i]
                    
                    # Compute loss
                    if self.acmodel.recurrent:
                        dist, value, memory = self.acmodel(sb.obs, memory * sb.mask)
                    else:
                        dist, value = self.acmodel(sb.obs)
                    
                    if self.action_option=="train_ppo_from_vlm":
                        selected_rgb_images = [rgb_images[idx] for idx in (inds + i)]
                        selected_mission=[missions[idx] for idx in (inds + i)]
                        shape = dist.probs.shape
                        dist_teachers = torch.empty(shape, device=dist.probs.device, dtype=dist.probs.dtype)
                        for n in range(len(dist.probs)):
                            n_workers=3
                            processed_image = create_single_image_without_wall(selected_rgb_images[n], edit_image=False, obs_size=192)
                            target_object = selected_mission[n][10:]
                            dist_teacher= get_probability_from_vlm(processed_image,target_object,n_workers,n)
                            dist_teachers[n]= torch.tensor(dist_teacher, dtype=dist_teachers.dtype, device=dist_teachers.device)
                            
                        # Calculate KL Divergence
                        epsilon = 1e-10
                        KL_div_tensor = torch.empty(len(dist_teachers))
                        for j in range(len(dist_teachers)):
                            prob_policy = F.softmax(dist.probs[j] + epsilon, dim=0)
                            _, max_index =torch.max(dist_teachers[j],0)
                            output_tensor = torch.zeros_like(dist_teachers[j])
                            output_tensor[max_index] = 0.8
                            kl_loss_max=torch.sum(output_tensor * (torch.log(output_tensor + epsilon) - torch.log(prob_policy + epsilon)))
                            kl_loss_max*=0.01
                            KL_div_tensor[j] = kl_loss_max
                            

                    entropy = dist.entropy().mean()

                    ratio = torch.exp(dist.log_prob(sb.action) - sb.log_prob)
                    surr1 = ratio * sb.advantage
                    surr2 = torch.clamp(ratio, 1.0 - self.clip_eps, 1.0 + self.clip_eps) * sb.advantage
                    policy_loss = -torch.min(surr1, surr2).mean()

                    value_clipped = sb.value + torch.clamp(value - sb.value, -self.clip_eps, self.clip_eps)
                    surr1 = (value - sb.returnn).pow(2)
                    surr2 = (value_clipped - sb.returnn).pow(2)
                    value_loss = torch.max(surr1, surr2).mean()

                    loss = policy_loss - self.entropy_coef * entropy + self.value_loss_coef * value_loss
                    if self.action_option=='train_ppo_from_vlm':
                        KL_loss=torch.mean(KL_div_tensor)
                        loss= loss+KL_loss
                    
                    # Update batch values
                    batch_entropy += entropy.item()
                    batch_value += value.mean().item()
                    batch_policy_loss += policy_loss.item()
                    batch_value_loss += value_loss.item()
                    if self.action_option=='train_ppo_from_vlm':
                        batch_KL_div+= KL_loss.item()
                    batch_loss += loss

                    # Update memories for next epoch
                    if self.acmodel.recurrent and i < self.recurrence - 1:
                        exps.memory[inds + i + 1] = memory.detach()

                # Update batch values
                batch_entropy /= self.recurrence
                batch_value /= self.recurrence
                batch_policy_loss /= self.recurrence
                batch_value_loss /= self.recurrence
                if self.action_option=='train_ppo_from_vlm':
                    batch_KL_div /=self.recurrence
                batch_loss /= self.recurrence

                # Update actor-critic
                self.optimizer.zero_grad()
                batch_loss.backward()
                grad_norm = sum(p.grad.data.norm(2).item() ** 2 for p in self.acmodel.parameters()) ** 0.5
                torch.nn.utils.clip_grad_norm_(self.acmodel.parameters(), self.max_grad_norm)
                self.optimizer.step()

                # Update log values
                log_entropies.append(batch_entropy)
                log_values.append(batch_value)
                log_policy_losses.append(batch_policy_loss)
                log_value_losses.append(batch_value_loss)
                log_grad_norms.append(grad_norm)
                if self.action_option=='train_ppo_from_vlm':
                    log_KL_divs.append(batch_KL_div)              
        
        # Log some values
        logs = {
            "entropy": numpy.mean(log_entropies),
            "value": numpy.mean(log_values),
            "policy_loss": numpy.mean(log_policy_losses),
            "value_loss": numpy.mean(log_value_losses),
            "grad_norm": numpy.mean(log_grad_norms),
            "KL_div" : numpy.mean(log_KL_divs)
        }

        return logs

    def _get_batches_starting_indexes(self):
        """Gives, for each batch, the indexes of the observations given to
        the model and the experiences used to compute the loss at first.

        First, the indexes are the integers from 0 to `self.num_frames` with a step of
        `self.recurrence`, shifted by `self.recurrence//2` one time in two for having
        more diverse batches. Then, the indexes are splited into the different batches.

        Returns
        -------
        batches_starting_indexes : list of list of int
            the indexes of the experiences to be used at first for each batch
        """

        indexes = numpy.arange(0, self.num_frames, self.recurrence)
        indexes = numpy.random.permutation(indexes)

        # Shift starting indexes by self.recurrence//2 half the time
        if self.batch_num % 2 == 1:
            indexes = indexes[(indexes + self.recurrence) % self.num_frames_per_proc != 0]
            indexes += self.recurrence // 2
        self.batch_num += 1

        num_indexes = self.batch_size // self.recurrence
        batches_starting_indexes = [indexes[i:i+num_indexes] for i in range(0, len(indexes), num_indexes)]

        return batches_starting_indexes
