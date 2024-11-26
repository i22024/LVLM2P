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

class A2CAlgo(BaseAlgo):
    """The Advantage Actor-Critic algorithm."""

    def __init__(self, envs, acmodel, device=None, num_frames_per_proc=None, discount=0.99, lr=0.01, gae_lambda=0.95,
                 entropy_coef=0.01, value_loss_coef=0.5, max_grad_norm=0.5, recurrence=4,
                 rmsprop_alpha=0.99, rmsprop_eps=1e-8, preprocess_obss=None,n_workers=12,action_option=None, env_name=None,
                 reshape_reward=None):
        num_frames_per_proc = num_frames_per_proc or 8

        super().__init__(envs, acmodel, device, num_frames_per_proc, discount, lr, gae_lambda, entropy_coef,
                         value_loss_coef, max_grad_norm, recurrence, preprocess_obss, reshape_reward,n_workers,action_option,env_name)

        self.optimizer = torch.optim.RMSprop(self.acmodel.parameters(), lr,
                                             alpha=rmsprop_alpha, eps=rmsprop_eps)

    def update_parameters(self, exps,rgb_images,missions):
        # Compute starting indexes
        inds = self._get_starting_indexes()

        # Initialize update values
        update_entropy = 0
        update_value = 0
        update_policy_loss = 0
        update_value_loss = 0
        update_loss = 0
        update_KL_div=0

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

            if self.action_option=="train_a2c_from_vlm":
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
                KL_div_tensor = torch.empty(len(dist_teacher))
                for j in range(len(dist_teachers)):
                    prob_policy = F.softmax(dist.probs[j] + epsilon, dim=0)
                    _, max_index =torch.max(dist_teachers[j],0)
                    output_tensor = torch.zeros_like(dist_teachers[j])
                    output_tensor[max_index] = 0.8
                    kl_loss_max=torch.sum(output_tensor * (torch.log(output_tensor + epsilon) - torch.log(prob_policy + epsilon)))
                    kl_loss_max*=0.01
                    KL_div_tensor[j] = kl_loss_max


            entropy = dist.entropy().mean()

            policy_loss = -(dist.log_prob(sb.action) * sb.advantage).mean()

            value_loss = (value - sb.returnn).pow(2).mean()

            loss = policy_loss - self.entropy_coef * entropy + self.value_loss_coef * value_loss
            if self.action_option=='train_a2c_from_vlm':
                KL_loss=KL_div_tensor.mean()
                loss= loss+KL_loss

            # Update batch values
            update_entropy += entropy.item()
            update_value += value.mean().item()
            update_policy_loss += policy_loss.item()
            update_value_loss += value_loss.item()
            if self.action_option=='train_a2c_from_vlm':
                update_KL_div+= KL_loss.item()
            update_loss += loss

        # Update update values
        update_entropy /= self.recurrence
        update_value /= self.recurrence
        update_policy_loss /= self.recurrence
        update_value_loss /= self.recurrence
        if self.action_option=='train_a2c_from_vlm':
            update_KL_div /=self.recurrence
        update_loss /= self.recurrence

        # Update actor-critic
        self.optimizer.zero_grad()
        update_loss.backward()
        update_grad_norm = sum(p.grad.data.norm(2) ** 2 for p in self.acmodel.parameters()) ** 0.5
        torch.nn.utils.clip_grad_norm_(self.acmodel.parameters(), self.max_grad_norm)
        self.optimizer.step()

        # Log some values
        logs = {
            "entropy": update_entropy,
            "value": update_value,
            "policy_loss": update_policy_loss,
            "value_loss": update_value_loss,
            "grad_norm": update_grad_norm,
            "KL_div" : update_KL_div
        }

        return logs

    def _get_starting_indexes(self):
        """Gives the indexes of the observations given to the model and the
        experiences used to compute the loss at first.

        The indexes are the integers from 0 to `self.num_frames` with a step of
        `self.recurrence`. If the model is not recurrent, they are all the
        integers from 0 to `self.num_frames`.

        Returns
        -------
        starting_indexes : list of int
            the indexes of the experiences to be used at first
        """

        starting_indexes = numpy.arange(0, self.num_frames, self.recurrence)
        return starting_indexes
