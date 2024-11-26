import time
from multiprocessing import Process, Pipe
from abc import ABC, abstractmethod
import torch
import numpy as np

from torch_ac.format import default_preprocess_obss
from torch_ac.utils import DictList, ParallelEnv
import torch.nn.functional as F

import multiprocessing
from multiprocessing import get_context
from multiprocessing import Pool

import sys
sys.path.append("LVLM2P/torch_ac/algos/vlm_policy")
from gemini_policy_2turn import query_probability_from_vlm_with_images, construct_vlm_with_key, construct_gemini_keys
from image_preprocess import create_single_image_without_wall


def worker(args):
    rank,n_workers ,obs_data = args  # args : (index, (obs, dist))
    obs, dist_prob = obs_data
    if dist_prob=="vlm_evaluate":
        processed_image = create_single_image_without_wall(obs['rgb_image'], edit_image=False, obs_size=192)
        target_object = obs['mission'][10:]
        vlm_prob_dist = get_probability_from_vlm(processed_image, target_object,n_workers,rank)
        
        vlm_action = vlm_prob_dist.index(max(vlm_prob_dist))
        vlm_prob_dist.extend([0.0,0.0,0.0])
        #print(f"vlm_prob_dist : {vlm_prob_dist}")

        return rank, vlm_action
    else:
        processed_image = create_single_image_without_wall(obs['rgb_image'], edit_image=False, obs_size=192)
        target_object = obs['mission'][10:]
        vlm_prob_dist = get_probability_from_vlm(processed_image, target_object,n_workers,rank)
        
        vlm_action = vlm_prob_dist.index(max(vlm_prob_dist))
        vlm_prob_dist.extend([0.0,0.0,0.0])
        KL_div = kl_divergence(vlm_prob_dist, dist_prob)
        return rank, vlm_action, KL_div

def parallel_process_observations(observations, dists, n_workers):
    print(f"n_workers : {n_workers}")
    if dists==None:
        with get_context("spawn").Pool(processes=n_workers) as pool:
            results = pool.map(worker, [(i,n_workers, (obs, "vlm_evaluate")) for i, obs in enumerate(observations)])
        
        indices, vlm_actions = zip(*results)
        return list(indices), list(vlm_actions)
    
    else:
        with get_context("spawn").Pool(processes=n_workers) as pool:
            results = pool.map(worker, [(i,n_workers, (obs, dists[i])) for i, obs in enumerate(observations)])

        indices, vlm_actions, KL_div_batch = zip(*results)
        return list(indices), list(vlm_actions), list(KL_div_batch)



def kl_divergence(p, q):
    epsilon = 1e-10
    p = np.array(p) + epsilon
    q = q.cpu().numpy() + epsilon #q = np.array(q) + epsilon

    # KL Divergence
    return np.sum(p * np.log(p / q))


class BaseAlgo(ABC):
    """The base class for RL algorithms."""

    def __init__(self, envs, acmodel, device, num_frames_per_proc, discount, lr, gae_lambda, entropy_coef,
                 value_loss_coef, max_grad_norm, recurrence, preprocess_obss, reshape_reward,n_workers, action_option,env_name):
        """
        Initializes a `BaseAlgo` instance.

        Parameters:
        ----------
        envs : list
            a list of environments that will be run in parallel
        acmodel : torch.Module
            the model
        num_frames_per_proc : int
            the number of frames collected by every process for an update
        discount : float
            the discount for future rewards
        lr : float
            the learning rate for optimizers
        gae_lambda : float
            the lambda coefficient in the GAE formula
            ([Schulman et al., 2015](https://arxiv.org/abs/1506.02438))
        entropy_coef : float
            the weight of the entropy cost in the final objective
        value_loss_coef : float
            the weight of the value loss in the final objective
        max_grad_norm : float
            gradient will be clipped to be at most this value
        recurrence : int
            the number of steps the gradient is propagated back in time
        preprocess_obss : function
            a function that takes observations returned by the environment
            and converts them into the format that the model can handle
        reshape_reward : function
            a function that shapes the reward, takes an
            (observation, action, reward, done) tuple as an input
        """
        # action_option
        self.action_option=action_option
        self.n_workers=3
        self.env_name=env_name

        # Store parameters
        self.env = ParallelEnv(envs)
        self.acmodel = acmodel
        self.device = device
        self.num_frames_per_proc = num_frames_per_proc
        self.discount = discount
        self.lr = lr
        self.gae_lambda = gae_lambda
        self.entropy_coef = entropy_coef
        self.value_loss_coef = value_loss_coef
        self.max_grad_norm = max_grad_norm
        self.recurrence = recurrence
        self.preprocess_obss = preprocess_obss or default_preprocess_obss
        self.reshape_reward = reshape_reward

        # Control parameters
        assert self.acmodel.recurrent or self.recurrence == 1
        assert self.num_frames_per_proc % self.recurrence == 0

        # Configure acmodel
        self.acmodel.to(self.device)
        self.acmodel.train()

        # Store helpers values
        self.num_procs = len(envs)
        self.num_frames = self.num_frames_per_proc * self.num_procs

        # Initialize experience values
        shape = (self.num_frames_per_proc, self.num_procs)
        
        self.obs = self.env.reset()
        self.obss = [None] * (shape[0])
        self.rgb_images=[[] for _ in range(shape[0])]
        self.missions=[[] for _ in range(shape[0])]
        if self.acmodel.recurrent:
            self.memory = torch.zeros(shape[1], self.acmodel.memory_size, device=self.device)
            self.memories = torch.zeros(*shape, self.acmodel.memory_size, device=self.device)
        self.mask = torch.ones(shape[1], device=self.device)
        self.masks = torch.zeros(*shape, device=self.device)
        self.actions = torch.zeros(*shape, device=self.device, dtype=torch.int)
        self.values = torch.zeros(*shape, device=self.device)
        self.rewards = torch.zeros(*shape, device=self.device)
        self.advantages = torch.zeros(*shape, device=self.device)
        self.log_probs = torch.zeros(*shape, device=self.device)
        self.KL_div_batchs=torch.zeros(*shape, device=self.device)

        # Initialize log values
        self.log_episode_return = torch.zeros(self.num_procs, device=self.device)
        self.log_episode_reshaped_return = torch.zeros(self.num_procs, device=self.device)
        self.log_episode_num_frames = torch.zeros(self.num_procs, device=self.device)

        self.log_done_counter = 0
        self.log_return = [0] * self.num_procs
        self.log_reshaped_return = [0] * self.num_procs
        self.log_num_frames = [0] * self.num_procs
        construct_gemini_keys(num_rollout_workers=self.n_workers)

    def collect_experiences(self):
        """Collects rollouts and computes advantages.

        Runs several environments concurrently. The next actions are computed
        in a batch mode for all environments at the same time. The rollouts
        and advantages from all environments are concatenated together.

        Returns
        -------
        exps : DictList
            Contains actions, rewards, advantages etc as attributes.
            Each attribute, e.g. `exps.reward` has a shape
            (self.num_frames_per_proc * num_envs, ...). k-th block
            of consecutive `self.num_frames_per_proc` frames contains
            data obtained from the k-th environment. Be careful not to mix
            data from different environments!
        logs : dict
            Useful stats about the training process, including the average
            reward, policy loss, value loss, etc.
        """
        if self.action_option=="train_ppo_from_vlm" or self.action_option=="train_a2c_from_vlm":
            for i in range(self.num_frames_per_proc):
                import time
                # Do one agent-environment interaction
                preprocessed_obs = self.preprocess_obss(self.obs, device=self.device)
                
                with torch.no_grad():
                    if self.acmodel.recurrent:
                        dist, value, memory = self.acmodel(preprocessed_obs, self.memory * self.mask.unsqueeze(1))
                    else:
                        dist, value = self.acmodel(preprocessed_obs)

                action = dist.sample()
                # policy probability
                obs, reward, done, info = self.env.step(action.cpu().numpy())
                #done = tuple(a | b for a, b in zip(terminated, truncated))
                self.obss[i] = self.obs
                for j in range(len(self.obs)):
                    self.rgb_images[i].append(self.obs[j]['rgb_image'])
                    self.missions[i].append(self.obs[j]['mission'])
                self.obs = obs
                if self.acmodel.recurrent:
                    self.memories[i] = self.memory
                    self.memory = memory
                self.masks[i] = self.mask
                self.mask = 1 - torch.tensor(done, device=self.device, dtype=torch.float)
                self.actions[i] = action
                self.values[i] = value
                
                #done = tuple(a | b for a, b in zip(terminated, truncated))
                if self.reshape_reward is not None:
                    self.rewards[i] = torch.tensor([
                        self.reshape_reward(obs_, action_, reward_, done_)
                        for obs_, action_, reward_, done_ in zip(obs, action, reward, done)
                    ], device=self.device)
                else:
                    self.rewards[i] = torch.tensor(reward, device=self.device)
                self.log_probs[i] = dist.log_prob(action)

                # Update log values
                self.log_episode_return += torch.tensor(reward, device=self.device, dtype=torch.float)
                self.log_episode_reshaped_return += self.rewards[i]
                self.log_episode_num_frames += torch.ones(self.num_procs, device=self.device)

                for i, done_ in enumerate(done):
                    if done_:                
                        self.log_done_counter += 1
                        self.log_return.append(self.log_episode_return[i].item())
                        self.log_reshaped_return.append(self.log_episode_reshaped_return[i].item())
                        self.log_num_frames.append(self.log_episode_num_frames[i].item())

                self.log_episode_return *= self.mask
                self.log_episode_reshaped_return *= self.mask
                self.log_episode_num_frames *= self.mask

            # Add advantage and return to experiences               
            preprocessed_obs = self.preprocess_obss(self.obs, device=self.device)
            with torch.no_grad():
                if self.acmodel.recurrent:
                    _, next_value, _ = self.acmodel(preprocessed_obs, self.memory * self.mask.unsqueeze(1))
                else:
                    _, next_value = self.acmodel(preprocessed_obs)

            for i in reversed(range(self.num_frames_per_proc)):
                next_mask = self.masks[i+1] if i < self.num_frames_per_proc - 1 else self.mask
                next_value = self.values[i+1] if i < self.num_frames_per_proc - 1 else next_value
                next_advantage = self.advantages[i+1] if i < self.num_frames_per_proc - 1 else 0

                delta = self.rewards[i] + self.discount * next_value * next_mask - self.values[i]
                self.advantages[i] = delta + self.discount * self.gae_lambda * next_advantage * next_mask

            # Define experiences:
            #   the whole experience is the concatenation of the experience
            #   of each process.
            # In comments below:
            #   - T is self.num_frames_per_proc,
            #   - P is self.num_procs,
            #   - D is the dimensionality.

            exps = DictList()
            
            exps.obs = [self.obss[i][j]
                        for j in range(self.num_procs)
                        for i in range(self.num_frames_per_proc)]
            rgb_images = [self.rgb_images[i][j]
                        for j in range(self.num_procs)
                        for i in range(self.num_frames_per_proc)]
            missions=[self.missions[i][j]
                        for j in range(self.num_procs)
                        for i in range(self.num_frames_per_proc)]
            if self.acmodel.recurrent:
                # T x P x D -> P x T x D -> (P * T) x D
                exps.memory = self.memories.transpose(0, 1).reshape(-1, *self.memories.shape[2:])
                # T x P -> P x T -> (P * T) x 1
                exps.mask = self.masks.transpose(0, 1).reshape(-1).unsqueeze(1)
            # for all tensors below, T x P -> P x T -> P * T
            exps.action = self.actions.transpose(0, 1).reshape(-1)
            exps.value = self.values.transpose(0, 1).reshape(-1)
            exps.reward = self.rewards.transpose(0, 1).reshape(-1)
            exps.advantage = self.advantages.transpose(0, 1).reshape(-1)
            exps.returnn = exps.value + exps.advantage
            exps.log_prob = self.log_probs.transpose(0, 1).reshape(-1)
            

            # Preprocess experiences
            exps.obs = self.preprocess_obss(exps.obs, device=self.device)

            # Log some values
            keep = max(self.log_done_counter, self.num_procs)

            logs = {
                "return_per_episode": self.log_return[-keep:],
                "reshaped_return_per_episode": self.log_reshaped_return[-keep:],
                "num_frames_per_episode": self.log_num_frames[-keep:],
                "num_frames": self.num_frames
            }

            self.log_done_counter = 0
            self.log_return = self.log_return[-self.num_procs:]
            self.log_reshaped_return = self.log_reshaped_return[-self.num_procs:]
            self.log_num_frames = self.log_num_frames[-self.num_procs:]

            return exps, logs,rgb_images,missions
            
        elif self.action_option=="vlm_action":
            
            print("collecting VLM prob/actions...")
            indices, vlm_actions = parallel_process_observations(self.obs,None,self.n_workers) # 0 8 16 32 x (1task) 4 : constant value 0.2 0.4 0.6 1

            action=torch.tensor(vlm_actions,device=self.device)
            
            # policy probability
            obs, reward, done, info = self.env.step(action.cpu().numpy())  

            # Update experiences values
            self.obs = obs
            self.mask = 1 - torch.tensor(done, device=self.device, dtype=torch.float)

            # Update log values
            self.log_episode_return += torch.tensor(reward, device=self.device, dtype=torch.float)
            self.log_episode_num_frames += torch.ones(self.num_procs, device=self.device)

            for i, done_ in enumerate(done):
                if done_:
                    self.log_done_counter += 1
                    self.log_return.append(self.log_episode_return[i].item())
                    self.log_num_frames.append(self.log_episode_num_frames[i].item())

            self.log_episode_return *= self.mask
            self.log_episode_num_frames *= self.mask

            # Log some values
            keep = max(self.log_done_counter, self.num_procs)

            logs = {
                "return_per_episode": self.log_return[-keep:],
                "reshaped_return_per_episode": self.log_reshaped_return[-keep:],
                "num_frames_per_episode": self.log_num_frames[-keep:],
                "num_frames": self.num_frames
            }

            return None, logs
        
        elif self.action_option=='ppo_baseline' or self.action_option=='a2c_baseline':
            for i in range(self.num_frames_per_proc):
                # Do one agent-environment interaction
                preprocessed_obs = self.preprocess_obss(self.obs, device=self.device)
                
                with torch.no_grad():
                    if self.acmodel.recurrent:   
                        dist, value, memory = self.acmodel(preprocessed_obs, self.memory * self.mask.unsqueeze(1))
                    else:
                        dist, value = self.acmodel(preprocessed_obs)

                action = dist.sample()
                
                # policy probability
                obs, reward, done, info = self.env.step(action.cpu().numpy())  # changed by dh
                
                # Update experiences values
                self.obss[i] = self.obs
                self.obs = obs
                if self.acmodel.recurrent:
                    self.memories[i] = self.memory
                    self.memory = memory
                self.masks[i] = self.mask
                self.mask = 1 - torch.tensor(done, device=self.device, dtype=torch.float)
                self.actions[i] = action
                self.values[i] = value

                if self.reshape_reward is not None:
                    self.rewards[i] = torch.tensor([
                        self.reshape_reward(obs_, action_, reward_, done_)
                        for obs_, action_, reward_, done_ in zip(obs, action, reward, done)
                    ], device=self.device)
                else:
                    self.rewards[i] = torch.tensor(reward, device=self.device)
                self.log_probs[i] = dist.log_prob(action)

                # Update log values
                self.log_episode_return += torch.tensor(reward, device=self.device, dtype=torch.float)
                self.log_episode_reshaped_return += self.rewards[i]
                self.log_episode_num_frames += torch.ones(self.num_procs, device=self.device)

                for i, done_ in enumerate(done):
                    if done_:
                        self.log_done_counter += 1
                        self.log_return.append(self.log_episode_return[i].item())
                        self.log_reshaped_return.append(self.log_episode_reshaped_return[i].item())
                        self.log_num_frames.append(self.log_episode_num_frames[i].item())

                self.log_episode_return *= self.mask
                self.log_episode_reshaped_return *= self.mask
                self.log_episode_num_frames *= self.mask

            # Add advantage and return to experiences
            preprocessed_obs = self.preprocess_obss(self.obs, device=self.device)
            with torch.no_grad():
                if self.acmodel.recurrent:
                    _, next_value, _ = self.acmodel(preprocessed_obs, self.memory * self.mask.unsqueeze(1))
                else:
                    _, next_value = self.acmodel(preprocessed_obs)

            for i in reversed(range(self.num_frames_per_proc)):
                next_mask = self.masks[i+1] if i < self.num_frames_per_proc - 1 else self.mask
                next_value = self.values[i+1] if i < self.num_frames_per_proc - 1 else next_value
                next_advantage = self.advantages[i+1] if i < self.num_frames_per_proc - 1 else 0

                delta = self.rewards[i] + self.discount * next_value * next_mask - self.values[i]
                self.advantages[i] = delta + self.discount * self.gae_lambda * next_advantage * next_mask

            # Define experiences:
            #   the whole experience is the concatenation of the experience
            #   of each process.
            # In comments below:
            #   - T is self.num_frames_per_proc,
            #   - P is self.num_procs,
            #   - D is the dimensionality.

            exps = DictList()
            exps.obs = [self.obss[i][j]
                        for j in range(self.num_procs)
                        for i in range(self.num_frames_per_proc)]
            if self.acmodel.recurrent:
                # T x P x D -> P x T x D -> (P * T) x D
                exps.memory = self.memories.transpose(0, 1).reshape(-1, *self.memories.shape[2:])
                # T x P -> P x T -> (P * T) x 1
                exps.mask = self.masks.transpose(0, 1).reshape(-1).unsqueeze(1)
            # for all tensors below, T x P -> P x T -> P * T
            exps.action = self.actions.transpose(0, 1).reshape(-1)
            exps.value = self.values.transpose(0, 1).reshape(-1)
            exps.reward = self.rewards.transpose(0, 1).reshape(-1)
            exps.advantage = self.advantages.transpose(0, 1).reshape(-1)
            exps.returnn = exps.value + exps.advantage
            exps.log_prob = self.log_probs.transpose(0, 1).reshape(-1)
            exps.KL_div  = self.KL_div_batchs.transpose(0, 1).reshape(-1)

            # Preprocess experiences
            exps.obs = self.preprocess_obss(exps.obs, device=self.device)

            # Log some values
            keep = max(self.log_done_counter, self.num_procs)

            logs = {
                "return_per_episode": self.log_return[-keep:],
                "reshaped_return_per_episode": self.log_reshaped_return[-keep:],
                "num_frames_per_episode": self.log_num_frames[-keep:],
                "num_frames": self.num_frames
            }

            self.log_done_counter = 0
            self.log_return = self.log_return[-self.num_procs:]
            self.log_reshaped_return = self.log_reshaped_return[-self.num_procs:]
            self.log_num_frames = self.log_num_frames[-self.num_procs:]

            return exps, logs

    @abstractmethod
    def update_parameters(self):
        pass


def get_probability_from_vlm(processed_image,target_object,n_workers,rank=0):
    construct_gemini_keys(num_rollout_workers=n_workers)
    gemini_flash, gemini_pro, gemini_key = construct_vlm_with_key( rank=rank)
    error = None
    vlm_flag = False
    attempt_cnt = 0
    save_idx=rank
    while not vlm_flag:  
        prob_dist, _, error = query_probability_from_vlm_with_images(gemini_flash,gemini_pro, image=processed_image,  target_object=target_object,save_idx=save_idx)
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
