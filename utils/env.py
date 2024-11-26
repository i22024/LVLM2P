'''
import gymnasium as gym


def make_env(env_key, seed=None, render_mode=None):
    env = gym.make(env_key, render_mode=render_mode)
    env.reset(seed=seed)
    return env
'''
import gym
import gym_minigrid
#print("gym path:")
#print(gym.__file__)
#print("gym_minigrid path:")
#print(gym_minigrid.__file__)


def make_env(env_key, seed=None):
    env = gym.make(env_key)
    env.seed(seed)
    return env
