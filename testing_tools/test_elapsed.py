import gymnasium as gym
import mani_skill.envs
from envs.seperate_instruments import SeparateInstrumentsEnv
import torch

env = gym.make("SeparateInstruments-v1", num_envs=4)
env.reset()
print(hasattr(env.unwrapped, "elapsed_steps"))
print(env.unwrapped.elapsed_steps)
