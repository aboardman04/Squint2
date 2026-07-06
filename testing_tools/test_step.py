import gymnasium as gym
import mani_skill.envs
from envs.seperate_instruments import SeparateInstrumentsEnv
import torch

env = gym.make("SeparateInstruments-v1", num_envs=4)
obs, _ = env.reset()

print("type of env:", type(env.unwrapped))
print("hasattr step:", hasattr(env.unwrapped, 'step'))
