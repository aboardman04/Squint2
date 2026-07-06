import gymnasium as gym
import mani_skill.envs
from envs.seperate_instruments import SeparateInstrumentsEnv
import torch

env = gym.make("SeparateInstruments-v1", num_envs=2)
env.reset()
for i in range(10):
    obs, reward, terminated, truncated, info = env.step(torch.zeros((2, 6)))
print("Reward mean:", reward.mean().item())
