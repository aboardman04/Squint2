import gymnasium as gym
import mani_skill.envs
from envs.seperate_instruments import SeparateInstrumentsEnv
import torch

env = gym.make("SeparateInstruments-v1", num_envs=4)
obs, _ = env.reset()

print("Num envs:", env.unwrapped.num_envs)
print("obj_1 pose shape:", env.unwrapped.obj_1.pose.p.shape)
