import gymnasium as gym
import mani_skill.envs
from envs.seperate_instruments import SeparateInstrumentsEnv
import torch

env = gym.make("SeparateInstruments-v1", num_envs=4)
env.reset()
env.reset(options={"env_idx": torch.tensor([0, 2], device=env.unwrapped.device)})
print(env.unwrapped.obj_1.pose.p.shape)
