import gymnasium as gym
import mani_skill.envs
from envs.seperate_instruments import SeparateInstrumentsEnv
import torch

env = gym.make("SeparateInstruments-v1", num_envs=2)
obs, _ = env.reset()

print("Setting velocity...")
try:
    vel = torch.zeros((2, 3), device=env.unwrapped.device)
    env.unwrapped.obj_1.set_linear_velocity(vel)
    env.unwrapped.obj_1.set_angular_velocity(vel)
    print("set_linear_velocity worked")
except Exception as e:
    print(f"Error: {e}")

