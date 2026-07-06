import gymnasium as gym
import mani_skill.envs
from envs.seperate_instruments import SeparateInstrumentsEnv
import torch
import cv2

env = gym.make("SeparateInstruments-v1", num_envs=1)
obs, _ = env.reset()

def get_dist(env):
    return torch.linalg.norm(env.unwrapped.obj_1.pose.p[..., :2] - env.unwrapped.obj_2.pose.p[..., :2], axis=1).item()

print(f"Initial distance: {get_dist(env)}")

for i in range(50):
    action = env.action_space.sample() * 0 # Do nothing
    obs, reward, terminated, truncated, info = env.step(action)
    if i % 10 == 0:
        print(f"Step {i} distance: {get_dist(env)}")

print(f"Final distance: {get_dist(env)}")
