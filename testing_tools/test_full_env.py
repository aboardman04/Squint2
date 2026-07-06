import gymnasium as gym
import mani_skill.envs
from envs.seperate_instruments import SeparateInstrumentsEnv
import torch

env = gym.make("SeparateInstruments-v1", num_envs=2)
env.reset()

print("Initial phase:", env.unwrapped.env_phase)
print("Initial settle steps:", env.unwrapped.settle_steps)

for i in range(15):
    env.step(torch.zeros((2, 6)))
    print(f"Step {i+1} phase:", env.unwrapped.env_phase.tolist(), "settle steps:", env.unwrapped.settle_steps.tolist())

dist = torch.linalg.norm(env.unwrapped.obj_1.pose.p[..., :2] - env.unwrapped.obj_2.pose.p[..., :2], axis=1)
print("Final distances:", dist.tolist())
