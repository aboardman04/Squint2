import gymnasium as gym
import mani_skill.envs
from envs.seperate_instruments import SeparateInstrumentsEnv
import torch
from mani_skill.utils.structs.pose import Pose

env = gym.make("SeparateInstruments-v1", num_envs=1)
env.reset()

# Force them to intersect heavily
env.unwrapped.obj_1.set_pose(Pose.create_from_pq(p=torch.tensor([[0.0, 0.0, 0.01]]), q=torch.tensor([[1.0, 0.0, 0.0, 0.0]])))
env.unwrapped.obj_2.set_pose(Pose.create_from_pq(p=torch.tensor([[0.0, 0.0, 0.01]]), q=torch.tensor([[1.0, 0.0, 0.0, 0.0]])))

print("Initial distance:", torch.linalg.norm(env.unwrapped.obj_1.pose.p - env.unwrapped.obj_2.pose.p, axis=1).item())

for i in range(5):
    env.step(torch.zeros(6))
    dist = torch.linalg.norm(env.unwrapped.obj_1.pose.p - env.unwrapped.obj_2.pose.p, axis=1).item()
    print(f"Step {i+1} distance:", dist)

