import gymnasium as gym
import mani_skill.envs
from envs.seperate_instruments import SeparateInstrumentsEnv
import torch
from mani_skill.utils.structs.pose import Pose

env = gym.make("SeparateInstruments-v1", num_envs=1)
env.reset()

for i in range(3):
    env.step(torch.zeros(6))

# Reconfigure
env.unwrapped.obj_1.set_pose(Pose.create_from_pq(p=torch.tensor([[0.0, 0.0, 0.01]]), q=torch.tensor([[1.0, 0.0, 0.0, 0.0]])))

for i in range(3):
    env.step(torch.zeros(6))

print("Done")
