import gymnasium as gym
import mani_skill.envs
from envs.seperate_instruments import SeparateInstrumentsEnv
import torch
from mani_skill.utils.structs.pose import Pose

env = gym.make("SeparateInstruments-v1", num_envs=2)
env.reset()

for i in range(4):
    env.step(torch.zeros((2, 6)))

# Artificially move them apart before the 5th step check
p1 = torch.tensor([[0.0,0,0], [1.0,0,0]], device=env.unwrapped.device)
p2 = torch.tensor([[0.0,0,0], [2.0,0,0]], device=env.unwrapped.device)
q = torch.tensor([[1.0,0,0,0], [1.0,0,0,0]], device=env.unwrapped.device)
env.unwrapped.obj_1.set_pose(Pose.create_from_pq(p=p1, q=q))
env.unwrapped.obj_2.set_pose(Pose.create_from_pq(p=p2, q=q))

print("Forced apart on step 4")

env.step(torch.zeros((2, 6))) # Step 5
print("Step 5 phase:", env.unwrapped.env_phase.tolist(), "settle steps:", env.unwrapped.settle_steps.tolist())

dist = torch.linalg.norm(env.unwrapped.obj_1.pose.p[..., :2] - env.unwrapped.obj_2.pose.p[..., :2], axis=1)
print("Distances after step 5:", dist.tolist())

for i in range(5):
    env.step(torch.zeros((2, 6)))
print("Step 10 phase:", env.unwrapped.env_phase.tolist(), "settle steps:", env.unwrapped.settle_steps.tolist())
