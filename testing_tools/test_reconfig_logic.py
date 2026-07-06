import gymnasium as gym
import mani_skill.envs
from envs.seperate_instruments import SeparateInstrumentsEnv
import torch

env = gym.make("SeparateInstruments-v1", num_envs=2)
env.reset()

env.unwrapped.env_phase[:] = 0
env.unwrapped.settle_steps[:] = 4

# Force it to fail
env.unwrapped.obj_1.set_linear_velocity(torch.tensor([[0.0, 0, 0], [100.0, 0, 0]], device=env.unwrapped.device))
env.unwrapped.obj_2.set_linear_velocity(torch.tensor([[0.0, 0, 0], [-100.0, 0, 0]], device=env.unwrapped.device))

env.step(torch.zeros((2, 6)))

print("Step 5 phase:", env.unwrapped.env_phase.tolist(), "settle steps:", env.unwrapped.settle_steps.tolist())

dist = torch.linalg.norm(env.unwrapped.obj_1.pose.p[..., :2] - env.unwrapped.obj_2.pose.p[..., :2], axis=1)
print("Distances after step 5:", dist.tolist())

for i in range(5):
    env.step(torch.zeros((2, 6)))
print("Step 10 phase:", env.unwrapped.env_phase.tolist(), "settle steps:", env.unwrapped.settle_steps.tolist())
