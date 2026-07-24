import gymnasium as gym
import torch
import mani_skill.envs
import envs.separate4_instruments_1
from mani_skill.utils.structs.pose import Pose

env = gym.make("SeparateInstruments-v2.5", num_envs=4)
env.reset()

# Mock settled
env.unwrapped.env_phase[:] = 0
env.unwrapped.settle_steps[0] = 30
env.unwrapped.settle_steps[1] = 30
env.unwrapped.settle_steps[2] = 30
env.unwrapped.settle_steps[3] = 0

# Run 1 step
action = env.action_space.sample()
env.step(action)

for wall, _ in env.unwrapped.settle_walls:
    print(wall.pose.p[:, 2])
