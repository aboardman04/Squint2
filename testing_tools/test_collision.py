import gymnasium as gym
import torch
import mani_skill.envs
import envs.separate4_instruments_1

env = gym.make("SeparateInstruments-v2.5", num_envs=2)
env.reset()

for wall, _ in env.unwrapped.settle_walls:
    print("Trying to disable collision...")
    wall.set_collision_group_bit(group=2, bit_idx=30, bit=1)
    
print("Success!")
