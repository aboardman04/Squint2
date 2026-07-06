import gymnasium as gym
import mani_skill.envs
from envs.seperate_instruments import SeparateInstrumentsEnv
import torch

env = gym.make("SeparateInstruments-v1", num_envs=1)
env.reset()
print("Forceps 1 bbox:")
mesh = env.unwrapped.obj_1._objs[0].get_collision_meshes()[0]
print(mesh)
