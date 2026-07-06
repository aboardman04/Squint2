import gymnasium as gym
import mani_skill.envs
from envs.seperate_instruments import SeparateInstrumentsEnv
import torch
from mani_skill.utils.structs.pose import Pose

env = gym.make("SeparateInstruments-v1", num_envs=4)
env.reset()

pose = env.unwrapped.obj_1.pose
try:
    env.unwrapped.obj_1.set_pose(Pose.create_from_pq(p=torch.zeros(2, 3), q=torch.zeros(2, 4)))
    print("set_pose with size 2 worked")
except Exception as e:
    print(f"set_pose failed: {e}")
