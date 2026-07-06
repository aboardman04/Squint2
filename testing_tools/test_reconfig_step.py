import gymnasium as gym
import mani_skill.envs
from envs.seperate_instruments import SeparateInstrumentsEnv
import torch
from mani_skill.utils.structs.pose import Pose

env = gym.make("SeparateInstruments-v1", num_envs=4)
env.reset()

full_p = env.unwrapped.obj_1.pose.p.clone()
full_q = env.unwrapped.obj_1.pose.q.clone()
full_p[0] = torch.tensor([0, 0, 1.0])

try:
    env.unwrapped.obj_1.set_pose(Pose.create_from_pq(p=full_p, q=full_q))
    print("set_pose with size 4 worked")
except Exception as e:
    print(f"set_pose failed: {e}")
