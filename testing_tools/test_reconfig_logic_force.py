import gymnasium as gym
import mani_skill.envs
from envs.seperate_instruments import SeparateInstrumentsEnv
import torch

env = gym.make("SeparateInstruments-v1", num_envs=2)
env.reset()

env.unwrapped.env_phase[:] = 0
env.unwrapped.settle_steps[:] = 4

# Override super().step behavior temporarily just to force fail
original_step = env.unwrapped.__class__.__bases__[0].step
def mock_step(self, action):
    res = original_step(self, action)
    # forcefully move objects apart
    self.obj_1.set_pose(mani_skill.utils.structs.pose.Pose.create_from_pq(p=torch.tensor([[0.0,0,0], [1.0,0,0]], device=self.device), q=torch.zeros((2,4), device=self.device)))
    return res

import types
env.unwrapped.__class__.__bases__[0].step = types.MethodType(mock_step, env.unwrapped)

try:
    env.step(torch.zeros((2, 6)))
    print("Step 5 phase:", env.unwrapped.env_phase.tolist(), "settle steps:", env.unwrapped.settle_steps.tolist())

    for i in range(5):
        env.step(torch.zeros((2, 6)))
        print(f"Step {6+i} phase:", env.unwrapped.env_phase.tolist(), "settle steps:", env.unwrapped.settle_steps.tolist())
finally:
    env.unwrapped.__class__.__bases__[0].step = original_step
