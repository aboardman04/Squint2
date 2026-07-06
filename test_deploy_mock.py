import os
import torch
import gymnasium as gym

# Mock real agent so it doesn't try to connect to /dev/ttyACM0
import deploy_utils.robot_config
class MockRealAgent:
    def __init__(self, *args, **kwargs): pass
    def reset(self, *args, **kwargs): pass
    def get_obs(self, *args, **kwargs): 
        return {"qpos": torch.zeros(1), "qvel": torch.zeros(1)}
    def step(self, *args, **kwargs): pass

deploy_utils.robot_config.create_real_robot = lambda: MockRealAgent()

from deploy_utils.manipulator import LeRobotRealAgent
class MockLeRobotRealAgent:
    def __init__(self, *args, **kwargs): pass
    def reset(self, *args, **kwargs): pass
    def get_proprioception(self):
        return {"qpos": torch.zeros(1, 6), "qvel": torch.zeros(1, 6)}
    def capture_sensor_data(self): pass
    def get_sensor_data(self):
        return {"base_camera": {"rgb": torch.zeros(1, 128, 128, 3, dtype=torch.uint8)}}
    def step(self, action): pass

import deploy_utils.manipulator
deploy_utils.manipulator.LeRobotRealAgent = MockLeRobotRealAgent

from envs.seperate_instruments import SeparateInstrumentsEnv
from mani_skill.envs.sim2real_env import Sim2RealEnv
from mani_skill.utils.wrappers.flatten import FlattenRGBDObservationWrapper

env = gym.make(
    "SeparateInstruments-v1",
    obs_mode="rgb",
    render_mode="sensors",
    control_mode="pd_joint_target_delta_pos",
)
env = FlattenRGBDObservationWrapper(env, rgb=True, depth=False, state=True)

real_agent = MockLeRobotRealAgent()

def real_reset_function(env, seed, options):
    print("MOCK RESET CALLED")
    return env.get_obs(), {"reconfigure": False}

real_env = Sim2RealEnv(
    env,
    real_agent=real_agent,
    real_reset_function=real_reset_function
)

real_obs, _ = real_env.reset()
print("RESET OK")
action = torch.zeros(env.action_space.shape)
real_obs, _, terminated, truncated, info = real_env.step(action)
print("STEP OK")
