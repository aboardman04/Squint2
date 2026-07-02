import gymnasium as gym
import mani_skill.envs
from envs.seperate_instruments import SeparateInstrumentsEnv

env = gym.make("SeparateInstruments-v1")
obs, _ = env.reset()
print("Reset successful!")
