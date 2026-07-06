import gymnasium as gym
import envs.seperate_instruments
env = gym.make("SeparateInstruments-v1", obs_mode="state")
env.reset()
env.step(env.action_space.sample())
print("Step success!")
