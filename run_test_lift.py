import gymnasium as gym
import mani_skill.envs
from envs.lift import LiftCube 

env = gym.make("SO101LiftCube-v1", render_mode="human")

obs, info = env.reset()

print("Viewer running. Press Ctrl+C in terminal or close the window to exit.")
try:
    while True:
        # Sample a random joint movement 
        action = env.action_space.sample()
        obs, reward, terminated, truncated, info = env.step(action)

        # Check if the human viewer window was manually closed
        if env.unwrapped.viewer is not None and env.unwrapped.viewer.closed:
            break
        env.render()

except KeyboardInterrupt:
    print("Closing environment...")

env.close()

