import gymnasium as gym
import numpy as np
import time

# 1. Import mani_skill to ensure ManiSkill's environment registry is loaded
import mani_skill.envs
from mani_skill.utils.registration import register_env

# 2. Import your custom environment module so Python registers "@register_env"
# Adjust this import path if your file is located elsewhere (e.g., from envs.separate_instruments import SeparateInstrumentsEnv)
try:
    # This registers your "SeparateInstruments-v2.5" class
    import envs
    print("Successfully imported custom envs package!")
except ImportError as e:
    print("Could not import custom envs package automatically. Error:", e)
    print("Make sure you run this script from the root of your squint2 directory.")

def main():
    # The registered name of your separate instruments environment
    env_id = "SeparateInstruments-v2.5"
    
    print(f"Creating environment: {env_id}...")
    
    # Create the environment.
    # Setting render_mode="human" forces ManiSkill to spawn the interactive Sapien GUI.
    try:
        env = gym.make(
            env_id,
            render_mode="human",
            obs_mode="rgb",  # Ensure camera visual outputs are initialized
            control_mode="pd_joint_delta_pos"  # Standard robot arm control mode
        )
    except Exception as e:
        print(f"Failed to load environment '{env_id}'. Error details:\n", e)
        print("\nChecking registered environments containing 'Separate':")
        from mani_skill.utils.registration import REGISTERED_ENVS
        for name in REGISTERED_ENVS.keys():
            if "Separate" in name or "separate" in name.lower():
                print(f" - {name}")
        return

    # Reset the environment to place the robot and spawn the instruments
    obs, info = env.reset()
    print("Environment successfully loaded and reset!")
    print(f"Observation keys: {obs.keys() if hasattr(obs, 'keys') else 'Array'}")

    print("\n--- Visualizer Running ---")
    print("Click inside the simulator GUI window to interact with the 3D scene.")
    print("Press 'q' or close the window to exit.")

    try:
        # Run a loop to keep the simulator window open and moving
        for step in range(1000):
            # Sample a completely random action from the robot's action space
            action = env.action_space.sample()
            
            # Step the environment forward with the random action
            obs, reward, terminated, truncated, info = env.step(action)
            
            # Render the environment (ManiSkill updates the human GUI viewer automatically here)
            try:
                env.render()
            except AttributeError:
                print("Viewer window closed or not available.")
                break
            
            # Slightly slow down execution so you can visually parse the actions
            time.sleep(0.05)
            
            if terminated or truncated:
                print(f"Episode ended at step {step}. Resetting environment...")
                env.reset()
                
    except KeyboardInterrupt:
        print("\nStopping simulation test loop.")
    finally:
        # Always close the environment to safely release memory and visual contexts
        env.close()
        print("Environment closed.")

if __name__ == "__main__":
    main()