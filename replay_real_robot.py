import time
import argparse
from pathlib import Path
import torch

from lerobot.datasets.lerobot_dataset import LeRobotDataset
from lerobot.robots.so_follower.config_so_follower import SO101FollowerConfig
from lerobot.robots.utils import make_robot_from_config

def main():
    parser = argparse.ArgumentParser(description="Replay a dataset episode on the real robot.")
    parser.add_argument("--repo_id", type=str, required=True, help="Dataset repo id (e.g. aboardman/combined_instrument_dataset_4-6-7)")
    parser.add_argument("--episode", type=int, default=0, help="Episode index to replay")
    parser.add_argument("--fps", type=float, default=30.0, help="Playback speed (frames per second)")
    parser.add_argument("--port", type=str, default="/dev/ttyACM0", help="Follower port")
    args = parser.parse_args()

    print(f"Loading dataset: {args.repo_id}...")
    dataset = LeRobotDataset(args.repo_id)
    
    # Check if episode is valid
    if args.episode >= dataset.num_episodes:
        print(f"Error: Dataset only has {dataset.num_episodes} episodes.")
        return
        
    print(f"Connecting to physical Follower robot on {args.port}...")
    follower_config = SO101FollowerConfig(
        port=args.port,
        use_degrees=True,
        cameras={}, # No cameras needed just to move the motors
        id="mind_blowing_mandy",
        calibration_dir=Path("/home/aboardman/.cache/huggingface/lerobot/calibration/robots/so_follower/")
    )
    follower = make_robot_from_config(follower_config)
    follower.connect()

    print("\nRobot connected! Saving initial position...")
    
    # Save the absolute initial position of the robot to return to later
    initial_pos_dict = follower.bus.sync_read("Present_Position")
    while initial_pos_dict is None:
        time.sleep(0.1)
        initial_pos_dict = follower.bus.sync_read("Present_Position")
        
    initial_pos = [
        initial_pos_dict["shoulder_pan"],
        initial_pos_dict["shoulder_lift"],
        initial_pos_dict["elbow_flex"],
        initial_pos_dict["wrist_flex"],
        initial_pos_dict["wrist_roll"],
        initial_pos_dict["gripper"]
    ]
    
    try:
        # Get the start and end indices of the requested episode
        episode_meta = dataset.meta.episodes[args.episode]
        start_idx = episode_meta["dataset_from_index"]
        end_idx = episode_meta["dataset_to_index"]
        num_frames = end_idx - start_idx
        
        print(f"Playing Episode {args.episode} ({num_frames} frames) at {args.fps} FPS.")
        
        # 1. Move to the first frame smoothly by interpolating
        # Access the underlying hf_dataset directly to avoid loading video frames (which triggers torchcodec)
        first_frame = dataset.hf_dataset[start_idx]
        first_action = first_frame["action"]
        if torch.is_tensor(first_action):
            first_action = first_action.numpy()
        if first_action.ndim > 1:
            first_action = first_action[0]
            
        print("Interpolating safely to the start position...")
        
        # We already fetched initial_pos, so we just copy it for the first interpolation
        current_pos = list(initial_pos)
        
        # Take 3 seconds (90 steps) to move to the starting position
        steps = 90
        motor_names = [
            "shoulder_pan",
            "shoulder_lift",
            "elbow_flex",
            "wrist_flex",
            "wrist_roll",
            "gripper"
        ]
        
        for step in range(steps):
            alpha = (step + 1) / steps
            interpolated_action = {}
            for i, name in enumerate(motor_names):
                val = current_pos[i] + alpha * (first_action[i] - current_pos[i])
                interpolated_action[f"{name}.pos"] = float(val)
                
            follower.send_action(interpolated_action)
            time.sleep(1/30.0)
            
        time.sleep(1.0)
        
        print("\nStarting playback! Press Ctrl+C to abort.")
        
        # 2. Play the trajectory
        sleep_time = 1.0 / args.fps
        for i in range(start_idx, end_idx):
            frame = dataset.hf_dataset[i]
            
            # The 'action' column is a tensor of shape (1, 6) or (6,)
            action = frame["action"]
            if torch.is_tensor(action):
                action = action.numpy()
            if action.ndim > 1:
                action = action[0]
                
            action_dict = {f"{name}.pos": float(action[j]) for j, name in enumerate(motor_names)}
                
            follower.send_action(action_dict)
            time.sleep(sleep_time)
            
            # Simple progress bar
            progress = int(20 * (i - start_idx) / num_frames)
            print(f"\rReplaying: [{'='*progress}{' '*(20-progress)}] {i-start_idx}/{num_frames}", end="")
            
        print("\n\nPlayback complete! Returning to initial position...")
        
        # 3. Return safely to the initial resting position
        current_pos_dict = follower.bus.sync_read("Present_Position")
        while current_pos_dict is None:
            time.sleep(0.1)
            current_pos_dict = follower.bus.sync_read("Present_Position")
            
        current_pos = [
            current_pos_dict["shoulder_pan"],
            current_pos_dict["shoulder_lift"],
            current_pos_dict["elbow_flex"],
            current_pos_dict["wrist_flex"],
            current_pos_dict["wrist_roll"],
            current_pos_dict["gripper"]
        ]
        
        for step in range(steps):
            alpha = (step + 1) / steps
            interpolated_action = {}
            for i, name in enumerate(motor_names):
                val = current_pos[i] + alpha * (initial_pos[i] - current_pos[i])
                interpolated_action[f"{name}.pos"] = float(val)
                
            follower.send_action(interpolated_action)
            time.sleep(1/30.0)
            
        print("Robot has returned to its initial position.")
        
    except KeyboardInterrupt:
        print("\n\nPlayback aborted by user.")
    except Exception as e:
        print(f"\n\nError during playback: {e}")
    finally:
        print("Disconnecting robot...")
        follower.disconnect()

if __name__ == "__main__":
    main()
