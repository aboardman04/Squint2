import argparse
import numpy as np
import cv2
import torch
import gymnasium as gym
import envs
from mani_skill.utils.wrappers.flatten import FlattenRGBDObservationWrapper

def main():
    parser = argparse.ArgumentParser(description="Replay a recorded deployment trajectory in the simulator.")
    parser.add_argument("--rec_name", type=str, required=True, help="Name of the recorded file to replay (e.g. 'test_0' or 'recorded_trajectories/test_0.npz')")
    parser.add_argument("--env_id", type=str, default="SO101LiftCube-v1")
    args = parser.parse_args()

    # Automatically resolve path if the user just provides the name
    traj_path = args.rec_name
    if not traj_path.endswith(".npz"):
        traj_path += ".npz"
    if not "/" in traj_path and not "\\" in traj_path:
        traj_path = f"recorded_trajectories/{traj_path}"

    print(f"Loading trajectory from {traj_path}...")
    data = np.load(traj_path)
    rgbs = data['rgb']
    qposes = data['qpos']
    num_frames = len(rgbs)
    print(f"Loaded {num_frames} frames.")

    print(f"Initializing simulation environment {args.env_id}...")
    env = gym.make(
        args.env_id,
        obs_mode="rgb+segmentation",
        render_mode="rgb_array",
        domain_randomization=False,
        domain_randomization_config={"apply_overlay": False},
        sensor_configs=dict(width=256, height=256)
    )
    env = FlattenRGBDObservationWrapper(env, rgb=True, depth=False, state=True)
    env.reset(seed=0)

    idx = 0
    paused = False
    
    print("\nControls:")
    print("  [Space] : Play / Pause")
    print("  [d]     : Step forward one frame (when paused)")
    print("  [a]     : Step backward one frame (when paused)")
    print("  [q]     : Quit")

    while True:
        real_rgb = rgbs[idx]
        qpos = qposes[idx]

        # Snap the ghost robot to the recorded joint positions
        env.unwrapped.agent.robot.set_qpos(qpos)
        
        # Step physics with 0 action just to update all visual transforms and cameras
        env.step(np.zeros_like(env.unwrapped.single_action_space.sample()))
        
        # Grab Sim Wrist Camera
        sim_obs = env.unwrapped.get_obs()
        sim_wrist = None
        for cam_key in ["base_camera", "wrist", "arm"]:
            if cam_key in sim_obs.get("sensor_data", {}):
                sim_wrist = sim_obs["sensor_data"][cam_key]["rgb"][0].cpu().numpy()
                break
        
        # Grab Sim 3rd Person POV
        sim_3rd = env.render()
        if isinstance(sim_3rd, torch.Tensor):
            sim_3rd = sim_3rd.cpu().numpy()
        if sim_3rd.ndim == 4:
            sim_3rd = sim_3rd[0]

        # Process images for display
        sim_wrist = cv2.cvtColor(sim_wrist, cv2.COLOR_RGB2BGR)
        sim_3rd = cv2.cvtColor(sim_3rd, cv2.COLOR_RGB2BGR)
        # Note: real_rgb is already BGR from the recording script

        # Resize for a nice 2x2 grid
        h, w = 400, 400
        real_disp = cv2.resize(real_rgb, (w, h), interpolation=cv2.INTER_NEAREST)
        sim_wrist_disp = cv2.resize(sim_wrist, (w, h), interpolation=cv2.INTER_NEAREST)
        sim_3rd_disp = cv2.resize(sim_3rd, (w, h), interpolation=cv2.INTER_NEAREST)
        
        # Create a blank panel for the 4th slot, or put some text there
        info_panel = np.zeros((h, w, 3), dtype=np.uint8)
        cv2.putText(info_panel, f"Frame: {idx+1}/{num_frames}", (20, 50), cv2.FONT_HERSHEY_SIMPLEX, 1, (255, 255, 255), 2)
        cv2.putText(info_panel, f"State: {'PAUSED' if paused else 'PLAYING'}", (20, 100), cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 0, 255) if paused else (0, 255, 0), 2)
        
        cv2.putText(real_disp, "Real Camera", (10, 30), cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 255, 0), 2)
        cv2.putText(sim_wrist_disp, "Sim Wrist", (10, 30), cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 255, 0), 2)
        cv2.putText(sim_3rd_disp, "Sim 3rd Person", (10, 30), cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 255, 0), 2)

        top_row = np.hstack((real_disp, sim_wrist_disp))
        bottom_row = np.hstack((info_panel, sim_3rd_disp))
        grid = np.vstack((top_row, bottom_row))

        cv2.imshow("Offline Replay Debugger", grid)

        # Handle keyboard input
        delay = 0 if paused else 100  # 100ms delay = ~10 FPS (matching control_freq)
        key = cv2.waitKey(delay) & 0xFF

        if key == ord('q') or key == 27: # q or ESC
            break
        elif key == ord(' '): # Spacebar
            paused = not paused
        elif key == ord('d') and paused:
            idx = min(idx + 1, num_frames - 1)
        elif key == ord('a') and paused:
            idx = max(idx - 1, 0)
        
        if not paused:
            idx += 1
            if idx >= num_frames:
                idx = num_frames - 1
                paused = True # Auto-pause at the end

    cv2.destroyAllWindows()

if __name__ == "__main__":
    main()
