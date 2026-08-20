import time
import numpy as np
import torch
import cv2
import gymnasium as gym

# Import ManiSkill environment
import envs
import mani_skill.envs
from mani_skill.utils.structs.types import Array

# Import LeRobot
from lerobot.robots.robot import Robot
from lerobot.robots.so_follower.config_so_follower import SO101FollowerConfig
from lerobot.robots.utils import make_robot_from_config
from lerobot.cameras.opencv.configuration_opencv import OpenCVCameraConfig

# NOTE: Since LeRobot doesn't have a distinct SO101LeaderConfig, 
# you typically use the FollowerConfig for the leader but on a different port, 
# or whichever config you used in your dataset collection.
from lerobot.robots.so_follower.config_so_follower import SO101FollowerConfig as LeaderConfig

from pathlib import Path

def create_teleop_robots():
    """Create the leader and follower robots."""
    
    # --- CONFIGURE YOUR LEADER ---
    leader_config = LeaderConfig(
        port="/dev/ttyACM1",  # CHANGE TO YOUR LEADER PORT
        use_degrees=True,
        # Leader usually doesn't need a camera for teleop
        cameras={},
        id="tinkering_tilda",
        # Set the calibration directory to where your leader calibration is
        calibration_dir=Path("/home/aboardman/.cache/huggingface/lerobot/calibration/teleoperators/so_leader/")
    )
    leader = make_robot_from_config(leader_config)

    # --- CONFIGURE YOUR FOLLOWER ---
    follower_config = SO101FollowerConfig(
        port="/dev/ttyACM0",  # CHANGE TO YOUR FOLLOWER PORT
        use_degrees=True,
        cameras={"base_camera": OpenCVCameraConfig(
            index_or_path="/dev/video4",  # CHANGE TO YOUR WEBCAM
            fps=30,
            width=640,
            height=480
        )},
        id="mind_blowing_mandy",
        calibration_dir=Path("/home/aboardman/.cache/huggingface/lerobot/calibration/robots/so_follower/")
    )
    follower = make_robot_from_config(follower_config)

    return leader, follower


def convert_real_to_sim(qpos_deg_dict):
    """
    Inverse of `convert_state_for_lerobot` in collect_sim_data.py.
    This exactly mirrors deploy_utils/manipulator.py logic.
    """
    # 1. Map dict to list in the correct order for ManiSkill
    # WARNING: This assumes ManiSkill's joint order is:
    # pan, lift, elbow, wrist_flex, wrist_roll, gripper
    order = [
        "shoulder_pan",
        "shoulder_lift",
        "elbow_flex",
        "wrist_flex",
        "wrist_roll",
        "gripper"
    ]
    
    qpos_deg = []
    for joint in order:
        val = qpos_deg_dict.get(joint)
        if val is None:
            # Fallback if names are slightly different (e.g. they have '.pos')
            val = qpos_deg_dict.get(f"{joint}.pos", 0.0)
        qpos_deg.append(val)
        
    qpos_deg = np.array(qpos_deg, dtype=np.float32)

    # 2. Convert Gripper from servo degrees to sim degrees
    _gripper_sim_min = -10.0
    _gripper_sim_max = 120.0
    _gripper_servo_min = -62.5
    _gripper_servo_max = 64.62
    _gripper_sim_range = _gripper_sim_max - _gripper_sim_min
    _gripper_servo_range = _gripper_servo_max - _gripper_servo_min

    servo_val = qpos_deg[5]
    qpos_deg[5] = (servo_val - _gripper_servo_min) / _gripper_servo_range * _gripper_sim_range + _gripper_sim_min

    # 3. Convert all from degrees to radians
    qpos_rad = np.deg2rad(qpos_deg)
    
    return qpos_rad


def main():
    print("Initializing environment...")
    # Initialize the simulation environment used in training
    env = gym.make(
        "LiftInstruments-v3",
        num_envs=1,
        obs_mode="rgb",
        control_mode="pd_joint_pos",
        render_mode="human" # This will open a window showing the digital twin
    )
    env.reset()

    print("Initializing physical robots...")
    leader, follower = create_teleop_robots()
    
    leader.connect()
    follower.connect()

    print("\nStarting Teleop Digital Twin! Press Ctrl+C to exit.")
    
    try:
        while True:
            # 1. Read Leader
            leader_obs = leader.get_observation()
            
            # The observation state is usually a dictionary of motor positions or a flat array.
            # Assuming it's a flat array or dictionary. If array, convert to dict:
            # (Adjust mapping based on what leader.get_observation() actually returns)
            if "observation.state" in leader_obs:
                target_state = leader_obs["observation.state"]
                # 2. Command Follower (teleop)
                follower.send_action(target_state)

            # 3. Read Follower
            # Instead of using get_observation, we can read directly from the bus for raw degrees
            follower_qpos_dict = follower.bus.sync_read("Present_Position")
            
            # 4. Convert Real (Degrees) to Sim (Radians)
            sim_qpos = convert_real_to_sim(follower_qpos_dict)
            
            # 5. Update Simulation Robot
            # Force the simulation robot to strictly mirror the physical follower
            env.unwrapped.agent.robot.set_qpos(sim_qpos)
            
            # Print the joint angles for terminal debugging
            names = ["pan", "lift", "elbow", "w_flex", "w_roll", "grip"]
            display_str = " | ".join([f"{n}: {deg:6.1f}°" for n, deg in zip(names, np.rad2deg(sim_qpos))])
            print(f"\r{display_str}", end="")
            
            # 6. Render the Digital Twin
            env.render()
            
            # Moderate frequency (e.g., 30 Hz)
            time.sleep(1/30.0)

    except KeyboardInterrupt:
        print("\nExiting...")
    finally:
        leader.disconnect()
        follower.disconnect()
        env.close()

if __name__ == "__main__":
    main()
