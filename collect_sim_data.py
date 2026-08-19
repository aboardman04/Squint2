import os
import sys
import argparse
import random
import warnings
import logging

import numpy as np
import torch
import gymnasium as gym

# ================================================================
# USER SETTINGS
# ================================================================

# Environment
ENV_ID = "LiftInstruments-v3"

# Trained Squint checkpoint
CHECKPOINT = "runs/lift_instruments_5.2/ckpt.pt"

# Number of SUCCESSFUL episodes to collect
NUM_SUCCESSFUL_EPISODES = 5

MAX_STEPS = 20

# Hugging Face dataset name
REPO_ID = "aboardman/so101_separate_instruments_sim_0.2"

# Set True only after verifying the local dataset
PUSH_TO_HUB = True

# Task description used in the dataset
TASK_DESCRIPTION = "Separate the surgical instruments"

# Camera settings -- match your real SO101 dataset
CAMERA_WIDTH = 640
CAMERA_HEIGHT = 480
DATASET_FPS = 30

# Name used by the real LeRobot dataset
IMAGE_FEATURE_NAME = "observation.images.arm"

# SO101 joint ordering used by the real dataset
SO101_JOINT_NAMES = [
    "shoulder_pan.pos",
    "shoulder_lift.pos",
    "elbow_flex.pos",
    "wrist_flex.pos",
    "wrist_roll.pos",
    "gripper.pos",
]

# Squint policy input size -- DO NOT change unless the policy was
# trained with a different size.
POLICY_IMAGE_SIZE = 16

# These should match the environment used during RL training
DOMAIN_RANDOMIZATION = True
RECONFIGURATION_FREQ = None

# Starting seed. Each episode gets a different seed.
BASE_SEED = 1000

# ================================================================
# SETUP
# ================================================================

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
os.environ["MKL_SERVICE_FORCE_INTEL"] = "1"
warnings.filterwarnings("ignore", category=DeprecationWarning)
logging.disable(level=logging.WARN)

from mani_skill.utils.wrappers.flatten import FlattenRGBDObservationWrapper
import envs
import mani_skill.envs
from train_squint import DeployAgent
from lerobot.datasets.lerobot_dataset import LeRobotDataset


# ================================================================
# SIMULATION DATA EXTRACTION
# ================================================================

def get_arm_image(obs):
    """Return the 640x480 arm-camera RGB image."""
    rgb = obs["rgb"]
    if torch.is_tensor(rgb):
        rgb = rgb.detach().cpu().numpy()
    if rgb.ndim == 4:
        rgb = rgb[0]

    expected = (CAMERA_HEIGHT, CAMERA_WIDTH, 3)
    if rgb.shape != expected:
        raise RuntimeError(
            f"Unexpected camera shape {rgb.shape}; expected {expected}. "
            "Modify get_arm_image() if your environment exposes cameras differently."
        )

    return rgb.astype(np.uint8)


def get_sim_state(env):
    """Return the six simulated SO101 joint positions."""
    qpos = env.unwrapped.agent.robot.get_qpos()
    if torch.is_tensor(qpos):
        qpos = qpos.detach().cpu().numpy()
    if qpos.ndim > 1:
        qpos = qpos[0]

    qpos = np.asarray(qpos, dtype=np.float32)

    if qpos.shape != (6,):
        raise RuntimeError(f"Expected 6 joints, got {qpos.shape}")

    return qpos


# ================================================================
# SIMULATION -> REAL DATA CONVERSION
# ================================================================
#
# These are the functions we will modify after verifying the
# ManiSkill SO101 state/action representation against your
# real LeRobot dataset.
# ================================================================

def convert_state_for_lerobot(sim_qpos):
    """Convert simulated qpos to the real dataset representation."""
    # Convert from radians to degrees
    state = np.rad2deg(np.asarray(sim_qpos, dtype=np.float32).copy())

    # Apply gripper mapping for SO101 (sim degrees -> servo degrees)
    _gripper_sim_min = -10.0
    _gripper_sim_max = 120.0
    _gripper_servo_min = -62.5
    _gripper_servo_max = 64.62
    _gripper_sim_range = _gripper_sim_max - _gripper_sim_min
    _gripper_servo_range = _gripper_servo_max - _gripper_servo_min

    sim_deg = state[5]
    state[5] = (sim_deg - _gripper_sim_min) / _gripper_sim_range * _gripper_servo_range + _gripper_servo_min

    return state


def convert_action_for_lerobot(sim_action, current_sim_qpos):
    """Convert Squint action to the real dataset representation."""
    action = np.asarray(sim_action, dtype=np.float32)
    if action.ndim > 1:
        action = action[0]

    if action.shape != (6,):
        raise RuntimeError(f"Expected 6 actions, got {action.shape}")

    # Convert from radians to degrees
    action_deg = np.rad2deg(action.copy())

    # Apply gripper mapping for SO101 (sim degrees -> servo degrees)
    _gripper_sim_min = -10.0
    _gripper_sim_max = 120.0
    _gripper_servo_min = -62.5
    _gripper_servo_max = 64.62
    _gripper_sim_range = _gripper_sim_max - _gripper_sim_min
    _gripper_servo_range = _gripper_servo_max - _gripper_servo_min

    sim_deg = action_deg[5]
    action_deg[5] = (sim_deg - _gripper_sim_min) / _gripper_sim_range * _gripper_servo_range + _gripper_servo_min
    
    return action_deg


# ================================================================
# SUCCESS
# ================================================================

def episode_is_successful(info):
    """Return the environment success flag."""
    if "success" in info:
        success = info["success"]
    elif "is_success" in info:
        success = info["is_success"]
    else:
        raise RuntimeError(
            "No success signal found in info. "
            "Expected info['success']."
        )

    if torch.is_tensor(success):
        success = success.detach().cpu().numpy()

    return bool(np.asarray(success).reshape(-1)[0])


# ================================================================
# LEROBOT DATASET
# ================================================================

def build_dataset_features():
    """Define the simulation dataset schema."""

    return {
        "observation.images.arm": {
            "dtype": "video",
            "shape": (CAMERA_HEIGHT, CAMERA_WIDTH, 3),
            "names": ["height", "width", "channel"],
        },
        "observation.state": {
            "dtype": "float32",
            "shape": (6,),
            "names": SO101_JOINT_NAMES,
        },
        "action": {
            "dtype": "float32",
            "shape": (6,),
            "names": SO101_JOINT_NAMES,
        },
    }


def build_frame(image, state, action):
    """Create one LeRobot timestep."""
    return {
        "observation.images.arm": image,
        "observation.state": state,
        "action": action,
        "task": TASK_DESCRIPTION,
    }


# ================================================================
# ENVIRONMENT
# ================================================================

def make_environment():
    """Create the simulation environment using training settings."""
    env = gym.make(
        ENV_ID,
        obs_mode="rgb+segmentation",
        render_mode="rgb_array",
        sim_backend="gpu",
        domain_randomization=DOMAIN_RANDOMIZATION,
        reconfiguration_freq=RECONFIGURATION_FREQ,
        sensor_configs={
            "width": CAMERA_WIDTH,
            "height": CAMERA_HEIGHT,
        },
        human_render_camera_configs={
            "width": CAMERA_WIDTH,
            "height": CAMERA_HEIGHT,
        },
        num_envs=1,
    )

    return FlattenRGBDObservationWrapper(
        env,
        rgb=True,
        depth=False,
        state=True,
    )


# ================================================================
# POLICY
# ================================================================

def load_policy(env, obs, device):
    """Load the trained Squint policy."""
    agent = DeployAgent(
        env,
        obs,
        target_image_size=POLICY_IMAGE_SIZE,
        device=device,
    )
    agent.load_checkpoint(CHECKPOINT)
    agent.eval()
    return agent


# ================================================================
# COLLECT ONE EPISODE
# ================================================================

def collect_episode(env, agent, dataset, seed, device):
    """Run one episode and save it only if successful."""

    print(f"\nEpisode seed: {seed}")

    obs, info = env.reset(seed=seed)
    success = False
    max_steps = MAX_STEPS

    if max_steps is None:
        raise RuntimeError("Environment has no max_episode_steps.")

    for step in range(max_steps):

        # Current observation
        image = get_arm_image(obs)
        sim_state = get_sim_state(env)
        state = convert_state_for_lerobot(sim_state)

        # Policy action
        policy_obs = {
            "rgb": obs["rgb"].to(device),
            "state": obs["state"].to(device),
        }

        with torch.no_grad():
            policy_action = agent(policy_obs)

        policy_action_np = (
            policy_action.detach().cpu().numpy()
        )

        # Convert action for dataset
        action = convert_action_for_lerobot(
            policy_action_np,
            sim_state,
        )

        # Record current state/action/image
        dataset.add_frame(
            build_frame(
                image=image,
                state=state,
                action=action,
            )
        )

        # Execute policy
        obs, reward, terminated, truncated, info = env.step(
            policy_action_np
        )

        success = episode_is_successful(info)

        if torch.is_tensor(terminated):
            terminated = terminated.detach().cpu().numpy()

        if torch.is_tensor(truncated):
            truncated = truncated.detach().cpu().numpy()

        terminated = bool(np.asarray(terminated).reshape(-1)[0])
        truncated = bool(np.asarray(truncated).reshape(-1)[0])

        if success:
            print(f"SUCCESS at step {step + 1}")
            break

        if terminated or truncated:
            break

    if success:
        dataset.save_episode()
        print(f"Saved episode ({step + 1} frames)")
        return True

    dataset.clear_episode_buffer()
    print(f"Discarded failed episode ({step + 1} frames)")
    return False


# ================================================================
# MAIN
# ================================================================

def main():

    device = torch.device(
        "cuda" if torch.cuda.is_available() else "cpu"
    )

    print("\n======================================")
    print("SQUINT SIMULATION DATA COLLECTION")
    print("======================================")
    print(f"Environment:    {ENV_ID}")
    print(f"Checkpoint:     {CHECKPOINT}")
    print(f"Camera:         {CAMERA_WIDTH}x{CAMERA_HEIGHT}")
    print(f"Policy input:   {POLICY_IMAGE_SIZE}x{POLICY_IMAGE_SIZE}")
    print(f"FPS:            {DATASET_FPS}")
    print(f"Randomization:  {DOMAIN_RANDOMIZATION}")
    print(f"Target episodes:{NUM_SUCCESSFUL_EPISODES}")
    print(f"Dataset:        {REPO_ID}")
    print(f"Device:         {device}")

    random.seed(BASE_SEED)
    np.random.seed(BASE_SEED)
    torch.manual_seed(BASE_SEED)

    env = make_environment()

    obs, info = env.reset(seed=BASE_SEED)
    agent = load_policy(env, obs, device)

    dataset = LeRobotDataset.create(
        repo_id=REPO_ID,
        fps=DATASET_FPS,
        robot_type="so101_follower",
        features=build_dataset_features(),
        use_videos=True,
        # streaming_encoding=False,
        image_writer_threads=2,
    )

    successful = 0
    attempted = 0

    try:

        while successful < NUM_SUCCESSFUL_EPISODES:

            seed = BASE_SEED + attempted
            attempted += 1

            if collect_episode(
                env,
                agent,
                dataset,
                seed,
                device,
            ):
                successful += 1

            print(
                f"Progress: {successful}/"
                f"{NUM_SUCCESSFUL_EPISODES} successful "
                f"({attempted} attempts)"
            )

            if successful < NUM_SUCCESSFUL_EPISODES:
                env.reset(seed=BASE_SEED + attempted)

        dataset.finalize()

        print("\n======================================")
        print("COLLECTION COMPLETE")
        print("======================================")
        print(f"Successful: {successful}")
        print(f"Attempts:   {attempted}")
        print(
            f"Success rate: "
            f"{100 * successful / attempted:.2f}%"
        )

        if PUSH_TO_HUB:
            print("\nUploading dataset...")
            dataset.push_to_hub()
            print("Upload complete.")

    finally:
        env.close()


if __name__ == "__main__":
    main()