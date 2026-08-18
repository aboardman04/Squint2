"""
Collect successful SO101 simulation demonstrations using a trained Squint policy.

IMPORTANT DESIGN:

    SAPIEN CAMERA
         |
         | 640 x 480 RGB
         |
         +-----------------------> DATASET
         |                           |
         |                           +--> observation.images.arm
         |                           +--> observation.state
         |                           +--> action
         |
         +--> DeployAgent
                  |
                  +--> downsamples to 16 x 16
                  |
                  +--> trained Squint policy
                  |
                  +--> action

The RL policy therefore receives the SAME size input it was trained with,
while the dataset receives the full-resolution camera image.

Failed episodes are discarded.
Successful episodes are written to a LeRobotDataset.

Before collecting thousands of episodes, run:

    --num_successful_episodes 5

and inspect the resulting dataset.
"""

import os
import sys
import argparse
import random
import warnings
import logging

import numpy as np
import torch
import gymnasium as gym

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

os.environ["MKL_SERVICE_FORCE_INTEL"] = "1"

warnings.filterwarnings("ignore", category=DeprecationWarning)
logging.disable(level=logging.WARN)
from mani_skill.utils.wrappers.flatten import FlattenRGBDObservationWrapper
import envs
import mani_skill.envs
from train_squint import DeployAgent
from lerobot.datasets.lerobot_dataset import LeRobotDataset

TASK_DESCRIPTION = "Lift the surgical instrument"
DATASET_FPS = 30
CAMERA_WIDTH = 640
CAMERA_HEIGHT = 480
IMAGE_FEATURE_NAME = "observation.images.arm"
STATE_FEATURE_NAME = "observation.state"
ACTION_FEATURE_NAME = "action"
JOINT_NAMES = [
    "shoulder_pan.pos",
    "shoulder_lift.pos",
    "elbow_flex.pos",
    "wrist_flex.pos",
    "wrist_roll.pos",
    "gripper.pos",
]
POLICY_IMAGE_SIZE = 16
NUM_SUCCESSFUL_EPISODES = 10
MAX_STEPS = None
TRAINING_DOMAIN_RANDOMIZATION = True
TRAINING_RECONFIGURATION_FREQ = None
BASE_SEED = 1000

# UPLOAD
# ------------------------------------------------
# Start with False.
# Once you have inspected the local dataset and verified that everything is correct, set this True or use --push_to_hub.

DEFAULT_PUSH_TO_HUB = False
DEFAULT_REPO_ID = "YOUR_USERNAME/so101-instrument-sim"
# OPTIONAL VIDEO PREVIEW (Keep False for large-scale collection)
SAVE_PREVIEW_VIDEO = False


def get_dataset_image(env):
    """
    Get the FULL-RESOLUTION camera image for the dataset.

    IMPORTANT:
        This is NOT the image fed to the Squint policy.

    The environment is configured at 640x480, so obs["rgb"]
    should contain the full-resolution camera observation.

    Returns:
        np.ndarray with shape (480, 640, 3), uint8
    """

    rgb = env.unwrapped.get_obs()["rgb"]

    # In case the environment returns a batch dimension.
    if rgb.ndim == 4:
        rgb = rgb[0]

    # Convert Torch -> NumPy.
    if torch.is_tensor(rgb):
        rgb = rgb.detach().cpu().numpy()

    rgb = np.asarray(rgb)

    if rgb.shape != (CAMERA_HEIGHT, CAMERA_WIDTH, 3):
        raise RuntimeError(
            f"Unexpected camera image shape: {rgb.shape}. "
            f"Expected {(CAMERA_HEIGHT, CAMERA_WIDTH, 3)}."
        )

    if rgb.dtype != np.uint8:
        rgb = rgb.astype(np.uint8)

    return rgb


def get_dataset_state(env):
    """
    Get the CLEAN SO101 joint positions.

    We intentionally do NOT use:

        obs["state"]

    because that state may contain additional information
    used by Squint.

    We also do NOT use noisy_qpos.

    Instead we directly read the simulated robot qpos.

    Returns:
        np.ndarray shape (N_JOINTS,), float32
    """

    qpos = env.unwrapped.agent.robot.get_qpos()

    if torch.is_tensor(qpos):
        qpos = qpos.detach().cpu().numpy()

    qpos = np.asarray(qpos)

    # Remove environment batch dimension.
    if qpos.ndim > 1:
        qpos = qpos[0]

    qpos = qpos.astype(np.float32)

    return qpos


def format_action_for_dataset(action, env):
    """
    Convert the Squint action into the action representation
    that will be stored in the LeRobot dataset.

    CURRENT VERSION:

        Store the raw action produced by Squint.

    IMPORTANT:
        This is probably the function we will modify after
        comparing the simulation action with your REAL
        LeRobot action representation.

    For example, if the real dataset stores absolute joint
    positions while Squint produces joint deltas, we can
    convert it here.

    Returns:
        np.ndarray shape (N_ACTIONS,), float32
    """

    action = np.asarray(action)

    # Remove environment batch dimension.
    if action.ndim > 1:
        action = action[0]

    return action.astype(np.float32)


def get_success(info):

    if "success" in info:
        success = info["success"]

    elif "is_success" in info:
        success = info["is_success"]

    else:
        return False

    if torch.is_tensor(success):
        success = success.detach().cpu().numpy()

    success = np.asarray(success)

    return bool(success.reshape(-1)[0])


def build_dataset_frame(
    image,
    state,
    action,
):

    frame = {
        IMAGE_FEATURE_NAME: image,
        STATE_FEATURE_NAME: state,
        ACTION_FEATURE_NAME: action,
        "task": TASK_DESCRIPTION,
    }

    return frame


def build_dataset_features(state_dim, action_dim):
    """
    Define the LeRobot dataset schema.

    THIS IS THE SECOND MAIN PLACE TO EDIT.

    The names/shapes here should eventually be made identical
    to your REAL SO101 dataset.
    """

    if len(JOINT_NAMES) != state_dim:
        raise ValueError(
            f"JOINT_NAMES contains {len(JOINT_NAMES)} names, "
            f"but simulation state has dimension {state_dim}."
        )

    features = {

        IMAGE_FEATURE_NAME: {
            "dtype": "video",

            # LeRobot accepts HWC image input.
            "shape": (
                CAMERA_HEIGHT,
                CAMERA_WIDTH,
                3,
            ),

            "names": [
                "height",
                "width",
                "channel",
            ],
        },

        STATE_FEATURE_NAME: {
            "dtype": "float32",
            "shape": (state_dim,),
            "names": JOINT_NAMES,
        },

        ACTION_FEATURE_NAME: {
            "dtype": "float32",
            "shape": (action_dim,),
            "names": JOINT_NAMES[:action_dim],
        },
    }

    return features

def make_environment(env_id):
    env = gym.make(
        env_id,
        obs_mode="rgb+segmentation",
        render_mode="rgb_array",
        sim_backend="gpu",
        domain_randomization=TRAINING_DOMAIN_RANDOMIZATION,
        reconfiguration_freq=TRAINING_RECONFIGURATION_FREQ,
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

    env = FlattenRGBDObservationWrapper(
        env,
        rgb=True,
        depth=False,
        state=True,
    )

    return env

def load_policy(env, obs, checkpoint, device):
    """
    Load the trained Squint policy.

    DeployAgent automatically downsamples the 640x480 RGB
    observation to POLICY_IMAGE_SIZE before passing it through
    the trained CNN.
    """

    print(
        f"Loading Squint policy from:\n"
        f"    {checkpoint}"
    )

    agent = DeployAgent(
        env,
        obs,
        target_image_size=POLICY_IMAGE_SIZE,
        device=device,
    )

    agent.load_checkpoint(checkpoint)

    agent.eval()

    return agent

def collect_episode(
    env,
    agent,
    dataset,
    seed,
    device,
):
    """
    Run one complete policy episode.

    Frames are temporarily added to the LeRobot episode buffer.

    If the episode succeeds:
        save_episode()

    If the episode fails:
        clear_episode_buffer()

    This means failed demonstrations never become part of the
    final dataset.
    """

    print(f"\nStarting episode with seed {seed}")

    obs, info = env.reset(seed=seed)

    episode_success = False
    episode_done = False

    if MAX_STEPS is not None:
        max_steps = MAX_STEPS
    else:
        # Fall back to the environment's configured horizon.
        max_steps = env.unwrapped.max_episode_steps

        if max_steps is None:
            raise RuntimeError(
                "MAX_STEPS is None and the environment does not "
                "provide max_episode_steps."
            )

    for step in range(max_steps):
        image = get_dataset_image(env)
        state = get_dataset_state(env)
        policy_obs = {
            "rgb": obs["rgb"].to(device),
            "state": obs["state"].to(device),
        }

        with torch.no_grad():
            action = agent(policy_obs)

        action_np = action.detach().cpu().numpy()

        dataset_action = format_action_for_dataset(
            action_np,
            env,
        )

        frame = build_dataset_frame(
            image=image,
            state=state,
            action=dataset_action,
        )

        dataset.add_frame(frame)

        obs, reward, terminated, truncated, info = env.step(
            action_np
        )

        episode_success = get_success(info)

        terminated_bool = bool(
            np.asarray(
                terminated.detach().cpu().numpy()
                if torch.is_tensor(terminated)
                else terminated
            ).reshape(-1)[0]
        )

        truncated_bool = bool(
            np.asarray(
                truncated.detach().cpu().numpy()
                if torch.is_tensor(truncated)
                else truncated
            ).reshape(-1)[0]
        )

        episode_done = terminated_bool or truncated_bool

        if episode_success:
            print(
                f"    SUCCESS at step {step + 1}"
            )
            break

        if episode_done:
            break

    if episode_success:

        dataset.save_episode()

        print(
            f"    SAVED successful episode "
            f"({step + 1} frames)"
        )

        return True

    else:

        dataset.clear_episode_buffer()

        print(
            f"    DISCARDED failed episode "
            f"({step + 1} frames)"
        )

        return False

def main():

    parser = argparse.ArgumentParser()

    parser.add_argument(
        "--env_id",
        type=str,
        default="LiftInstruments-v2",
    )

    parser.add_argument(
        "--checkpoint",
        type=str,
        required=True,
        help="Path to trained Squint ckpt.pt",
    )

    parser.add_argument(
        "--num_successful_episodes",
        type=int,
        default=NUM_SUCCESSFUL_EPISODES,
    )

    parser.add_argument(
        "--repo_id",
        type=str,
        default=DEFAULT_REPO_ID,
        help="Hugging Face dataset repo ID",
    )

    parser.add_argument(
        "--push_to_hub",
        action="store_true",
        help="Push dataset to Hugging Face after collection",
    )

    parser.add_argument(
        "--seed",
        type=int,
        default=BASE_SEED,
    )

    args = parser.parse_args()

    device = torch.device(
        "cuda"
        if torch.cuda.is_available()
        else "cpu"
    )

    print("\n==============================================")
    print("SQUINT → SMOLVLA SIMULATION DATA COLLECTION")
    print("==============================================")

    print(f"Environment:       {args.env_id}")
    print(f"Checkpoint:        {args.checkpoint}")
    print(f"Camera:            {CAMERA_WIDTH} x {CAMERA_HEIGHT}")
    print(f"Policy input:      {POLICY_IMAGE_SIZE} x {POLICY_IMAGE_SIZE}")
    print(f"Dataset FPS:       {DATASET_FPS}")
    print(f"Randomization:     {TRAINING_DOMAIN_RANDOMIZATION}")
    print(f"Reconfiguration:   {TRAINING_RECONFIGURATION_FREQ}")
    print(f"Target episodes:   {args.num_successful_episodes}")
    print(f"Dataset:           {args.repo_id}")
    print(f"Device:            {device}")

    random.seed(args.seed)
    np.random.seed(args.seed)
    torch.manual_seed(args.seed)

    env = make_environment(args.env_id)

    obs, info = env.reset(seed=args.seed)

    agent = load_policy(
        env,
        obs,
        args.checkpoint,
        device,
    )

    # ============================================================
    # DETERMINE DIMENSIONS
    # ============================================================

    state = get_dataset_state(env)

    # Get action dimension from environment.
    action_dim = int(
        np.prod(
            env.unwrapped.single_action_space.shape
        )
    )

    state_dim = int(state.shape[0])

    print("\nDataset dimensions:")
    print(f"    state:  {state_dim}")
    print(f"    action: {action_dim}")

    # ============================================================
    # DATASET FEATURES
    # ============================================================

    features = build_dataset_features(
        state_dim=state_dim,
        action_dim=action_dim,
    )

    print("\nDataset features:")

    for name, feature in features.items():
        print(f"    {name}: {feature}")


    dataset = LeRobotDataset.create(
        repo_id=args.repo_id,
        fps=DATASET_FPS,
        robot_type="so101_follower",
        features=features,
        use_videos=True,
        streaming_encoding=False,
        image_writer_threads=2,
    )
    
    # COLLECTION
    successful = 0
    attempted = 0

    try:

        while successful < args.num_successful_episodes:

            seed = args.seed + attempted

            attempted += 1

            success = collect_episode(
                env=env,
                agent=agent,
                dataset=dataset,
                seed=seed,
                device=device,
            )

            if success:
                successful += 1

            print(
                f"\nProgress:"
                f" {successful}/{args.num_successful_episodes} "
                f"successful"
                f" | {attempted} attempts"
            )

            # Reset after every episode.
            if successful < args.num_successful_episodes:
                obs, info = env.reset(
                    seed=args.seed + attempted
                )

        print("\n==============================================")
        print("COLLECTION COMPLETE")
        print("==============================================")

        print(f"Successful episodes: {successful}")
        print(f"Total attempts:      {attempted}")

        if attempted > 0:
            print(
                f"Success rate:        "
                f"{100.0 * successful / attempted:.2f}%"
            )

        # FINALIZE DATASET
        print("\nFinalizing LeRobot dataset...")

        dataset.finalize()

        print(
            f"Dataset contains "
            f"{dataset.num_episodes} episodes."
        )

        print(
            f"Dataset contains "
            f"{dataset.num_frames} frames."
        )

        # ========================================================
        # OPTIONAL HUB UPLOAD
        # ========================================================
        if args.push_to_hub:

            print("\nUploading dataset to Hugging Face...")

            dataset.push_to_hub(
                tags=[
                    "SO101",
                    "ManiSkill",
                    "simulation",
                    "Squint",
                    "SmolVLA",
                ],
            )

            print(
                f"\nDataset uploaded to:\n"
                f"https://huggingface.co/datasets/{args.repo_id}"
            )

        else:

            print(
                "\nDataset was NOT uploaded."
            )

            print(
                "Use --push_to_hub when you are ready."
            )

    finally:
        if dataset.has_pending_frames():

            print("\nCleaning up unfinished episode...")

            dataset.clear_episode_buffer()
        env.close()

if __name__ == "__main__":
    main()