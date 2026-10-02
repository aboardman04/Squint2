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
ENV_ID = "SO101LiftCube-v1"

# Trained Squint checkpoint
CHECKPOINT = "squint2/runs/Lift_cube/ckpt.pt"

# Number of SUCCESSFUL episodes to collect
NUM_SUCCESSFUL_EPISODES = 5

MAX_STEPS = 60

# Hugging Face dataset name
REPO_ID = "aboardman/so101_lift_cube_sim_ee_1"

# Set True only after verifying the local dataset
PUSH_TO_HUB = True

# Task description used in the dataset
TASK_DESCRIPTION = "Lift the white cube"

# Camera settings -- match your real SO101 dataset
CAMERA_WIDTH = 640
CAMERA_HEIGHT = 480
DATASET_FPS = 30

# Name used by the real LeRobot dataset
IMAGE_FEATURE_NAME = "observation.images.arm"

# ================================================================
# JOINT / EE REPRESENTATIONS
# ================================================================

# Six values stored in joint-space datasets.
#
# The first five are the arm joints used by FK.
# The sixth is the gripper.
SO101_JOINT_NAMES = [
    "shoulder_pan.pos",
    "shoulder_lift.pos",
    "elbow_flex.pos",
    "wrist_flex.pos",
    "wrist_roll.pos",
    "gripper.pos",
]

# These are the five joints used by LeRobot RobotKinematics.
#
# IMPORTANT:
# The gripper is NOT part of the 5-DOF arm FK.
SO101_KINEMATIC_JOINT_NAMES = [
    "shoulder_pan",
    "shoulder_lift",
    "elbow_flex",
    "wrist_flex",
    "wrist_roll",
]

# EE representation used by current LeRobot EE processors.
SO101_EE_NAMES = [
    "ee.x",
    "ee.y",
    "ee.z",
    "ee.wx",
    "ee.wy",
    "ee.wz",
    "ee.gripper_pos",
]

# ================================================================
# LE ROBOT KINEMATICS
# ================================================================

# IMPORTANT:
#
# This should point to the SAME calibrated URDF that you use
# for your real SO101 LeRobot kinematics.
#
# Because the real and simulation URDFs you supplied have the
# same kinematic chain and gripper_frame_link, this gives us
# a common coordinate convention.
LEROBOT_URDF_PATH = "lerobot/SO101/so101_new_calib.urdf"

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
from utils import DownsampleObsWrapper

import envs
import mani_skill.envs

from train_squint import DeployAgent

from lerobot.datasets.lerobot_dataset import LeRobotDataset
from lerobot.model.kinematics import RobotKinematics


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
    """
    Return the six simulated SO101 joint positions in radians.

    Order:
        shoulder_pan
        shoulder_lift
        elbow_flex
        wrist_flex
        wrist_roll
        gripper
    """

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
# LE ROBOT FORWARD KINEMATICS
# ================================================================

def create_kinematics():
    """
    Create the exact kinematic model used for EE dataset coordinates.

    The important pieces are:

        URDF:
            so101_new_calib.urdf

        target frame:
            gripper_frame_link

        joints:
            five arm joints

    This is the same convention used by LeRobot's SO101 EE
    processing pipeline.
    """

    if not os.path.exists(LEROBOT_URDF_PATH):
        raise FileNotFoundError(
            f"Could not find LeRobot SO101 URDF:\n"
            f"    {LEROBOT_URDF_PATH}\n\n"
            "Set LEROBOT_URDF_PATH to the location of your "
            "so101_new_calib.urdf."
        )

    print("\nCreating LeRobot kinematics:")
    print(f"  URDF:         {LEROBOT_URDF_PATH}")
    print("  EE frame:     gripper_frame_link")
    print("  Arm joints:   " + ", ".join(SO101_KINEMATIC_JOINT_NAMES))

    return RobotKinematics(
        urdf_path=LEROBOT_URDF_PATH,
        target_frame_name="gripper_frame_link",
        joint_names=SO101_KINEMATIC_JOINT_NAMES,
    )


def sim_qpos_to_ee(sim_qpos, kinematics):
    """
    Convert simulated SO101 joint positions to the LeRobot EE frame.

    Input:
        sim_qpos:
            [shoulder_pan,
             shoulder_lift,
             elbow_flex,
             wrist_flex,
             wrist_roll,
             gripper]

        Units:
            radians for all six simulated joints.

    Output:
        [x, y, z, wx, wy, wz, gripper_pos]

    Position:
        meters

    Orientation:
        rotation vector in radians

    Gripper:
        LeRobot servo/degrees representation.
    """

    sim_qpos = np.asarray(sim_qpos, dtype=np.float32)

    if sim_qpos.shape != (6,):
        raise RuntimeError(
            f"Expected six simulated joints, got {sim_qpos.shape}"
        )

    # ------------------------------------------------------------
    # IMPORTANT:
    #
    # Do NOT manually calculate the gripper_frame_link offset here.
    #
    # RobotKinematics reads the URDF and calculates:
    #
    # base_link
    #     ↓
    # shoulder
    #     ↓
    # ...
    #     ↓
    # gripper_link
    #     ↓
    # gripper_frame_link
    #
    # This means the coordinate system is exactly the same one
    # LeRobot uses for the real robot.
    # ------------------------------------------------------------

    arm_qpos = sim_qpos[:5].astype(np.float64)

    transform = kinematics.forward_kinematics(arm_qpos)

    # Position of gripper_frame_link in the URDF base frame.
    position = transform[:3, 3]

    # Convert rotation matrix → rotation vector.
    #
    # LeRobot uses wx, wy, wz as a rotation vector.
    from lerobot.utils.rotation import Rotation

    rotvec = Rotation.from_matrix(transform[:3, :3]).as_rotvec()

    # Gripper remains a separate joint value.
    #
    # The arm FK has only five joints.
    gripper_sim_rad = sim_qpos[5]

    gripper_sim_deg = float(np.rad2deg(gripper_sim_rad))

    # Convert the simulated gripper convention to the real
    # LeRobot servo convention.
    gripper_pos = convert_gripper_for_lerobot(gripper_sim_deg)

    ee = np.array(
        [
            position[0],
            position[1],
            position[2],
            rotvec[0],
            rotvec[1],
            rotvec[2],
            gripper_pos,
        ],
        dtype=np.float32,
    )

    return ee


# ================================================================
# SIMULATION -> REAL DATA CONVERSION
# ================================================================

def convert_gripper_for_lerobot(sim_deg):
    """
    Convert simulated SO101 gripper angle to the real LeRobot
    gripper representation.

    This preserves the existing mapping from the original
    collection script.
    """

    _gripper_sim_min = -10.0
    _gripper_sim_max = 120.0

    _gripper_servo_min = -62.5
    _gripper_servo_max = 64.62

    sim_range = _gripper_sim_max - _gripper_sim_min

    # Invert the servo range because the simulated and real
    # gripper directions are opposite.
    servo_range_inverted = (
        _gripper_servo_min - _gripper_servo_max
    )

    servo_deg = (
        (sim_deg - _gripper_sim_min)
        / sim_range
        * servo_range_inverted
        + _gripper_servo_max
    )

    return float(np.clip(servo_deg, 2.0, 100.0))


def convert_state_for_lerobot(sim_qpos):
    """
    Convert simulated joint state to the six-value joint-space
    LeRobot representation.
    """

    state = np.rad2deg(
        np.asarray(sim_qpos, dtype=np.float32).copy()
    )

    state[5] = convert_gripper_for_lerobot(state[5])

    return state


def convert_action_for_lerobot(target_qpos_sim):
    """
    Convert absolute target qpos from ManiSkill to the six-value
    joint-space LeRobot action representation.
    """

    action = np.asarray(
        target_qpos_sim,
        dtype=np.float32,
    )

    if action.ndim > 1:
        action = action[0]

    if action.shape != (6,):
        raise RuntimeError(
            f"Expected 6 actions, got {action.shape}"
        )

    action_deg = np.rad2deg(action.copy())

    action_deg[5] = convert_gripper_for_lerobot(action_deg[5])

    return action_deg


# ================================================================
# DATA REPRESENTATION
# ================================================================

def build_dataset_features(data_space):
    """
    Build the LeRobot dataset schema.

    data_space:
        "joint" -> six joint state/action values
        "ee"    -> seven EE state/action values
    """

    features = {
        IMAGE_FEATURE_NAME: {
            "dtype": "video",
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
        }
    }

    if data_space == "joint":

        features["observation.state"] = {
            "dtype": "float32",
            "shape": (6,),
            "names": SO101_JOINT_NAMES,
        }

        features["action"] = {
            "dtype": "float32",
            "shape": (6,),
            "names": SO101_JOINT_NAMES,
        }

    elif data_space == "ee":

        features["observation.state"] = {
            "dtype": "float32",
            "shape": (7,),
            "names": SO101_EE_NAMES,
        }

        features["action"] = {
            "dtype": "float32",
            "shape": (7,),
            "names": SO101_EE_NAMES,
        }

    else:
        raise ValueError(
            f"Unknown data space: {data_space}"
        )

    return features


def build_frame(
    image,
    state,
    action,
):
    """Create one LeRobot timestep."""

    return {
        IMAGE_FEATURE_NAME: image,
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

    env = FlattenRGBDObservationWrapper(
        env,
        rgb=True,
        depth=False,
        state=True,
    )

    return env


# ================================================================
# POLICY
# ================================================================

def load_policy(env, obs, device):
    """Load the trained Squint policy."""

    sample_obs = {
        "rgb": torch.zeros(
            (
                1,
                POLICY_IMAGE_SIZE,
                POLICY_IMAGE_SIZE,
                3,
            ),
            dtype=torch.uint8,
            device=device,
        ),
        "state": obs["state"],
    }

    agent = DeployAgent(
        env,
        sample_obs,
        target_image_size=POLICY_IMAGE_SIZE,
        device=device,
    )

    agent.load_checkpoint(CHECKPOINT)
    agent.eval()

    return agent


# ================================================================
# COLLECT ONE EPISODE
# ================================================================

def collect_episode(
    env,
    agent,
    dataset,
    seed,
    device,
    data_space,
    kinematics,
):
    """
    Run one episode and save it only if successful.

    The policy itself always operates exactly as before.

    data_space only controls how the recorded dataset is represented.
    """

    print(f"\nEpisode seed: {seed}")

    obs, info = env.reset(seed=seed)

    success = False
    max_steps = MAX_STEPS

    if max_steps is None:
        raise RuntimeError(
            "Environment has no max_episode_steps."
        )

    for step in range(max_steps):

        # --------------------------------------------------------
        # Current observation
        # --------------------------------------------------------

        image = get_arm_image(obs)

        sim_state = get_sim_state(env)

        # --------------------------------------------------------
        # Convert CURRENT STATE for dataset
        # --------------------------------------------------------

        if data_space == "joint":

            state = convert_state_for_lerobot(
                sim_state
            )

        elif data_space == "ee":

            state = sim_qpos_to_ee(
                sim_state,
                kinematics,
            )

        else:
            raise RuntimeError(
                f"Unknown data space: {data_space}"
            )

        # --------------------------------------------------------
        # Policy action
        # --------------------------------------------------------

        policy_obs = {
            "rgb": obs["rgb"].to(device),
            "state": obs["state"].to(device),
        }

        with torch.no_grad():
            policy_action = agent(policy_obs)

        policy_action_np = (
            policy_action.detach()
            .cpu()
            .numpy()
        )

        # --------------------------------------------------------
        # Execute policy
        # --------------------------------------------------------

        next_obs, reward, terminated, truncated, info = env.step(
            policy_action_np
        )

        # --------------------------------------------------------
        # Grab absolute target qpos from controller
        # --------------------------------------------------------

        target_qpos = (
            env.unwrapped
            .agent
            .controller
            ._target_qpos
        )

        if torch.is_tensor(target_qpos):
            target_qpos = (
                target_qpos
                .detach()
                .cpu()
                .numpy()
            )

        if target_qpos.ndim > 1:
            target_qpos = target_qpos[0]

        target_qpos = np.asarray(
            target_qpos,
            dtype=np.float32,
        )

        if target_qpos.shape != (6,):
            raise RuntimeError(
                f"Expected target qpos shape (6,), "
                f"got {target_qpos.shape}"
            )

        # --------------------------------------------------------
        # Convert ACTION for dataset
        # --------------------------------------------------------

        if data_space == "joint":

            action = convert_action_for_lerobot(
                target_qpos
            )

        elif data_space == "ee":

            action = sim_qpos_to_ee(
                target_qpos,
                kinematics,
            )

        else:
            raise RuntimeError(
                f"Unknown data space: {data_space}"
            )

        # --------------------------------------------------------
        # Record timestep
        # --------------------------------------------------------

        dataset.add_frame(
            build_frame(
                image=image,
                state=state,
                action=action,
            )
        )

        # --------------------------------------------------------
        # Continue
        # --------------------------------------------------------

        obs = next_obs

        success = episode_is_successful(info)

        if torch.is_tensor(terminated):
            terminated = (
                terminated
                .detach()
                .cpu()
                .numpy()
            )

        if torch.is_tensor(truncated):
            truncated = (
                truncated
                .detach()
                .cpu()
                .numpy()
            )

        terminated = bool(
            np.asarray(terminated)
            .reshape(-1)[0]
        )

        truncated = bool(
            np.asarray(truncated)
            .reshape(-1)[0]
        )

        if success:
            print(
                f"SUCCESS at step {step + 1}"
            )
            break

        if terminated or truncated:
            break

    # ------------------------------------------------------------
    # Save/discard episode
    # ------------------------------------------------------------

    if success:

        dataset.save_episode()

        print(
            f"Saved episode ({step + 1} frames)"
        )

        return True

    dataset.clear_episode_buffer()

    print(
        f"Discarded failed episode "
        f"({step + 1} frames)"
    )

    return False


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
        success = (
            success
            .detach()
            .cpu()
            .numpy()
        )

    return bool(
        np.asarray(success)
        .reshape(-1)[0]
    )


# ================================================================
# ARGUMENTS
# ================================================================

def parse_args():
    parser = argparse.ArgumentParser(
        description=(
            "Collect successful SO101 simulation episodes "
            "in joint-space or LeRobot-compatible EE-space."
        )
    )

    parser.add_argument(
        "--data-space",
        type=str,
        choices=["joint", "ee"],
        default="joint",
        help=(
            "Representation stored in observation.state "
            "and action. Use 'joint' for joint-space data "
            "or 'ee' for LeRobot-compatible EE-space data."
        ),
    )

    parser.add_argument(
        "--repo-id",
        type=str,
        default=REPO_ID,
        help=(
            "Hugging Face dataset repository ID."
        ),
    )

    parser.add_argument(
        "--num-episodes",
        type=int,
        default=NUM_SUCCESSFUL_EPISODES,
        help=(
            "Number of successful episodes to collect."
        ),
    )

    parser.add_argument(
        "--checkpoint",
        type=str,
        default=CHECKPOINT,
        help=(
            "Path to the Squint checkpoint."
        ),
    )

    parser.add_argument(
        "--urdf-path",
        type=str,
        default=LEROBOT_URDF_PATH,
        help=(
            "Path to the calibrated SO101 URDF used "
            "for LeRobot EE forward kinematics."
        ),
    )

    return parser.parse_args()


# ================================================================
# MAIN
# ================================================================

def main():

    args = parse_args()

    # Update globals from CLI arguments.
    global CHECKPOINT
    global LEROBOT_URDF_PATH

    CHECKPOINT = args.checkpoint
    LEROBOT_URDF_PATH = args.urdf_path

    data_space = args.data_space

    device = torch.device(
        "cuda"
        if torch.cuda.is_available()
        else "cpu"
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
    print(f"Target episodes:{args.num_episodes}")
    print(f"Dataset:        {args.repo_id}")
    print(f"Data space:     {data_space}")
    print(f"Device:         {device}")

    if data_space == "ee":
        print("\nEE representation:")
        print("  ee.x")
        print("  ee.y")
        print("  ee.z")
        print("  ee.wx")
        print("  ee.wy")
        print("  ee.wz")
        print("  ee.gripper_pos")
        print("\nEE frame:")
        print("  gripper_frame_link")
        print(f"\nKinematics URDF:\n  {LEROBOT_URDF_PATH}")

    random.seed(BASE_SEED)
    np.random.seed(BASE_SEED)
    torch.manual_seed(BASE_SEED)

    # ------------------------------------------------------------
    # Create environment
    # ------------------------------------------------------------

    env = make_environment()

    obs, info = env.reset(
        seed=BASE_SEED
    )

    # ------------------------------------------------------------
    # Create LeRobot kinematics
    # ------------------------------------------------------------

    #
    # We create this even though joint-space mode does not need it.
    # This keeps the code simple and lets us verify the URDF early.
    #
    kinematics = create_kinematics()

    # ------------------------------------------------------------
    # Load policy
    # ------------------------------------------------------------

    agent = load_policy(
        env,
        obs,
        device,
    )

    # ------------------------------------------------------------
    # Create dataset
    # ------------------------------------------------------------

    dataset = LeRobotDataset.create(
        repo_id=args.repo_id,
        fps=DATASET_FPS,
        robot_type="so101_follower",
        features=build_dataset_features(
            data_space
        ),
        use_videos=True,
        image_writer_threads=2,
    )

    successful = 0
    attempted = 0

    try:

        while successful < args.num_episodes:

            seed = (
                BASE_SEED
                + attempted
            )

            attempted += 1

            if collect_episode(
                env=env,
                agent=agent,
                dataset=dataset,
                seed=seed,
                device=device,
                data_space=data_space,
                kinematics=kinematics,
            ):
                successful += 1

            print(
                f"Progress: {successful}/"
                f"{args.num_episodes} successful "
                f"({attempted} attempts)"
            )

            if successful < args.num_episodes:
                env.reset(
                    seed=BASE_SEED + attempted
                )

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

# python collect_sim_data.py \
#     --data-space joint/ee \
#     --repo-id aboardman/so101_lift_instruments_sim_joint
