"""
Collect successful simulation rollouts from a trained Squint policy.

Usage:

    Joint-space dataset:
        python collect_sim_data.py \
            --data-space joint \
            --repo-id aboardman/so101_lift_cube_sim_joint

    EE-space dataset:
python squint2/collect_sim_data_jonly.py \                                                                                                                
    --data-space ee \                                                                                                                                     
    --urdf /home/aboardman/lerobot/SO101/so101_new_calib.urdf \                                                                                           
    --repo-id aboardman/so101_lift_cube_sim_ee_10 \                                                                                                       
    --push-to-hub                                                                                                                                         

The EE representation is:
    ee.x
    ee.y
    ee.z
    ee.wx
    ee.wy
    ee.wz
    ee.gripper_pos

The EE pose is calculated with the same LeRobot RobotKinematics model using:
    target_frame_name = "gripper_frame_link"

The policy itself still runs using its original action representation.
"""

import os
import sys
import argparse
import random
import warnings
import logging
from pathlib import Path

import numpy as np
import torch
import gymnasium as gym


# ================================================================
# SETUP
# ================================================================
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
os.environ["MKL_SERVICE_FORCE_INTEL"] = "1"
warnings.filterwarnings(
    "ignore",
    category=DeprecationWarning,
)
logging.disable(level=logging.WARN)


# ================================================================
# USER DEFAULT SETTINGS
# ================================================================
DEFAULT_ENV_ID = "SO101LiftCube-v1"
DEFAULT_CHECKPOINT = "squint2/runs/Lift_cube/ckpt.pt"
DEFAULT_NUM_SUCCESSFUL_EPISODES = 100
DEFAULT_MAX_STEPS = 60
DEFAULT_REPO_ID = "aboardman/so101_lift_cube_sim"
DEFAULT_TASK_DESCRIPTION = "Lift the metal instrument"
DEFAULT_CAMERA_WIDTH = 640
DEFAULT_CAMERA_HEIGHT = 480
DEFAULT_DATASET_FPS = 30
DEFAULT_POLICY_IMAGE_SIZE = 16
DEFAULT_DOMAIN_RANDOMIZATION = True
DEFAULT_RECONFIGURATION_FREQ = None
DEFAULT_BASE_SEED = 1000

# ================================================================
# SO101 JOINT NAMES
# ================================================================
SO101_JOINT_NAMES = [
    "shoulder_pan.pos",
    "shoulder_lift.pos",
    "elbow_flex.pos",
    "wrist_flex.pos",
    "wrist_roll.pos",
    "gripper.pos",
]

# ================================================================
# EE FEATURE NAMES
# ================================================================
EE_NAMES = [
    "x",
    "y",
    "z",
    "wx",
    "wy",
    "wz",
    "gripper_pos",
]

# ================================================================
# IMPORT PROJECT / LE-ROBOT COMPONENTS
# ================================================================
from mani_skill.utils.wrappers.flatten import (
    FlattenRGBDObservationWrapper,
)
from utils import DownsampleObsWrapper
import envs
import mani_skill.envs
from train_squint import DeployAgent
from lerobot.datasets.lerobot_dataset import LeRobotDataset
# Only needed when recording EE-space data.
try:
    from lerobot.model.kinematics import RobotKinematics
except ImportError:
    RobotKinematics = None

# ================================================================
# ARGUMENTS
# ================================================================

def parse_args():
    parser = argparse.ArgumentParser(
        description=(
            "Run a trained Squint policy normally and record "
            "the resulting simulation trajectory."
        )
    )

    parser.add_argument(
        "--data-space",
        type=str,
        choices=["joint", "ee"],
        default="joint",
        help=(
            "Representation saved to the LeRobot dataset. "
            "This does NOT change how the policy controls the robot."
        ),
    )

    parser.add_argument(
        "--env-id",
        type=str,
        default=DEFAULT_ENV_ID,
        help="ManiSkill environment ID.",
    )

    parser.add_argument(
        "--checkpoint",
        type=str,
        default=DEFAULT_CHECKPOINT,
        help="Path to the trained Squint checkpoint.",
    )

    parser.add_argument(
        "--repo-id",
        type=str,
        default=DEFAULT_REPO_ID,
        help="Hugging Face LeRobot dataset repository.",
    )

    parser.add_argument(
        "--num-episodes",
        type=int,
        default=DEFAULT_NUM_SUCCESSFUL_EPISODES,
        help="Number of successful episodes to collect.",
    )

    parser.add_argument(
        "--max-steps",
        type=int,
        default=DEFAULT_MAX_STEPS,
        help="Maximum number of steps per episode.",
    )

    parser.add_argument(
        "--camera-width",
        type=int,
        default=DEFAULT_CAMERA_WIDTH,
        help="Recorded camera width.",
    )

    parser.add_argument(
        "--camera-height",
        type=int,
        default=DEFAULT_CAMERA_HEIGHT,
        help="Recorded camera height.",
    )

    parser.add_argument(
        "--fps",
        type=int,
        default=DEFAULT_DATASET_FPS,
        help="Dataset FPS.",
    )

    parser.add_argument(
        "--policy-image-size",
        type=int,
        default=DEFAULT_POLICY_IMAGE_SIZE,
        help="Image size expected by the trained policy.",
    )

    parser.add_argument(
        "--task",
        type=str,
        default=DEFAULT_TASK_DESCRIPTION,
        help="Task description stored in the dataset.",
    )

    parser.add_argument(
        "--base-seed",
        type=int,
        default=DEFAULT_BASE_SEED,
        help="Starting random seed.",
    )

    parser.add_argument(
        "--push-to-hub",
        action="store_true",
        help="Push the finished dataset to Hugging Face.",
    )

    parser.add_argument(
        "--urdf",
        type=str,
        default=None,
        help=(
            "Path to the SO101 URDF used by LeRobot RobotKinematics. "
            "Required when --data-space ee is selected."
        ),
    )

    parser.add_argument(
        "--domain-randomization",
        action=argparse.BooleanOptionalAction,
        default=DEFAULT_DOMAIN_RANDOMIZATION,
        help="Enable/disable ManiSkill domain randomization.",
    )

    return parser.parse_args()


# ================================================================
# IMAGE EXTRACTION
# ================================================================

def get_arm_image(obs, camera_width, camera_height):
    """
    Extract the 640x480 arm camera image.

    This function only reads the observation.
    It does not modify the observation.
    """

    rgb = obs["rgb"]

    if torch.is_tensor(rgb):
        rgb = rgb.detach().cpu().numpy()

    # Remove batch dimension if present.
    if rgb.ndim == 4:
        rgb = rgb[0]

    expected_shape = (
        camera_height,
        camera_width,
        3,
    )

    if rgb.shape != expected_shape:
        raise RuntimeError(
            f"Unexpected camera shape {rgb.shape}; "
            f"expected {expected_shape}."
        )

    return rgb.astype(np.uint8)


# ================================================================
# GET SIMULATION JOINT STATE
# ================================================================

def get_sim_state(env):
    """
    Read the actual robot joint positions from the simulation.

    Returns:

        [shoulder_pan,
         shoulder_lift,
         elbow_flex,
         wrist_flex,
         wrist_roll,
         gripper]

    in simulation radians.

    IMPORTANT:
        This is only reading the robot state.
        It does not modify the robot.
    """

    qpos = env.unwrapped.agent.robot.get_qpos()

    if torch.is_tensor(qpos):
        qpos = qpos.detach().cpu().numpy()

    if qpos.ndim > 1:
        qpos = qpos[0]

    qpos = np.asarray(
        qpos,
        dtype=np.float32,
    )

    if qpos.shape != (6,):
        raise RuntimeError(
            f"Expected 6 robot joints, got {qpos.shape}"
        )

    return qpos


# ================================================================
# SIM JOINTS -> LEROBOT JOINT DATA
# ================================================================

def convert_state_for_lerobot(sim_qpos):
    """
    Convert simulation joint state to the joint representation
    used by the real LeRobot SO101 dataset.

    This function is ONLY for recording.

    Simulation:
        radians

    Dataset:
        degrees
    """

    state = np.rad2deg(
        np.asarray(
            sim_qpos,
            dtype=np.float32,
        ).copy()
    )

    # ------------------------------------------------------------
    # Gripper conversion
    # ------------------------------------------------------------

    gripper_sim_min = -10.0
    gripper_sim_max = 120.0

    gripper_servo_min = -62.5
    gripper_servo_max = 64.62

    gripper_sim_range = (
        gripper_sim_max - gripper_sim_min
    )

    gripper_servo_range_inverted = (
        gripper_servo_min - gripper_servo_max
    )

    sim_gripper_deg = state[5]

    servo_deg = (
        (
            sim_gripper_deg - gripper_sim_min
        )
        / gripper_sim_range
        * gripper_servo_range_inverted
        + gripper_servo_max
    )

    state[5] = np.clip(
        servo_deg,
        2.0,
        100.0,
    )

    return state.astype(np.float32)


# ================================================================
# SIM ACTION -> LEROBOT JOINT DATA
# ================================================================

def convert_action_for_lerobot(target_qpos_sim):
    """
    Convert the actual ManiSkill controller target into the
    joint representation stored in the LeRobot dataset.

    IMPORTANT:

        This does NOT modify target_qpos_sim.

        It creates a copy only for recording.
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

    action_deg = np.rad2deg(
        action.copy()
    )

    # ------------------------------------------------------------
    # Gripper conversion
    # ------------------------------------------------------------

    gripper_sim_min = -10.0
    gripper_sim_max = 120.0

    gripper_servo_min = -62.5
    gripper_servo_max = 64.62

    gripper_sim_range = (
        gripper_sim_max - gripper_sim_min
    )

    gripper_servo_range_inverted = (
        gripper_servo_min - gripper_servo_max
    )

    sim_gripper_deg = action_deg[5]

    servo_deg = (
        (
            sim_gripper_deg - gripper_sim_min
        )
        / gripper_sim_range
        * gripper_servo_range_inverted
        + gripper_servo_max
    )

    action_deg[5] = np.clip(
        servo_deg,
        2.0,
        100.0,
    )

    return action_deg.astype(np.float32)


# ================================================================
# SO101 JOINTS -> EE
# ================================================================

def joints_to_ee(
    joint_values_deg,
    kinematics,
):
    """
    Convert the first five SO101 arm joint positions into:

        [x, y, z, wx, wy, wz]

    using LeRobot's RobotKinematics.

    joint_values_deg must be:

        shoulder_pan
        shoulder_lift
        elbow_flex
        wrist_flex
        wrist_roll

    in degrees.

    EE orientation is represented as a rotation vector.
    """

    joint_values_deg = np.asarray(
        joint_values_deg,
        dtype=np.float64,
    )

    if joint_values_deg.shape != (5,):
        raise RuntimeError(
            "Expected 5 arm joints for EE FK, "
            f"got {joint_values_deg.shape}"
        )

    transform = kinematics.forward_kinematics(
        joint_values_deg
    )

    position = transform[:3, 3]

    # LeRobot EE orientation is represented as a
    # rotation vector rather than Euler angles.
    from scipy.spatial.transform import Rotation

    rotation_vector = (
        Rotation.from_matrix(
            transform[:3, :3]
        ).as_rotvec()
    )

    return np.array(
        [
            position[0],
            position[1],
            position[2],
            rotation_vector[0],
            rotation_vector[1],
            rotation_vector[2],
        ],
        dtype=np.float32,
    )


# ================================================================
# FULL JOINT STATE -> EE STATE
# ================================================================

def convert_frame_to_ee(
    values,
    kinematics,
):
    """
    Convert:

        [5 arm joints, gripper]

    into:

        [x, y, z, wx, wy, wz, gripper_pos]

    This function is RECORDING ONLY.
    """

    values = np.asarray(
        values,
        dtype=np.float64,
    )

    if values.shape != (6,):
        raise RuntimeError(
            f"Expected 6 joint values, got {values.shape}"
        )

    arm_joints = values[:5]

    gripper = values[5]

    ee_pose = joints_to_ee(
        arm_joints,
        kinematics,
    )

    return np.concatenate(
        [
            ee_pose,
            np.array(
                [gripper],
                dtype=np.float32,
            ),
        ]
    ).astype(np.float32)


# ================================================================
# SUCCESS CHECK
# ================================================================

def episode_is_successful(info):
    """
    Read the environment success flag.
    """

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

    return bool(
        np.asarray(success)
        .reshape(-1)[0]
    )


# ================================================================
# DATASET FEATURES
# ================================================================

def build_dataset_features(
    data_space,
    camera_width,
    camera_height,
):
    """
    Build the LeRobot dataset schema.

    The actual policy does not care about this schema.
    """

    features = {
        "observation.images.arm": {
            "dtype": "video",
            "shape": (
                camera_height,
                camera_width,
                3,
            ),
            "names": [
                "height",
                "width",
                "channel",
            ],
        },
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
            "names": [
                f"ee.{name}"
                for name in EE_NAMES
            ],
        }

        features["action"] = {
            "dtype": "float32",
            "shape": (7,),
            "names": [
                f"ee.{name}"
                for name in EE_NAMES
            ],
        }

    return features


# ================================================================
# BUILD DATASET FRAME
# ================================================================

def build_frame(
    image,
    state,
    action,
    task_description,
):
    """
    Build one LeRobot frame.

    Again, this is completely separate from policy control.
    """

    return {
        "observation.images.arm": image,
        "observation.state": state,
        "action": action,
        "task": task_description,
    }


# ================================================================
# ENVIRONMENT
# ================================================================

def make_environment(
    env_id,
    camera_width,
    camera_height,
    domain_randomization,
    reconfiguration_freq,
):
    """
    Create the environment.

    This should match the environment used during policy training.
    """

    env = gym.make(
        env_id,

        obs_mode="rgb+segmentation",

        render_mode="rgb_array",

        sim_backend="gpu",

        domain_randomization=domain_randomization,

        reconfiguration_freq=reconfiguration_freq,

        sensor_configs={
            "width": camera_width,
            "height": camera_height,
        },

        human_render_camera_configs={
            "width": camera_width,
            "height": camera_height,
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
# LOAD POLICY
# ================================================================

def load_policy(
    env,
    obs,
    device,
    checkpoint,
    policy_image_size,
):
    """
    Load the trained Squint policy.

    Nothing about dataset representation is passed to the policy.
    """

    sample_obs = {
        "rgb": torch.zeros(
            (
                1,
                policy_image_size,
                policy_image_size,
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
        target_image_size=policy_image_size,
        device=device,
    )

    agent.load_checkpoint(
        checkpoint
    )

    agent.eval()

    return agent


# ================================================================
# GET CONTROLLER TARGET
# ================================================================

def get_controller_target_qpos(env):
    """
    Read the current ManiSkill controller target.

    This is used ONLY to record the action.

    We do not modify the target.
    """

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

    target_qpos = np.asarray(
        target_qpos,
        dtype=np.float32,
    )

    if target_qpos.ndim > 1:
        target_qpos = target_qpos[0]

    if target_qpos.shape != (6,):
        raise RuntimeError(
            "Expected controller target to contain "
            f"6 joints, got {target_qpos.shape}"
        )

    return target_qpos.copy()


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
    max_steps,
    camera_width,
    camera_height,
    task_description,
    kinematics=None,
):
    """
    Run ONE policy episode.

    CRITICAL CONTROL ORDER:

        1. Read observation
        2. Run policy
        3. env.step(policy_action)
        4. Read resulting/controller state
        5. Convert state for recording
        6. Save frame

    EE conversion NEVER enters the control loop.
    """

    print()
    print(
        f"Episode seed: {seed}"
    )

    obs, info = env.reset(
        seed=seed
    )

    success = False

    for step in range(max_steps):

        # ========================================================
        # 1. GET IMAGE FOR POLICY
        # ========================================================

        # IMPORTANT:
        # This is the exact image used by the policy.
        #
        # We do not replace it with the saved 640x480 image.
        policy_obs = {
            "rgb": obs["rgb"].to(device),
            "state": obs["state"].to(device),
        }

        # ========================================================
        # 2. RUN POLICY
        # ========================================================

        with torch.no_grad():
            policy_action = agent(
                policy_obs
            )

        # ========================================================
        # 3. ORIGINAL POLICY -> MANISKILL
        # ========================================================

        # THIS IS THE CRITICAL LINE.
        #
        # The data-space setting does NOT affect this action.
        #
        # No EE conversion.
        # No IK.
        # No joint conversion.
        #
        # The policy action goes directly to the same
        # environment/controller used during the original rollout.

        policy_action_np = (
            policy_action
            .detach()
            .cpu()
            .numpy()
        )

        next_obs, reward, terminated, truncated, info = (
            env.step(policy_action_np)
        )

        # ========================================================
        # 4. RECORD THE IMAGE
        # ========================================================

        image = get_arm_image(
            next_obs,
            camera_width,
            camera_height,
        )

        # ========================================================
        # 5. RECORD ACTUAL ROBOT STATE
        # ========================================================

        # This reads the robot AFTER the policy action was executed.

        sim_qpos = get_sim_state(
            env
        )

        # ========================================================
        # 6. READ CONTROLLER TARGET
        # ========================================================

        # This is the target that the original controller was
        # given. We only READ it for dataset recording.

        target_qpos = (
            get_controller_target_qpos(
                env
            )
        )

        # ========================================================
        # 7. CONVERT FOR DATASET
        # ========================================================

        if data_space == "joint":

            state = (
                convert_state_for_lerobot(
                    sim_qpos
                )
            )

            action = (
                convert_action_for_lerobot(
                    target_qpos
                )
            )

        elif data_space == "ee":

            if kinematics is None:
                raise RuntimeError(
                    "EE data requested but no "
                    "kinematics model was provided."
                )

            # ----------------------------------------------------
            # IMPORTANT:
            #
            # These two calculations ONLY generate numbers for
            # the dataset.
            #
            # They are NOT sent back to ManiSkill.
            # ----------------------------------------------------

            state_joint_deg = (
                convert_state_for_lerobot(
                    sim_qpos
                )
            )

            action_joint_deg = (
                convert_action_for_lerobot(
                    target_qpos
                )
            )

            state = convert_frame_to_ee(
                state_joint_deg,
                kinematics,
            )

            action = convert_frame_to_ee(
                action_joint_deg,
                kinematics,
            )

        else:
            raise RuntimeError(
                f"Unknown data space: {data_space}"
            )

        # ========================================================
        # 8. SAVE DATASET FRAME
        # ========================================================

        dataset.add_frame(
            build_frame(
                image=image,
                state=state,
                action=action,
                task_description=task_description,
            )
        )

        # ========================================================
        # 9. UPDATE OBSERVATION
        # ========================================================

        obs = next_obs

        # ========================================================
        # 10. CHECK SUCCESS
        # ========================================================

        success = episode_is_successful(
            info
        )

        # ========================================================
        # 11. CHECK TERMINATION
        # ========================================================

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

    # ============================================================
    # SAVE OR DISCARD EPISODE
    # ============================================================

    if success:

        dataset.save_episode()

        print(
            f"Saved episode "
            f"({step + 1} frames)"
        )

        return True

    else:

        dataset.clear_episode_buffer()

        print(
            f"Discarded failed episode "
            f"({step + 1} frames)"
        )

        return False


# ================================================================
# MAIN
# ================================================================

def main():

    args = parse_args()

    # ============================================================
    # DEVICE
    # ============================================================

    device = torch.device(
        "cuda"
        if torch.cuda.is_available()
        else "cpu"
    )

    # ============================================================
    # PRINT CONFIGURATION
    # ============================================================

    print()
    print("=" * 70)
    print("SQUINT SIMULATION DATA COLLECTION")
    print("=" * 70)

    print(
        f"Environment:       {args.env_id}"
    )

    print(
        f"Checkpoint:        {args.checkpoint}"
    )

    print(
        f"Data space:        {args.data_space}"
    )

    print(
        f"Camera:            "
        f"{args.camera_width}x"
        f"{args.camera_height}"
    )

    print(
        f"Policy input:      "
        f"{args.policy_image_size}x"
        f"{args.policy_image_size}"
    )

    print(
        f"FPS:               {args.fps}"
    )

    print(
        f"Randomization:     "
        f"{args.domain_randomization}"
    )

    print(
        f"Target episodes:   "
        f"{args.num_episodes}"
    )

    print(
        f"Dataset:           {args.repo_id}"
    )

    print(
        f"Device:            {device}"
    )

    if args.data_space == "ee":
        print(
            f"EE URDF:           {args.urdf}"
        )

    print()
    print(
        "IMPORTANT:"
    )
    print(
        "The policy action is sent directly to "
        "ManiSkill."
    )
    print(
        "EE conversion is recording-only."
    )

    # ============================================================
    # VALIDATE EE CONFIGURATION
    # ============================================================

    kinematics = None

    if args.data_space == "ee":

        if RobotKinematics is None:
            raise ImportError(
                "Could not import RobotKinematics. "
                "Make sure your installed LeRobot version "
                "contains lerobot.model.kinematics."
            )

        if args.urdf is None:
            raise ValueError(
                "--urdf is required when "
                "--data-space ee is used."
            )

        urdf_path = Path(
            args.urdf
        )

        if not urdf_path.exists():
            raise FileNotFoundError(
                "Could not find the specified URDF:\n"
                f"{urdf_path}"
            )

        # --------------------------------------------------------
        # IMPORTANT:
        #
        # This kinematics model is used ONLY to calculate the
        # EE numbers that are written to the dataset.
        #
        # It is NOT used for control.
        # --------------------------------------------------------

        kinematics = RobotKinematics(
            urdf_path=str(
                urdf_path
            ),
            target_frame_name=(
                "gripper_frame_link"
            ),
            joint_names=[
                "shoulder_pan",
                "shoulder_lift",
                "elbow_flex",
                "wrist_flex",
                "wrist_roll",
            ],
        )

        print()
        print(
            "EE kinematics initialized."
        )

        print(
            "Target frame: "
            "gripper_frame_link"
        )

    # ============================================================
    # RANDOM SEEDS
    # ============================================================

    random.seed(
        args.base_seed
    )

    np.random.seed(
        args.base_seed
    )

    torch.manual_seed(
        args.base_seed
    )

    # ============================================================
    # CREATE ENVIRONMENT
    # ============================================================

    env = make_environment(
        env_id=args.env_id,
        camera_width=args.camera_width,
        camera_height=args.camera_height,
        domain_randomization=args.domain_randomization,
        reconfiguration_freq=DEFAULT_RECONFIGURATION_FREQ,
    )

    # ============================================================
    # INITIAL RESET
    # ============================================================

    obs, info = env.reset(
        seed=args.base_seed
    )

    # ============================================================
    # LOAD POLICY
    # ============================================================

    agent = load_policy(
        env=env,
        obs=obs,
        device=device,
        checkpoint=args.checkpoint,
        policy_image_size=args.policy_image_size,
    )

    # ============================================================
    # CREATE DATASET
    # ============================================================

    dataset = LeRobotDataset.create(
        repo_id=args.repo_id,

        fps=args.fps,

        robot_type="so101_follower",

        features=build_dataset_features(
            data_space=args.data_space,
            camera_width=args.camera_width,
            camera_height=args.camera_height,
        ),

        use_videos=True,

        image_writer_threads=2,
    )

    # ============================================================
    # COLLECTION LOOP
    # ============================================================

    successful = 0
    attempted = 0

    try:

        while successful < args.num_episodes:

            seed = (
                args.base_seed
                + attempted
            )

            attempted += 1

            success = collect_episode(
                env=env,

                agent=agent,

                dataset=dataset,

                seed=seed,

                device=device,

                data_space=args.data_space,

                max_steps=args.max_steps,

                camera_width=args.camera_width,

                camera_height=args.camera_height,

                task_description=args.task,

                kinematics=kinematics,
            )

            if success:
                successful += 1

            print()
            print(
                f"Progress: "
                f"{successful}/"
                f"{args.num_episodes} successful "
                f"({attempted} attempts)"
            )

            # ----------------------------------------------------
            # Prepare next episode.
            # ----------------------------------------------------

            if successful < args.num_episodes:

                env.reset(
                    seed=(
                        args.base_seed
                        + attempted
                    )
                )

        # ========================================================
        # FINALIZE DATASET
        # ========================================================

        dataset.finalize()

        # ========================================================
        # SUMMARY
        # ========================================================

        print()
        print("=" * 70)
        print("COLLECTION COMPLETE")
        print("=" * 70)

        print(
            f"Successful episodes: "
            f"{successful}"
        )

        print(
            f"Attempts:            "
            f"{attempted}"
        )

        if attempted > 0:
            print(
                f"Success rate:        "
                f"{100.0 * successful / attempted:.2f}%"
            )

        print(
            f"Data space:          "
            f"{args.data_space}"
        )

        print(
            f"Dataset:             "
            f"{args.repo_id}"
        )

        # ========================================================
        # PUSH TO HUB
        # ========================================================

        if args.push_to_hub:

            print()
            print(
                "Uploading dataset to Hugging Face..."
            )

            dataset.push_to_hub()

            print(
                "Upload complete."
            )

    finally:

        env.close()


# ================================================================
# ENTRY POINT
# ================================================================

if __name__ == "__main__":
    main()
