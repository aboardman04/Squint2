from dataclasses import dataclass
from typing import Any, Sequence, Union

import dacite
import numpy as np
import sapien
import torch
import torch.random
from transforms3d.euler import euler2quat

from mani_skill.agents.robots import Fetch, Panda
from mani_skill.sensors.camera import CameraConfig
from mani_skill.utils import common, sapien_utils
from mani_skill.utils.registration import register_env
from mani_skill.utils.scene_builder.table import TableSceneBuilder
from mani_skill.utils.structs import Pose
from mani_skill.utils.structs.types import Array, GPUMemoryConfig, SimConfig
import mani_skill.envs.utils.randomization as randomization

# Base randomization imports
from .base_random_env import DefaultCameraEnv, DefaultRandomizationConfig
from .robot.so101 import SO101
import sys
import os
try:
    import env_cal
except ImportError:
    env_cal = None

@dataclass
class SeparateRandomizationConfig(DefaultRandomizationConfig):
    robot_qpos_noise_std: float = np.deg2rad(5)
    item_friction_range: Sequence[float] = (0.1, 0.5)
    item_density_range: Sequence[float] = (200, 200)
    randomize_item_color: bool = False


@register_env("SeparateInstruments-v5", max_episode_steps=500)
class Separate(DefaultCameraEnv):
    SUPPORTED_ROBOTS = ["so101", "panda", "fetch"]
    SUPPORTED_OBS_MODES = [
        "none",
        "state",
        "state_dict",
        "rgb",
        "rgb+segmentation",
        "rgb+state",
        "rgb+segmentation+state",
        "rgb+depth+segmentation",
        "rgb+depth+segmentation+state",
    ]
    agent: Union[SO101, Panda, Fetch]

    instrument_spawn_xy_range = 0.02
    instrument_spawn_z_base = 0.008
    instrument_spawn_z_spacing = 0.007
    num_instruments = 4
    block_half_size = [0.0075, 0.085, 0.085]

    def __init__(
        self,
        *args,
        robot_uids="so101",
        control_mode="pd_joint_target_delta_pos",
        robot_init_qpos_noise=0.02,
        domain_randomization=False,
        domain_randomization_config=None,
        **kwargs,
    ):
        self.robot_init_qpos_noise = robot_init_qpos_noise
        self.base_z_rot = 0
        self.rest_qpos = SO101.keyframes["start"].qpos.tolist()

        self.domain_randomization_config = SeparateRandomizationConfig()
        merged_domain_randomization_config = (self.domain_randomization_config.dict())
        if isinstance(domain_randomization_config, dict):
            common.dict_merge(merged_domain_randomization_config, domain_randomization_config)
            self.domain_randomization_config = dacite.from_dict(
                data_class=SeparateRandomizationConfig,
                data=merged_domain_randomization_config,
                config=dacite.Config(strict=True),
            )
        elif isinstance(domain_randomization_config, SeparateRandomizationConfig):
            self.domain_randomization_config = domain_randomization_config

        super().__init__(
            *args,
            robot_uids=robot_uids,
            domain_randomization=domain_randomization,
            domain_randomization_config=self.domain_randomization_config,
            **kwargs,
        )

    @property
    def _default_sim_config(self):
        return SimConfig(
            gpu_memory_config=GPUMemoryConfig(found_lost_pairs_capacity=2**25, max_rigid_patch_count=2**18)
        )

    @property
    def _default_sensor_configs(self):
        pose = sapien_utils.look_at(eye=[0.3, 0, 0.6], target=[-0.1, 0, 0.1])
        return [
            CameraConfig(
                "base_camera",
                pose=pose,
                width=128,
                height=128,
                fov=np.pi / 2,
                near=0.01,
                far=100,
            )
        ]

    @property
    def _default_human_render_camera_configs(self):
        pose = sapien_utils.look_at([0.6, 0.7, 0.6], [0.0, 0.0, 0.35])
        return CameraConfig(
            "render_camera",
            pose=pose,
            width=512,
            height=512,
            fov=1,
            near=0.01,
            far=100,
        )

    def _load_agent(self, options: dict):
        super()._load_agent(
            options,
            sapien.Pose(p=[0, 0, 0], q=euler2quat(0, 0, self.base_z_rot)),
            build_separate=True
            if self.domain_randomization
            and self.domain_randomization_config.robot_color == "random"
            else False,
        )

    def _build_instrument(self, obj_path: str, name: str, initial_pose: sapien.Pose):
        steel_material = sapien.render.RenderMaterial(
            base_color=[0.44, 0.44, 0.44, 1.0],
            roughness=0.15,
            metallic=1.0,
        )
        physx_material = sapien.physx.PhysxMaterial(
            static_friction=0.6,
            dynamic_friction=0.5,
            restitution=0.1,
        )

        builder = self.scene.create_actor_builder()
        builder.add_visual_from_file(filename=obj_path, material=steel_material)
        builder.add_multiple_convex_collisions_from_file(
            filename=obj_path,
            decomposition="coacd",
            material=physx_material,
        )
        builder.initial_pose = initial_pose
        return builder.build(name=name)

    def _sample_instrument_poses(self, b: int, base_pos: torch.Tensor):
        poses = []
        for i in range(self.num_instruments):
            xyz = torch.zeros((b, 3), device=self.device)
            xyz[:, 0] = (base_pos[:, 0] + (torch.rand(b, device=self.device) * 2 - 1) * self.instrument_spawn_xy_range)
            xyz[:, 1] = (base_pos[:, 1] + (torch.rand(b, device=self.device) * 2 - 1) * self.instrument_spawn_xy_range)
            xyz[:, 2] = (base_pos[:, 2] + self.instrument_spawn_z_base + i * self.instrument_spawn_z_spacing)

            yaw = torch.rand(b, device=self.device) * 2 * torch.pi
            q = torch.zeros((b, 4), device=self.device)
            q[:, 0] = torch.cos(yaw / 2)
            q[:, 3] = torch.sin(yaw / 2)
            poses.extend([xyz, q])
        return tuple(poses)

    def _load_scene(self, options: dict):
        
        self.table_scene = TableSceneBuilder(self)
        self.table_scene.build()

        self.table_pose = Pose.create_from_pq(p=[-0.12 + 0.737, 0, -0.9196429], q=euler2quat(0, 0, np.pi / 2))

        blue_material = sapien.render.RenderMaterial(base_color=[0.1, 0.2, 0.85, 1.0], roughness=0.6, metallic=0.0)
        self.table_mat_half_size = [0.40, 0.80, 0.001]
        builder = self.scene.create_actor_builder()
        builder.add_box_visual(half_size=self.table_mat_half_size, material=blue_material)
        builder.initial_pose = sapien.Pose()
        self.table_mat = builder.build_kinematic("table_mat")

        builder = self.scene.create_actor_builder()
        builder.initial_pose = sapien.Pose()
        self.camera_mount = builder.build_kinematic("camera_mount")

        builder = self.scene.create_actor_builder()
        builder.initial_pose = sapien.Pose()
        self.wrist_camera_mount = builder.build_kinematic("wrist_camera_mount")

        bin_path = "/home/aboardman/squint2/deploy_utils/blender_objs/box.obj"
        bin_q = euler2quat(np.pi / 2, 0.0, np.pi / 2)
        bin_steel_material = sapien.render.RenderMaterial(base_color=[1, 1, 1, 1.0], roughness=0.15, metallic=0.5)
        physx_material = sapien.physx.PhysxMaterial(static_friction=0.6, dynamic_friction=0.5, restitution=0.1)
        builder = self.scene.create_actor_builder()
        builder.add_visual_from_file(filename=bin_path, material=bin_steel_material)
        builder.add_multiple_convex_collisions_from_file(filename=bin_path, decomposition="coacd", material=physx_material)
        builder.initial_pose = sapien.Pose(p=[0.0, 0.0, float(self.block_half_size[2]) - 0.03], q=list(bin_q))
        self.bin = builder.build_kinematic("bin")

        inst1_path = "/home/aboardman/squint2/deploy_utils/blender_objs/dressing_forceps.obj"
        inst2_path = "/home/aboardman/squint2/deploy_utils/blender_objs/allis.obj"
        self.obj_1 = self._build_instrument(
            inst1_path,
            name="forceps_1",
            initial_pose=sapien.Pose(p=[-0.1, -0.05, 0.1], q=[1, 0, 0, 0]),
        )
        self.obj_2 = self._build_instrument(
            inst1_path,
            name="forceps_2",
            initial_pose=sapien.Pose(p=[0.1, -0.05, 0.1]),
        )
        self.obj_3 = self._build_instrument(
            inst2_path,
            name="allis_1",
            initial_pose=sapien.Pose(p=[-0.1, 0.05, 0.1]),
        )
        self.obj_4 = self._build_instrument(
            inst2_path,
            name="allis_2",
            initial_pose=sapien.Pose(p=[0.1, 0.05, 0.1]),
        )
        self.objects = [self.obj_1, self.obj_2, self.obj_3, self.obj_4]
        self.obj = self.obj_1

    def _initialize_episode(self, env_idx: torch.Tensor, options: dict):
        super()._initialize_episode(env_idx, options)
        with torch.device(self.device):
            b = len(env_idx)
            
            # 1. Initialize scene (THIS WAS OVERWRITING THE TABLE & ROBOT POSES)
            self.table_scene.initialize(env_idx)
            
            # 2. OVERRIDE Table Scene Poses immediately after initialize()
            self.table_scene.table.set_pose(self.table_pose)

            # 3. Force Robot Base back to origin (0, 0, 0)
            robot_pose = Pose.create_from_pq(
                p=[0, 0, 0], #torch.zeros((b, 3), device=self.device),
                q=euler2quat(0, 0, self.base_z_rot),
            )
            self.agent.robot.set_pose(robot_pose)

            # 4. Position Table Mat relative to the new surface
            mat_pos = torch.zeros((b, 3), device=self.device)
            mat_pos[:, 0] = 0.250
            mat_pos[:, 1] = 0.0
            mat_pos[:, 2] = float(self.table_mat_half_size[2])
            mat_q = (
                torch.tensor([1.0, 0.0, 0.0, 0.0], device=self.device)
                .unsqueeze(0)
                .repeat(b, 1)
            )
            self.table_mat.set_pose(Pose.create_from_pq(p=mat_pos, q=mat_q))

            # 5. Position Bin 0.3m directly in front of the robot base
            bin_pos = torch.zeros((b, 3), device=self.device)
            bin_pos[:, 0] = 0.3
            bin_pos[:, 1] = 0.0
            bin_pos[:, 2] = 0.001 
            
            bin_q = euler2quat(np.pi / 2, 0.0, 0.0)
            q_tensor = (
                torch.tensor(bin_q, device=self.device, dtype=bin_pos.dtype)
                .unsqueeze(0)
                .repeat(b, 1)
            )
            self.bin.set_pose(Pose.create_from_pq(p=bin_pos, q=q_tensor))

            # 6. Spawn instruments relative to bin position
            p1, q1, p2, q2, p3, q3, p4, q4 = self._sample_instrument_poses(b, bin_pos)
            self.obj_1.set_pose(Pose.create_from_pq(p=p1, q=q1))
            self.obj_2.set_pose(Pose.create_from_pq(p=p2, q=q2))
            self.obj_3.set_pose(Pose.create_from_pq(p=p3, q=q3))
            self.obj_4.set_pose(Pose.create_from_pq(p=p4, q=q4))
            self.obj = self.obj_1

    def _get_obs_agent(self):
        qpos = self.agent.robot.get_qpos()
        if (
            self.domain_randomization
            and self.domain_randomization_config.robot_qpos_noise_std > 0
        ):
            noise = (
                torch.randn_like(qpos)
                * self.domain_randomization_config.robot_qpos_noise_std
            )
            qpos = qpos + noise
        obs = dict(noisy_qpos=qpos)
        controller_state = self.agent.controller.get_state()
        if len(controller_state) > 0:
            obs.update(controller=controller_state)
        return obs

    # Distance / Clearance Helper Stubs
    def compute_gripper_to_bin_clearance(self) -> torch.Tensor:
        return torch.linalg.norm(self.agent.tcp_pos - self.bin.pose.p, dim=-1)

    def compute_gripper_to_table_clearance(self) -> torch.Tensor:
        return torch.abs(self.agent.tcp_pos[..., 2] - 0.02)

    def compute_grasped_to_bin_clearance(self, obj) -> torch.Tensor:
        return torch.linalg.norm(obj.pose.p - self.bin.pose.p, dim=-1)

    def is_object_visible(self, obj) -> torch.Tensor:
        return torch.ones(self.num_envs, dtype=torch.bool, device=self.device)

    def is_object_occluded(self, obj) -> torch.Tensor:
        return torch.zeros(self.num_envs, dtype=torch.bool, device=self.device)

    def is_inside_bin(self, obj) -> torch.Tensor:
        bin_xy = self.bin.pose.p[..., :2]
        obj_xy = obj.pose.p[..., :2]
        return torch.linalg.norm(obj_xy - bin_xy, dim=-1) < (
            self.block_half_size[1] + 0.05
        )

    def is_touching_other_object(self, obj) -> torch.Tensor:
        return torch.zeros(self.num_envs, dtype=torch.bool, device=self.device)

    def is_grasped(self, obj) -> torch.Tensor:
        return self.agent.is_grasping(obj)

    def _get_obs_extra(self, info: dict):
        obs = dict(tcp_pose=self.agent.tcp_pose.raw_pose)

        DROP_LOCATION = torch.tensor([0.45, 0.20, 0.02], device=self.device)

        gripper_to_bin = self.compute_gripper_to_bin_clearance()
        gripper_to_table = self.compute_gripper_to_table_clearance()

        all_distances = []
        all_occluded = []

        for obj in self.objects:
            dist = torch.linalg.norm(obj.pose.p - self.agent.tcp_pos, dim=-1)
            occluded = self.is_object_occluded(obj)
            all_distances.append(dist)
            all_occluded.append(occluded)

        distances_tensor = torch.stack(all_distances, dim=0)
        occluded_tensor = torch.stack(all_occluded, dim=0)

        unoccluded_distances = torch.where(
            ~occluded_tensor,
            distances_tensor,
            torch.tensor(float("inf"), device=self.device),
        )

        target_obj_idx = torch.argmin(unoccluded_distances, dim=0)

        all_are_occluded = torch.all(occluded_tensor, dim=0)
        closest_overall_idx = torch.argmin(distances_tensor, dim=0)

        target_obj_idx = torch.where(
            all_are_occluded,
            closest_overall_idx,
            target_obj_idx,
        )

        obs["target_obj"] = target_obj_idx.long() + 1

        for i, obj in enumerate(self.objects):
            prefix = f"obj{i+1}"

            obj_pos = obj.pose.p
            obj_quat = obj.pose.q

            visible = self.is_object_visible(obj)
            occluded = self.is_object_occluded(obj)
            inside_bin = self.is_inside_bin(obj)
            touching = self.is_touching_other_object(obj)
            grasped = self.is_grasped(obj)

            tcp_dist = distances_tensor[i]
            drop_distance = torch.linalg.norm(obj_pos - DROP_LOCATION, dim=-1)

            grasped_bin_dist = self.compute_grasped_to_bin_clearance(obj)
            gripper_min_dist = torch.minimum(gripper_to_table, gripper_to_bin)

            nearest_collision_float = torch.where(
                grasped, grasped_bin_dist, gripper_min_dist
            )
            nearest_collision = nearest_collision_float.round().long()

            obs.update(
                {
                    f"{prefix}_visible": visible.float(),
                    f"{prefix}_occluded": occluded.float(),
                    f"{prefix}_inside_bin": inside_bin.float(),
                    f"{prefix}_touching": touching.float(),
                    f"{prefix}_grasped": grasped.float(),
                    f"{prefix}_nearest_collision": nearest_collision,
                    f"{prefix}_distance_to_gripper": tcp_dist,
                    f"{prefix}_orientation": obj_quat,
                    f"{prefix}_distance_to_drop": drop_distance * grasped.float(),
                }
            )

        return obs

    def evaluate(self):
        poses_xy = [obj.pose.p[..., :2] for obj in self.objects]

        all_separated = torch.ones(
            self.num_envs, dtype=torch.bool, device=self.device
        )
        for i in range(self.num_instruments):
            for j in range(i + 1, self.num_instruments):
                dist = torch.linalg.norm(poses_xy[i] - poses_xy[j], dim=-1)
                all_separated = all_separated & (dist > 0.15)

        bin_xy = self.bin.pose.p[..., :2]
        if bin_xy.ndim == 1:
            bin_xy = bin_xy.unsqueeze(0)

        bin_half_size = self.block_half_size[1]

        all_outside_bin = torch.ones(
            self.num_envs, dtype=torch.bool, device=self.device
        )
        all_on_table = torch.ones(
            self.num_envs, dtype=torch.bool, device=self.device
        )

        for obj in self.objects:
            obj_xy = obj.pose.p[..., :2]
            obj_z = obj.pose.p[..., 2]

            dist_to_bin = torch.linalg.norm(obj_xy - bin_xy, dim=-1)
            outside_bin = dist_to_bin > (bin_half_size + 0.05)
            on_table = (obj_z < 0.04) & (obj_z > -0.01)

            all_outside_bin = all_outside_bin & outside_bin
            all_on_table = all_on_table & on_table

        success = all_separated & all_outside_bin & all_on_table

        return {
            "success": success,
            "all_separated": all_separated,
            "all_outside_bin": all_outside_bin,
            "all_on_table": all_on_table,
        }

    def compute_dense_reward(self, obs: Any, action: Any, info: dict):
        reward = torch.zeros((self.num_envs,), device=self.device)
        DROP_LOCATION = torch.tensor([0.45, 0.20, 0.02], device=self.device)

        batch_idx = torch.arange(self.num_envs, device=self.device)

        obs_extra = obs["extra"] if "extra" in obs else obs
        target_idx = obs_extra["target_obj"] - 1

        target_grasped = torch.stack(
            [obs_extra[f"obj{i+1}_grasped"] for i in range(len(self.objects))],
            dim=1,
        )[batch_idx, target_idx]
        target_tcp_dist = torch.stack(
            [
                obs_extra[f"obj{i+1}_distance_to_gripper"]
                for i in range(len(self.objects))
            ],
            dim=1,
        )[batch_idx, target_idx]
        target_nearest_collision = torch.stack(
            [
                obs_extra[f"obj{i+1}_nearest_collision"]
                for i in range(len(self.objects))
            ],
            dim=1,
        )[batch_idx, target_idx]

        collision_safe_margin = 2.0
        collision_penalty = torch.clamp(
            collision_safe_margin - target_nearest_collision.float(), min=0.0
        )
        reward -= 0.5 * collision_penalty

        reach_reward = (1.0 - torch.tanh(5.0 * target_tcp_dist)) * (
            1.0 - target_grasped
        )
        reward += 1.0 * reach_reward

        reward += 2.5 * target_grasped

        target_obj_positions = torch.stack(
            [obj.pose.p for obj in self.objects], dim=1
        )
        target_pos = target_obj_positions[batch_idx, target_idx]
        drop_dist = torch.linalg.norm(target_pos - DROP_LOCATION, dim=-1)

        placement_reward = (1.0 - torch.tanh(3.0 * drop_dist)) * target_grasped
        reward += 4.0 * placement_reward

        in_drop_zone = (drop_dist < 0.05).float()
        successful_placement = in_drop_zone * (1.0 - target_grasped)
        reward += 10.0 * successful_placement

        reward[info["success"]] += 15.0

        return reward

    def compute_normalized_dense_reward(self, obs: Any, action: Any, info: dict):
        return self.compute_dense_reward(obs=obs, action=action, info=info) / 32.5