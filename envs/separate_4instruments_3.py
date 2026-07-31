from dataclasses import asdict, dataclass
from typing import Any, Optional, Sequence, Union

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
from mani_skill.utils.structs.types import GPUMemoryConfig, SimConfig

# Base randomization imports
from .base_random_env import DefaultCameraEnv, DefaultRandomizationConfig
from .robot.so101 import SO101


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
    DROP_LOCATION = np.array([0.25, -0.30, 0.01])
    DROP_ZONE_HEIGHT = 0.15
    DROP_ZONE_WIDTH = 0.20

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
        merged_domain_randomization_config = self.domain_randomization_config.dict()
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
            control_mode=control_mode,
            domain_randomization=domain_randomization,
            domain_randomization_config=self.domain_randomization_config,
            **kwargs,
        )

    @property
    def _default_sim_config(self):
        return SimConfig(
            gpu_memory_config=GPUMemoryConfig(found_lost_pairs_capacity=2**25, max_rigid_patch_count=2**18)
        )

    def _load_agent(self, options: dict):
        # load the robot arm at this initial pose
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
            xyz[:, 0] = (
                base_pos[:, 0]
                + (torch.rand(b, device=self.device) * 2 - 1)
                * self.instrument_spawn_xy_range
            )
            xyz[:, 1] = (
                base_pos[:, 1]
                + (torch.rand(b, device=self.device) * 2 - 1)
                * self.instrument_spawn_xy_range
            )
            xyz[:, 2] = (
                base_pos[:, 2]
                + self.instrument_spawn_z_base
                + i * self.instrument_spawn_z_spacing
            )

            yaw = torch.rand(b, device=self.device) * 2 * torch.pi
            q = torch.zeros((b, 4), device=self.device)
            q[:, 0] = torch.cos(yaw / 2)
            q[:, 3] = torch.sin(yaw / 2)
            poses.extend([xyz, q])
        return tuple(poses)

    def _build_drop_zone_outline(self, half_width: float, half_height: float, thickness: float = 0.0025):
        """Creates a thin rectangular border marking the drop zone on the table surface."""
        builder = self.scene.create_actor_builder()
        green_material = sapien.render.RenderMaterial(
            base_color=[0.0, 0.8, 0.2, 0.8],
            roughness=0.1,
            metallic=0.0
        )
        builder.add_box_visual(
            pose=sapien.Pose(p=[0.0, half_height, 0.0]),
            half_size=[half_width + thickness, thickness, thickness],
            material=green_material,
        )
        builder.add_box_visual(
            pose=sapien.Pose(p=[0.0, -half_height, 0.0]),
            half_size=[half_width + thickness, thickness, thickness],
            material=green_material,
        )
        builder.add_box_visual(
            pose=sapien.Pose(p=[half_width, 0.0, 0.0]),
            half_size=[thickness, half_height, thickness],
            material=green_material,
        )
        builder.add_box_visual(
            pose=sapien.Pose(p=[-half_width, 0.0, 0.0]),
            half_size=[thickness, half_height, thickness],
            material=green_material,
        )
        builder.initial_pose = sapien.Pose()
        return builder.build_kinematic("drop_zone_outline")

    def _load_scene(self, options: dict):
        self.table_scene = TableSceneBuilder(self)
        self.table_scene.build()
        self.table_pose = Pose.create_from_pq(p=[-0.12 + 0.737, 0, -0.9196429], q=euler2quat(0, 0, np.pi / 2))
        
        self.drop_zone_visual = self._build_drop_zone_outline(
            half_width=self.DROP_ZONE_WIDTH, 
            half_height=self.DROP_ZONE_HEIGHT, 
            thickness=0.00025
        )

        blue_material = sapien.render.RenderMaterial(base_color=[0.1, 0.2, 0.85, 1.0], roughness=0.6, metallic=0.0)
        self.table_mat_half_size = [0.40, 0.80, 0.001]
        builder = self.scene.create_actor_builder()
        builder.add_box_visual(half_size=self.table_mat_half_size, material=blue_material)
        builder.initial_pose = sapien.Pose()
        self.table_mat = builder.build_kinematic("table_mat")

        # builder = self.scene.create_actor_builder()
        # builder.initial_pose = sapien.Pose()
        # self.camera_mount = builder.build_kinematic("camera_mount")

        # builder = self.scene.create_actor_builder()
        # builder.initial_pose = sapien.Pose()
        # self.wrist_camera_mount = builder.build_kinematic("wrist_camera_mount")

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

        self._load_camera_mount()

        self._randomize_robot_color()


    def _initialize_episode(self, env_idx: torch.Tensor, options: dict):
        super()._initialize_episode(env_idx, options)
        with torch.device(self.device):
            b = len(env_idx)
            self.table_scene.initialize(env_idx)
            self.table_scene.table.set_pose(self.table_pose)

            if hasattr(self.table_scene, "table"):
                table_z = self.table_scene.table.pose.p[..., 2]
                if table_z.ndim == 0:
                    table_z = table_z.unsqueeze(0)
                table_z = table_z + 0.92
            else:
                table_z = torch.full((b,), float(self.block_half_size[2]) + 0.92, device=self.device)
            mat_pos = torch.zeros((b, 3), device=self.device)
            mat_pos[:, 0] = 0.450
            mat_pos[:, 1] = -0.275
            mat_pos[:, 2] = table_z + float(self.table_mat_half_size[2])
            mat_q = (torch.tensor([1.0, 0.0, 0.0, 0.0], device=self.device).unsqueeze(0).repeat(b, 1))
            self.table_mat.set_pose(Pose.create_from_pq(p=mat_pos, q=mat_q))

            center = self.agent.robot.pose.p + torch.tensor([0.3, 0.0, 0.0], device=self.device)
            center = center[env_idx]
            bin_pos = center.clone()
            bin_pos[:, 2] = float(self.block_half_size[2]) - 0.03
            bin_q = euler2quat(np.pi / 2, 0.0, 0.0)
            q_tensor = torch.tensor(bin_q, device=self.device, dtype=bin_pos.dtype)
            q_tensor = q_tensor.unsqueeze(0).repeat(b, 1)
            bin_pose = Pose.create_from_pq(p=bin_pos, q=q_tensor)
            self.bin.set_pose(bin_pose)

            spawn_base = bin_pos.clone()
            spawn_base[:, 2] += 0.03
            p1, q1, p2, q2, p3, q3, p4, q4 = self._sample_instrument_poses(b, spawn_base)
            self.obj_1.set_pose(Pose.create_from_pq(p=p1, q=q1))
            self.obj_2.set_pose(Pose.create_from_pq(p=p2, q=q2))
            self.obj_3.set_pose(Pose.create_from_pq(p=p3, q=q3))
            self.obj_4.set_pose(Pose.create_from_pq(p=p4, q=q4))
            self.obj = self.obj_1

            drop_pos = torch.tensor(self.DROP_LOCATION, device=self.device, dtype=torch.float32).repeat(b, 1)
            drop_q = torch.tensor([1.0, 0.0, 0.0, 0.0], device=self.device).repeat(b, 1)
            self.drop_zone_visual.set_pose(Pose.create_from_pq(p=drop_pos, q=drop_q))

    def _get_obs_agent(self):
        qpos = self.agent.robot.get_qpos()
        if (self.domain_randomization and self.domain_randomization_config.robot_qpos_noise_std > 0):
            noise = (torch.randn_like(qpos) * self.domain_randomization_config.robot_qpos_noise_std)
            qpos = qpos + noise
        obs = dict(noisy_qpos=qpos)
        controller_state = self.agent.controller.get_state()
        if len(controller_state) > 0:
            obs.update(controller=controller_state)
        return obs

    def is_object_visible(self, obj):
        """
        Returns a boolean tensor [num_envs] indicating whether the object's center lies inside the wrist camera's viewing frustum.
        """
        cam_pose = self.wrist_camera_mount.pose
        cam_xyz = cam_pose.inv() * obj.pose

        x = cam_xyz.p[:, 0]
        y = cam_xyz.p[:, 1]
        z = cam_xyz.p[:, 2]

        hfov = self.WRIST_CAMERA_FOV
        vfov = self.WRIST_CAMERA_FOV
        tan_half_h = torch.tan(torch.tensor(hfov / 2, device=self.device))
        tan_half_v = torch.tan(torch.tensor(vfov / 2, device=self.device))
        visible = ((z > 0.01) & (torch.abs(x / z) < tan_half_h) & (torch.abs(y / z) < tan_half_v))
        return visible

    def is_object_occluded(self, obj):
        """
        Returns True if another object is closer to the wrist camera and projects to almost the same image location.
        """
        cam_pose = self.wrist_camera_mount.pose
        cam_xyz = cam_pose.inv() * obj.pose
        x = cam_xyz.p[:, 0]
        y = cam_xyz.p[:, 1]
        z = cam_xyz.p[:, 2]
        occluded = torch.zeros(self.num_envs, dtype=torch.bool, device=self.device)
        # if not self.is_object_visible(obj).any():
        #     return ~self.is_object_visible(obj)
        target_u = x / z
        target_v = y / z

        PIXEL_THRESHOLD = 0.03

        for other in self.objects:
            if other == obj:
                continue
            other_cam = cam_pose.inv() * other.pose
            ox = other_cam.p[:, 0]
            oy = other_cam.p[:, 1]
            oz = other_cam.p[:, 2]
            other_visible = ((oz > 0.01))
            other_u = ox / oz
            other_v = oy / oz
            overlap = (torch.abs(other_u - target_u) < PIXEL_THRESHOLD) & (torch.abs(other_v - target_v) < PIXEL_THRESHOLD)
            closer = oz < z
            occluded |= overlap & closer & other_visible
        return occluded

    def is_inside_bin(self, obj) -> torch.Tensor:
        bin_xy = self.bin.pose.p[..., :2]
        obj_xy = obj.pose.p[..., :2]
        return torch.linalg.norm(obj_xy - bin_xy, dim=-1) < (self.block_half_size[1] + 0.05)

    def _get_obs_extra(self, info: dict):
        obs = dict(tcp_pose=self.agent.tcp_pose.raw_pose)

        DROP_LOCATION = torch.tensor(self.DROP_LOCATION, device=self.device, dtype=torch.float32)

        gripper_to_bin = torch.linalg.norm(self.agent.tcp_pos - self.bin.pose.p, dim=-1)
        gripper_to_table = torch.abs(self.agent.tcp_pos[..., 2] - 0.02)

        all_distances = []
        all_occluded = []

        for obj in self.objects:
            dist = torch.linalg.norm(obj.pose.p - self.agent.tcp_pos, dim=-1)
            occluded = self.is_object_occluded(obj)
            all_distances.append(dist)
            all_occluded.append(occluded)

        distances_tensor = torch.stack(all_distances, dim=0)
        occluded_tensor = torch.stack(all_occluded, dim=0)

        unoccluded_distances = torch.where(~occluded_tensor, distances_tensor, torch.tensor(float("inf"), device=self.device))

        target_obj_idx = torch.argmin(unoccluded_distances, dim=0)
        all_are_occluded = torch.all(occluded_tensor, dim=0)
        closest_overall_idx = torch.argmin(distances_tensor, dim=0)
        target_obj_idx = torch.where(all_are_occluded, closest_overall_idx, target_obj_idx)
        obs["target_obj"] = target_obj_idx.long()

        for i, obj in enumerate(self.objects):
            prefix = f"obj{i+1}"

            obj_pos = obj.pose.p
            obj_quat = obj.pose.q

            visible = self.is_object_visible(obj)
            occluded = self.is_object_occluded(obj)
            inside_bin = self.is_inside_bin(obj)
            touching = torch.zeros(self.num_envs, dtype=torch.bool, device=self.device)
            grasped = self.agent.is_grasping(obj)

            tcp_dist = distances_tensor[i]
            drop_distance = torch.linalg.norm(obj_pos - DROP_LOCATION, dim=-1)

            grasped_bin_dist = torch.linalg.norm(obj.pose.p - self.bin.pose.p, dim=-1)
            gripper_min_dist = torch.minimum(gripper_to_table, gripper_to_bin)

            nearest_collision_float = torch.where(grasped, grasped_bin_dist, gripper_min_dist)
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

        all_separated = torch.ones(self.num_envs, dtype=torch.bool, device=self.device)
        for i in range(self.num_instruments):
            for j in range(i + 1, self.num_instruments):
                dist = torch.linalg.norm(poses_xy[i] - poses_xy[j], dim=-1)
                all_separated = all_separated & (dist > 0.15)

        bin_xy = self.bin.pose.p[..., :2]
        if bin_xy.ndim == 1:
            bin_xy = bin_xy.unsqueeze(0)

        bin_half_size = self.block_half_size[1]

        all_outside_bin = torch.ones(self.num_envs, dtype=torch.bool, device=self.device)
        all_on_table = torch.ones(self.num_envs, dtype=torch.bool, device=self.device)

        for obj in self.objects:
            obj_xy = obj.pose.p[..., :2]
            obj_z = obj.pose.p[..., 2]

            dist_to_bin = torch.linalg.norm(obj_xy - bin_xy, dim=-1)
            outside_bin = dist_to_bin > (bin_half_size + 0.05)
            on_table = (obj_z < 0.04) & (obj_z > -0.01)

            all_outside_bin = all_outside_bin & outside_bin
            all_on_table = all_on_table & on_table

        success = all_outside_bin & all_on_table

        return {
            "success": success,
            #"all_separated": all_separated,
            "all_outside_bin": all_outside_bin,
            "all_on_table": all_on_table,
        }

    def compute_dense_reward(self, obs: Any, action: Any, info: dict):
        reward = torch.zeros((self.num_envs,), device=self.device)
        DROP_LOCATION = torch.tensor(self.DROP_LOCATION, device=self.device, dtype=torch.float32)

        batch_idx = torch.arange(self.num_envs, device=self.device)

        # Handle different obs dictionary structures (ManiSkill wraps extra info in "extra")
        obs_extra = obs["extra"] if "extra" in obs else obs
        target_idx = obs_extra["target_obj"]
        target_visible = torch.stack(
            [obs_extra[f"obj{i+1}_visible"] for i in range(len(self.objects))],
            dim=1,
        )[batch_idx, target_idx]

        reward += 0.5 * target_visible

        target_grasped = torch.stack([obs_extra[f"obj{i+1}_grasped"] for i in range(len(self.objects))], dim=1)[batch_idx, target_idx]
        target_tcp_dist = torch.stack([obs_extra[f"obj{i+1}_distance_to_gripper"] for i in range(len(self.objects))], dim=1)[batch_idx, target_idx]
        target_nearest_collision = torch.stack([obs_extra[f"obj{i+1}_nearest_collision"] for i in range(len(self.objects))], dim=1)[batch_idx, target_idx]

        # Penalizes getting dangerously close (< 2.0 integer clearance units) to table or bin
        collision_safe_margin = 0.02
        collision_penalty = torch.clamp(collision_safe_margin - target_nearest_collision.float(), min=0.0)
        reward -= 0.5 * collision_penalty

        # Continuous reach reward (0 to 1) when not holding target
        reach_reward = (1.0 - torch.tanh(5.0 * target_tcp_dist)) * (1.0 - target_grasped)
        reward += 1.0 * reach_reward

        # High discrete reward for active target grasp
        reward += 2.5 * target_grasped

        # Distance from target object to desired drop location
        target_obj_positions = torch.stack([obj.pose.p for obj in self.objects], dim=1) 
        target_pos = target_obj_positions[batch_idx, target_idx]
        drop_dist = torch.linalg.norm(target_pos - DROP_LOCATION, dim=-1)

        # Continuous placement reward (strongest when grasped and moved toward target area)
        placement_reward = (1.0 - torch.tanh(3.0 * drop_dist)) * target_grasped
        reward += 4.0 * placement_reward

        # Bonus for dropping/releasing target within drop zone threshold (< 0.05m)
        in_drop_zone = (drop_dist < 0.05).float()
        successful_placement = in_drop_zone * (1.0 - target_grasped)
        reward += 10.0 * successful_placement

        # 5. Success Bonus
        reward[info["success"]] += 15.0

        return reward

    def compute_normalized_dense_reward(self, obs: Any, action: Any, info: dict):
        return self.compute_dense_reward(obs=obs, action=action, info=info) / 32.5

