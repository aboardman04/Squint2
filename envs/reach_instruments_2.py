from dataclasses import asdict, dataclass
from typing import Any, Optional, Sequence, Union

import dacite
import numpy as np
import sapien
import torch
from transforms3d.euler import euler2quat

from mani_skill.agents.robots import Fetch, Panda
from mani_skill.sensors.camera import CameraConfig
from mani_skill.utils import common, sapien_utils
from mani_skill.utils.registration import register_env
from mani_skill.utils.scene_builder.table import TableSceneBuilder
from mani_skill.utils.structs import Pose
from mani_skill.utils.structs.types import GPUMemoryConfig, SimConfig
from .base_random_env import DefaultCameraEnv, DefaultRandomizationConfig
from .robot.so101 import SO101


@dataclass
class SeparateRandomizationConfig(DefaultRandomizationConfig):
    robot_qpos_noise_std: float = np.deg2rad(5)
    item_friction_range: Sequence[float] = (0.1, 0.5)
    item_density_range: Sequence[float] = (200, 200)
    randomize_item_color: bool = False


@register_env("ReachInstruments-v1", max_episode_steps=50)
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
        domain_randomization_config: Union[
            SeparateRandomizationConfig, dict
        ] = SeparateRandomizationConfig(),
        domain_randomization=False,
        **kwargs,
    ):
        self.base_z_rot = 0
        self.rest_qpos = SO101.keyframes["start"].qpos.tolist()

        self.domain_randomization_config = SeparateRandomizationConfig()
        merged_domain_randomization_config = asdict(self.domain_randomization_config)
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

    # @property
    # def _default_sim_config(self):
    #     return SimConfig(
    #         gpu_memory_config=GPUMemoryConfig(found_lost_pairs_capacity=2**25, max_rigid_patch_count=2**18)
    #     )

    def _load_agent(self, options: dict):
        super()._load_agent(
            options,
            sapien.Pose(p=[0, 0, 0], q=euler2quat(0, 0, self.base_z_rot)),
            build_separate=True
            if self.domain_randomization
            and getattr(self.domain_randomization_config, "robot_color", None) == "random"
            else False,
        )

    def _get_mesh_center(self, obj_path: str) -> np.ndarray:
        vertices = []
        with open(obj_path, "r", encoding="utf-8") as f:
            for line in f:
                line = line.strip()
                if line.startswith("v "):
                    parts = line.split()
                    if len(parts) >= 4:
                        vertices.append(np.array(parts[1:4], dtype=np.float32))
        if not vertices:
            return np.zeros(3, dtype=np.float32)
        return np.mean(np.stack(vertices, axis=0), axis=0)

    def _build_instrument(self, obj_path: str, name: str, initial_pose: sapien.Pose):
        steel_material = sapien.render.RenderMaterial(base_color=[0.44, 0.44, 0.44, 1.0], roughness=0.15, metallic=1.0)
        physx_material = sapien.physx.PhysxMaterial(static_friction=0.6, dynamic_friction=0.5, restitution=0.1)
        builder = self.scene.create_actor_builder()
        builder.add_visual_from_file(filename=obj_path, material=steel_material)
        builder.add_multiple_convex_collisions_from_file(filename=obj_path, material=physx_material)

        mesh_center = self._get_mesh_center(obj_path)
        pose_pos = np.array(initial_pose.p, dtype=np.float32)
        translated_pose = sapien.Pose(
            p=(pose_pos - mesh_center).tolist(),
            q=initial_pose.q,
        )
        builder.initial_pose = translated_pose
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

    # def _build_drop_zone_outline(self, half_width: float, half_height: float, thickness: float = 0.0025):
    #     builder = self.scene.create_actor_builder()
    #     green_material = sapien.render.RenderMaterial(base_color=[0.0, 0.8, 0.2, 0.8], roughness=0.1, metallic=0.0)
    #     builder.add_box_visual(pose=sapien.Pose(p=[0.0, half_height, 0.0]), half_size=[half_width + thickness, thickness, thickness], material=green_material)
    #     builder.add_box_visual(pose=sapien.Pose(p=[0.0, -half_height, 0.0]), half_size=[half_width + thickness, thickness, thickness], material=green_material)
    #     builder.add_box_visual(pose=sapien.Pose(p=[half_width, 0.0, 0.0]), half_size=[thickness, half_height, thickness], material=green_material)
    #     builder.add_box_visual(pose=sapien.Pose(p=[-half_width, 0.0, 0.0]), half_size=[thickness, half_height, thickness], material=green_material)
    #     builder.initial_pose = sapien.Pose()
    #     return builder.build_kinematic("drop_zone_outline")

    def _load_scene(self, options: dict):
        self.table_scene = TableSceneBuilder(self)
        self.table_scene.build()
        self.table_pose = Pose.create_from_pq(p=[-0.12 + 0.737, 0, -0.9196429], q=euler2quat(0, 0, np.pi / 2))
        
        # self.drop_zone_visual = self._build_drop_zone_outline(half_width=self.DROP_ZONE_WIDTH, half_height=self.DROP_ZONE_HEIGHT, thickness=0.00025)

        blue_material = sapien.render.RenderMaterial(base_color=[0.1, 0.2, 0.85, 1.0], roughness=0.6, metallic=0.0)
        self.table_mat_half_size = [0.40, 0.80, 0.001]
        builder = self.scene.create_actor_builder()
        builder.add_box_visual(half_size=self.table_mat_half_size, material=blue_material)
        builder.initial_pose = sapien.Pose()
        self.table_mat = builder.build_kinematic("table_mat")

        # bin_path = "/home/aboardman/squint2/deploy_utils/blender_objs/box.obj"
        # bin_q = euler2quat(np.pi / 2, 0.0, np.pi / 2)
        # bin_steel_material = sapien.render.RenderMaterial(base_color=[1, 1, 1, 1.0], roughness=0.15, metallic=0.5)
        # physx_material = sapien.physx.PhysxMaterial(static_friction=0.6, dynamic_friction=0.5, restitution=0.1)
        # builder = self.scene.create_actor_builder()
        # builder.add_visual_from_file(filename=bin_path, material=bin_steel_material)
        # builder.add_multiple_convex_collisions_from_file(filename=bin_path, decomposition="coacd", material=physx_material)
        # builder.initial_pose = sapien.Pose(p=[0.0, 0.0, float(self.block_half_size[2]) - 0.03], q=list(bin_q))
        # self.bin = builder.build_kinematic("bin")

        inst1_path = "/home/aboardman/squint2/deploy_utils/blender_objs/dressing_forceps.obj"
        # inst2_path = "/home/aboardman/squint2/deploy_utils/blender_objs/allis.obj"
        self.obj_1 = self._build_instrument(
            inst1_path,
            name="forceps_1",
            initial_pose=sapien.Pose(p=[-0.1, -0.05, 0.1], q=[1, 0, 0, 0]),
        )
        # self.obj_2 = self._build_instrument(
        #     inst1_path,
        #     name="forceps_2",
        #     initial_pose=sapien.Pose(p=[0.1, -0.05, 0.2]),
        # )
        # self.obj_3 = self._build_instrument(
        #     inst2_path,
        #     name="allis_1",
        #     initial_pose=sapien.Pose(p=[-0.1, 0.05, 0.3]),
        # )
        # self.obj_4 = self._build_instrument(
        #     inst2_path,
        #     name="allis_2",
        #     initial_pose=sapien.Pose(p=[0.1, 0.05, 0.4]),
        # )
        # self.objects = [self.obj_1, self.obj_2, self.obj_3, self.obj_4]
        self.obj = self.obj_1
        self.objects = [self.obj_1]

        if hasattr(self, "_load_camera_mount"):
            self._load_camera_mount()

        if hasattr(self, "_randomize_robot_color"):
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
            # bin_q = euler2quat(np.pi / 2, 0.0, 0.0)
            # q_tensor = torch.tensor(bin_q, device=self.device, dtype=bin_pos.dtype)
            # q_tensor = q_tensor.unsqueeze(0).repeat(b, 1)
            # bin_pose = Pose.create_from_pq(p=bin_pos, q=q_tensor)
            # self.bin.set_pose(bin_pose)

            spawn_base = bin_pos.clone()
            spawn_base[:, 2] += 0.03
            p1, q1, p2, q2, p3, q3, p4, q4 = self._sample_instrument_poses(b, spawn_base)
            self.obj_1.set_pose(Pose.create_from_pq(p=p1, q=q1))
            # self.obj_2.set_pose(Pose.create_from_pq(p=p2, q=q2))
            # self.obj_3.set_pose(Pose.create_from_pq(p=p3, q=q3))
            # self.obj_4.set_pose(Pose.create_from_pq(p=p4, q=q4))
            self.obj = self.obj_1

            # drop_pos = torch.tensor(self.DROP_LOCATION, device=self.device, dtype=torch.float32).repeat(b, 1)
            # drop_q = torch.tensor([1.0, 0.0, 0.0, 0.0], device=self.device).repeat(b, 1)
            # self.drop_zone_visual.set_pose(Pose.create_from_pq(p=drop_pos, q=drop_q))
            
            self.target_obj_idx = torch.zeros((b,), dtype=torch.long, device=self.device)
            # self.completed_objects = torch.zeros((b, self.num_instruments), dtype=torch.bool, device=self.device)
            # self.target_act_obj_idx = torch.full((b,), -1, dtype=torch.long, device=self.device)

    def _get_obs_agent(self):
        qpos = self.agent.robot.get_qpos()
        if self.domain_randomization and getattr(self.domain_randomization_config, "robot_qpos_noise_std", 0) > 0:
            noise = (torch.randn_like(qpos) * self.domain_randomization_config.robot_qpos_noise_std)
            qpos = qpos + noise
        obs = dict(noisy_qpos=qpos)
        controller_state = self.agent.controller.get_state()
        if len(controller_state) > 0:
            obs.update(controller=controller_state)
        return obs

    def is_object_visible(self, obj):
        cam_mount = getattr(self, "wrist_camera_mount", self.agent.robot)
        cam_pose = cam_mount.pose
        cam_xyz = cam_pose.inv() * obj.pose

        x = cam_xyz.p[:, 0]
        y = cam_xyz.p[:, 1]
        z = cam_xyz.p[:, 2]

        fov = getattr(self, "WRIST_CAMERA_FOV", np.pi / 3)
        tan_half = torch.tan(torch.tensor(fov / 2, device=self.device))
        visible = ((z > 0.01) & (torch.abs(x / z) < tan_half) & (torch.abs(y / z) < tan_half))
        return visible

    # def is_object_occluded(self, obj):
    #     cam_mount = getattr(self, "wrist_camera_mount", self.agent.robot)
    #     cam_pose = cam_mount.pose
    #     cam_xyz = cam_pose.inv() * obj.pose
    #     x = cam_xyz.p[:, 0]
    #     y = cam_xyz.p[:, 1]
    #     z = cam_xyz.p[:, 2]
    #     occluded = torch.zeros(self.num_envs, dtype=torch.bool, device=self.device)
    #     target_u = x / z
    #     target_v = y / z

    #     PIXEL_THRESHOLD = 0.03

    #     for other in self.objects:
    #         if other == obj:
    #             continue
    #         other_cam = cam_pose.inv() * other.pose
    #         ox = other_cam.p[:, 0]
    #         oy = other_cam.p[:, 1]
    #         oz = other_cam.p[:, 2]
    #         other_visible = (oz > 0.01)
    #         other_u = ox / oz
    #         other_v = oy / oz
    #         overlap = (torch.abs(other_u - target_u) < PIXEL_THRESHOLD) & (torch.abs(other_v - target_v) < PIXEL_THRESHOLD)
    #         closer = oz < z
    #         occluded |= overlap & closer & other_visible
    #     return occluded

    # def is_inside_bin(self, obj) -> torch.Tensor:
    #     bin_xy = self.bin.pose.p[..., :2]
    #     obj_xy = obj.pose.p[..., :2]
    #     return torch.linalg.norm(obj_xy - bin_xy, dim=-1) < (self.block_half_size[1] + 0.05)

    # def _assign_target_object(self):
    #     obj_positions = torch.stack([obj.pose.p for obj in self.objects], dim=1)
    #     tcp_pos = self.agent.tcp_pos.unsqueeze(1)
    #     distances = torch.linalg.norm(obj_positions - tcp_pos, dim=-1)
    #     occluded = torch.stack([self.is_object_occluded(obj) for obj in self.objects], dim=1)
    #     available = ~self.completed_objects.bool()

    #     unobcluded_distances = torch.where(
    #         (~occluded) & available,
    #         distances,
    #         torch.full_like(distances, float("inf")),
    #     )
    #     target_obj_idx_unobstructed = torch.argmin(unobcluded_distances, dim=1)

    #     available_distances = torch.where(
    #         available,
    #         distances,
    #         torch.full_like(distances, float("inf")),
    #     )
    #     target_obj_idx_closest = torch.argmin(available_distances, dim=1)

    #     has_unobstructed = ((~occluded) & available).any(dim=1)
    #     self.target_obj_idx = torch.where(
    #         has_unobstructed,
    #         target_obj_idx_unobstructed,
    #         target_obj_idx_closest,
    #     ).long()

    #     no_available_targets = (~available).all(dim=1)
    #     self.target_obj_idx = torch.where(
    #         no_available_targets,
    #         torch.full_like(self.target_obj_idx, -1),
    #         self.target_obj_idx,
    #     )

    # def _update_active_target(self):
    #     batch_idx = torch.arange(self.num_envs, device=self.device)
    #     visible = torch.stack(
    #         [self.is_object_visible(obj) for obj in self.objects],
    #         dim=1,
    #     )

    #     valid_mask = self.target_obj_idx >= 0
    #     target_visible = torch.zeros(self.num_envs, dtype=torch.bool, device=self.device)
    #     if valid_mask.any():
    #         target_visible[valid_mask] = visible[batch_idx[valid_mask], self.target_obj_idx[valid_mask]]

    #     self.target_act_obj_idx = torch.where(
    #         target_visible,
    #         self.target_obj_idx,
    #         self.target_act_obj_idx,
    #     )

    def _get_obs_extra(self, info: dict):
        obs = dict(tcp_pose=self.agent.tcp_pose.raw_pose)

        obj_positions = torch.stack([obj.pose.p for obj in self.objects], dim=1)
        tcp_pos_expanded = self.agent.tcp_pos.unsqueeze(1)
        instrument_distances = torch.linalg.norm(obj_positions - tcp_pos_expanded, dim=-1)

        # grasped_states = []
        # for obj in self.objects:
        #     is_grasped = self.agent.is_grasping(obj)
        #     if isinstance(is_grasped, bool):
        #         is_grasped = torch.tensor([is_grasped], device=self.device).repeat(self.num_envs)
        #     grasped_states.append(is_grasped.float())

        # grasped_tensor = torch.stack(grasped_states, dim=1)
        # any_grasped = (grasped_tensor > 0.5).any(dim=1)

        gripper_to_table_dist = torch.abs(self.agent.tcp_pos[..., 2] - 0.02)
        # gripper_to_bin_dist = torch.linalg.norm(self.agent.tcp_pos[..., :2] - self.bin.pose.p[..., :2], dim=-1)

        gripper_touching_table = (gripper_to_table_dist < 0.005).float()
        # gripper_touching_bin = ((gripper_to_bin_dist < (self.block_half_size[1] + 0.01)) & (self.agent.tcp_pos[..., 2] < 0.1)).float()

        # grasped_obj_touching_bin = torch.zeros(self.num_envs, device=self.device)
        # for i, obj in enumerate(self.objects):
        #     obj_to_bin_dist = torch.linalg.norm(obj.pose.p[..., :2] - self.bin.pose.p[..., :2], dim=-1)
        #     obj_in_bin_wall = (obj_to_bin_dist < (self.block_half_size[1] + 0.01)) & (obj.pose.p[..., 2] < 0.05)
        #     grasped_obj_touching_bin = torch.where(
        #         (grasped_tensor[:, i] > 0.5) & obj_in_bin_wall, 
        #         torch.tensor(1.0, device=self.device), 
        #         grasped_obj_touching_bin
        #     )

        # grasped_obj_lifted = any_grasped & (grasped_obj_touching_bin < 0.5)

        obs.update({
            "instrument_distances": instrument_distances,
            # "is_any_grasped": any_grasped.float(),
            "gripper_touching_table": gripper_touching_table,
            # "gripper_touching_bin": gripper_touching_bin,
            # "grasped_obj_touching_bin": grasped_obj_touching_bin,
            # "grasped_obj_lifted": grasped_obj_lifted.float(),
            "target_obj": self.target_obj_idx,
            # "target_act_obj": self.target_act_obj_idx,
        })

        num_objects = instrument_distances.shape[1]
        for i in range(num_objects):
            obs[f"obj{i+1}_distance_to_gripper"] = instrument_distances[:, i]
            # obs[f"obj{i+1}_grasped"] = grasped_tensor[:, i]

        return obs

    # def _target_in_target_area(self, target_idx=None):
    #     batch_idx = torch.arange(self.num_envs, device=self.device)
    #     if target_idx is None:
    #         target_idx = self.target_act_obj_idx.clamp(min=0)

    #     obj_positions = torch.stack([obj.pose.p for obj in self.objects], dim=1)
    #     target_pos = obj_positions[batch_idx, target_idx]

    #     within_target_x = torch.abs(target_pos[:, 0] - self.DROP_LOCATION[0]) <= (self.DROP_ZONE_WIDTH / 2.0)
    #     within_target_y = torch.abs(target_pos[:, 1] - self.DROP_LOCATION[1]) <= (self.DROP_ZONE_HEIGHT / 2.0)
    #     resting_on_table = (target_pos[:, 2] < 0.04) & (target_pos[:, 2] > -0.01)

    #     return within_target_x & within_target_y & resting_on_table

    def evaluate(self):
        # bin_xy = self.bin.pose.p[..., :2]
        # if bin_xy.ndim == 1:
        #     bin_xy = bin_xy.unsqueeze(0)

        # self._assign_target_object()
        # self._update_active_target()

        batch_idx = torch.arange(self.num_envs, device=self.device)
        obs_extra = self._get_obs_extra(info={})
        target_idx = self.target_obj_idx.clamp(min=0)
        reach_threshold = 0.05

        # all_outside_bin = torch.ones(self.num_envs, dtype=torch.bool, device=self.device)
        # all_on_table = torch.ones(self.num_envs, dtype=torch.bool, device=self.device)

        target_in_view = torch.stack([self.is_object_visible(obj) for obj in self.objects], dim=1)[batch_idx, target_idx]
        target_distances = torch.stack(
            [obs_extra[f"obj{i+1}_distance_to_gripper"] for i in range(len(self.objects))],
            dim=1,
        )
        target_reached = target_distances[batch_idx, target_idx] < reach_threshold
        success = target_reached & target_in_view

        return {"success": success}

    def compute_dense_reward(self, obs: Any, action: Any, info: dict):
        reward = torch.zeros((self.num_envs,), device=self.device)
        # DROP_LOCATION = torch.tensor(self.DROP_LOCATION, device=self.device, dtype=torch.float32)
        batch_idx = torch.arange(self.num_envs, device=self.device)

        obs_extra = obs["extra"] if isinstance(obs, dict) and "extra" in obs else obs
        target_idx = obs_extra["target_obj"]
        # target_act_idx = obs_extra["target_act_obj"]

        if not isinstance(target_idx, torch.Tensor):
            target_idx = torch.tensor(target_idx, device=self.device, dtype=torch.long)
        target_idx = target_idx.clamp(min=0)
        active_idx_clamped = target_idx

        vis_list = torch.stack([self.is_object_visible(o) for o in self.objects], dim=1)
        target_visible = vis_list[batch_idx, active_idx_clamped]

        has_target = target_idx >= 0
        locating_time_penalty = torch.where(
            has_target & ~target_visible,
            torch.full_like(reward, -0.01),
            torch.zeros_like(reward),
        )
        reward += locating_time_penalty

        target_tcp_dist = torch.stack([obs_extra[f"obj{i+1}_distance_to_gripper"] for i in range(len(self.objects))], dim=1)[batch_idx, active_idx_clamped]

        # gripper_touching_bin = obs_extra.get("gripper_touching_bin", torch.zeros_like(reward)) > 0.5
        gripper_touching_table = obs_extra.get("gripper_touching_table", torch.zeros_like(reward)) > 0.5
        # grasped_obj_lifted = obs_extra.get("grasped_obj_lifted", torch.zeros_like(reward)) > 0.5
        # grasped_obj_touching_bin = obs_extra.get("grasped_obj_touching_bin", torch.zeros_like(reward)) > 0.5

        collision = gripper_touching_table #| gripper_touching_bin | (grasped_obj_lifted & grasped_obj_touching_bin)

        target_visible = self.is_object_visible(self.objects[1]) # default fallback
        if has_target.any():
            target_obj_idx_clamped = target_idx.clamp(min=0)
            vis_list = torch.stack([self.is_object_visible(o) for o in self.objects], dim=1)
            target_visible = vis_list[batch_idx, target_obj_idx_clamped]

        visibility_reward = (1 - torch.tanh(5 * target_tcp_dist)) * target_visible.float()
        reach_reward = (1 - torch.tanh(5 * target_tcp_dist))
        
        reward += visibility_reward + reach_reward

        reward -= 2.5 * collision.float()

        if "success" in info:
            reward[info["success"]] += 15.0

        return reward

    def compute_normalized_dense_reward(self, obs: Any, action: Any, info: dict):
        return self.compute_dense_reward(obs=obs, action=action, info=info) / 32.5