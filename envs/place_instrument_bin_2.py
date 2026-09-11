from dataclasses import asdict, dataclass
from typing import Any, Sequence, Union

import dacite
import numpy as np
import sapien
import torch
from transforms3d.euler import euler2quat
import env_cal
from mani_skill.agents.robots import Fetch, Panda
from mani_skill.utils import common
from mani_skill.utils.registration import register_env
from mani_skill.utils.scene_builder.table import TableSceneBuilder
from mani_skill.utils.structs import Pose
from .base_random_env import DefaultCameraEnv, DefaultRandomizationConfig
from .robot.so101 import SO101


@dataclass
class SeparateRandomizationConfig(DefaultRandomizationConfig):
    robot_qpos_noise_std: float = np.deg2rad(5)
    item_friction_range: Sequence[float] = (0.1, 0.5)
    item_density_range: Sequence[float] = (200, 200)#(7000, 7850)
    randomize_item_color: bool = False


@register_env("PlaceInstruments-v2", max_episode_steps=100)
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
    instrument_separation = 0 #0.12
    num_instruments = 4

    #parameters to help improve physics of grasping instruments
    SIM_FREQ = 200
    CONTROL_FREQ = 20

    block_half_size = [0.0075, 0.085, 0.085]
    DROP_LOCATION = np.array([0.25, -0.30, 0.01])
    DROP_ZONE_HEIGHT = 0.15
    DROP_ZONE_WIDTH = 0.20

    def __init__(
        self,
        *args,
        robot_uids="so101",
        control_mode="pd_joint_target_delta_pos",
        domain_randomization_config: Union[SeparateRandomizationConfig, dict] = SeparateRandomizationConfig(),
        domain_randomization=False,
            sim_config=dict(
                sim_freq=SIM_FREQ,
                control_freq=CONTROL_FREQ,
                scene_config=dict(
                    solver_position_iterations=20,
                    solver_velocity_iterations=2,
                    enable_ccd=True,
                ),
            ),
        **kwargs,
    ):
        self.base_z_rot = 0
        self.rest_qpos = SO101.keyframes["start"].qpos.tolist()

        self.domain_randomization_config = SeparateRandomizationConfig()
        merged_domain_randomization_config = asdict(self.domain_randomization_config)
        if isinstance(domain_randomization_config, dict):
            common.dict_merge(
                merged_domain_randomization_config, domain_randomization_config
            )
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

    def _load_agent(self, options: dict):
        super()._load_agent(
            options,
            sapien.Pose(p=[0, 0, 0], q=euler2quat(0, 0, self.base_z_rot)),
            build_separate=True
            if self.domain_randomization
            and getattr(self.domain_randomization_config, "robot_color", None) == "random"
            else False,
        )

    def _load_camera_mount(self):
        """Matches the wrist camera alignment from LiftCube exactly."""
        super()._load_camera_mount()
        if hasattr(self, "wrist_camera") and self.wrist_camera is not None:
            pos = getattr(env_cal, "WRIST_CAMERA_BASE_POS", (-0.0130, 0.0520, -0.0520))
            rot = getattr(env_cal, "WRIST_CAMERA_BASE_ROT_RAD", (np.deg2rad(-101.0), np.deg2rad(81.0), np.deg2rad(-31.0)))
            fov = getattr(env_cal, "WRIST_CAMERA_FOV", np.deg2rad(71.0))
            
            self.wrist_camera.set_local_pose(sapien.Pose(p=pos, q=euler2quat(*rot)))
            if hasattr(self.wrist_camera, "set_fov"):
                self.wrist_camera.set_fov(fov)

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


    def _build_instrument(self, obj_path: str, name: str, initial_pose: sapien.Pose): #, density: Union[float, np.ndarray, list] = 1000.0):
        # Use color from env_cal if available, otherwise default steel color
        base_color = (
            env_cal.INSTRUMENT_COLOR
            if env_cal and hasattr(env_cal, "INSTRUMENT_COLOR")
            else [0.44, 0.44, 0.44, 1.0]
        )
        steel_material = sapien.render.RenderMaterial(
            base_color=base_color, roughness=0.15, metallic=0.5
        )
        physx_material = sapien.physx.PhysxMaterial(
            static_friction=0.5, dynamic_friction=0.4, restitution=0.003
        )
        builder = self.scene.create_actor_builder()
        builder.add_visual_from_file(filename=obj_path, material=steel_material)

        try:
            builder.add_multiple_convex_collisions_from_file(
                filename=obj_path,
                decomposition="coacd",
                material=physx_material,
                contact_offset=0.002,
                rest_offset=0.001,
                # density=density,
            )
        except TypeError:
            builder.add_multiple_convex_collisions_from_file(
                filename=obj_path,
                decomposition="coacd",
                material=physx_material,
                # density=density,
            )

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
        cfg = self.domain_randomization_config
        frictions = (
            np.ones(self.num_envs)
            * (cfg.item_friction_range[0] + cfg.item_friction_range[1])
            / 2
        )
        densities = (
            np.ones(self.num_envs)
            * (cfg.item_density_range[0] + cfg.item_density_range[1])
            / 2
        )

        self.table_scene = TableSceneBuilder(self)
        self.table_scene.build()
        self.table_pose = Pose.create_from_pq(
            p=[-0.12 + 0.737, 0, -0.9196429], q=euler2quat(0, 0, np.pi / 2)
        )

        self.drop_zone_visual = self._build_drop_zone_outline(
            half_width=self.DROP_ZONE_WIDTH, 
            half_height=self.DROP_ZONE_HEIGHT, 
            thickness=0.00025
        )

        bin_path = "/home/aboardman/squint2/deploy_utils/blender_objs/bin_2.obj"
        bin_q = euler2quat(0, np.pi/2, 0.0)
        bin_steel_material = sapien.render.RenderMaterial(base_color=[1, 1, 1, 1.0], roughness=0.15, metallic=0.5)
        physx_material = sapien.physx.PhysxMaterial(static_friction=0.6, dynamic_friction=0.5, restitution=0.1)
        builder = self.scene.create_actor_builder()
        builder.add_visual_from_file(filename=bin_path, material=bin_steel_material)
        builder.add_multiple_convex_collisions_from_file(filename=bin_path, decomposition="coacd", material=physx_material)
        builder.initial_pose = sapien.Pose(p=[0.0, 0.0, float(self.block_half_size[2]) - 0.05], q=list(bin_q))
        self.bin = builder.build_kinematic("bin")

        table_mat_color = (
            env_cal.TABLE_COLOR
            if env_cal and hasattr(env_cal, "TABLE_COLOR")
            else [0.1, 0.2, 0.85, 1.0]
        )
        mat_material = sapien.render.RenderMaterial(
            base_color=table_mat_color, roughness=0.6, metallic=0.0
        )
        physx_material = sapien.physx.PhysxMaterial(
            static_friction=0.6, dynamic_friction=0.5, restitution=0.1
        )
        self.table_mat_half_size = [0.40, 0.80, 0.001]
        builder = self.scene.create_actor_builder()
        builder.add_box_visual(
            half_size=self.table_mat_half_size, material=mat_material
        )
        builder.add_box_collision(
            half_size=self.table_mat_half_size, material=physx_material
        )
        builder.initial_pose = sapien.Pose()
        self.table_mat = builder.build_kinematic("table_mat")

        inst1_path = "/home/aboardman/squint2/deploy_utils/blender_objs/dressing_forceps.obj"
        self.obj_1 = self._build_instrument(
            inst1_path,
            name="forceps_1",
            initial_pose=sapien.Pose(p=[-0.1, -0.05, 0.1], q=[1, 1, 0, 0]),
        )
        self.obj_2 = self._build_instrument(
            inst1_path,
            name="forceps_2",
            initial_pose=sapien.Pose(p=[0.1, -0.05, 0.1], q=[1, 1, 0, 0]),
        )

        inst2_path = "/home/aboardman/squint2/deploy_utils/blender_objs/allis.obj"
        self.obj_3 = self._build_instrument(
            inst2_path,
            name="allis_1",
            initial_pose=sapien.Pose(p=[-0.1, 0.05, 0.1], q=[1, 1, 0, 0]),
        )
        self.obj_4 = self._build_instrument(
            inst2_path,
            name="allis_2",
            initial_pose=sapien.Pose(p=[0.1, 0.05, 0.1], q=[1, 1, 0, 0]),
        )

        self.objects = [self.obj_1, self.obj_2, self.obj_3, self.obj_4]
        self.target_object = self.obj_1

        self._load_camera_mount()
        self._randomize_robot_color()
        self.rest_qpos = common.to_tensor(self.rest_qpos, device=self.device)

        self.item_frictions = common.to_tensor(frictions, device=self.device)
        self.item_densities = common.to_tensor(densities, device=self.device)

    def _initialize_episode(self, env_idx: torch.Tensor, options: dict):
        super()._initialize_episode(env_idx, options)
        with torch.device(self.device):
            b = len(env_idx)
            self.table_scene.initialize(env_idx)
            self.table_scene.table.set_pose(self.table_pose)

            self.agent.robot.set_qpos(
                self.rest_qpos
                + torch.randn(size=(b, self.rest_qpos.shape[-1]))
                * self.domain_randomization_config.initial_qpos_noise_scale
            )
            self.agent.robot.set_pose(
                Pose.create_from_pq(p=[0, 0, 0], q=euler2quat(0, 0, self.base_z_rot))
            )

            if hasattr(self.table_scene, "table"):
                table_z = self.table_scene.table.pose.p[..., 2]
                if table_z.ndim == 0:
                    table_z = table_z.unsqueeze(0)
                table_z = table_z + 0.92
            else:
                table_z = torch.full((b,), 0.92, device=self.device)

            mat_pos = torch.zeros((b, 3), device=self.device)
            mat_pos[:, 0] = 0.450
            mat_pos[:, 1] = -0.275
            mat_pos[:, 2] = table_z + float(self.table_mat_half_size[2])
            mat_q = (
                torch.tensor([1.0, 0.0, 0.0, 0.0], device=self.device)
                .unsqueeze(0)
                .repeat(b, 1)
            )
            self.table_mat.set_pose(Pose.create_from_pq(p=mat_pos, q=mat_q))

            center = self.agent.robot.pose.p + torch.tensor([0.3, 0.0, 0.0], device=self.device)
            center = center[env_idx]
            bin_pos = center.clone()
            bin_pos[:, 2] = float(self.block_half_size[2]) - 0.03
            bin_q = euler2quat(np.pi / 2, 0.0, np.pi/2)
            q_tensor = torch.tensor(bin_q, device=self.device, dtype=bin_pos.dtype)
            q_tensor = q_tensor.unsqueeze(0).repeat(b, 1)
            bin_pose = sapien.Pose(p=[0.3, 0.0, float(self.block_half_size[2]) - 0.03], q=list(bin_q))
            self.bin.set_pose(bin_pose)

            spawn_base = bin_pos.clone() #spawn_base = center[env_idx]
            spawn_base[:, 2] += 0.03

            p1, q1, p2, q2, p3, q3, p4, q4 = self._sample_instrument_poses(b, spawn_base)
            # p2[:, 0] = p1[:, 0] + self.instrument_separation
            # p2[:, 1] = p1[:, 1]

            self.obj_1.set_pose(Pose.create_from_pq(p=p1, q=q1))
            self.obj_2.set_pose(Pose.create_from_pq(p=p2, q=q2))
            self.obj_3.set_pose(Pose.create_from_pq(p=p3, q=q3))
            self.obj_4.set_pose(Pose.create_from_pq(p=p4, q=q4))
            # self.target_object = self.obj_1

            drop_pos = torch.tensor(self.DROP_LOCATION, device=self.device, dtype=torch.float32).repeat(b, 1)
            drop_q = torch.tensor([1.0, 0.0, 0.0, 0.0], device=self.device).repeat(b, 1)
            self.drop_zone_visual.set_pose(Pose.create_from_pq(p=drop_pos, q=drop_q))

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

    def _get_obs_extra(self, info: dict):
        obs = dict()
        # Stack positions of all objects to compute distances to TCP
        # Shape: (num_envs, num_instruments, 3)
        all_obj_positions = torch.stack([obj.pose.p for obj in self.objects], dim=1)
        tcp_pos_expanded = self.agent.tcp_pos.unsqueeze(1) # Shape: (num_envs, 1, 3)
        
        # Distance to each instrument
        dists_to_items = torch.linalg.norm(all_obj_positions - tcp_pos_expanded, dim=-1)
        min_dist_to_item, min_dist_idx = torch.min(dists_to_items, dim=1)
        
        # Closest item position and pose
        closest_obj_pos = all_obj_positions[torch.arange(self.num_envs, device=self.device), min_dist_idx]
        
        tcp_velocity = (self.agent.finger1_tip.linear_velocity + self.agent.finger2_tip.linear_velocity) / 2

        obs.update(
            tcp_pose=self.agent.tcp_pose.raw_pose,
            target_item_pose=closest_obj_pos,
            tcp_to_item_pos=closest_obj_pos - self.agent.tcp_pos,
            tcp_to_item_dist=min_dist_to_item,
            tcp_velocity=tcp_velocity,
            is_item_grasped=info.get("is_item_grasped", torch.stack([self.agent.is_grasping(obj) for obj in self.objects], dim=1).any(dim=1)),
            robot_touching_mat=self.agent.is_touching(self.table_mat).float(),
            dist_to_rest_qpos=self.agent.controller._target_qpos[:, :-1] - self.rest_qpos[:-1],
        )

        if self.domain_randomization:
            gripper_params = self.get_gripper_params()
            obs.update(
                clean_qpos=self.agent.robot.get_qpos(),
                item_friction=self.item_frictions,
                item_density=self.item_densities,
                gripper_stiffness=gripper_params["gripper_stiffness"],
                gripper_damping=gripper_params["gripper_damping"],
            )

        return obs

    def evaluate(self):
        # 1. Distances and reaching logic
        all_obj_positions = torch.stack([obj.pose.p for obj in self.objects], dim=1)  # Shape: (num_envs, 4, 3)
        tcp_pos_expanded = self.agent.tcp_pos.unsqueeze(1)
        dists_to_items = torch.linalg.norm(all_obj_positions - tcp_pos_expanded, dim=-1)
        min_dist_to_item, _ = torch.min(dists_to_items, dim=1)
        reached_object = min_dist_to_item < 0.03

        # 2. Check grasping and lifting status across all instruments
        is_item_grasped = torch.zeros(self.num_envs, dtype=torch.bool, device=self.device)
        item_lifted = torch.zeros(self.num_envs, dtype=torch.bool, device=self.device)

        for obj in self.objects:
            is_item_grasped = is_item_grasped | self.agent.is_grasping(obj)
            item_lifted = item_lifted | (obj.pose.p[..., -1] >= 0.04)

        # 3. Distance to rest qpos
        target_qpos = self.agent.controller._target_qpos.clone()
        distance_to_rest_qpos = torch.linalg.norm(
            target_qpos[:, :-1] - self.rest_qpos[:-1], axis=-1
        )

        # 4. Drop Zone evaluation
        drop_xy = torch.tensor(self.DROP_LOCATION[:2], device=self.device, dtype=torch.float32)
        # Shape: (num_envs, 1) for proper broadcasting against (num_envs, 4)
        table_z = (self.table_mat.pose.p[..., 2] + float(self.table_mat_half_size[2])).unsqueeze(-1)

        # Instrument inside XY boundaries: Shape (num_envs, 4)
        inst_in_drop_xy = (
            (torch.abs(all_obj_positions[..., 0] - drop_xy[0]) <= self.DROP_ZONE_WIDTH) &
            (torch.abs(all_obj_positions[..., 1] - drop_xy[1]) <= self.DROP_ZONE_HEIGHT)
        )
        # Instrument within 2cm of table surface: Shape (num_envs, 4)
        inst_near_table = torch.abs(all_obj_positions[..., 2] - table_z) <= 0.02

        # True if AT LEAST ONE instrument is inside the drop zone and near the table surface
        instrument_in_drop_zone = (inst_in_drop_xy & inst_near_table).any(dim=1)

        robot_touching_mat = self.agent.is_touching(self.table_mat)

        # Success condition: At least one instrument successfully placed in the drop zone
        success = instrument_in_drop_zone

        return {
            "is_item_grasped": is_item_grasped,
            "reached_object": reached_object,
            "distance_to_rest_qpos": distance_to_rest_qpos,
            "robot_touching_mat": robot_touching_mat,
            "item_lifted": item_lifted,
            "instrument_in_drop_zone": instrument_in_drop_zone,
            "success": success,
        }

    def compute_dense_reward(self, obs: Any, action: torch.Tensor, info: dict):
        # Determine closest instrument for reaching/grasping phase
        all_obj_positions = torch.stack([obj.pose.p for obj in self.objects], dim=1)
        tcp_pos_expanded = self.agent.tcp_pose.p.unsqueeze(1)
        dists_to_items = torch.linalg.norm(all_obj_positions - tcp_pos_expanded, dim=-1)
        min_tcp_to_item_dist, min_item_idx = torch.min(dists_to_items, dim=1)

        # Active item tracking
        env_indices = torch.arange(self.num_envs, device=self.device)
        active_obj_pos = all_obj_positions[env_indices, min_item_idx]

        # ==================== LIFT LOGIC ====================
        reaching_reward = 1.0 - torch.tanh(5.0 * min_tcp_to_item_dist)
        reward = reaching_reward.clone()

        approach_weight = torch.clamp(min_tcp_to_item_dist / 0.15, 0.0, 1.0)
        tcp_velocity = torch.linalg.norm(
            (
                self.agent.finger1_tip.linear_velocity
                + self.agent.finger2_tip.linear_velocity
            )
            / 2,
            axis=1,
        )

        slow_zone = torch.clamp(0.08 - min_tcp_to_item_dist, 0.0, 0.08) / 0.08
        desired_speed = 0.08
        speed_penalty = (slow_zone * torch.clamp(tcp_velocity - desired_speed, min=0.0,))
        reward -= 0.5 * speed_penalty

        is_grasped = info["is_item_grasped"].float()
        reward += is_grasped

        stable_grasp = (is_grasped * torch.exp(-5.0 * tcp_velocity))
        reward += 0.5 * stable_grasp

        action_diff = torch.linalg.norm(action, axis=1)
        smoothness_penalty = action_diff * (1.0 - approach_weight)
        lift_phase_penalty = is_grasped * torch.linalg.norm((self.agent.finger1_tip.linear_velocity + self.agent.finger2_tip.linear_velocity) / 2, axis=1)

        reward -= 0.1 * smoothness_penalty
        reward -= 0.05 * lift_phase_penalty

        reward -= 3.0 * info["robot_touching_mat"].float()
        reward -= 1.0 * (~info["item_lifted"]).float()
        # ==============================================================

        # ==================== PLACE LOGIC (GATED) ====================
        lift_completed = info["item_lifted"] & info["is_item_grasped"]
        lift_completed_mask = lift_completed.float()

        drop_xy = torch.tensor(self.DROP_LOCATION[:2], device=self.device, dtype=torch.float32)
        # Single Z calculation per env for active object
        table_z = self.table_mat.pose.p[..., 2] + float(self.table_mat_half_size[2])

        # Stage A: XY Alignment Reward
        inst_xy_dist = torch.linalg.norm(active_obj_pos[:, :2] - drop_xy, dim=-1)
        xy_reach_reward = 1.0 - torch.tanh(3.0 * inst_xy_dist)
        
        # Check if instrument is inside XY Drop Zone bounds
        in_xy_drop_zone = (
            (torch.abs(active_obj_pos[:, 0] - drop_xy[0]) <= self.DROP_ZONE_WIDTH) &
            (torch.abs(active_obj_pos[:, 1] - drop_xy[1]) <= self.DROP_ZONE_HEIGHT)
        )

        # Stage B: Minimize Z-distance once in XY drop zone
        inst_z_dist = torch.abs(active_obj_pos[:, 2] - table_z)
        z_minimize_reward = 1.0 - torch.tanh(10.0 * inst_z_dist)

        # Stage C: Gripper Opening Reward when <= 2cm from table
        near_table_surface = inst_z_dist <= 0.02
        gripper_qpos = self.agent.robot.get_qpos()[:, -1]
        gripper_open_reward = torch.clamp(gripper_qpos / 0.04, 0.0, 1.0)

        # Cumulative Place Reward formulation
        place_reward = 2.0 + 3.0 * xy_reach_reward
        place_reward += in_xy_drop_zone.float() * (3.0 + 4.0 * z_minimize_reward)
        place_reward += (in_xy_drop_zone & near_table_surface).float() * (5.0 + 3.0 * gripper_open_reward)

        # Activate Place Reward ONLY when lift is completed
        reward += lift_completed_mask * place_reward

        # Penalty for robot touching blue mat
        reward -= 2.0 * info["robot_touching_mat"].float()

        if "success" in info:
            reward[info["success"]] += 15.0

        return reward