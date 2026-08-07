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
    instrument_separation = 0.12
    num_instruments = 2
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
        self.target_goal_thresh = 0.05
        self.target_switch_pause_steps = 10
        self.max_targets = 2

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

    def _load_scene(self, options: dict):
        cfg = self.domain_randomization_config
        frictions = np.ones(self.num_envs) * (cfg.item_friction_range[0] + cfg.item_friction_range[1]) / 2
        densities = np.ones(self.num_envs) * (cfg.item_density_range[0] + cfg.item_density_range[1]) / 2

        self.table_scene = TableSceneBuilder(self)
        self.table_scene.build()
        self.table_pose = Pose.create_from_pq(p=[-0.12 + 0.737, 0, -0.9196429], q=euler2quat(0, 0, np.pi / 2))
        
        blue_material = sapien.render.RenderMaterial(base_color=[0.1, 0.2, 0.85, 1.0], roughness=0.6, metallic=0.0)
        physx_material = sapien.physx.PhysxMaterial(static_friction=0.6, dynamic_friction=0.5, restitution=0.1)
        self.table_mat_half_size = [0.40, 0.80, 0.001]
        builder = self.scene.create_actor_builder()
        builder.add_box_visual(half_size=self.table_mat_half_size, material=blue_material)
        builder.add_box_collision(half_size=self.table_mat_half_size, material=physx_material)
        builder.initial_pose = sapien.Pose()
        self.table_mat = builder.build_kinematic("table_mat")

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
        self.obj_1 = self._build_instrument(
            inst1_path,
            name="forceps_1",
            initial_pose=sapien.Pose(p=[-0.1, -0.05, 0.1], q=[1, 0, 0, 0]),
        )
        self.obj_2 = self._build_instrument(
            inst1_path,
            name="forceps_2",
            initial_pose=sapien.Pose(p=[0.1, -0.05, 0.1], q=[1, 0, 0, 0]),
        )
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
        self.obj = self.obj_1
        self.objects = [self.obj_1, self.obj_2]

        goal_builder = self.scene.create_actor_builder()
        goal_builder.add_sphere_visual(
            radius=0.01,
            material=sapien.render.RenderMaterial(base_color=[0, 1, 0, 1]),
        )
        goal_builder.initial_pose = sapien.Pose(p=[0, 0, 0.1])
        self.goal_site = goal_builder.build_kinematic(name="goal_site")
        self._hidden_objects.append(self.goal_site)

        second_goal_builder = self.scene.create_actor_builder()
        second_goal_builder.add_sphere_visual(
            radius=0.01,
            material=sapien.render.RenderMaterial(base_color=[0, 1, 0, 1]),
        )
        second_goal_builder.initial_pose = sapien.Pose(p=[0, 0, 0.1])
        self.second_goal_site = second_goal_builder.build_kinematic(name="second_goal_site")
        self._hidden_objects.append(self.second_goal_site)

        self._load_camera_mount()
        self._randomize_robot_color()

        self.item_dimensions = torch.ones((self.num_envs, 3), device=self.device)
        self.item_frictions = common.to_tensor(frictions, device=self.device)
        self.item_densities = common.to_tensor(densities, device=self.device)

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
            bin_q = euler2quat(np.pi / 2, 0.0, np.pi / 2)
            q_tensor = torch.tensor(bin_q, device=self.device, dtype=bin_pos.dtype)
            q_tensor = q_tensor.unsqueeze(0).repeat(b, 1)
            bin_pose = Pose.create_from_pq(p=bin_pos, q=q_tensor)
            self.bin.set_pose(bin_pose)

            spawn_base = bin_pos.clone()
            spawn_base[:, 2] += 0.03
            p1, q1, p2, q2 = self._sample_instrument_poses(b, spawn_base)
            p2[:, 0] = p1[:, 0] + self.instrument_separation
            p2[:, 1] = p1[:, 1]
            self.obj_1.set_pose(Pose.create_from_pq(p=p1, q=q1))
            self.obj_2.set_pose(Pose.create_from_pq(p=p2, q=q2))
            self.obj = self.obj_1

            goal_xyz = self.obj.pose.p.clone()
            goal_xyz[:, 2] += 0.01
            self.goal_site.set_pose(Pose.create_from_pq(goal_xyz))

            second_goal_xyz = self.obj_2.pose.p.clone()
            second_goal_xyz[:, 2] += 0.01
            self.second_goal_site.set_pose(Pose.create_from_pq(second_goal_xyz))
            
            self.target_obj_idx = torch.zeros((b,), dtype=torch.long, device=self.device)
            self.reached_objects = torch.zeros((b, self.max_targets), dtype=torch.bool, device=self.device)
            self.target_switch_timer = torch.zeros((b,), dtype=torch.int32, device=self.device)

            init_distances = torch.stack(
                [
                    torch.linalg.norm(self.obj_1.pose.p - self.agent.tcp_pos, dim=-1),
                    torch.linalg.norm(self.obj_2.pose.p - self.agent.tcp_pos, dim=-1),
                ],
                dim=1,
            )
            self.target_obj_idx = self._select_target_idx(init_distances, self.reached_objects)

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

    def _select_target_idx(self, distances: torch.Tensor, reached_objects: torch.Tensor) -> torch.Tensor:
        remaining = ~reached_objects
        candidate_distances = torch.where(remaining, distances, torch.full_like(distances, float("inf")))
        return torch.argmin(candidate_distances, dim=1).long()

    def _update_target_progress(self, distances: torch.Tensor):
        batch_idx = torch.arange(self.num_envs, device=self.device)
        current_target = self.target_obj_idx.clamp(min=0)
        target_reached = distances[batch_idx, current_target] < 0.01
        static = self.agent.is_static()

        just_reached = target_reached & static & (~self.reached_objects[batch_idx, current_target])
        self.reached_objects[batch_idx, current_target] = self.reached_objects[batch_idx, current_target] | just_reached

        if just_reached.any():
            self.target_switch_timer[batch_idx[just_reached]] = 0

        active_pause_mask = self.reached_objects[batch_idx, current_target] & target_reached & static
        if active_pause_mask.any():
            self.target_switch_timer[batch_idx[active_pause_mask]] += 1

        reset_mask = (~active_pause_mask) & (~just_reached)
        if reset_mask.any():
            self.target_switch_timer[batch_idx[reset_mask]] = 0

        has_more_targets = (~self.reached_objects).any(dim=1)
        switch_mask = active_pause_mask & (self.target_switch_timer[batch_idx] >= self.target_switch_pause_steps) & has_more_targets
        if switch_mask.any():
            next_targets = self._select_target_idx(distances[switch_mask], self.reached_objects[switch_mask])
            self.target_obj_idx[switch_mask] = next_targets
            self.target_switch_timer[batch_idx[switch_mask]] = 0

    def _get_obs_extra(self, info: dict):
        obs = dict(tcp_pose=self.agent.tcp_pose.raw_pose)

        obj_positions = torch.stack([obj.pose.p for obj in self.objects], dim=1)
        tcp_pos_expanded = self.agent.tcp_pos.unsqueeze(1)
        instrument_distances = torch.linalg.norm(obj_positions - tcp_pos_expanded, dim=-1)
        self._update_target_progress(instrument_distances)

        # grasped_states = []
        # for obj in self.objects:
        #     is_grasped = self.agent.is_grasping(obj)
        #     if isinstance(is_grasped, bool):
        #         is_grasped = torch.tensor([is_grasped], device=self.device).repeat(self.num_envs)
        #     grasped_states.append(is_grasped.float())

        # grasped_tensor = torch.stack(grasped_states, dim=1)
        # any_grasped = (grasped_tensor > 0.5).any(dim=1)

        robot_touching_mat = self.agent.is_touching(self.table_mat).float()
        # gripper_to_bin_dist = torch.linalg.norm(self.agent.tcp_pos[..., :2] - self.bin.pose.p[..., :2], dim=-1)

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
            "robot_touching_mat": robot_touching_mat,
            # "gripper_touching_bin": gripper_touching_bin,
            # "grasped_obj_touching_bin": grasped_obj_touching_bin,
            # "grasped_obj_lifted": grasped_obj_lifted.float(),
            "target_obj": self.target_obj_idx,
            "reached_objects": self.reached_objects,
            "target_switch_timer": self.target_switch_timer,
            # "target_act_obj": self.target_act_obj_idx,
        })

        num_objects = instrument_distances.shape[1]

        if self.domain_randomization:
            gripper_params = self.get_gripper_params()
            obs.update(
                clean_qpos=self.agent.robot.get_qpos(),
                item_friction=self.item_frictions,
                item_density=self.item_densities,
                gripper_stiffness=gripper_params["gripper_stiffness"],
                gripper_damping=gripper_params["gripper_damping"],
            )

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
        batch_idx = torch.arange(self.num_envs, device=self.device)
        obs_extra = self._get_obs_extra(info={})

        target_idx = self.target_obj_idx.clamp(min=0)
        reach_threshold = 0.01

        target_distances = torch.stack(
            [obs_extra[f"obj{i+1}_distance_to_gripper"] for i in range(len(self.objects))],
            dim=1,
        )
        target_reached = target_distances[batch_idx, target_idx] < reach_threshold

        tcp_to_goal_dist = torch.linalg.norm(
            self.goal_site.pose.p - self.agent.tcp_pose.p, axis=1
        )
        is_reached = tcp_to_goal_dist <= self.target_goal_thresh
        is_robot_static = self.agent.is_static()

        robot_touching_instrument = torch.stack(
            [self.agent.is_touching(obj) for obj in self.objects], dim=1
        ).any(dim=1)
        robot_touching_instrument_1 = self.agent.is_touching(self.obj_1)
        robot_touching_instrument_2 = self.agent.is_touching(self.obj_2)
        robot_touching_mat = self.agent.is_touching(self.table_mat)

        all_targets_reached = self.reached_objects.sum(dim=1) >= self.max_targets
        success = all_targets_reached & is_robot_static & (~robot_touching_instrument_1) & (~robot_touching_instrument_2) & (~robot_touching_mat)

        return {
            "success": success,
            "is_reached": is_reached,
            "is_robot_static": is_robot_static,
            "robot_touching_instrument": robot_touching_instrument,
            "robot_touching_mat": robot_touching_mat,
            "target_obj": self.target_obj_idx,
            "reached_objects": self.reached_objects,
        }

    def compute_dense_reward(self, obs: Any, action: Any, info: dict):
        reward = torch.zeros((self.num_envs,), device=self.device)
        batch_idx = torch.arange(self.num_envs, device=self.device)

        obs_extra = obs["extra"] if isinstance(obs, dict) and "extra" in obs else obs
        target_idx = obs_extra["target_obj"]

        if not isinstance(target_idx, torch.Tensor):
            target_idx = torch.tensor(target_idx, device=self.device, dtype=torch.long)
        target_idx = target_idx.clamp(min=0)
        active_idx_clamped = target_idx

        gripper_min, gripper_max = self.agent.robot.get_qlimits()[0, -1, :]
        reward += (gripper_max - self.agent.robot.get_qpos()[:, -1]) / (gripper_max - gripper_min)

        target_tcp_dist = torch.stack(
            [obs_extra[f"obj{i+1}_distance_to_gripper"] for i in range(len(self.objects))],
            dim=1,
        )[batch_idx, active_idx_clamped]
        target_reached = target_tcp_dist < 0.01
        is_robot_static = self.agent.is_static()

        robot_touching_instrument = torch.stack(
            [self.agent.is_touching(obj) for obj in self.objects], dim=1
        ).any(dim=1)
        robot_touching_instrument_1 = self.agent.is_touching(self.obj_1)
        robot_touching_instrument_2 = self.agent.is_touching(self.obj_2)
        robot_touching_mat = obs_extra.get("robot_touching_mat", torch.zeros_like(reward)) > 0.5

        reach_reward = 1 - torch.tanh(5 * target_tcp_dist)
        reward += reach_reward
        reward += target_reached.float()
        reward += 1.5 * (target_reached & is_robot_static).float()

        reward -= 2.5 * torch.logical_or(robot_touching_instrument_1, robot_touching_instrument_2).float()
        reward -= 3.0 * robot_touching_mat.float()

        if "success" in info:
            reward[info["success"]] += 15.0

        return reward

    def compute_normalized_dense_reward(self, obs: Any, action: Any, info: dict):
        return self.compute_dense_reward(obs=obs, action=action, info=info) / 32.5