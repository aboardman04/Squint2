from typing import Any, Dict, List, Union
import torch
import numpy as np

import mani_skill.envs.utils.randomization as randomization
from mani_skill.agents.robots import Fetch, Panda
from mani_skill.envs.sapien_env import BaseEnv
from mani_skill.sensors.camera import CameraConfig
from mani_skill.utils import register_env
from mani_skill.utils.building import actor_builder
from mani_skill.utils.structs.pose import Pose


@register_env("PlaceInstruments-v3", max_episode_steps=100)
class PlaceInstrumentsEnv(BaseEnv):
    """
    Task: Pick up surgical/medical instruments and place them inside a designated drop zone on a mat.
    """

    SUPPORTED_ROBOTS = ["panda", "fetch"]
    agent: Union[Panda, Fetch]

    # Task geometry constants
    DROP_LOCATION = [0.15, 0.2, 0.0]  # Local relative XY target on the table
    DROP_ZONE_WIDTH = 0.08            # Half-width of drop zone (bounds check)
    DROP_ZONE_HEIGHT = 0.08           # Half-height of drop zone (bounds check)

    def __init__(self, *args, robot_uids="panda", **kwargs):
        super().__init__(*args, robot_uids=robot_uids, **kwargs)

    def _load_agent(self, options: dict):
        super()._load_agent(options, start_rigid_building=True)

    def _load_scene(self):
        # Build Table / Base
        self.table = self.scene.create_table(
            pose=Pose.create_from_pq([0, 0, 0]),
            surface_friction=0.8
        )

        # Build Blue Target Mat (Drop Surface)
        self.table_mat_half_size = [0.12, 0.12, 0.002]
        builder = self.scene.create_actor_builder()
        builder.add_box_visual(
            half_size=self.table_mat_half_size,
            color=[0.1, 0.3, 0.8, 1.0]
        )
        builder.add_box_collision(half_size=self.table_mat_half_size)
        builder.initial_pose = Pose.create_from_pq(self.DROP_LOCATION)
        self.table_mat = builder.build(name="target_mat")

        # Load 4 Instruments/Objects
        self.objects = []
        for i in range(4):
            obj_builder = self.scene.create_actor_builder()
            obj_builder.add_box_visual(
                half_size=[0.015, 0.015, 0.03],
                color=[0.8, 0.2, 0.2, 1.0]
            )
            obj_builder.add_box_collision(half_size=[0.015, 0.015, 0.03])
            obj = obj_builder.build(name=f"instrument_{i}")
            self.objects.append(obj)

    def _initialize_episode(self, env_idx: torch.Tensor, options: dict):
        with torch.device(self.device):
            b = len(env_idx)
            # Initialize robot position
            self.table_mat.set_pose(Pose.create_from_pq(self.DROP_LOCATION))

            # Scatter objects across table
            for i, obj in enumerate(self.objects):
                init_xy = torch.zeros((b, 2), device=self.device)
                init_xy[:, 0] = torch.rand(b, device=self.device) * 0.2 - 0.2
                init_xy[:, 1] = torch.rand(b, device=self.device) * 0.3 - 0.15
                init_z = torch.ones((b, 1), device=self.device) * 0.03
                
                xyz = torch.cat([init_xy, init_z], dim=-1)
                obj.set_pose(Pose.create_from_pq(xyz))

            # Store default rest qpos for stabilization penalties
            if hasattr(self.agent, "keypoint_qpos"):
                self.rest_qpos = self.agent.keypoint_qpos
            else:
                self.rest_qpos = self.agent.robot.get_qpos()[0]

    def evaluate(self) -> Dict[str, torch.Tensor]:
        # 1. Distances and reaching logic across all 4 instruments
        all_obj_positions = torch.stack([obj.pose.p for obj in self.objects], dim=1)  # Shape: (num_envs, 4, 3)
        tcp_pos_expanded = self.agent.tcp_pose.p.unsqueeze(1)                        # Shape: (num_envs, 1, 3)
        dists_to_items = torch.linalg.norm(all_obj_positions - tcp_pos_expanded, dim=-1)
        min_dist_to_item, _ = torch.min(dists_to_items, dim=1)
        reached_object = min_dist_to_item < 0.03

        # 2. Check grasping and lifting status
        is_item_grasped = torch.zeros(self.num_envs, dtype=torch.bool, device=self.device)
        item_lifted = torch.zeros(self.num_envs, dtype=torch.bool, device=self.device)

        for obj in self.objects:
            is_item_grasped = is_item_grasped | self.agent.is_grasping(obj)
            item_lifted = item_lifted | (obj.pose.p[..., -1] >= 0.04)

        # 3. Rest position offset
        current_qpos = self.agent.robot.get_qpos()
        distance_to_rest_qpos = torch.linalg.norm(
            current_qpos[:, :-1] - self.rest_qpos[:-1], dim=-1
        )

        # 4. Drop Zone evaluation (Dimensions fixed with .unsqueeze(-1))
        drop_xy = torch.tensor(self.DROP_LOCATION[:2], device=self.device, dtype=torch.float32)
        table_z = (self.table_mat.pose.p[..., 2] + float(self.table_mat_half_size[2])).unsqueeze(-1)  # Shape: (num_envs, 1)

        # Check spatial placement criteria per instrument
        inst_in_drop_xy = (
            (torch.abs(all_obj_positions[..., 0] - drop_xy[0]) <= self.DROP_ZONE_WIDTH) &
            (torch.abs(all_obj_positions[..., 1] - drop_xy[1]) <= self.DROP_ZONE_HEIGHT)
        )
        inst_near_table = torch.abs(all_obj_positions[..., 2] - table_z) <= 0.02

        # True if AT LEAST ONE instrument is inside the drop zone boundary and on/near table
        instrument_in_drop_zone = (inst_in_drop_xy & inst_near_table).any(dim=1)
        robot_touching_mat = self.agent.is_touching(self.table_mat)

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

    def _get_obs_extra(self, info: dict) -> Dict[str, torch.Tensor]:
        # Global target drop zone coordinate
        table_z_val = self.table_mat.pose.p[:, 2] + float(self.table_mat_half_size[2])
        drop_center = torch.tensor(self.DROP_LOCATION, device=self.device, dtype=torch.float32).repeat(self.num_envs, 1)
        drop_center[:, 2] = table_z_val

        # Relative displacement vector: (drop_zone - instrument_pose)
        all_obj_positions = torch.stack([obj.pose.p for obj in self.objects], dim=1)  # (num_envs, 4, 3)
        drop_center_expanded = drop_center.unsqueeze(1)                              # (num_envs, 1, 3)
        obj_to_drop_vec = drop_center_expanded - all_obj_positions                  # (num_envs, 4, 3)

        # Z-distance relative to mat surface
        table_z_expanded = table_z_val.unsqueeze(-1)                                # (num_envs, 1)
        obj_surface_z_dist = all_obj_positions[..., 2] - table_z_expanded            # (num_envs, 4)

        # Binary bounding box indicator
        inst_in_drop_xy = (
            (torch.abs(obj_to_drop_vec[..., 0]) <= self.DROP_ZONE_WIDTH) &
            (torch.abs(obj_to_drop_vec[..., 1]) <= self.DROP_ZONE_HEIGHT)
        )

        return {
            "is_item_grasped": info["is_item_grasped"].float().unsqueeze(-1),
            "drop_zone_center": drop_center,
            "obj_to_drop_vec": obj_to_drop_vec.reshape(self.num_envs, -1),
            "obj_surface_z_dist": obj_surface_z_dist,
            "inst_in_drop_xy": inst_in_drop_xy.float(),
        }

    def compute_dense_reward(self, obs: Any, action: torch.Tensor, info: dict) -> torch.Tensor:
        # Determine closest instrument for reaching/grasping phase
        all_obj_positions = torch.stack([obj.pose.p for obj in self.objects], dim=1)
        tcp_pos_expanded = self.agent.tcp_pose.p.unsqueeze(1)
        dists_to_items = torch.linalg.norm(all_obj_positions - tcp_pos_expanded, dim=-1)
        min_tcp_to_item_dist, min_item_idx = torch.min(dists_to_items, dim=1)

        env_indices = torch.arange(self.num_envs, device=self.device)
        active_obj_pos = all_obj_positions[env_indices, min_item_idx]

        # ==================== LIFT PHASE ====================
        reaching_reward = 1.0 - torch.tanh(5.0 * min_tcp_to_item_dist)
        reward = reaching_reward.clone()

        approach_weight = torch.clamp(min_tcp_to_item_dist / 0.15, 0.0, 1.0)
        tcp_velocity = torch.linalg.norm(self.agent.tcp_pose.p, dim=-1)

        slow_zone = torch.clamp(0.08 - min_tcp_to_item_dist, 0.0, 0.08) / 0.08
        desired_speed = 0.08
        speed_penalty = slow_zone * torch.clamp(tcp_velocity - desired_speed, min=0.0)
        reward -= 0.5 * speed_penalty

        is_grasped = info["is_item_grasped"].float()
        reward += is_grasped

        stable_grasp = is_grasped * torch.exp(-5.0 * tcp_velocity)
        reward += 0.5 * stable_grasp

        action_diff = torch.linalg.norm(action, dim=1)
        smoothness_penalty = action_diff * (1.0 - approach_weight)
        reward -= 0.1 * smoothness_penalty

        reward -= 3.0 * info["robot_touching_mat"].float()
        reward -= 1.0 * (~info["item_lifted"]).float()

        # ==================== PLACE PHASE (GATED) ====================
        lift_completed = info["item_lifted"] & info["is_item_grasped"]
        lift_completed_mask = lift_completed.float()

        drop_xy = torch.tensor(self.DROP_LOCATION[:2], device=self.device, dtype=torch.float32)
        table_z = self.table_mat.pose.p[..., 2] + float(self.table_mat_half_size[2])

        # Stage A: XY Alignment
        inst_xy_dist = torch.linalg.norm(active_obj_pos[:, :2] - drop_xy, dim=-1)
        xy_reach_reward = 1.0 - torch.tanh(3.0 * inst_xy_dist)
        
        in_xy_drop_zone = (
            (torch.abs(active_obj_pos[:, 0] - drop_xy[0]) <= self.DROP_ZONE_WIDTH) &
            (torch.abs(active_obj_pos[:, 1] - drop_xy[1]) <= self.DROP_ZONE_HEIGHT)
        )

        # Stage B: Z-Descent once aligned
        inst_z_dist = torch.abs(active_obj_pos[:, 2] - table_z)
        z_minimize_reward = 1.0 - torch.tanh(10.0 * inst_z_dist)

        # Stage C: Gripper release near surface
        near_table_surface = inst_z_dist <= 0.02
        gripper_qpos = self.agent.robot.get_qpos()[:, -1]
        gripper_open_reward = torch.clamp(gripper_qpos / 0.04, 0.0, 1.0)

        place_reward = 2.0 + 3.0 * xy_reach_reward
        place_reward += in_xy_drop_zone.float() * (3.0 + 4.0 * z_minimize_reward)
        place_reward += (in_xy_drop_zone & near_table_surface).float() * (5.0 + 3.0 * gripper_open_reward)

        # Apply gated placement reward
        reward += lift_completed_mask * place_reward

        if "success" in info:
            reward[info["success"]] += 15.0

        return reward