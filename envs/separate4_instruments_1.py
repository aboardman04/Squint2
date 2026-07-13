#This env has uptated rewards and penalties functions to reward looking at the 
# instruments to make sure they are seperated and to penalize extreme movements

from dataclasses import asdict, dataclass
from typing import Any, Union, Optional, Sequence

import dacite
import numpy as np
import sapien
import torch
import torch.random
from transforms3d.euler import euler2quat

import mani_skill.envs.utils.randomization as randomization
import sys
import os
try:
    import env_cal
except ImportError:
    env_cal = None

from mani_skill.agents.robots import Fetch, Panda
from mani_skill.sensors.camera import CameraConfig
from mani_skill.utils import common, sapien_utils
from mani_skill.utils.building import actor_builder
from mani_skill.utils.building import actors
from mani_skill.utils.registration import register_env
from mani_skill.utils.scene_builder.table import TableSceneBuilder
from mani_skill.utils.structs import Pose
from mani_skill.utils.structs.types import Array, GPUMemoryConfig, SimConfig

from .base_random_env import DefaultCameraEnv, DefaultRandomizationConfig
from .robot.so100 import SO100
from .robot.so101 import SO101


@dataclass
class SeparateInstrumentsRandomizationConfig(DefaultRandomizationConfig):
    robot_qpos_noise_std: float = np.deg2rad(5)


@register_env("SeparateInstruments-v2.5", max_episode_steps=50)
class SeparateInstrumentsEnv(DefaultCameraEnv):

    SUPPORTED_ROBOTS = ["so100", "so101", "panda", "fetch"]
    SUPPORTED_OBS_MODES = ["none", "state", "state_dict", "rgb", "rgb+segmentation", "rgb+state", "rgb+segmentation+state",
                           "rgb+depth+segmentation", "rgb+depth+segmentation+state"]
    agent: Union[SO100, SO101, Panda, Fetch]

    goal_radius = 0.1
    instrument_half_size = 0.075

    def __init__(
        self,
        *args,
        robot_uids="so101",
        control_mode="pd_joint_target_delta_pos",
        domain_randomization_config: Union[
            SeparateInstrumentsRandomizationConfig, dict
        ] = SeparateInstrumentsRandomizationConfig(),
        domain_randomization=False,
        spawn_box_pos=[0.3, 0],
        spawn_box_half_size=0.2 / 2,
        **kwargs,
    ):
        # Robot-specific configuration
        if robot_uids == "so100":
            self.base_z_rot = np.pi / 2
            self.rest_qpos = [0, 0, 0, np.pi / 2, np.pi / 2, 0]
        elif robot_uids == "so101":
            self.base_z_rot = 0
            self.rest_qpos = SO101.keyframes["start"].qpos.tolist()
        else:
            self.base_z_rot = 0
            self.rest_qpos = None

        self.domain_randomization_config = SeparateInstrumentsRandomizationConfig()
        merged_domain_randomization_config = self.domain_randomization_config.dict()
        if isinstance(domain_randomization_config, dict):
            common.dict_merge(merged_domain_randomization_config, domain_randomization_config)
            self.domain_randomization_config = dacite.from_dict(
                data_class=SeparateInstrumentsRandomizationConfig,
                data=merged_domain_randomization_config,
                config=dacite.Config(strict=True),
            )
        elif isinstance(domain_randomization_config, SeparateInstrumentsRandomizationConfig):
            self.domain_randomization_config = domain_randomization_config

        self.spawn_box_pos = spawn_box_pos
        self.spawn_box_half_size = spawn_box_half_size

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
            gpu_memory_config=GPUMemoryConfig(
                found_lost_pairs_capacity=2**25, max_rigid_patch_count=2**18
            )
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

    def _load_scene(self, options: dict):
        self.table_scene = TableSceneBuilder(self)
        self.table_scene.build()
        self._color_table()
        
        #------ Build Instruments ------
        obj_path = "/home/aboardman/squint2/deploy_utils/blender_objs/dressing_forceps.obj"

        steel_material = sapien.render.RenderMaterial(
            base_color=[0.44, 0.44, 0.44, 1.0], 
            roughness=0.15,                     
            metallic=1.0                        
        )

        physx_material = sapien.physx.PhysxMaterial(
            static_friction=0.6,
            dynamic_friction=0.5,
            restitution=0.1
        )
        
        # --- Forceps 1 ---
        builder1 = self.scene.create_actor_builder()
        builder1.add_visual_from_file(filename=obj_path, material=steel_material)
        builder1.add_multiple_convex_collisions_from_file(
            filename=obj_path, decomposition="coacd", material=physx_material
        ) 
        builder1.initial_pose = sapien.Pose(p=[-0.1, -0.05, 0.1], q=[1, 0, 0, 0])
        self.obj_1 = builder1.build(name="forceps_1")

        # --- Forceps 2 ---
        builder2 = self.scene.create_actor_builder()
        builder2.add_visual_from_file(filename=obj_path, material=steel_material)
        builder2.add_multiple_convex_collisions_from_file(
            filename=obj_path, decomposition="coacd", material=physx_material
        )
        builder2.initial_pose = sapien.Pose(p=[0.1, -0.05, 0.1])
        self.obj_2 = builder2.build(name="forceps_2")

        # --- Forceps 3 (UPDATED) ---
        builder3 = self.scene.create_actor_builder()
        builder3.add_visual_from_file(filename=obj_path, material=steel_material)
        builder3.add_multiple_convex_collisions_from_file(
            filename=obj_path, decomposition="coacd", material=physx_material
        )
        builder3.initial_pose = sapien.Pose(p=[-0.1, 0.05, 0.1])
        self.obj_3 = builder3.build(name="forceps_3")

        # --- Forceps 4 (UPDATED) ---
        builder4 = self.scene.create_actor_builder()
        builder4.add_visual_from_file(filename=obj_path, material=steel_material)
        builder4.add_multiple_convex_collisions_from_file(
            filename=obj_path, decomposition="coacd", material=physx_material
        )
        builder4.initial_pose = sapien.Pose(p=[0.1, 0.05, 0.1])
        self.obj_4 = builder4.build(name="forceps_4")

        if self.apply_greenscreen:
            self.remove_object_from_greenscreen(self.agent.robot)
            self.remove_object_from_greenscreen(self.obj_1)
            self.remove_object_from_greenscreen(self.obj_2)
            self.remove_object_from_greenscreen(self.obj_3)
            self.remove_object_from_greenscreen(self.obj_4)

        if self.rest_qpos is not None:
            self.rest_qpos = common.to_tensor(self.rest_qpos, device=self.device)
            
        self.table_pose = Pose.create_from_pq(
            p=[-0.12 + 0.737, 0, -0.9196429], q=euler2quat(0, 0, np.pi / 2)
        )

        self._load_camera_mount()
        self._randomize_robot_color()

    def _sample_instrument_poses(self, b: int, base_pos: torch.Tensor):
        # Sample for base instrument 1
        xyz_1 = torch.zeros((b, 3), device=self.device)
        xyz_1[:, :2] = (
            torch.rand((b, 2), device=self.device) * self.spawn_box_half_size * 2
            - self.spawn_box_half_size
        )
        xyz_1[:, :2] += base_pos[:, :2]
        xyz_1[..., 2] = 0.008

        # Create localized directional anchors for the other 3 objects
        yaw1 = torch.rand(b, device=self.device) * 2 * torch.pi
        
        def make_quat(yaw_angles):
            q = torch.zeros((b, 4), device=self.device)
            q[:, 0] = torch.cos(yaw_angles / 2)
            q[:, 3] = torch.sin(yaw_angles / 2)
            return q

        q1 = make_quat(yaw1)
        q2 = make_quat(yaw1 + (torch.rand(b, device=self.device) - 0.5) * 0.1)
        q3 = make_quat(yaw1 + (torch.rand(b, device=self.device) - 0.5) * 0.2)
        q4 = make_quat(yaw1 + (torch.rand(b, device=self.device) - 0.5) * 0.3)

        perp_dir = torch.stack([-torch.sin(yaw1), torch.cos(yaw1)], dim=1)
        par_dir = torch.stack([torch.cos(yaw1), torch.sin(yaw1)], dim=1)

        # Distribute the other 3 items relative to the first anchor
        xyz_2 = xyz_1.clone()
        side_dist2 = (torch.rand(b, device=self.device) * 0.01 + 0.01) * torch.sign(torch.randn(b, device=self.device))
        fwd_dist2 = (torch.rand(b, device=self.device) - 0.5) * 0.10
        xyz_2[:, :2] += perp_dir * side_dist2.unsqueeze(1) + par_dir * fwd_dist2.unsqueeze(1)
        xyz_2[..., 2] = 0.008

        xyz_3 = xyz_1.clone()
        side_dist3 = (torch.rand(b, device=self.device) * 0.01 + 0.02) * torch.sign(torch.randn(b, device=self.device))
        fwd_dist3 = (torch.rand(b, device=self.device) - 0.5) * 0.10
        xyz_3[:, :2] += perp_dir * side_dist3.unsqueeze(1) + par_dir * fwd_dist3.unsqueeze(1)
        xyz_3[..., 2] = 0.008

        xyz_4 = xyz_1.clone()
        side_dist4 = (torch.rand(b, device=self.device) * 0.01 + 0.03) * torch.sign(torch.randn(b, device=self.device))
        fwd_dist4 = (torch.rand(b, device=self.device) - 0.5) * 0.10
        xyz_4[:, :2] += perp_dir * side_dist4.unsqueeze(1) + par_dir * fwd_dist4.unsqueeze(1)
        xyz_4[..., 2] = 0.008
        
        return xyz_1, q1, xyz_2, q2, xyz_3, q3, xyz_4, q4

    def _initialize_episode(self, env_idx: torch.Tensor, options: dict):
        super()._initialize_episode(env_idx, options)
        with torch.device(self.device):
            b = len(env_idx)
            self.table_scene.initialize(env_idx)
            self.table_scene.table.set_pose(self.table_pose)

            if self.rest_qpos is not None:
                self.agent.robot.set_qpos(
                    self.rest_qpos + torch.randn(size=(b, self.rest_qpos.shape[-1])) * self.domain_randomization_config.initial_qpos_noise_scale
                )
            self.agent.robot.set_pose(
                Pose.create_from_pq(p=[0, 0, 0], q=euler2quat(0, 0, self.base_z_rot))
            )

            spawn_box_pos_tensor = self.agent.robot.pose.p + torch.tensor(
                [self.spawn_box_pos[0], self.spawn_box_pos[1], 0]
            )

            # Sample and set poses for all 4 Forceps
            p1, q1, p2, q2, p3, q3, p4, q4 = self._sample_instrument_poses(b, spawn_box_pos_tensor[env_idx])
            self.obj_1.set_pose(Pose.create_from_pq(p=p1, q=q1))
            self.obj_2.set_pose(Pose.create_from_pq(p=p2, q=q2))
            self.obj_3.set_pose(Pose.create_from_pq(p=p3, q=q3))
            self.obj_4.set_pose(Pose.create_from_pq(p=p4, q=q4))

            if not hasattr(self, "env_phase"):
                self.env_phase = torch.zeros(self.num_envs, dtype=torch.int32, device=self.device)
                self.settle_steps = torch.zeros(self.num_envs, dtype=torch.int32, device=self.device)
                self.spawn_box_pos_tensor = self.agent.robot.pose.p + torch.tensor(
                    [self.spawn_box_pos[0], self.spawn_box_pos[1], 0], device=self.device
                )
            
            self.env_phase[env_idx] = 0
            self.settle_steps[env_idx] = 0

    def step(self, action: Union[None, np.ndarray, torch.Tensor, dict]):
        if not hasattr(self, "env_phase"):
            self.env_phase = torch.zeros(self.num_envs, dtype=torch.int32, device=self.device)
            self.settle_steps = torch.zeros(self.num_envs, dtype=torch.int32, device=self.device)
            self.spawn_box_pos_tensor = self.agent.robot.pose.p + torch.tensor(
                [self.spawn_box_pos[0], self.spawn_box_pos[1], 0], device=self.device
            )

        settle_duration = 5
        settling = self.env_phase == 0

        if action is not None and settling.any():
            if isinstance(action, torch.Tensor):
                action = action.clone()
                if action.dim() == 1:
                    action = action.unsqueeze(0)
                    action[settling] = 0.0
                    action = action.squeeze(0)
                else:
                    action[settling] = 0.0
            elif isinstance(action, np.ndarray):
                action = action.copy()
                settling_np = settling.cpu().numpy() if isinstance(settling, torch.Tensor) else settling
                
                if action.ndim == 1:
                    action = np.expand_dims(action, 0)
                    action[settling_np] = 0.0
                    action = action[0]
                else:
                    action[settling_np] = 0.0

        if settling.any():
            # Update velocity damping across all 4 bodies
            for obj in [self.obj_1, self.obj_2, self.obj_3, self.obj_4]:
                vel = obj.linear_velocity.clone()
                avel = obj.angular_velocity.clone()
                vel[settling] *= 0.1
                avel[settling] *= 0.1
                obj.set_linear_velocity(vel)
                obj.set_angular_velocity(avel)

        result = DefaultCameraEnv.step(self, action)

        if settling.any():
            self.settle_steps[settling] += 1
            done_settling = (self.env_phase == 0) & (self.settle_steps >= settle_duration)
            
            if done_settling.any():
                # Check maximum compaction during initialization between sequential clusters
                d12 = torch.linalg.norm(self.obj_1.pose.p[..., :2] - self.obj_2.pose.p[..., :2], axis=1)
                d23 = torch.linalg.norm(self.obj_2.pose.p[..., :2] - self.obj_3.pose.p[..., :2], axis=1)
                d34 = torch.linalg.norm(self.obj_3.pose.p[..., :2] - self.obj_4.pose.p[..., :2], axis=1)
                
                is_close = (d12 < 0.08) & (d23 < 0.08) & (d34 < 0.08)
                
                success_idx = done_settling & is_close
                self.env_phase[success_idx] = 1
                
                fail_idx = done_settling & ~is_close
                num_fail = fail_idx.sum().item()
                if num_fail > 0:
                    reconfig_idxs = torch.where(fail_idx)[0]
                    p1, q1, p2, q2, p3, q3, p4, q4 = self._sample_instrument_poses(num_fail, self.spawn_box_pos_tensor[reconfig_idxs])
                    
                    def patch_tensor(obj, sampled_p, sampled_q):
                        full_p = obj.pose.p.clone()
                        full_q = obj.pose.q.clone()
                        full_p[reconfig_idxs] = sampled_p
                        full_q[reconfig_idxs] = sampled_q
                        obj.set_pose(Pose.create_from_pq(p=full_p, q=full_q))
                        
                        v = obj.linear_velocity.clone()
                        av = obj.angular_velocity.clone()
                        v[reconfig_idxs] = 0.0
                        av[reconfig_idxs] = 0.0
                        obj.set_linear_velocity(v)
                        obj.set_angular_velocity(av)

                    patch_tensor(self.obj_1, p1, q1)
                    patch_tensor(self.obj_2, p2, q2)
                    patch_tensor(self.obj_3, p3, q3)
                    patch_tensor(self.obj_4, p4, q4)
                    
                    self.settle_steps[fail_idx] = 0

        return result

    def evaluate(self):
        # Global success evaluates whether every separate pairing breaks the 0.20m boundary
        poses = [self.obj_1.pose.p[..., :2], self.obj_2.pose.p[..., :2], self.obj_3.pose.p[..., :2], self.obj_4.pose.p[..., :2]]
        all_separated = torch.tensor(True, device=self.device, dtype=torch.bool)
        
        for i in range(4):
            for j in range(i + 1, 4):
                dist = torch.linalg.norm(poses[i] - poses[j], axis=1)
                all_separated = all_separated & (dist > 0.20)

        robot_touching_table = self.agent.is_touching(self.table_scene.table)
        return {
            "success": all_separated,
            "robot_touching_table": robot_touching_table
        }

    def _get_obs_agent(self):
        qpos = self.agent.robot.get_qpos()
        if self.domain_randomization and self.domain_randomization_config.robot_qpos_noise_std > 0:
            noise = torch.randn_like(qpos) * self.domain_randomization_config.robot_qpos_noise_std
            qpos = qpos + noise
        obs = dict(noisy_qpos=qpos)
        controller_state = self.agent.controller.get_state()
        if len(controller_state) > 0:
            obs.update(controller=controller_state)
        return obs

    def _get_obs_extra(self, info: dict):
        obs = dict(
            qpos=self.agent.robot.get_qpos(),
            tcp_pose=self.agent.tcp_pose.raw_pose,
        )
        if self.obs_mode_struct.state:
            obs.update(
                obj_1_pose=self.obj_1.pose.raw_pose,
                obj_2_pose=self.obj_2.pose.raw_pose, 
                obj_3_pose=self.obj_3.pose.raw_pose, 
                obj_4_pose=self.obj_4.pose.raw_pose, 
            )
        return obs

    def compute_dense_reward(self, obs: Any, action: Array, info: dict):
        # 1. Fixed Time Penalty
        reward = torch.full((self.num_envs,), -0.05, device=self.device)
        if hasattr(self, "env_phase"):
            settling = self.env_phase == 0
            reward[settling] = 0.0

        # Calculate multi-body centroid
        midpoint_objs = (self.obj_1.pose.p + self.obj_2.pose.p + self.obj_3.pose.p + self.obj_4.pose.p) / 4.0
        tcp_to_objs_dist = torch.linalg.norm(midpoint_objs - self.agent.tcp_pos, axis=1)
        
        # Gather all distinct relative distance pairs
        obj_poses = [self.obj_1.pose.p[..., :2], self.obj_2.pose.p[..., :2], self.obj_3.pose.p[..., :2], self.obj_4.pose.p[..., :2]]
        pair_distances = []
        for i in range(4):
            for j in range(i + 1, 4):
                pair_distances.append(torch.linalg.norm(obj_poses[i] - obj_poses[j], axis=1))

        # Linear and angular velocity tracking across the cluster
        vel_mags = [torch.linalg.norm(obj.linear_velocity, axis=1) for obj in [self.obj_1, self.obj_2, self.obj_3, self.obj_4]]
        total_cluster_speed = sum(vel_mags)

        # 2. Reaching Reward
        reaching_reward = 1.0 - torch.tanh(5.0 * tcp_to_objs_dist)
        reward += reaching_reward

        # 3. Gaze/Looking Reward (Stays locked onto the 4-object cluster centroid)
        if hasattr(self, "cameras") and len(self.cameras) > 0:
            cam = list(self.cameras.values())[0]
            cam_pose = cam.pose
            cam_mat = cam_pose.to_transformation_matrix() 
            cam_forward = cam_mat[:, :3, 2] 
            
            cam_to_midpoint = midpoint_objs - cam_pose.p
            cam_to_midpoint = cam_to_midpoint / (torch.linalg.norm(cam_to_midpoint, axis=1, keepdim=True) + 1e-6)
            
            gaze_alignment = torch.clamp(torch.sum(cam_forward * cam_to_midpoint, dim=1), min=0.0)
            reward += 1.5 * gaze_alignment

        # 4. Controlled "Slow Pushing" Reward (Evaluated across the total speed of all 4 parts)
        arm_is_close = tcp_to_objs_dist < 0.15
        slow_control_multiplier = torch.clamp(1.0 - (total_cluster_speed / 0.6), min=0.0, max=1.0)
        
        # Reward increasing the mean separation distance across every pair configuration slowly
        mean_separation = sum(pair_distances) / len(pair_distances)
        pushing_progress = torch.clamp(mean_separation, max=0.25)
        slow_push_reward = 2.0 * pushing_progress * slow_control_multiplier
        reward += torch.where(arm_is_close, slow_push_reward, torch.zeros_like(reward))

        # 5. Non-Overlapping Proximity Optimization (Multi-body Sweet Spot Matrix)
        min_safe_separation = 0.08  
        proximity_bonus_total = torch.zeros_like(reward)
        
        for dist in pair_distances:
            is_separated = dist > min_safe_separation
            # Accumulate reward points for each separate pair optimized to the proximity margin
            pair_bonus = torch.exp(-8.0 * (dist - min_safe_separation))
            proximity_bonus_total += torch.where(is_separated, pair_bonus, torch.zeros_like(reward))
            
        reward += (5.0 / 6.0) * proximity_bonus_total  # Normalized against the 6 unique pairing combinations

        # 6. Success Bonus
        reward[info["success"]] = 15.0

        # 7. Operational Penalties
        if "robot_touching_table" in info:
            reward -= 2.0 * info["robot_touching_table"].float()
            
        for vel in vel_mags:
            excessive_vel = torch.clamp(vel - 0.4, min=0.0)
            reward -= 3.0 * (excessive_vel ** 2)

        return reward

    def compute_normalized_dense_reward(self, obs: Any, action: Array, info: dict):
        max_reward = 25.0 
        return self.compute_dense_reward(obs=obs, action=action, info=info) / max_reward