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


@register_env("SeparateInstruments-v1", max_episode_steps=50)
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
        builder1.initial_pose = sapien.Pose(p=[-0.1, 0.0, 0.1], q=[1, 0, 0, 0])
        self.obj_1 = builder1.build(name="forceps_1")

        # --- Forceps 2 ---
        builder2 = self.scene.create_actor_builder()
        builder2.add_visual_from_file(filename=obj_path, material=steel_material)
        builder2.add_multiple_convex_collisions_from_file(
            filename=obj_path, decomposition="coacd", material=physx_material
        )
        builder2.initial_pose = sapien.Pose(p=[0.1, 0.0, 0.1])
        self.obj_2 = builder2.build(name="forceps_2")

        if self.apply_greenscreen:
            self.remove_object_from_greenscreen(self.agent.robot)
            self.remove_object_from_greenscreen(self.obj_1)
            self.remove_object_from_greenscreen(self.obj_2)

        if self.rest_qpos is not None:
            self.rest_qpos = common.to_tensor(self.rest_qpos, device=self.device)
            
        self.table_pose = Pose.create_from_pq(
            p=[-0.12 + 0.737, 0, -0.9196429], q=euler2quat(0, 0, np.pi / 2)
        )

        self._load_camera_mount()
        self._randomize_robot_color()

    def _sample_instrument_poses(self, b: int, base_pos: torch.Tensor):
        xyz_1 = torch.zeros((b, 3), device=self.device)
        xyz_1[:, :2] = (
            torch.rand((b, 2), device=self.device) * self.spawn_box_half_size * 2
            - self.spawn_box_half_size
        )
        xyz_1[:, :2] += base_pos[:, :2]
        xyz_1[..., 2] = 0.008

        yaw1 = torch.rand(b, device=self.device) * 2 * torch.pi
        yaw2 = yaw1 + (torch.rand(b, device=self.device) - 0.5) * 0.1

        q1 = torch.zeros((b, 4), device=self.device)
        q1[:, 0] = torch.cos(yaw1 / 2)
        q1[:, 3] = torch.sin(yaw1 / 2)

        q2 = torch.zeros((b, 4), device=self.device)
        q2[:, 0] = torch.cos(yaw2 / 2)
        q2[:, 3] = torch.sin(yaw2 / 2)

        perp_dir = torch.stack([-torch.sin(yaw1), torch.cos(yaw1)], dim=1)
        par_dir = torch.stack([torch.cos(yaw1), torch.sin(yaw1)], dim=1)

        side_dist = (torch.rand(b, device=self.device) * 0.01 + 0.01) * torch.sign(torch.randn(b, device=self.device))
        fwd_dist = (torch.rand(b, device=self.device) - 0.5) * 0.10

        xyz_2 = xyz_1.clone()
        xyz_2[:, :2] += perp_dir * side_dist.unsqueeze(1) + par_dir * fwd_dist.unsqueeze(1)
        xyz_2[..., 2] = 0.008
        
        return xyz_1, q1, xyz_2, q2

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

            # 1. Sample and set poses for Forceps
            p1, q1, p2, q2 = self._sample_instrument_poses(b, spawn_box_pos_tensor[env_idx])
            self.obj_1.set_pose(Pose.create_from_pq(p=p1, q=q1))
            self.obj_2.set_pose(Pose.create_from_pq(p=p2, q=q2))

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
                # Use numpy indexing for numpy actions
                if isinstance(settling, torch.Tensor):
                    settling_np = settling.cpu().numpy()
                else:
                    settling_np = settling
                
                if action.ndim == 1:
                    action = np.expand_dims(action, 0)
                    action[settling_np] = 0.0
                    action = action[0]
                else:
                    action[settling_np] = 0.0

        if settling.any():
            vel1 = self.obj_1.linear_velocity.clone()
            avel1 = self.obj_1.angular_velocity.clone()
            vel2 = self.obj_2.linear_velocity.clone()
            avel2 = self.obj_2.angular_velocity.clone()
            
            vel1[settling] *= 0.1
            avel1[settling] *= 0.1
            vel2[settling] *= 0.1
            avel2[settling] *= 0.1
            
            self.obj_1.set_linear_velocity(vel1)
            self.obj_1.set_angular_velocity(avel1)
            self.obj_2.set_linear_velocity(vel2)
            self.obj_2.set_angular_velocity(avel2)

        result = DefaultCameraEnv.step(self, action)

        if settling.any():
            self.settle_steps[settling] += 1
            done_settling = (self.env_phase == 0) & (self.settle_steps >= settle_duration)
            
            if done_settling.any():
                dist = torch.linalg.norm(self.obj_1.pose.p[..., :2] - self.obj_2.pose.p[..., :2], axis=1)
                is_close = dist < 0.08
                
                success_idx = done_settling & is_close
                self.env_phase[success_idx] = 1
                
                fail_idx = done_settling & ~is_close
                num_fail = fail_idx.sum().item()
                if num_fail > 0:
                    reconfig_idxs = torch.where(fail_idx)[0]
                    p1, q1, p2, q2 = self._sample_instrument_poses(num_fail, self.spawn_box_pos_tensor[reconfig_idxs])
                    
                    full_p1 = self.obj_1.pose.p.clone()
                    full_q1 = self.obj_1.pose.q.clone()
                    full_p2 = self.obj_2.pose.p.clone()
                    full_q2 = self.obj_2.pose.q.clone()
                    
                    full_p1[reconfig_idxs] = p1
                    full_q1[reconfig_idxs] = q1
                    full_p2[reconfig_idxs] = p2
                    full_q2[reconfig_idxs] = q2
                    
                    self.obj_1.set_pose(Pose.create_from_pq(p=full_p1, q=full_q1))
                    self.obj_2.set_pose(Pose.create_from_pq(p=full_p2, q=full_q2))
                    
                    vel1 = self.obj_1.linear_velocity.clone()
                    avel1 = self.obj_1.angular_velocity.clone()
                    vel2 = self.obj_2.linear_velocity.clone()
                    avel2 = self.obj_2.angular_velocity.clone()
                    vel1[reconfig_idxs] = 0.0
                    avel1[reconfig_idxs] = 0.0
                    vel2[reconfig_idxs] = 0.0
                    avel2[reconfig_idxs] = 0.0
                    self.obj_1.set_linear_velocity(vel1)
                    self.obj_1.set_angular_velocity(avel1)
                    self.obj_2.set_linear_velocity(vel2)
                    self.obj_2.set_angular_velocity(avel2)
                    
                    self.settle_steps[fail_idx] = 0

        return result

    def evaluate(self):
        distance_between_objs = torch.linalg.norm(
            self.obj_1.pose.p[..., :2] - self.obj_2.pose.p[..., :2], axis=1
        )
        is_separated = distance_between_objs > 0.20
        robot_touching_table = self.agent.is_touching(self.table_scene.table)
        return {
            "success": is_separated,
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
            )
        return obs

    def compute_dense_reward(self, obs: Any, action: Array, info: dict):
        # 1. Fixed Time Penalty (Critical for speed optimization)
        reward = torch.full((self.num_envs,), -0.05, device=self.device)
        
        if hasattr(self, "env_phase"):
            settling = self.env_phase == 0
            reward[settling] = 0.0

        # 2. Reaching Reward
        midpoint_objs = (self.obj_1.pose.p + self.obj_2.pose.p) / 2.0
        tcp_to_objs_dist = torch.linalg.norm(midpoint_objs - self.agent.tcp_pos, axis=1)
        
        reaching_reward = 1.0 - torch.tanh(5.0 * tcp_to_objs_dist)
        reward += reaching_reward

        # 3. Separation Reward (Drives the fast splitting motion)
        distance_between_objs = torch.linalg.norm(
            self.obj_1.pose.p[..., :2] - self.obj_2.pose.p[..., :2], axis=1
        )
        
        target_separation = 0.20
        separation_reward = 3.0 * (1.0 - torch.tanh(3.0 * (target_separation - distance_between_objs)))
        
        arm_is_close = tcp_to_objs_dist < 0.15
        reward += torch.where(arm_is_close, separation_reward, torch.zeros_like(reward))

        # 4. Success Bonus (Massive payout to heavily favor early termination)
        reward[info["success"]] = 10.0

        # 5. Operational Penalties
        if "robot_touching_table" in info:
            reward -= 2.0 * info["robot_touching_table"].float()

        return reward

    def compute_normalized_dense_reward(self, obs: Any, action: Array, info: dict):
        max_reward = 14.0 
        return self.compute_dense_reward(obs=obs, action=action, info=info) / max_reward
