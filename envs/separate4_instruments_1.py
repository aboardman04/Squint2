# This env has updated rewards and penalties functions to reward looking at the 
# instruments to make sure they are separated and to penalize extreme movements

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


@register_env("SeparateInstruments-v2.5", max_episode_steps=20)
class SeparateInstrumentsEnv(DefaultCameraEnv):

    SUPPORTED_ROBOTS = ["so100", "so101", "panda", "fetch"]
    SUPPORTED_OBS_MODES = ["none", "state", "state_dict", "rgb", "rgb+segmentation", "rgb+state", "rgb+segmentation+state",
                           "rgb+depth+segmentation", "rgb+depth+segmentation+state"]
    agent: Union[SO100, SO101, Panda, Fetch]

    instrument_half_size = 0.025 #0.075

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

    def _build_settle_tray(self):
        """Create an invisible tray used only while the instruments settle."""
        wall_thickness = 0.003
        wall_height = 0.04
        half_size = 0.08
        invisible = sapien.render.RenderMaterial(
            base_color=[1, 1, 1, 0.5]
        )
        self.settle_walls = []
        wall_specs = [
            ([wall_thickness, half_size, wall_height], [-half_size, 0, wall_height]), # left
            ([wall_thickness, half_size, wall_height], [half_size, 0, wall_height]),  # right
            ([half_size, wall_thickness, wall_height], [0, half_size, wall_height]),  # front
            ([half_size, wall_thickness, wall_height], [0, -half_size, wall_height]), # back        
        ]
        for i, (half_extents, local_pos) in enumerate(wall_specs):
            builder = self.scene.create_actor_builder()
            builder.add_box_collision(half_size=half_extents)
            builder.add_box_visual(half_size=half_extents, material=invisible)
            builder.initial_pose = sapien.Pose(p=local_pos)
            wall = builder.build_kinematic(name=f"settle_wall_{i}")
            self.settle_walls.append((wall, torch.tensor(local_pos, device=self.device)))

    def _load_scene(self, options: dict):
        self.table_scene = TableSceneBuilder(self)
        self.table_scene.build()
        self._color_table()
        self._build_settle_tray()

    #-------------------------------
    #------ Build Instruments ------
    #-------------------------------

        obj_path = "/home/aboardman/squint2/deploy_utils/blender_objs/dressing_forceps.obj"
        obj2_path = "/home/aboardman/squint2/deploy_utils/blender_objs/allis.obj"

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

        # --- Allis 1 ---
        builder3 = self.scene.create_actor_builder()
        builder3.add_visual_from_file(filename=obj2_path, material=steel_material)
        builder3.add_multiple_convex_collisions_from_file(
            filename=obj2_path, decomposition="coacd", material=physx_material
        )
        builder3.initial_pose = sapien.Pose(p=[-0.1, 0.05, 0.1])
        self.obj_3 = builder3.build(name="allis_1")

        # --- Allis 2 ---
        builder4 = self.scene.create_actor_builder()
        builder4.add_visual_from_file(filename=obj2_path, material=steel_material)
        builder4.add_multiple_convex_collisions_from_file(
            filename=obj2_path, decomposition="coacd", material=physx_material
        )
        builder4.initial_pose = sapien.Pose(p=[0.1, 0.05, 0.1])
        self.obj_4 = builder4.build(name="allis_2")

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
        """
        Generate four independently randomized instruments within a small
        spawn region so they naturally settle into a pile.
        """
        xy_range = 0.02          # ±2 cm
        z_base = 0.008           # Just above table
        z_spacing = 0.005        # 5 mm stagger

        poses = []

        for i in range(4):
            xyz = torch.zeros((b, 3), device=self.device)

            xyz[:, 0] = (
                base_pos[:, 0]
                + (torch.rand(b, device=self.device) * 2 - 1) * xy_range
            )

            xyz[:, 1] = (
                base_pos[:, 1]
                + (torch.rand(b, device=self.device) * 2 - 1) * xy_range
            )

            xyz[:, 2] = z_base + i * z_spacing

            yaw = torch.rand(b, device=self.device) * 2 * torch.pi

            q = torch.zeros((b, 4), device=self.device)
            q[:, 0] = torch.cos(yaw / 2)
            q[:, 3] = torch.sin(yaw / 2)

            poses.extend([xyz, q])

        return tuple(poses)

    def _initialize_episode(self, env_idx: torch.Tensor, options: dict):
        super()._initialize_episode(env_idx, options)

        with torch.device(self.device):
            b = len(env_idx)

            self.table_scene.initialize(env_idx)
            self.table_scene.table.set_pose(self.table_pose)

            if self.rest_qpos is not None:
                self.agent.robot.set_qpos(
                    self.rest_qpos
                    + torch.randn(
                        size=(b, self.rest_qpos.shape[-1]),
                        device=self.device,
                    )
                    * self.domain_randomization_config.initial_qpos_noise_scale
                )

            self.agent.robot.set_pose(
                Pose.create_from_pq(
                    p=[0, 0, 0],
                    q=euler2quat(0, 0, self.base_z_rot),
                )
            )

            spawn_box_pos_tensor = (
                self.agent.robot.pose.p
                + torch.tensor(
                    [self.spawn_box_pos[0], self.spawn_box_pos[1], 0],
                    device=self.device,
                )
            )
            center = spawn_box_pos_tensor[env_idx]
            
            # Correctly slice batch dimension for table height
            tray_z = self.table_scene.table.pose.p[:, 2] + 0.06
            
            for wall, offset in self.settle_walls:
                pos = center.clone()
                pos[:, 0] += offset[0]
                pos[:, 1] += offset[1]
                pos[:, 2] = tray_z
                wall.set_pose(
                    Pose.create_from_pq(
                        p=pos,
                        q=torch.tensor([1.0, 0.0, 0.0, 0.0], device=self.device).repeat(b, 1),
                    )                    
                )

            p1, q1, p2, q2, p3, q3, p4, q4 = self._sample_instrument_poses(
                b,
                spawn_box_pos_tensor[env_idx],
            )

            self.obj_1.set_pose(Pose.create_from_pq(p=p1, q=q1))
            self.obj_2.set_pose(Pose.create_from_pq(p=p2, q=q2))
            self.obj_3.set_pose(Pose.create_from_pq(p=p3, q=q3))
            self.obj_4.set_pose(Pose.create_from_pq(p=p4, q=q4))

            if not hasattr(self, "env_phase"):
                self.env_phase = torch.zeros(
                    self.num_envs,
                    dtype=torch.int32,
                    device=self.device,
                )

                self.settle_steps = torch.zeros(
                    self.num_envs,
                    dtype=torch.int32,
                    device=self.device,
                )

            self.env_phase[env_idx] = 0
            self.settle_steps[env_idx] = 0

    def step(self, action):
        if not hasattr(self, "env_phase"):
            self.env_phase = torch.zeros(
                self.num_envs,
                dtype=torch.int32,
                device=self.device,
            )
            self.settle_steps = torch.zeros(
                self.num_envs,
                dtype=torch.int32,
                device=self.device,
            )
        settle_duration = 30
        settling = self.env_phase == 0

        # Disable robot actions while objects settle.
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
                settling_np = settling.cpu().numpy()               
                if action.ndim == 1:
                    action = np.expand_dims(action, 0)
                    action[settling_np] = 0.0
                    action = action[0]
                else:
                    action[settling_np] = 0.0

        # Let physics settle the pile.
        result = DefaultCameraEnv.step(self, action)
        if settling.any():
            self.settle_steps[settling] += 1
            finished = (
                (self.env_phase == 0)
                & (self.settle_steps >= settle_duration)
            )
            self.env_phase[finished] = 1
            if finished.any():
                for wall, _ in self.settle_walls:
                    # In GPU sim, wall.set_pose updates all environments. 
                    # We must modify only the finished environments in the full batch.
                    curr_p = wall.pose.p.clone()
                    curr_q = wall.pose.q.clone()
                    
                    curr_p[finished, 0] = 0.0
                    curr_p[finished, 1] = 0.0
                    curr_p[finished, 2] = -10.0
                    
                    curr_q[finished, 0] = 1.0
                    curr_q[finished, 1] = 0.0
                    curr_q[finished, 2] = 0.0
                    curr_q[finished, 3] = 0.0
                    
                    wall.set_pose(
                        Pose.create_from_pq(
                            p=curr_p,
                            q=curr_q,
                        )
                    )
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
        # add randomization/noise to the simulation to help with sim to real conversion
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
        reward = torch.full(
            (self.num_envs,),
            -0.01,
            device=self.device,
        )
        # Ignore reward while pile settles
        if hasattr(self, "env_phase"):
            settling = self.env_phase == 0
            reward[settling] = 0.0
        objects = [
            self.obj_1,
            self.obj_2,
            self.obj_3,
            self.obj_4,
        ]
        table_height = self.table_scene.table.pose.p[:,2]
        # Reach nearest instrument
        tcp = self.agent.tcp_pos
        obj_positions = torch.stack(
            [o.pose.p for o in objects],
            dim=1,
        )
        tcp_expand = tcp.unsqueeze(1)
        distances = torch.linalg.norm(
            obj_positions - tcp_expand,
            dim=2,
        )
        nearest_dist = torch.min(distances, dim=1).values
        reward += 1.0 - torch.tanh(5.0 * nearest_dist)
        # Reward lifting
        for obj in objects:
            lift_height = torch.clamp(
                obj.pose.p[:, 2] - table_height,
                min=0.0,
            )
            reward += 3.0 * torch.clamp(
                lift_height / 0.05,
                max=1.0,
            )
        # Penalize disturbing other instruments
        for obj in objects:
            speed = torch.linalg.norm(
                obj.linear_velocity,
                dim=1,
            )
            reward -= 0.5 * torch.clamp(
                speed - 0.10,
                min=0.0,
            )
        # Reward isolated placement
        isolated_bonus = torch.zeros_like(reward)
        for i in range(4):
            isolated = torch.ones(
                self.num_envs,
                dtype=torch.bool,
                device=self.device,
            )
            for j in range(4):
                if i == j:
                    continue
                d = torch.linalg.norm(
                    objects[i].pose.p[:, :2]
                    - objects[j].pose.p[:, :2],
                    dim=1,
                )
                isolated &= d > 0.12
            low_enough = (
                objects[i].pose.p[:, 2]
                < table_height + 0.03
            )
            isolated_bonus += (
                isolated.float()
                * low_enough.float()
            )
        reward += 5.0 * isolated_bonus
        # Penalize excessive speed
        for obj in objects:
            speed = torch.linalg.norm(
                obj.linear_velocity,
                dim=1,
            )
            reward -= 2.0 * torch.clamp(
                speed - 0.30,
                min=0.0,
            )
        # Penalize robot touching table
        if "robot_touching_table" in info:
            reward -= (
                3.0
                * info["robot_touching_table"].float()
            )
        # Keep separated objects inside workspace
        workspace_center = self.agent.robot.pose.p[:, :2]
        for obj in objects:
            radius = torch.linalg.norm(
                obj.pose.p[:, :2]
                - workspace_center,
                dim=1,
            )
            reward -= torch.clamp(
                radius - 0.25,
                min=0.0,
            ) * 5.0
        # Large success reward
        reward[info["success"]] = 100.0
        return reward

    def compute_normalized_dense_reward(
        self,
        obs,
        action,
        info,
    ):
        return (
            self.compute_dense_reward(
                obs,
                action,
                info,
            )
            / 100.0
        )