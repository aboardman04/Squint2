from typing import Any, Union

import numpy as np
import sapien
import torch
import torch.random
from transforms3d.euler import euler2quat

from mani_skill.agents.robots import Fetch, Panda
from mani_skill.envs.sapien_env import BaseEnv
from mani_skill.sensors.camera import CameraConfig
from mani_skill.utils import common, sapien_utils
from mani_skill.utils.building import actors
from mani_skill.utils.registration import register_env
from mani_skill.utils.scene_builder.table import TableSceneBuilder
from mani_skill.utils.structs import Pose
from mani_skill.utils.structs.types import Array, GPUMemoryConfig, SimConfig

# ----- my additional imports -----
from .base_random_env import DefaultCameraEnv, DefaultRandomizationConfig
from .robot.so101 import SO101


@register_env("SeparateInstruments-v5", max_episode_steps=500)
class SeparateInstrumentsEnv(DefaultCameraEnv):
    
    SUPPORTED_ROBOTS = ["so101", "panda", "fetch"]

    agent: Union[SO101, Panda, Fetch]

    goal_radius = 0.1
    instrument_spawn_xy_range = 0.02
    instrument_spawn_z_base = 0.008
    instrument_spawn_z_spacing = 0.007
    num_instruments = 4


    def __init__(self, *args, robot_uids="so101", robot_init_qpos_noise=0.02, **kwargs):
        self.robot_init_qpos_noise = robot_init_qpos_noise
        super().__init__(*args, robot_uids=robot_uids, **kwargs)

    # Specify default simulation/gpu memory configurations to override any default values
    @property
    def _default_sim_config(self):
        return SimConfig(
            gpu_memory_config=GPUMemoryConfig(
                found_lost_pairs_capacity=2**25, max_rigid_patch_count=2**18
            )
        )

    @property
    def _default_sensor_configs(self):
        # registers one 128x128 camera looking at the robot, cube, and target
        # a smaller sized camera will be lower quality, but render faster
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
        # registers a more high-definition (512x512) camera used just for rendering when render_mode="rgb_array" or calling env.render_rgb_array()
        pose = sapien_utils.look_at([0.6, 0.7, 0.6], [0.0, 0.0, 0.35])
        return CameraConfig(
            "render_camera", pose=pose, width=512, height=512, fov=1, near=0.01, far=100
        )

    def _load_agent(self, options: dict):
        # set a reasonable initial pose for the agent that doesn't intersect other objects
        super()._load_agent(options, sapien.Pose(p=[-0.615, 0, 0]))

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
            xyz[:, 0] = base_pos[:, 0] + (torch.rand(b, device=self.device) * 2 - 1) * self.instrument_spawn_xy_range
            xyz[:, 1] = base_pos[:, 1] + (torch.rand(b, device=self.device) * 2 - 1) * self.instrument_spawn_xy_range
            xyz[:, 2] = self.instrument_spawn_z_base + i * self.instrument_spawn_z_spacing

            yaw = torch.rand(b, device=self.device) * 2 * torch.pi
            q = torch.zeros((b, 4), device=self.device)
            q[:, 0] = torch.cos(yaw / 2)
            q[:, 3] = torch.sin(yaw / 2)
            poses.extend([xyz, q])
        return tuple(poses)

    def _load_scene(self, options: dict):
        self.table_scene = TableSceneBuilder(
            env=self, robot_init_qpos_noise=self.robot_init_qpos_noise
        )
        self.table_scene.build()

        # Create camera mount actors expected by the base randomization helpers.
        builder = self.scene.create_actor_builder()
        builder.initial_pose = sapien.Pose()
        self.camera_mount = builder.build_kinematic("camera_mount")

        builder = self.scene.create_actor_builder()
        builder.initial_pose = sapien.Pose()
        self.wrist_camera_mount = builder.build_kinematic("wrist_camera_mount")

        obj1_path = "/home/aboardman/squint2/deploy_utils/blender_objs/dressing_forceps.obj"
        obj2_path = "/home/aboardman/squint2/deploy_utils/blender_objs/allis.obj"

        self.obj_1 = self._build_instrument(
            obj1_path,
            name="forceps_1",
            initial_pose=sapien.Pose(p=[-0.1, -0.05, 0.1], q=[1, 0, 0, 0]),
        )
        self.obj_2 = self._build_instrument(
            obj1_path,
            name="forceps_2",
            initial_pose=sapien.Pose(p=[0.1, -0.05, 0.1]),
        )
        self.obj_3 = self._build_instrument(
            obj2_path,
            name="allis_1",
            initial_pose=sapien.Pose(p=[-0.1, 0.05, 0.1]),
        )
        self.obj_4 = self._build_instrument(
            obj2_path,
            name="allis_2",
            initial_pose=sapien.Pose(p=[0.1, 0.05, 0.1]),
        )
        self.objects = [self.obj_1, self.obj_2, self.obj_3, self.obj_4]
        self.obj = self.obj_1

    def _initialize_episode(self, env_idx: torch.Tensor, options: dict):
        with torch.device(self.device):
            b = len(env_idx)
            self.table_scene.initialize(env_idx)

            center = self.agent.robot.pose.p + torch.tensor([0.3, 0.0, 0.0], device=self.device)
            center = center[env_idx]

            p1, q1, p2, q2, p3, q3, p4, q4 = self._sample_instrument_poses(b, center)
            self.obj_1.set_pose(Pose.create_from_pq(p=p1, q=q1))
            self.obj_2.set_pose(Pose.create_from_pq(p=p2, q=q2))
            self.obj_3.set_pose(Pose.create_from_pq(p=p3, q=q3))
            self.obj_4.set_pose(Pose.create_from_pq(p=p4, q=q4))
            self.obj = self.obj_1

    def evaluate(self):
        poses_xy = [obj.pose.p[..., :2] for obj in self.objects]
        all_separated = torch.ones(self.num_envs, dtype=torch.bool, device=self.device)

        for i in range(4):
            for j in range(i + 1, 4):
                dist = torch.linalg.norm(poses_xy[i] - poses_xy[j], axis=1)
                all_separated = all_separated & (dist > 0.20)

        return {
            "success": all_separated,
        }

    def _get_obs_extra(self, info: dict):
        obs = dict(
            tcp_pose=self.agent.tcp_pose.raw_pose,
        )
        if self.obs_mode_struct.use_state:
            obs.update(
                goal_pos=self.goal_region.pose.p,
                obj_1_pose=self.obj_1.pose.raw_pose,
                obj_2_pose=self.obj_2.pose.raw_pose,
                obj_3_pose=self.obj_3.pose.raw_pose,
                obj_4_pose=self.obj_4.pose.raw_pose,
            )
        return obs

    def compute_dense_reward(self, obs: Any, action: Array, info: dict):
        reward = torch.zeros((self.num_envs,), device=self.device)

        tcp = self.agent.tcp_pos
        obj_positions = torch.stack([obj.pose.p for obj in self.objects], dim=1)
        tcp_expand = tcp.unsqueeze(1)
        distances = torch.linalg.norm(obj_positions - tcp_expand, dim=2)
        nearest_dist = torch.min(distances, dim=1).values
        reward += 1.0 - torch.tanh(5.0 * nearest_dist)

        separation_bonus = torch.zeros_like(reward)
        for i in range(4):
            for j in range(i + 1, 4):
                d = torch.linalg.norm(
                    self.objects[i].pose.p[..., :2] - self.objects[j].pose.p[..., :2],
                    dim=1,
                )
                separation_bonus += (d > 0.12).float()
        reward += 0.1 * separation_bonus

        reward[info["success"]] = 4.0
        return reward

    def compute_normalized_dense_reward(self, obs: Any, action: Array, info: dict):
        return self.compute_dense_reward(obs=obs, action=action, info=info) / 4.0
        return self.compute_dense_reward(obs=obs, action=action, info=info) / max_reward