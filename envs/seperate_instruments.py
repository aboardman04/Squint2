from typing import Any, Union

import numpy as np
import sapien
import torch
import torch.random
from transforms3d.euler import euler2quat

# Core ManiSkill imports
from mani_skill.agents.robots import Fetch, Panda
from mani_skill.envs.sapien_env import BaseEnv
from mani_skill.sensors.camera import CameraConfig
from mani_skill.utils import common, sapien_utils
from mani_skill.utils.building import actor_builder
from mani_skill.utils.building import actors
from mani_skill.utils.registration import register_env
from mani_skill.utils.scene_builder.table import TableSceneBuilder
from mani_skill.utils.structs import Pose
from mani_skill.utils.structs.types import Array, GPUMemoryConfig, SimConfig

from .robot.so101 import SO101


@register_env("SeparateInstruments-v1", max_episode_steps=50)
class SeparateInstrumentsEnv(BaseEnv):

    SUPPORTED_ROBOTS = ["so101", "panda", "fetch"]

    agent: Union[SO101, Panda, Fetch]

    goal_radius = 0.1
    instrument_half_size = 0.075

    def __init__(self, *args, robot_uids="so101", robot_init_qpos_noise=0.02, **kwargs):
        self.robot_init_qpos_noise = robot_init_qpos_noise
        super().__init__(*args, robot_uids=robot_uids, **kwargs)

    @property
    def _default_sim_config(self):
        return SimConfig(
            gpu_memory_config=GPUMemoryConfig(
                found_lost_pairs_capacity=2**25, max_rigid_patch_count=2**18
            )
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
            "render_camera", pose=pose, width=512, height=512, fov=1, near=0.01, far=100
        )

    def _load_agent(self, options: dict):
        super()._load_agent(options, sapien.Pose(p=[-0.35, 0, 0]))

    def _load_scene(self, options: dict):
        self.table_scene = TableSceneBuilder(
            env=self, robot_init_qpos_noise=self.robot_init_qpos_noise
        )
        self.table_scene.build()
        
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

    def _initialize_episode(self, env_idx: torch.Tensor, options: dict):
        with torch.device(self.device):
            b = len(env_idx)
            self.table_scene.initialize(env_idx)

            # Randomize positions for Forceps 1
            xyz_1 = torch.zeros((b, 3))
            xyz_1[..., 0] = torch.rand(b) * 0.15 - 0.15  
            xyz_1[..., 1] = torch.rand(b) * 0.2 - 0.1
            xyz_1[..., 2] = 0.005 
            
            # Randomize positions for Forceps 2
            xyz_2 = torch.zeros((b, 3))
            xyz_2[..., 0] = torch.rand(b) * 0.15        
            xyz_2[..., 1] = torch.rand(b) * 0.2 - 0.1
            xyz_2[..., 2] = 0.005

            q = [1, 0, 0, 0]
            
            self.obj_1.set_pose(Pose.create_from_pq(p=xyz_1, q=q))
            self.obj_2.set_pose(Pose.create_from_pq(p=xyz_2, q=q))

    def evaluate(self):
        distance_between_objs = torch.linalg.norm(
            self.obj_1.pose.p[..., :2] - self.obj_2.pose.p[..., :2], axis=1
        )
        is_separated = distance_between_objs > 0.20
        return {
            "success": is_separated,
        }

    def _get_obs_extra(self, info: dict):
        # FIX: Adjusted to match your working local SO101 structural properties
        obs = dict(
            qpos=self.agent.robot.get_qpos(),
            tcp_pose=self.agent.tcp_pose.raw_pose,
        )
        if self.obs_mode_struct.use_state:
            obs.update(
                obj_1_pose=self.obj_1.pose.raw_pose,
                obj_2_pose=self.obj_2.pose.raw_pose, 
            )
        return obs

    def compute_dense_reward(self, obs: Any, action: Array, info: dict):
        # 1. Reaching Reward: Use self.agent.tcp_pos for spatial distance logic
        midpoint_objs = (self.obj_1.pose.p + self.obj_2.pose.p) / 2.0
        tcp_to_objs_dist = torch.linalg.norm(midpoint_objs - self.agent.tcp_pos, axis=1)
        reaching_reward = 1.0 - torch.tanh(5.0 * tcp_to_objs_dist)
        reward = reaching_reward

        # 2. Separation Reward
        distance_between_objs = torch.linalg.norm(
            self.obj_1.pose.p[..., :2] - self.obj_2.pose.p[..., :2], axis=1
        )
        
        target_separation = 0.25
        separation_reward = torch.clamp(distance_between_objs / target_separation, max=1.0)
        
        arm_is_close = tcp_to_objs_dist < 0.15
        reward += separation_reward * arm_is_close

        # 3. Success Bonus
        reward[info["success"]] = 4.0
        return reward

    def compute_normalized_dense_reward(self, obs: Any, action: Array, info: dict):
        max_reward = 4.0
        return self.compute_dense_reward(obs=obs, action=action, info=info) / max_reward