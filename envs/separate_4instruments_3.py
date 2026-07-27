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


@register_env("SeparateInstruments-v6", max_episode_steps=500)
class SeparateInstrumentsEnv(DefaultCameraEnv):
    
    SUPPORTED_ROBOTS = ["so101", "panda", "fetch"]

    agent: Union[SO101, Panda, Fetch]

    goal_radius = 0.1
    radius = 0.02
    instrument_spawn_xy_range = 0.02
    instrument_spawn_z_base = 0.008
    instrument_spawn_z_spacing = 0.007
    num_instruments = 4
    inner_side_half_len = 0.07  # side length of the bin's inner square
    short_side_half_size = 0.0075  # length of the shortest edge of the block
    block_half_size = [
        short_side_half_size,
        2 * short_side_half_size + inner_side_half_len,
        2 * short_side_half_size + inner_side_half_len,
    ]  # The bottom block of the bin, which is larger: The list represents the half length of the block along the [x, y, z] axis respectively.
    edge_block_half_size = [
        short_side_half_size,
        2 * short_side_half_size + inner_side_half_len,
        2 * short_side_half_size,
    ]  # The edge block of the bin, which is smaller. The representations are similar to the above one


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

    # def _build_bin(self, radius):
    #     builder = self.scene.create_actor_builder()

    #     # init the locations of the basic blocks
    #     dx = self.block_half_size[1] - self.block_half_size[0]
    #     dy = self.block_half_size[1] - self.block_half_size[0]
    #     dz = self.edge_block_half_size[2] + self.block_half_size[0]

    #     # build the bin bottom and edge blocks
    #     poses = [
    #         sapien.Pose([0, 0, 0]),
    #         sapien.Pose([-dx, 0, dz]),
    #         sapien.Pose([dx, 0, dz]),
    #         sapien.Pose([0, -dy, dz]),
    #         sapien.Pose([0, dy, dz]),
    #     ]
    #     half_sizes = [
    #         [self.block_half_size[1], self.block_half_size[2], self.block_half_size[0]],
    #         self.edge_block_half_size,
    #         self.edge_block_half_size,
    #         [
    #             self.edge_block_half_size[1],
    #             self.edge_block_half_size[0],
    #             self.edge_block_half_size[2],
    #         ],
    #         [
    #             self.edge_block_half_size[1],
    #             self.edge_block_half_size[0],
    #             self.edge_block_half_size[2],
    #         ],
    #     ]
    #     for pose, half_size in zip(poses, half_sizes):
    #         builder.add_box_collision(pose, half_size)
    #         builder.add_box_visual(pose, half_size)

    #     # build the kinematic bin
    #     return builder.build_kinematic(name="bin")


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
            # spawn relative to the provided base_pos z so we can place instruments
            # inside the bin (base_pos should represent a useful reference z)
            xyz[:, 2] = base_pos[:, 2] + self.instrument_spawn_z_base + i * self.instrument_spawn_z_spacing

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

        # Create a thin blue mat on the table surface
        blue_material = sapien.render.RenderMaterial(
            base_color=[0.1, 0.2, 0.85, 1.0], roughness=0.6, metallic=0.0
        )
        patch_half_size = [0.20, 0.375, 0.002]
        table_z = (
            self.table_scene.table.pose.p[2]
            if hasattr(self.table_scene, "table")
            else float(self.block_half_size[2])
        )
        builder = self.scene.create_actor_builder()
        builder.add_box_visual(half_size=patch_half_size, material=blue_material)
        builder.initial_pose = sapien.Pose(
            p=[0.20, 0.225, table_z + patch_half_size[2]], q=[1, 0, 0, 0]
        )
        self.table_mat = builder.build_kinematic("table_mat")

        # Create camera mount actors expected by the base randomization helpers.
        builder = self.scene.create_actor_builder()
        builder.initial_pose = sapien.Pose()
        self.camera_mount = builder.build_kinematic("camera_mount")

        builder = self.scene.create_actor_builder()
        builder.initial_pose = sapien.Pose()
        self.wrist_camera_mount = builder.build_kinematic("wrist_camera_mount")

        #self.bin = self._build_bin(self.radius)
        bin_path = "/home/aboardman/squint2/deploy_utils/blender_objs/box.obj"
        # compute a quaternion so the box's open side faces up and the long side
        # is rotated to be perpendicular to the robot-forward direction.
        # Tweak these Euler angles if the model's axes differ.
        # (roll, pitch, yaw) where yaw=pi/2 aligns the long axis along +Y.
        bin_q = euler2quat(np.pi / 2, 0.0, np.pi / 2)
        # build the bin as a kinematic actor so it won't be pushed or fly
        bin_steel_material = sapien.render.RenderMaterial(
            base_color=[0.831, 0.827, 0.800, 1.0], roughness=0.15, metallic=1.0
        )
        physx_material = sapien.physx.PhysxMaterial(
            static_friction=0.6, dynamic_friction=0.5, restitution=0.1
        )
        builder = self.scene.create_actor_builder()
        builder.add_visual_from_file(filename=bin_path, material=bin_steel_material)
        builder.add_multiple_convex_collisions_from_file(
            filename=bin_path, decomposition="coacd", material=physx_material
        )
        # set initial pose so the bin sits on the table (center z = half-height)
        # place initial builder pose slightly lower so the bin base rests on table
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

    def _initialize_episode(self, env_idx: torch.Tensor, options: dict):
        with torch.device(self.device):
            b = len(env_idx)
            self.table_scene.initialize(env_idx)

            center = self.agent.robot.pose.p + torch.tensor([0.3, 0.0, 0.0], device=self.device)
            if center.ndim == 1:
                center = center.unsqueeze(0)
            if center.shape[0] == 1:
                center = center.expand(len(env_idx), -1)
            else:
                center = center[env_idx]
            # place the bin first so instruments can be spawned relative to it
            bin_pos = center.clone()
            # lower the bin slightly so its bottom should touch the table
            bin_pos[:, 2] = float(self.block_half_size[2]) - 0.03
            # create a per-env quaternion tensor matching the initial rotation
            bin_q = euler2quat(np.pi / 2, 0.0, np.pi / 2)
            q_tensor = torch.tensor(bin_q, device=self.device, dtype=bin_pos.dtype)
            q_tensor = q_tensor.unsqueeze(0).repeat(b, 1)
            bin_pose = Pose.create_from_pq(p=bin_pos, q=q_tensor)
            self.bin.set_pose(bin_pose)

            # spawn instruments slightly above the bin center so they rest inside
            spawn_base = bin_pos.clone()
            spawn_base[:, 2] += 0.03
            p1, q1, p2, q2, p3, q3, p4, q4 = self._sample_instrument_poses(b, spawn_base)
            self.obj_1.set_pose(Pose.create_from_pq(p=p1, q=q1))
            self.obj_2.set_pose(Pose.create_from_pq(p=p2, q=q2))
            self.obj_3.set_pose(Pose.create_from_pq(p=p3, q=q3))
            self.obj_4.set_pose(Pose.create_from_pq(p=p4, q=q4))
            self.obj = self.obj_1

    # def _get_obs_extra(self, info: dict):
    #     obs = dict(
    #         tcp_pose=self.agent.tcp_pose.raw_pose,
    #     )
    #     if self.obs_mode_struct.use_state:
    #         obs.update(
    #             goal_pos=self.goal_region.pose.p,
    #             obj_1_pose=self.obj_1.pose.raw_pose,
    #             obj_2_pose=self.obj_2.pose.raw_pose,
    #             obj_3_pose=self.obj_3.pose.raw_pose,
    #             obj_4_pose=self.obj_4.pose.raw_pose,
    #         )
    #     return obs

    def evaluate(self):
        # Extract XY positions for all instruments
        poses_xy = [obj.pose.p[..., :2] for obj in self.objects]
        poses_z = [obj.pose.p[..., 2] for obj in self.objects]
        
        # 1. Check pairwise instrument separation (distance > 0.15m)
        all_separated = torch.ones(self.num_envs, dtype=torch.bool, device=self.device)
        for i in range(self.num_instruments):
            for j in range(i + 1, self.num_instruments):
                dist = torch.linalg.norm(poses_xy[i] - poses_xy[j], dim=1)
                all_separated = all_separated & (dist > 0.15)

        # 2. Check if all instruments are outside the bin bounds
        # (Assuming the bin center is near self.bin.pose.p)
        bin_xy = self.bin.pose.p[..., :2]
        bin_half_size = self.block_half_size[1]  # roughly half width of the bin
        
        all_outside_bin = torch.ones(self.num_envs, dtype=torch.bool, device=self.device)
        all_on_table = torch.ones(self.num_envs, dtype=torch.bool, device=self.device)

        for obj in self.objects:
            obj_xy = obj.pose.p[..., :2]
            obj_z = obj.pose.p[..., 2]
            
            # Distance from bin center in XY plane
            dist_to_bin = torch.linalg.norm(obj_xy - bin_xy, dim=1)
            outside_bin = dist_to_bin > (bin_half_size + 0.05)
            
            # On table check (z position close to table height ~0.02m)
            on_table = (obj_z < 0.04) & (obj_z > -0.01)

            all_outside_bin = all_outside_bin & outside_bin
            all_on_table = all_on_table & on_table

        # Success condition: all instruments outside, on the table, and not touching each other
        success = all_separated & all_outside_bin & all_on_table

        return {
            "success": success,
            "all_separated": all_separated,
            "all_outside_bin": all_outside_bin,
            "all_on_table": all_on_table,
        }

    def compute_dense_reward(self, obs: Any, action: Array, info: dict):
        reward = torch.zeros((self.num_envs,), device=self.device)
        
        tcp_pos = self.agent.tcp_pos
        bin_xy = self.bin.pose.p[..., :2]
        bin_half_size = self.block_half_size[1]

        # -------------------------------------------------------------
        # 1. Picking & Reaching Stage
        # -------------------------------------------------------------
        # Find nearest instrument to the TCP
        obj_positions = torch.stack([obj.pose.p for obj in self.objects], dim=1) # [B, 4, 3]
        tcp_expand = tcp_pos.unsqueeze(1) # [B, 1, 3]
        distances = torch.linalg.norm(obj_positions - tcp_expand, dim=2) # [B, 4]
        nearest_dist = torch.min(distances, dim=1).values
        
        # Continuous reach reward (0 to 1)
        reach_reward = 1.0 - torch.tanh(5.0 * nearest_dist)
        reward += 0.5 * reach_reward

        # Check for active grasping / lifting of any instrument inside the bin
        lift_reward = torch.zeros((self.num_envs,), device=self.device)
        for obj in self.objects:
            obj_z = obj.pose.p[..., 2]
            obj_xy = obj.pose.p[..., :2]
            dist_to_bin = torch.linalg.norm(obj_xy - bin_xy, dim=1)
            
            # If instrument is inside bin and lifted off the bottom
            is_inside_bin = dist_to_bin <= (bin_half_size + 0.05)
            is_lifted = obj_z > 0.05
            lift_reward += (is_inside_bin & is_lifted).float()
            
        reward += 0.5 * lift_reward

        # -------------------------------------------------------------
        # 2. Placement Outside the Box & On the Table
        # -------------------------------------------------------------
        num_placed_successfully = torch.zeros((self.num_envs,), device=self.device)

        for i, obj in enumerate(self.objects):
            obj_xy = obj.pose.p[..., :2]
            obj_z = obj.pose.p[..., 2]
            obj_vel = obj.linear_velocity  # [B, 3]
            
            dist_to_bin = torch.linalg.norm(obj_xy - bin_xy, dim=1)
            is_outside = dist_to_bin > (bin_half_size + 0.05)
            is_near_table = (obj_z < 0.04) & (obj_z > -0.01)

            # Continuous reward for carrying objects further away from bin center
            reward += 0.2 * torch.clamp(dist_to_bin - bin_half_size, min=0.0, max=0.3)

            # Sizable reward per instrument placed outside bin on the table
            is_placed = is_outside & is_near_table
            num_placed_successfully += is_placed.float()

            # ---------------------------------------------------------
            # 3. Soft Placement Reward (Low Velocity near Table)
            # ---------------------------------------------------------
            # When object is near table, reward low z-velocity (placing gently rather than dropping)
            z_speed = torch.abs(obj_vel[..., 2])
            gentle_landing_reward = is_near_table.float() * (1.0 - torch.tanh(3.0 * z_speed))
            reward += 0.3 * gentle_landing_reward

        # Sizable reward per successfully placed instrument
        reward += 2.0 * num_placed_successfully

        # -------------------------------------------------------------
        # 4. Pairwise Non-Touching / Separation Bonus
        # -------------------------------------------------------------
        separation_bonus = torch.zeros_like(reward)
        for i in range(self.num_instruments):
            for j in range(i + 1, self.num_instruments):
                dist = torch.linalg.norm(
                    self.objects[i].pose.p[..., :2] - self.objects[j].pose.p[..., :2],
                    dim=1,
                )
                # Gradual bonus as distance increases beyond 0.15m threshold
                separation_bonus += torch.clamp((dist - 0.05) / 0.10, min=0.0, max=1.0)
        
        reward += 0.5 * separation_bonus

        # -------------------------------------------------------------
        # 5. Excess Force Penalties (Robot & Environment)
        # -------------------------------------------------------------
        max_allowed_force = 20.0  # Newtons threshold
        force_penalty = torch.zeros((self.num_envs,), device=self.device)

        for obj in self.objects:
            # Query net contact force acting on the instrument
            net_force = obj.get_net_contact_forces() # Returns shape [B, 3]
            force_mag = torch.linalg.norm(net_force, dim=-1)
            
            # Apply quadratic penalty for forces exceeding the safe threshold
            excess_force = torch.clamp(force_mag - max_allowed_force, min=0.0)
            force_penalty += 0.01 * (excess_force ** 2)

        reward -= force_penalty

        # -------------------------------------------------------------
        # 6. Task Completion Bonus
        # -------------------------------------------------------------
        reward[info["success"]] += 10.0

        return reward

    def compute_normalized_dense_reward(self, obs: Any, action: Array, info: dict):
        # Max theoretical reward scale per step (~18.0 with task completion)
        return self.compute_dense_reward(obs=obs, action=action, info=info) / 18.0