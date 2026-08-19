# I am trying to improve the physics of the instruments in this version, will also add envi visual constomization options

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
    item_density_range: Sequence[float] = (200, 200)
    randomize_item_color: bool = False


@register_env("LiftInstruments-v3", max_episode_steps=50)
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

    SIM_FREQ = 200
    CONTROL_FREQ = 20

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


    def _build_instrument(self, obj_path: str, name: str, initial_pose: sapien.Pose):
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
                contact_offset=0.003,
                rest_offset=0.002,
            )
        except TypeError:
            builder.add_multiple_convex_collisions_from_file(
                filename=obj_path,
                decomposition="coacd",
                material=physx_material,
            )

        # try:
        #     # Switch decomposition to "vhacd" for tighter, more solid collision geometry on complex meshes,
        #     # and increase contact_offset to detect collisions earlier and prevent tunneling.
        #     builder.add_multiple_convex_collisions_from_file(
        #         filename=obj_path, 
        #         # decomposition="vhacd", 
        #         material=physx_material,
        #         contact_offset=0.02,  # Increased from 0.010 to register contacts earlier
        #         rest_offset=0.001
        #     )
        # except TypeError:
        #     # Fallback if specific decomposition keyword arguments differ in your version
        #     builder.add_multiple_convex_collisions_from_file(
        #         filename=obj_path, 
        #         # decomposition="vhacd", 
        #         material=physx_material
        #     )

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

        inst1_path = (
            "/home/aboardman/squint2/deploy_utils/blender_objs/dressing_forceps.obj"
        )
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
        self.objects = [self.obj_1, self.obj_2]
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

            center = self.agent.robot.pose.p + torch.tensor(
                [0.3, 0.0, 0.0], device=self.device
            )
            spawn_base = center[env_idx]
            spawn_base[:, 2] += 0.03

            p1, q1, p2, q2 = self._sample_instrument_poses(b, spawn_base)
            p2[:, 0] = p1[:, 0] + self.instrument_separation
            p2[:, 1] = p1[:, 1]

            self.obj_1.set_pose(Pose.create_from_pq(p=p1, q=q1))
            self.obj_2.set_pose(Pose.create_from_pq(p=p2, q=q2))
            self.target_object = self.obj_1

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
        tcp_to_item_dist = torch.linalg.norm(self.target_object.pose.p - self.agent.tcp_pos, axis=-1)
        tcp_velocity = (self.agent.finger1_tip.linear_velocity + self.agent.finger2_tip.linear_velocity) / 2

        obs.update(
            tcp_pose=self.agent.tcp_pose.raw_pose,
            target_item_pose=self.target_object.pose.raw_pose,
            tcp_to_item_pos=self.target_object.pose.p - self.agent.tcp_pos,
            tcp_to_item_dist=tcp_to_item_dist,
            tcp_velocity=tcp_velocity,
            is_item_grasped=info.get("is_item_grasped", self.agent.is_grasping(self.target_object)),
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
        tcp_to_item_dist = torch.linalg.norm(
            self.target_object.pose.p - self.agent.tcp_pos, axis=-1
        )
        reached_object = tcp_to_item_dist < 0.03
        is_item_grasped = self.agent.is_grasping(self.target_object)

        target_qpos = self.agent.controller._target_qpos.clone()
        distance_to_rest_qpos = torch.linalg.norm(
            target_qpos[:, :-1] - self.rest_qpos[:-1], axis=-1
        )
        reached_rest_qpos = distance_to_rest_qpos < 0.2

        item_lifted = self.target_object.pose.p[..., -1] >= 0.04
        robot_touching_mat = self.agent.is_touching(self.table_mat)

        success = item_lifted & is_item_grasped & reached_rest_qpos

        return {
            "is_item_grasped": is_item_grasped,
            "reached_object": reached_object,
            "distance_to_rest_qpos": distance_to_rest_qpos,
            "robot_touching_mat": robot_touching_mat,
            "item_lifted": item_lifted,
            "success": success,
        }

    def compute_dense_reward(self, obs: Any, action: torch.Tensor, info: dict):
        tcp_to_item_dist = torch.linalg.norm(
            self.target_object.pose.p - self.agent.tcp_pose.p, axis=1
        )
        reaching_reward = 1.0 - torch.tanh(5.0 * tcp_to_item_dist)
        reward = reaching_reward.clone()

        approach_weight = torch.clamp(tcp_to_item_dist / 0.15, 0.0, 1.0)
        tcp_velocity = torch.linalg.norm(
            (
                self.agent.finger1_tip.linear_velocity
                + self.agent.finger2_tip.linear_velocity
            )
            / 2,
            axis=1,
        )
        
        # speed_bonus = approach_weight * torch.min(tcp_vel, torch.tensor(1.0, device=self.device))
        # reward += 0.2 * speed_bonus

        slow_zone = torch.clamp(0.08 - tcp_to_item_dist, 0.0, 0.08) / 0.08
        desired_speed = 0.08
        speed_penalty = (slow_zone * torch.clamp(tcp_velocity - desired_speed, min=0.0,))
        reward -= 0.5 * speed_penalty

        is_grasped = info["is_item_grasped"].float()
        reward += is_grasped

        stable_grasp = (is_grasped * torch.exp(-5.0 * tcp_velocity))
        reward += 0.5 * stable_grasp

        place_reward = torch.exp(-2 * info["distance_to_rest_qpos"])
        reward += place_reward * info["is_item_grasped"]

        action_diff = torch.linalg.norm(action, axis=1)
        smoothness_penalty = action_diff * (1.0 - approach_weight)
        lift_phase_penalty = is_grasped * torch.linalg.norm((self.agent.finger1_tip.linear_velocity + self.agent.finger2_tip.linear_velocity) / 2, axis=1)

        reward -= 0.1 * smoothness_penalty
        reward -= 0.05 * lift_phase_penalty

        reward -= 3.0 * info["robot_touching_mat"].float()
        reward -= 1.0 * (~info["item_lifted"]).float()

        if "success" in info:
            reward[info["success"]] += 15.0

        return reward

    def compute_normalized_dense_reward(self, obs: Any, action: torch.Tensor, info: dict):
        return self.compute_dense_reward(obs=obs, action=action, info=info) / 18.0