"""
Live wrist camera tuning for sim2real alignment (SO101).

Side-by-side view: Real | Sim | Blended overlay.
Trackbars adjust wrist camera pose (x, y, z, roll, pitch, yaw) and FOV.
Keys: p=print params, r=rest pose, s=start pose (sim+real), f=apply FOV, q=quit.
"""

import sys
import os
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
os.environ["MKL_SERVICE_FORCE_INTEL"] = "1"

import warnings
warnings.filterwarnings("ignore", category=DeprecationWarning)

import signal
import atexit
import argparse

import cv2
import numpy as np
import torch
import gymnasium as gym
import sapien
from transforms3d.euler import euler2quat
from transforms3d.quaternions import qmult

from mani_skill.utils.wrappers.flatten import FlattenRGBDObservationWrapper
from mani_skill.utils.structs import Pose

from deploy_utils.manipulator import LeRobotRealAgent
from deploy_utils.robot_config import create_real_robot

import envs


class LiveTwoCameraTuner:
    def __init__(self, env_id: str, sim_width: int = 480, sim_height: int = 480):
        self.env_id = env_id
        self.sim_width = sim_width
        self.sim_height = sim_height

        # Overhead Camera pose defaults (overwritten by sim extraction)
        self.cam_x = self.cam_y = self.cam_z = 0.0
        self.cam_roll = self.cam_pitch = self.cam_yaw = 0.0
        self.cam_fov = 60.0
        self._last_fov = self.cam_fov
        self._fov_pending = False

        # Trackbar scaling
        self.pos_scale = 1000  # mm

        self.sim_env = None
        self.real_robot = None
        self.real_agent = None

        self._create_sim_env()
        self._setup_real_robot()
        self._move_real_to_sim_pose()
        self._setup_exit()
        self._setup_ui()
        self._setup_v4l2_trackbars()

    def _setup_v4l2_trackbars(self):
        self.v4l2_win = "Overhead Camera V4L2 Controls"
        cv2.namedWindow(self.v4l2_win, cv2.WINDOW_NORMAL)
        
        # Overhead camera default device
        self.v4l2_device = "/dev/video2"
        
        # Initialize standard values
        self.v4l2_exposure = 150
        self.v4l2_wb = 4600
        self.v4l2_brightness = 5
        self.v4l2_contrast = 45
        self.v4l2_saturation = 70
        
        # Ensure auto-modes are turned off first
        subprocess.run(f"v4l2-ctl -d {self.v4l2_device} --set-ctrl=auto_exposure=1", shell=True, stderr=subprocess.DEVNULL)
        subprocess.run(f"v4l2-ctl -d {self.v4l2_device} --set-ctrl=white_balance_automatic=0", shell=True, stderr=subprocess.DEVNULL)
        
        def set_exposure(val):
            val = max(1, val)  # min 1
            self.v4l2_exposure = val
            subprocess.run(f"v4l2-ctl -d {self.v4l2_device} --set-ctrl=exposure_time_absolute={val}", shell=True, stderr=subprocess.DEVNULL)
            
        def set_wb(val):
            val = max(2800, val) # min 2800
            self.v4l2_wb = val
            subprocess.run(f"v4l2-ctl -d {self.v4l2_device} --set-ctrl=white_balance_temperature={val}", shell=True, stderr=subprocess.DEVNULL)
            
        def set_brightness(val):
            val = val - 64 # Shift range from [0, 128] to [-64, 64]
            self.v4l2_brightness = val
            subprocess.run(f"v4l2-ctl -d {self.v4l2_device} --set-ctrl=brightness={val}", shell=True, stderr=subprocess.DEVNULL)
            
        def set_contrast(val):
            self.v4l2_contrast = val
            subprocess.run(f"v4l2-ctl -d {self.v4l2_device} --set-ctrl=contrast={val}", shell=True, stderr=subprocess.DEVNULL)
            
        def set_saturation(val):
            self.v4l2_saturation = val
            subprocess.run(f"v4l2-ctl -d {self.v4l2_device} --set-ctrl=saturation={val}", shell=True, stderr=subprocess.DEVNULL)
            
        cv2.createTrackbar("Exposure", self.v4l2_win, self.v4l2_exposure, 1000, set_exposure)
        cv2.createTrackbar("White Bal", self.v4l2_win, self.v4l2_wb, 6500, set_wb)
        cv2.createTrackbar("Brightness", self.v4l2_win, self.v4l2_brightness + 64, 128, set_brightness)
        cv2.createTrackbar("Contrast", self.v4l2_win, self.v4l2_contrast, 64, set_contrast)
        cv2.createTrackbar("Saturation", self.v4l2_win, self.v4l2_saturation, 128, set_saturation)

    # --- Sim environment ---

    def _create_sim_env(self, preserve_fov=False):
        desired_fov = self.cam_fov if preserve_fov else None
        if self.sim_env is not None:
            self.sim_env.close()

        sensor_configs = {"width": self.sim_width, "height": self.sim_height}
        if preserve_fov and desired_fov is not None:
            sensor_configs["fov"] = np.radians(desired_fov)

        self.sim_env = gym.make(
            self.env_id,
            obs_mode="rgb+segmentation",
            render_mode="sensors",
            num_envs=1,
            domain_randomization=False,
            domain_randomization_config={"initial_qpos_noise_scale": 0.0},
            sensor_configs=sensor_configs,
        )
        self.sim_env = FlattenRGBDObservationWrapper(self.sim_env, rgb=True, depth=False, state=True)
        self.sim_env.reset(seed=0)
        self._extract_camera_params()

        if preserve_fov and desired_fov is not None:
            self.cam_fov = desired_fov
        self._last_fov = self.cam_fov

    def _extract_camera_params(self):
        """Extract overhead camera params from the sim environment."""
        env = self.sim_env.unwrapped
        
        if hasattr(env, "OVERHEAD_CAMERA_BASE_POS") and hasattr(env, "OVERHEAD_CAMERA_BASE_ROT_RAD"):
            pos = env.OVERHEAD_CAMERA_BASE_POS
            rot = env.OVERHEAD_CAMERA_BASE_ROT_RAD
            self.cam_x, self.cam_y, self.cam_z = float(pos[0]), float(pos[1]), float(pos[2])
            self.cam_roll = float(np.degrees(rot[0]))
            self.cam_pitch = float(np.degrees(rot[1]))
            self.cam_yaw = float(np.degrees(rot[2]))
            if hasattr(env, "OVERHEAD_CAMERA_FOV"):
                self.cam_fov = float(np.degrees(env.OVERHEAD_CAMERA_FOV))
            return

    # --- Real robot ---

    def _setup_real_robot(self):
        self.real_robot = create_real_robot()
        self.real_robot.connect()
        self.real_agent = LeRobotRealAgent(self.real_robot)

    def _move_real_to_sim_pose(self):
        if self.real_agent is None or self.sim_env is None:
            return
        qpos = self.sim_env.unwrapped.agent.robot.get_qpos()
        if hasattr(qpos, "cpu"):
            qpos = qpos.cpu()
        if isinstance(qpos, torch.Tensor):
            qpos = qpos.squeeze()
        self.real_agent.reset(qpos)

    # --- Camera update ---

    def _get_camera_pose(self):
        r, p, y = np.radians(self.cam_roll), np.radians(self.cam_pitch), np.radians(self.cam_yaw)
        q = qmult(euler2quat(0, p, y, axes="rxyz"), euler2quat(r, 0, 0, axes="rxyz"))
        return sapien.Pose(p=[self.cam_x, self.cam_y, self.cam_z], q=q)

    def _update_camera(self):
        env = self.sim_env.unwrapped
        global_pose = self._get_camera_pose()

        # Update the overhead camera directly
        for name, cam in env._sensors.items():
            if name == "overhead_camera":
                cam.camera.local_pose = global_pose
                break

    # --- Image capture ---

    def _get_real_image(self):
        self.real_agent.capture_sensor_data()
        obs = self.real_agent.get_sensor_data()
        
        cam_key = None
        for key in ["overhead_camera", "base_camera", "camera2"]:
            if key in obs and "rgb" in obs[key]:
                cam_key = key
                break
                
        if cam_key is None:
            return None
            
        rgb = obs[cam_key]["rgb"]
        if hasattr(rgb, "cpu"):
            rgb = rgb.cpu().numpy()
        if rgb.ndim == 4:
            rgb = rgb[0]

        # Center-crop to square
        h, w = rgb.shape[:2]
        if h != w:
            s = min(h, w)
            c = (max(h, w) - s) // 2
            rgb = rgb[c : c + s, :, :] if h > w else rgb[:, c : c + s, :]

        rgb = cv2.resize(rgb, (self.sim_width, self.sim_height))
        return cv2.cvtColor(rgb, cv2.COLOR_RGB2BGR)

    def _get_sim_image(self):
        obs = self.sim_env.unwrapped.get_obs()
        if "sensor_data" in obs and "overhead_camera" in obs["sensor_data"]:
            rgb = obs["sensor_data"]["overhead_camera"]["rgb"][0].cpu().numpy()
            return cv2.cvtColor(rgb, cv2.COLOR_RGB2BGR)
        rendered = self.sim_env.render()
        arr = rendered.cpu().numpy() if not isinstance(rendered, np.ndarray) else rendered
        return cv2.cvtColor(arr, cv2.COLOR_RGB2BGR)

    def _make_comparison(self, real, sim):
        if real is None or sim is None:
            return None
        h, w = real.shape[:2]
        sim_r = cv2.resize(sim, (w, h))
        blended = cv2.addWeighted(real, 0.5, sim_r, 0.5, 0)
        comp = np.hstack([real, sim_r, blended])

        font = cv2.FONT_HERSHEY_SIMPLEX
        font_scale = 1.5
        thickness = 3
        white = (255, 255, 255)
        black = (0, 0, 0)
        y_pos = 50

        # Draw text with black outline for visibility
        for text, x_offset in [("Real", 10), ("Sim", w + 10), ("Blended", 2 * w + 10)]:
            cv2.putText(comp, text, (x_offset, y_pos), font, font_scale, black, thickness + 2)
            cv2.putText(comp, text, (x_offset, y_pos), font, font_scale, white, thickness)

        # Camera params at bottom
        params = (f"pos=[{self.cam_x:.3f},{self.cam_y:.3f},{self.cam_z:.3f}] "
                  f"rot=[{self.cam_roll:.0f},{self.cam_pitch:.0f},{self.cam_yaw:.0f}] "
                  f"fov={self.cam_fov:.0f}")
        cv2.putText(comp, params, (10, comp.shape[0] - 15), font, 0.7, black, 3)
        cv2.putText(comp, params, (10, comp.shape[0] - 15), font, 0.7, white, 2)
        return comp

    # --- UI ---

    def _setup_ui(self):
        self.win = "Live Overhead Camera Tuner | p:print r:rest s:start f:FOV q:quit"
        cv2.namedWindow(self.win, cv2.WINDOW_NORMAL)
        # Expanded ranges for overhead camera (pos_scale is 1000, meaning 1 unit = 1mm)
        # Let's map sliders directly to 0-2000 mm (0 to +2.0m) or (-1.0 to 1.0m)
        # Overhead camera usually between -1.0 to +1.0 for X and Y, and 0 to 1.0 for Z
        def set_x(v): self.cam_x = (v - 1000) / self.pos_scale
        def set_y(v): self.cam_y = (v - 1000) / self.pos_scale
        def set_z(v): self.cam_z = v / self.pos_scale
        
        cv2.createTrackbar("X (mm)", self.win, int(self.cam_x * self.pos_scale) + 1000, 2000, set_x)
        cv2.createTrackbar("Y (mm)", self.win, int(self.cam_y * self.pos_scale) + 1000, 2000, set_y)
        cv2.createTrackbar("Z (mm)", self.win, int(self.cam_z * self.pos_scale), 2000, set_z)
        cv2.createTrackbar("Roll", self.win, int(self.cam_roll + 180), 360, lambda v: setattr(self, "cam_roll", v - 180))
        cv2.createTrackbar("Pitch", self.win, int(self.cam_pitch + 180), 360, lambda v: setattr(self, "cam_pitch", v - 180))
        cv2.createTrackbar("Yaw", self.win, int(self.cam_yaw + 180), 360, lambda v: setattr(self, "cam_yaw", v - 180))
        cv2.createTrackbar("FOV", self.win, int(self.cam_fov), 120, self._on_fov)

    def _on_fov(self, val):
        new = max(10, val)
        if new != self.cam_fov:
            self.cam_fov = new
            self._fov_pending = True

    def _setup_exit(self):
        def cleanup(sig=None, frame=None):
            try:
                if self.real_agent and self.sim_env:
                    self.real_agent.reset(self.sim_env.unwrapped.agent.keyframes["rest"].qpos)
            except Exception:
                pass
            try:
                self.real_robot and self.real_robot.disconnect()
            except Exception:
                pass
            try:
                self.sim_env and self.sim_env.close()
            except Exception:
                pass
            if sig is not None:
                sys.exit(0)

        signal.signal(signal.SIGINT, cleanup)
        atexit.register(cleanup)
        self._cleanup = cleanup

    def print_params(self):
        print(f"\n{'='*60}")
        print("Overhead camera params for TwoCameraEnv (two_camera_base_random_env.py):")
        print(f"  OVERHEAD_CAMERA_BASE_POS = [{self.cam_x:.4f}, {self.cam_y:.4f}, {self.cam_z:.4f}]")
        print(f"  OVERHEAD_CAMERA_BASE_ROT_RAD = (np.deg2rad({self.cam_roll:.1f}), np.deg2rad({self.cam_pitch:.1f}), np.deg2rad({self.cam_yaw:.1f}))")
        print(f"  OVERHEAD_CAMERA_FOV = np.deg2rad({self.cam_fov:.1f})")
        print(f"\nHardware Camera settings (v4l2-ctl for {self.v4l2_device}):")
        print(f"  Exposure: {self.v4l2_exposure}")
        print(f"  White Balance: {self.v4l2_wb}")
        print(f"  Brightness: {self.v4l2_brightness}")
        print(f"  Contrast: {self.v4l2_contrast}")
        print(f"  Saturation: {self.v4l2_saturation}")
        print(f"{'='*60}\n")

    def run(self):
        print("\nControls:")
        print("  p  - Print current camera parameters")
        print("  r  - Move sim+real to rest pose")
        print("  s  - Move sim+real to start pose")
        print("  f  - Apply pending FOV change")
        print("  q  - Quit")
        print("  Trackbars - Adjust X/Y/Z, Roll/Pitch/Yaw, FOV\n")

        while True:
            self._update_camera()
            comp = self._make_comparison(self._get_real_image(), self._get_sim_image())

            if comp is not None:
                if self.cam_fov != self._last_fov:
                    fov_text = f"FOV: {self._last_fov:.0f}->{self.cam_fov:.0f} (press 'f')"
                    cv2.putText(comp, fov_text, (10, 90), cv2.FONT_HERSHEY_SIMPLEX, 0.9, (0, 0, 0), 4)
                    cv2.putText(comp, fov_text, (10, 90), cv2.FONT_HERSHEY_SIMPLEX, 0.9, (0, 255, 255), 2)
                cv2.imshow(self.win, comp)
            else:
                err = np.zeros((480, 640 * 3, 3), dtype=np.uint8)
                cv2.putText(err, "Waiting for camera...", (700, 240), cv2.FONT_HERSHEY_SIMPLEX, 1.5, (255, 255, 255), 3)
                cv2.imshow(self.win, err)

            key = cv2.waitKey(30) & 0xFF
            if key == ord("q"):
                break
            elif key == ord("p"):
                self.print_params()
            elif key == ord("r"):
                try:
                    rest_qpos = self.sim_env.unwrapped.agent.keyframes["rest"].qpos
                    qpos = rest_qpos if isinstance(rest_qpos, torch.Tensor) else torch.tensor(rest_qpos, dtype=torch.float32)
                    if qpos.dim() == 1:
                        qpos = qpos.unsqueeze(0)
                    env = self.sim_env.unwrapped
                    env.agent.robot.set_qpos(qpos)
                    if env.gpu_sim_enabled:
                        env.scene._gpu_apply_all()
                    self.real_agent.reset(rest_qpos)
                    print("Moved sim+real to rest pose")
                except Exception as e:
                    print(f"Rest pose error: {e}")
            elif key == ord("s"):
                try:
                    self.sim_env.reset(seed=0)
                    self._move_real_to_sim_pose()
                    print("Moved sim+real to start pose")
                except Exception as e:
                    print(f"Start pose error: {e}")
            elif key == ord("f") and self._fov_pending:
                self._create_sim_env(preserve_fov=True)
                self._fov_pending = False

        cv2.destroyAllWindows()
        self._cleanup()


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Live wrist camera tuning (SO101)")
    parser.add_argument("--env-id", default="2Camera-SO101LiftCube-v1", help="Sim environment ID")
    parser.add_argument("--sim-width", type=int, default=480)
    parser.add_argument("--sim-height", type=int, default=480)
    args = parser.parse_args()
    LiveCameraTuner(args.env_id, args.sim_width, args.sim_height).run()