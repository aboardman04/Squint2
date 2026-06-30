"""
Central configuration file for Visual Sim-to-Real calibration.
Changing values here will automatically update all environment files.
"""

import numpy as np

# ==========================================
# 🎨 COLOR SETTINGS (RGB format: 0.0 to 1.0)
# ==========================================
ROBOT_COLOR = [0.1, 0.1, 0.1, 1.0]     # Black
BLOCK_COLOR = [0.91, 1.0, 0.94]        # Minty-White
TABLE_COLOR = [0.145, 0.624, 0.839, 1.0] # Blue
BOX_COLOR   = [0.996, 0.918, 0.243, 1.0] # Yellow

# ==========================================
# 📷 WRIST CAMERA ALIGNMENT
# ==========================================
WRIST_CAMERA_BASE_POS = (-0.0110, 0.0520, -0.0520)
WRIST_CAMERA_BASE_ROT_RAD = (np.deg2rad(-101.0), np.deg2rad(81.0), np.deg2rad(-31.0))
WRIST_CAMERA_FOV = np.deg2rad(71.0)

# ==========================================
# 📹 OVERHEAD CAMERA ALIGNMENT
# ==========================================
OVERHEAD_CAMERA_BASE_POS = [0.6000, 0.0000, 0.4000]
OVERHEAD_CAMERA_BASE_ROT_RAD = (np.deg2rad(0), np.deg2rad(45), np.deg2rad(180))
OVERHEAD_CAMERA_FOV = np.deg2rad(60.0)

# ==========================================
# ⚙️ WRIST CAMERA HARDWARE V4L2 SETTINGS (/dev/video4)
# ==========================================
V4L2_WRIST_EXPOSURE = 243
V4L2_WRIST_WB = 2800
V4L2_WRIST_BRIGHTNESS = -10
V4L2_WRIST_CONTRAST = 23
V4L2_WRIST_SATURATION = 41

# ==========================================
# ⚙️ OVERHEAD CAMERA HARDWARE V4L2 SETTINGS (/dev/video2)
# ==========================================
V4L2_OVERHEAD_EXPOSURE = 150
V4L2_OVERHEAD_WB = 4600
V4L2_OVERHEAD_BRIGHTNESS = 5
V4L2_OVERHEAD_CONTRAST = 45
V4L2_OVERHEAD_SATURATION = 70
