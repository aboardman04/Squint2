"""
Central configuration file for Visual Sim-to-Real calibration.
Changing values here will automatically update all environment files.
"""

import numpy as np

# ==========================================
# 🎨 COLOR SETTINGS (RGB format: 0 to 255)
# ==========================================
def rgb(r, g, b, a=255):
    return [r/255.0, g/255.0, b/255.0, a/255.0]

ROBOT_COLOR = rgb(25, 25, 25)          # Black
BLOCK_COLOR = rgb(232, 255, 240)[:3]   # Minty-White (RGB only)
TABLE_COLOR = rgb(20, 170, 217)        # Blue
BOX_COLOR   = rgb(254, 234, 62)        # Yellow

# ==========================================
# 📷 WRIST CAMERA ALIGNMENT
# ==========================================
WRIST_CAMERA_BASE_POS = (-0.0130, 0.0520, -0.0520)
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
V4L2_WRIST_EXPOSURE = 143
V4L2_WRIST_WB = 2800
V4L2_WRIST_BRIGHTNESS = -15
V4L2_WRIST_CONTRAST = 20
V4L2_WRIST_SATURATION = 46

# ==========================================
# ⚙️ OVERHEAD CAMERA HARDWARE V4L2 SETTINGS (/dev/video2)
# ==========================================
V4L2_OVERHEAD_EXPOSURE = 150
V4L2_OVERHEAD_WB = 4600
V4L2_OVERHEAD_BRIGHTNESS = 5
V4L2_OVERHEAD_CONTRAST = 45
V4L2_OVERHEAD_SATURATION = 70
