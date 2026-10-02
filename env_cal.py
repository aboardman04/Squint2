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

ROBOT_COLOR      = rgb(0, 3, 4)          # Black
BLOCK_COLOR      = rgb(252, 255, 255)[:3]   # Minty-White (RGB only)
TABLE_COLOR      = rgb(0, 150, 207)        # Blue
BOX_COLOR        = rgb(254, 234, 62)        # Yellow
INSTRUMENT_COLOR = rgb(150, 180, 165)       # Steel/Gray Base Color for Instruments

# ==========================================
# 📷 WRIST CAMERA ALIGNMENT
# ==========================================
WRIST_CAMERA_BASE_POS = (-0.0100, 0.0520, -0.0520)
WRIST_CAMERA_BASE_ROT_RAD = (np.deg2rad(-102.0), np.deg2rad(80.0), np.deg2rad(-29.0))
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
V4L2_WRIST_EXPOSURE = 94
V4L2_WRIST_WB = 3017
V4L2_WRIST_BRIGHTNESS = -26
V4L2_WRIST_CONTRAST = 64
V4L2_WRIST_SATURATION = 56

# ==========================================
# ⚙️ OVERHEAD CAMERA HARDWARE V4L2 SETTINGS (/dev/video2)
# ==========================================
V4L2_OVERHEAD_EXPOSURE = 150
V4L2_OVERHEAD_WB = 4600
V4L2_OVERHEAD_BRIGHTNESS = 5
V4L2_OVERHEAD_CONTRAST = 45
V4L2_OVERHEAD_SATURATION = 70