import sys
import torch
import numpy as np

# We need to find ManiSkill
sys.path.append('.')
from mani_skill.utils.structs.pose import Pose
from transforms3d.euler import euler2quat

table_pose = Pose.create_from_pq(
    p=[-0.12 + 0.737, 0, -0.9196429], q=euler2quat(0, 0, np.pi / 2)
)
print("table_pose.p shape:", table_pose.p.shape)
print("table_pose.p device:", table_pose.p.device)

# env_idx mock
env_idx = torch.tensor([0, 1], device='cuda')
try:
    tray_z = table_pose.p[env_idx, 2] + 0.04
    print("Success:", tray_z)
except Exception as e:
    print("Error:", type(e), e)
