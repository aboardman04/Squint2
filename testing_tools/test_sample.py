import torch

def _sample_instrument_poses(b, base_pos, device):
    spawn_box_half_size = 0.1
    xyz_1 = torch.zeros((b, 3), device=device)
    xyz_1[:, :2] = (
        torch.rand((b, 2), device=device) * spawn_box_half_size * 2
        - spawn_box_half_size
    )
    xyz_1[:, :2] += base_pos[:, :2]
    xyz_1[..., 2] = 0.008

    yaw1 = torch.rand(b, device=device) * 2 * torch.pi
    yaw2 = yaw1 + (torch.rand(b, device=device) - 0.5) * 0.1

    q1 = torch.zeros((b, 4), device=device)
    q1[:, 0] = torch.cos(yaw1 / 2)
    q1[:, 3] = torch.sin(yaw1 / 2)

    q2 = torch.zeros((b, 4), device=device)
    q2[:, 0] = torch.cos(yaw2 / 2)
    q2[:, 3] = torch.sin(yaw2 / 2)

    perp_dir = torch.stack([-torch.sin(yaw1), torch.cos(yaw1)], dim=1)
    par_dir = torch.stack([torch.cos(yaw1), torch.sin(yaw1)], dim=1)

    side_dist = (torch.rand(b, device=device) * 0.01 + 0.01) * torch.sign(torch.randn(b, device=device))
    fwd_dist = (torch.rand(b, device=device) - 0.5) * 0.10

    xyz_2 = xyz_1.clone()
    xyz_2[:, :2] += perp_dir * side_dist.unsqueeze(1) + par_dir * fwd_dist.unsqueeze(1)
    xyz_2[..., 2] = 0.008
    
    return xyz_1, q1, xyz_2, q2

b = 2
device = 'cpu'
base_pos = torch.zeros(2, 3)
p1, q1, p2, q2 = _sample_instrument_poses(b, base_pos, device)
print(p1.shape, q1.shape)
