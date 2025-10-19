import torch


# https://en.wikipedia.org/wiki/Rotation_matrix#Basic_3D_rotations
def rot_X(theta, dtype=torch.float32, device="cpu"):
    """ Rotation matrix around X-axis """
    c = torch.cos(theta)
    s = torch.sin(theta)
    return torch.tensor([
        [1, 0,  0, 0],
        [0, c, -s, 0],
        [0, s,  c, 0],
        [0, 0,  0, 1]
    ], dtype=dtype, device=device)


def rot_Y(theta, dtype=torch.float32, device="cpu"):
    """ Rotation matrix around Y-axis """
    c = torch.cos(theta)
    s = torch.sin(theta)
    return torch.tensor([
        [ c, 0, s, 0],
        [ 0, 1, 0, 0],
        [-s, 0, c, 0],
        [ 0, 0, 0, 1]
    ], dtype=dtype, device=device)


def rot_Z(theta, dtype=torch.float32, device="cpu"):
    """ Rotation matrix around Z-axis """
    c = torch.cos(theta)
    s = torch.sin(theta)
    return torch.tensor([
        [ c, -s, 0, 0],
        [ s,  c, 0, 0],
        [ 0,  0, 1, 0],
        [ 0,  0, 0, 1]
    ], dtype=dtype, device=device)


def transl(x=0., y=0., z=0., dtype=torch.float32, device="cpu"):
    """ Translation """
    return torch.tensor([
        [ 1, 0, 0, x],
        [ 0, 1, 0, y],
        [ 0, 0, 1, z],
        [ 0, 0, 0, 1]
    ], dtype=dtype, device=device)


def orbit_poses_about_vert(poses: torch.Tensor, angles_X: torch.Tensor, angles_Y: torch.Tensor, dists_Z: torch.Tensor):
    """
    Rotate a pose around the Y axis (-: left; +: right) and X axis (-: up; +: down) 
    located at a point at a given distance in Z direction.
    """
    dtype = poses.dtype
    device = poses.device
    
    T = []
    for angle_x, angle_y, dist_z in zip(angles_X, angles_Y, dists_Z):
        R_x = rot_X(torch.deg2rad(angle_x), dtype, device)
        R_y = rot_Y(torch.deg2rad(angle_y), dtype, device)
        t_to = transl(z=-dist_z, dtype=dtype, device=device)
        t_back = transl(z=dist_z, dtype=dtype, device=device)
        T.append(t_back @ R_x @ R_y @ t_to)
        
    T = torch.stack(T)      # [B, 4, 4]
    rotated_poses = T @ poses   # [..., B, ..., 4, 4]

    return rotated_poses   


def orbit_poses_about(poses: torch.Tensor, c2ref: torch.Tensor, angles_X: torch.Tensor, angles_Y: torch.Tensor):
    """Rotate a pose matrix about the given rotation reference frame using the provided X and Y angles."""
    dtype = poses.dtype
    device = poses.device
    
    T = []
    for angle_x, angle_y in zip(angles_X, angles_Y):
        R_x = rot_X(torch.deg2rad(angle_x), dtype=dtype, device=device)  # [B*NV, 4, 4]
        R_y = rot_Y(torch.deg2rad(angle_y), dtype=dtype, device=device)  # [B*NV, 4, 4]
        T.append(R_x @ R_y)
    T = torch.stack(T)
    
    ref2c = torch.inverse(c2ref.float()).to(dtype)
    # Transform the poses to the reference frame, apply the rotations and transform back.
    rotated_poses = ref2c @ T @ c2ref @ poses
    
    return rotated_poses
    

def shift_poses(poses: torch.Tensor, shifts_X: torch.Tensor, shifts_Y: torch.Tensor, shifts_Z: torch.Tensor):
    """Modifies the positions of the input poses."""
    T = []
    for x, y, z in zip(shifts_X, shifts_Y, shifts_Z):
        T.append(transl(x, y, z, dtype=poses.dtype, device=poses.device))
        
    T = torch.stack(T)
    rotated_poses = T @ poses   # [..., B, ..., 4, 4]

    return rotated_poses   


def orientate_poses(poses: torch.Tensor, angles_X: torch.Tensor, angles_Y: torch.Tensor, angles_Z: torch.Tensor):
    """Rotates the rotation part of the poses matrices. Orientates the poses without changing positions."""
    dtype = poses.dtype
    device = poses.device
    
    R = []
    for angle_x, angle_y, angle_z in zip(angles_X, angles_Y, angles_Z):
        R_x = rot_X(torch.deg2rad(angle_x), dtype, device)[:3, :3]
        R_y = rot_Y(torch.deg2rad(angle_y), dtype, device)[:3, :3]
        R_z = rot_Z(torch.deg2rad(angle_z), dtype, device)[:3, :3]
        R.append(R_x @ R_y @ R_z)
        
    R = torch.stack(R)      # [B, 4, 4]
    rotated_poses = poses.clone()
    rotated_poses[..., :3, :3] = R @ poses[..., :3, :3]   # [..., B, ..., 4, 4]

    return rotated_poses   


def rodrigues_rotation_matrix(axis: torch.Tensor, angle: torch.Tensor) -> torch.Tensor:
    """Compute the rotation matrix from axis and angle using Rodrigues' rotation formula."""
    axis = axis / torch.linalg.norm(axis)
    K = torch.tensor([
        [0, -axis[2], axis[1]],
        [axis[2], 0, -axis[0]],
        [-axis[1], axis[0], 0]
    ])
    I = torch.eye(3)
    R = I + torch.sin(angle) * K + (1 - torch.cos(angle)) * (K @ K)
    return R