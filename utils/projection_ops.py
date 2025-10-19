from typing import Union, Tuple, Optional
import cv2
import numpy as np
import torch
import torch.nn.functional as F
from utils.constants import DEFINITIONS


EPS = 1e-5


def scale_intrinsics(K, scale_width, scale_height, normalized=False):
    """Scales normalized intrinsics to a new aspect ratio.

    Args:
        K (torch.Tensor): intrinsics
        scale_width (float): Ratio `old_width/new_width` for normalized coordinates and `new_width/old_width` otherwise. 
        scale_height (float): Height ratio like above.
    """
    K = K.clone()
    K[..., 0, 0] *= scale_width
    K[..., 1, 1] *= scale_height
    if normalized:
        K[..., 0, 2] = (K[..., 0, 2] * 0.5 + 0.5) * scale_width * 2. - 1.
        K[..., 1, 2] = (K[..., 1, 2] * 0.5 + 0.5) * scale_height * 2. - 1.
    else:
        K[..., 0, 2] = K[..., 0, 2] * scale_width 
        K[..., 1, 2] = K[..., 1, 2] * scale_height 
    return K  


def unnormalize_intrinsics(K, width, height):
    """Maps intrinsics from normalized in [-1, 1] to pixels in [0, H or W]."""
    K = K.clone()
    K[..., 0, 0] = (K[..., 0, 0] * 0.5) * width
    K[..., 1, 1] = (K[..., 1, 1] * 0.5) * height
    K[..., 0, 2] = (K[..., 0, 2] * 0.5 + 0.5) * width
    K[..., 1, 2] = (K[..., 1, 2] * 0.5 + 0.5) * height
    return K


def crop_intrinsics(K, x0, y0, unnormalized_W=None, unnormalized_H=None, new_unnormalized_W=None, new_unnormalized_H=None):
    """Crops normalized or unnormalized intrinsics according to the given horizontal and vertical offsets."""
    K = K.clone()
    cx = K[..., 0, 2]
    cy = K[..., 1, 2]
    
    if unnormalized_H and unnormalized_W:
        # [-1, 1] --> [0, H or W]
        cx = (cx * 0.5 + 0.5) * unnormalized_W
        cy = (cy * 0.5 + 0.5) * unnormalized_H
    
    cx -= x0.view(cx.shape)
    cy -= y0.view(cx.shape)
        
    if unnormalized_H and unnormalized_W:
        # [0, H or W] --> [-1, 1]
        cx = (cx / new_unnormalized_W) * 2. - 1.
        cy = (cy / new_unnormalized_H) * 2. - 1.
        K[..., 0, 0] *= unnormalized_W / new_unnormalized_W
        K[..., 1, 1] *= unnormalized_H / new_unnormalized_H
    
    K[..., 0, 2] = cx
    K[..., 1, 2] = cy
    
    return K


# @torch.cuda.amp.custom_fwd(cast_inputs=torch.float32)
def distance_to_z(depths: torch.Tensor, projs: torch.Tensor):
    """Converts depth-along-camera-rays to depth in Z-direction."""
    n, nv, h, w = depths.shape
    device = depths.device
    dtype = depths.dtype
    # AMP does not automatically upcast the input to FP32.
    # inv_K = torch.inverse(projs)
    inv_K = torch.inverse(projs.float()).to(projs.dtype)

    grid_x = torch.linspace(-1, 1, w, device=device, dtype=dtype).view(1, 1, 1, -1).expand(-1, -1, h, -1)
    grid_y = torch.linspace(-1, 1, h, device=device, dtype=dtype).view(1, 1, -1, 1).expand(-1, -1, -1, w)
    img_points = torch.stack((grid_x, grid_y, torch.ones_like(grid_x)), dim=2).expand(n, nv, -1, -1, -1)
    cam_points = (inv_K @ img_points.view(n, nv, 3, -1)).view(n, nv, 3, h, w)
    factors = cam_points[:, :, 2, :, :] / torch.norm(cam_points, dim=2, dtype=dtype)

    return depths * factors


# @torch.cuda.amp.custom_fwd(cast_inputs=torch.float32)
def unproject_to_world_points(depth, intrinsics, poses, rgb=None):
    """
    Unproject a depth map (and RGB pixels) to a 3D point cloud.
    :param depth: [B, H, W]
    :param intrinsics: [B, 3, 3]
    :param pose: [B, 4, 4]
    :praram rgb: [B, 3, H, W]
    """
    B, H, W = depth.shape

    y, x = torch.meshgrid(torch.linspace(-1, 1, H), torch.linspace(-1, 1, W), indexing="ij")    # [H, W], [H, W] 
    image_coords = torch.stack([x, y, torch.ones_like(x)], dim=-1).to(intrinsics.device)        # [H, W, 3]
    image_coords = image_coords.reshape(1, -1, 3).repeat(B, 1, 1)                               # [B, H*W, 3]
    # Transform to camera coordinates
    K_inv = torch.linalg.inv(intrinsics.float()).to(intrinsics.dtype)
    cam_coords = (K_inv @ image_coords.transpose(1, 2)).transpose(1, 2)
    # Scale the normalized direction vectors with per-pixel depth
    # cam_coords = cam_coords / torch.norm(cam_coords, p=2, dim=-1)[:, :, None] * depth.reshape(B, -1, 1)    # [B, H*W, 3]
    cam_coords = cam_coords * depth.reshape(B, -1, 1)    # [B, H*W, 3]

    # Transform to world coordinates
    homogenous_coords = torch.cat([cam_coords, torch.ones_like(cam_coords[:, :, 0:1])], dim=-1)         # [B, H*W, 4]
    world_coords = (poses @ homogenous_coords.transpose(1, 2)).transpose(1, 2)
    
    if rgb is not None:
        # Per-3D-point RGB color.
        rgb = rgb.clone().permute(0, 2, 3, 1).reshape(B, -1, 3)
    
    return world_coords[:, :, :3], rgb


# Matrix inversion needs fp32 inputs.
# @torch.cuda.amp.custom_fwd(cast_inputs=torch.float32)
def project_world_points(pointcloud, intrinsics, poses):
    """
    Project a point cloud to a list of pixels with depths.
    :param pointcloud: [B, N, 3]
    :param intrinsics: [B, 3, 3]
    :param pose: [B, 4, 4]    
    """
    B, N, _ = pointcloud.shape
    homogeneous_coords = torch.cat([pointcloud, torch.ones_like(pointcloud[:, :, 0:1])], dim=-1)    # [B, N, 4]
    poses_inv = torch.linalg.inv(poses.float()).to(poses.dtype)
    proj_mats = intrinsics @ poses_inv[:, 0:3, :]                                     # [B, 3, 4]
    image_coords = (proj_mats @ homogeneous_coords.transpose(1, 2)).transpose(1, 2)[:, :, :3]
    z = image_coords[:, :, 2]
    xy = (image_coords / z.unsqueeze(-1))[:, :, :2]
    
    camera_origins = poses[:, :3, 3].unsqueeze(1)                       # [B, 1, 3]
    # depths = torch.norm(pointcloud - camera_origins, p=2, dim=-1)       # [B, N]
    
    return xy, z       # [B, N, 3] and [B, N]


def align_depth(depth1: torch.Tensor, depth2: torch.Tensor, valid_mask: Optional[torch.Tensor] = None, **kwargs):
    scales, shifts = comp_align_scale_shift(depth1, depth2, valid_mask, **kwargs)
    # print("scale", scales, "shift", shifts)
    aligned_depth1 = depth1 * scales[:, None, None, None].expand(depth1.shape) + shifts[:, None, None, None].expand(depth1.shape)
    return aligned_depth1, scales, shifts


def align_inv_depth(depth1: torch.Tensor, depth2: torch.Tensor, valid_mask: Optional[torch.Tensor] = None, **kwargs):
    depth1 = 1 / (depth1.abs() + EPS)       # TODO maybe add clone
    depth2 = 1 / (depth2.abs() + EPS)
        
    scales, shifts = comp_align_scale_shift(depth1, depth2, valid_mask, **kwargs)
    # print("scale", scales, "shift", shifts)
    aligned_depth1 = depth1 * scales[:, None, None, None].expand(depth1.shape) + shifts[:, None, None, None].expand(depth1.shape)
    aligned_depth1 = 1 / (aligned_depth1.abs() + EPS)

    return aligned_depth1, scales, shifts


def comp_align_scale_shift(depth1: torch.Tensor, depth2: torch.Tensor, valid_mask: Optional[torch.Tensor] = None, mode="lstsq", scale_only=False, max_depth_to_align=None, min_valid_fraction: float = 0.):
    """Computes scale (and shift) to align depth1 to depth2. Inputs are shape [B, num_pixels] or [B, 1, H, W]."""
    B = depth1.shape[0]
    
    # Ensure no division by zero or overflowing values.
    valid_mask_min = torch.logical_and(depth1 > 0.1, depth2 > 0.1)
    if valid_mask is not None:
        valid_mask = torch.logical_and(valid_mask, valid_mask_min)
    else:
        valid_mask = valid_mask_min
    
    if max_depth_to_align is not None:
        valid_mask_max = torch.logical_and(depth1 < max_depth_to_align, depth2 < max_depth_to_align)
        valid_mask = torch.logical_and(valid_mask, valid_mask_max)
    
    shifts = torch.zeros((B,), dtype=depth1.dtype, device=depth1.device)
    if mode == "lstsq":
        A = depth1[valid_mask].reshape(B, -1, 1)
        if not scale_only:
            A = torch.concat([A, torch.ones_like(A)], dim=-1)       # [B, n_pts, 2]
        B = depth2[valid_mask].reshape(B, -1, 1)                   # [B, n_pts, 1]
        X = torch.linalg.lstsq(A, B).solution                   # [B, 2, 1]
        # Replace NaNs with 0. NaNs occur if A is not full rank.
        X = torch.nan_to_num(X)
        scales = X[:, 0, 0]
        shifts = X[:, 1, 0] if not scale_only else torch.zeros_like(scales)
        
    elif mode in ["mean", "median"]:
        ratios = torch.zeros_like(depth1)
        ratios[valid_mask] = depth2[valid_mask] / depth1[valid_mask]
        
        scales = []
        for i in range(B):
            ratios_valid_i = ratios[i][valid_mask[i]]
            if (ratios_valid_i.numel() / depth1[i].numel()) > min_valid_fraction:
                if mode == "mean":
                    scales.append(ratios_valid_i.mean())
                elif mode == "median":
                    scales.append(ratios_valid_i.median())
            else:
                scales.append(torch.tensor(1., device=depth1.device, dtype=depth1.dtype))
        scales = torch.stack(scales)
        
    else:
        NotImplementedError(f"Depth alignment mode {mode} if not available.")
        
    return scales, shifts