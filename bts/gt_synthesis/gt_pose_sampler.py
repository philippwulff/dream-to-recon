import os
from typing import Tuple, Dict, Any, Optional, Literal, List

import torch
import torch.nn as nn
import numpy as np

from utils.transformation_ops import orientate_poses, shift_poses, orbit_poses_about, orbit_poses_about_vert
from utils.projection_ops import unnormalize_intrinsics, crop_intrinsics
from utils.utils import get_interval_sample
from configs.structured_configs.synthetic_gt_config import CameraSamplerConfig
from copy import copy
from utils.utils import asdict_lowercase_keys_override
import torch.nn.functional as F
import logging
from bts.models.pseudo_volume import PseudoVolume
from bts.renderer.nerf import NeRFRenderer
from bts.common.ray_sampler import ImageRaySampler
from utils.occlusion_ops import sobel_filter
from utils.projection_ops import distance_to_z
from torch.optim import Adam, SGD
from torch.optim.lr_scheduler import StepLR, MultiStepLR, CyclicLR, LinearLR, PolynomialLR
import matplotlib.pyplot as plt
from utils.plotting import render_profile, OUT_RES, plot_frustums, color_tensor, plot_profile, set_spines, PAGE_WIDTH_INCHES
import math
from utils.occlusion_ops import morphological_op
from matplotlib.patches import Arc, Rectangle
import os
import moviepy.video.io.ImageSequenceClip
from moviepy.video.fx.all import crop
import re
from collections import defaultdict
import time    
from configs.structured_configs.data_config import VisVolume
from utils.plotting import cmap_magma, set_thesis_rcparams
from bts.gt_synthesis.common import compute_occlusion_center_error, compute_occlusion_edges_error, gather_bottom_k_error


logger = logging.getLogger(__name__)


EPS = 1e-5


# @torch.cuda.amp.custom_fwd(cast_inputs=torch.float32)
def comp_reprojected_img_extents(projs_enc: torch.Tensor, projs_new: torch.Tensor, depths_enc: torch.Tensor, c2w_enc: torch.Tensor, c2w_new: torch.Tensor, w_new: int, h_new: int, valid_depths_enc: torch.Tensor | None = None) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor]:
    """Helper function that returns the min and max image coordinates of the encoding image in the new image."""
    B, NV, _, h, w = depths_enc.shape
    
    # Corners have duplicates, but that does not matter.
    top_edge = torch.stack([torch.linspace(-1, 1, steps=w), -torch.ones(w)], dim=1)
    bottom_edge = torch.stack([torch.linspace(-1, 1, steps=w), torch.ones(w)], dim=1)
    left_edge = torch.stack([-torch.ones(h), torch.linspace(-1, 1, steps=h)], dim=1)
    right_edge = torch.stack([torch.ones(h), torch.linspace(-1, 1, steps=h)], dim=1)
    edge_pixels = torch.cat([top_edge, bottom_edge, left_edge, right_edge], dim=0).to(projs_enc.device)
    # Edge pixels expanded to batch size
    edges = edge_pixels.unsqueeze(0).unsqueeze(0).repeat(B, NV, 1, 1)
    edges = torch.cat([edges, torch.ones(B, NV, edges.size(2), 1, device=projs_enc.device)], dim=-1)

    # Get the corresponding depth values for edge pixels
    x_index = ((edge_pixels[:, 0] + 1) * 0.5 * (w - 1)).long()
    y_index = ((edge_pixels[:, 1] + 1) * 0.5 * (h - 1)).long()
    edges_depths = depths_enc[..., y_index, x_index]
    
    # Check for valid depths and apply mask
    if valid_depths_enc is not None:
        valid_mask = valid_depths_enc[..., y_index, x_index]
        # Mask the depths where they are invalid, set invalid depths to NaN for filtering
        edges_depths[~valid_mask] = torch.nan
        edges[~valid_mask[:, :, 0, :, None].repeat(1, 1, 1, 3)] = torch.nan
        # edges = edges[valid_mask]
        # edges_depths = edges_depths[valid_mask]
        
    corners = edges
    corners_depth = edges_depths

    projs_enc_inv = torch.inverse(projs_enc.float()).to(projs_enc.dtype) 
    corners = (projs_enc_inv @ corners.transpose(3, 2)).transpose(3, 2)
    # Scale by depths (along Z-axis) and make homogeneous coordinates.
    corners = corners * corners_depth[:, :, 0, :, None]
    corners = torch.concatenate([corners, torch.ones_like(corners[..., :1])], dim=-1)
    # This automatically works with c2w_new [B, nv, 4, 4] and c2w_enc [B, 1, 4, 4]
    w2c_new = torch.inverse(c2w_new.float()).to(c2w_new.dtype)
    corners_nv = (projs_new @ (w2c_new @ c2w_enc)[..., :3, :] @ corners.transpose(3, 2)).transpose(3, 2)
    corners_nv = (corners_nv / (corners_nv[..., 2:3].clamp_min(1e-3)))
    # Un-normalize pixel coordinates
    corners_nv = corners_nv[..., :2] * 0.5 + 0.5
    corners_nv[..., 0] *= w_new
    corners_nv[..., 1] *= h_new
    
    INF = 100_000
    min_left_and_top = torch.min(torch.nan_to_num(corners_nv, INF), dim=-2)[0]
    max_right_and_bottom = torch.max(torch.nan_to_num(corners_nv, -INF), dim=-2)[0]
    l = min_left_and_top[:, :, 0]
    t = min_left_and_top[:, :, 1]
    r = max_right_and_bottom[:, :, 0]
    b = max_right_and_bottom[:, :, 1]
    
    # p = corners_nv[0, 0, :, :].clamp(0, 1000).cpu()
    # plt.scatter(p[:, 0], p[:, 1])
    # plt.savefig("temp.png")
    
    
    
    return l, t, r, b       # each is [B, NV] in pixels



class CameraSampler(nn.Module):
    def __init__(
        self, 
        num_novel_views: int = 1, 
        make_projs_policy: Literal["none", "center_crop", "crop_to_visible"] = "none", 
        edge_dist_for_proj_sample: float = 0.0, 
        generator: Optional[torch.Generator] = None, 
        cam_incl_adjust: Optional[torch.Tensor] = None,
        apply_cam_incl_adjust: bool = False,
        *args, 
        **kwargs,
        ) -> None:
        
        super().__init__()
        self.nv = num_novel_views
        self.generator = generator
        self.edge_dist_for_proj_sample = edge_dist_for_proj_sample
        self.make_projs_policy = make_projs_policy
        # Register the tensor as a buffer, so that it is moved to the device.
        self.register_buffer("cam_incl_adjust", cam_incl_adjust, persistent=False)
        self.apply_cam_incl_adjust = apply_cam_incl_adjust
        self.sample_projs_first = False
        if self.sample_projs_first:
            assert self.make_projs_policy == "center_crop", "Intrinsics sampling policy must use center crops if 'sample_projs_first' is true."
        if apply_cam_incl_adjust:
            logger.info("No camera incline matrix given, but incline adjustement is enabled. Expecting to receive the incline matrix as a keyword argument in the forward pass.")

    @classmethod
    def from_conf(cls, cfg: CameraSamplerConfig | Dict[str, Any], **kwargs):
        return cls(
            **asdict_lowercase_keys_override(cfg, **kwargs)
        )

    def forward(self, poses: torch.Tensor, projs: torch.Tensor, nv: Optional[int] = None, generator: Optional[torch.Generator] = None, cam_incl_adjust: Optional[torch.Tensor] = None, *args, **kwargs) -> Tuple[torch.Tensor, torch.Tensor, Dict[str, Any]]:
        """Returns new poses and projs given the current ones."""
        
        old_nv = copy(self.nv)
        old_generator = copy(self.generator)
        if nv is not None:
            self.nv = nv
        if generator is not None:
            self.generator = generator
            
        # if self.apply_cam_incl_adjust and (self.cam_incl_adjust is not None or cam_incl_adjust is not None):
        #     T_adjust = cam_incl_adjust if cam_incl_adjust is not None else self.cam_incl_adjust
        #     poses = T_adjust.to(poses) @ poses
            
        projs_new = projs.repeat(1, self.nv, 1, 1)
        if self.sample_projs_first:
            # Some pose samplers want the new projs to sample poses.
            try:
                projs_new, debug_vars = self._sample_projs(poses, projs.repeat(1, self.nv, 1, 1), None, projs_new, **kwargs)
                kwargs["projs_new"] = projs_new
            except Exception as e:
                raise ValueError(f"Sampling intrinsics first failed with exception: {e}. Try disabling 'sample_projs_first'.")
        
        poses_new = self._sample_poses(poses, projs=projs, **kwargs)
        
        if self.apply_cam_incl_adjust and (self.cam_incl_adjust is not None or cam_incl_adjust is not None):
            T_adjust = cam_incl_adjust if cam_incl_adjust is not None else self.cam_incl_adjust
            poses_new = T_adjust.to(poses_new) @ poses_new

        if not self.sample_projs_first:
            projs_new, debug_vars = self._sample_projs(poses, projs.repeat(1, self.nv, 1, 1), poses_new, projs_new, **kwargs)
        
        self.nv = old_nv
        self.generator = old_generator
        
        return poses_new, projs_new, debug_vars
    
    def _sample_poses(self, poses: torch.Tensor, projs: torch.Tensor, **kwargs) -> torch.Tensor:
        """Samples SE(4) poses. Implemented in subclasses.

        Args:
            poses (torch.Tensor): [B, 1, 4, 4]
        Returns:
            torch.Tensor: [B, nv, 4, 4]
        """
        raise NotImplementedError
    
    def _sample_projs(self, poses: torch.Tensor, projs: torch.Tensor, poses_new: torch.Tensor, projs_new: torch.Tensor, depths: torch.Tensor, hw: Tuple[int, int], hw_rot: Tuple[int, int], **kwargs) -> Tuple[torch.Tensor, Any]:
        """Samples the camera intrinsics from the set of intrinsics showing visible regions of the 3D scene."""
        B, _, _, _ = poses.shape
        device = poses.device
        dtype = poses.dtype
        
        H, W = hw
        H_ROT, W_ROT = hw_rot
        H_UNNORM, W_UNNORM = kwargs.get("hw_unnorm", (H, W))
        #  = 376, 1408  ## TODO FIXME     
        
        debug_vars = {}
        match self.make_projs_policy:
            case "none":
                projs_new_cropped = projs_new
            case "center_crop":
                l_sample = torch.full((B, self.nv), fill_value=(W-W_ROT)/2, device=device, dtype=dtype)
                t_sample = torch.full((B, self.nv), fill_value=(H-H_ROT)/2, device=device, dtype=dtype)
                
                projs_new_cropped = crop_intrinsics(projs_new, l_sample, t_sample, unnormalized_W=W, unnormalized_H=H, new_unnormalized_W=W_ROT, new_unnormalized_H=H_ROT)
                debug_vars = {
                    # "l": torch.full_like(l_sample, -(W-W_ROT)/2), 
                    # "t": torch.full_like(l_sample, -(H-H_ROT)/2), 
                    # "r": torch.full_like(l_sample, W_ROT+(W-W_ROT)/2), 
                    # "b": torch.full_like(l_sample, H_ROT+(H-H_ROT)/2), 
                    "l_sample": l_sample, "t_sample": t_sample
                }
            case "crop_to_visible":
                # Determine the min and max visible image pixels when viewed from poses_rot (using same intrinsics).
                l, t, r, b = comp_reprojected_img_extents(projs, projs_new, depths, poses, poses_new, W, H)         # FIXME these should maybe be W_RECT and H_RECT???
                
                # this forces the new proj to lie within the input proj
                projs_unnorm = unnormalize_intrinsics(projs, W_UNNORM, H_UNNORM)
                l = torch.maximum(projs_unnorm[..., 0, 2] - W_UNNORM / 2, l)
                t = torch.maximum(projs_unnorm[..., 1, 2] - H_UNNORM / 2, t)
                r = torch.minimum(projs_unnorm[..., 0, 2] + W_UNNORM / 2, r)
                b = torch.minimum(projs_unnorm[..., 1, 2] + H_UNNORM / 2, b)
                
                # Sample crop from visible pixels.
                w_interval = (r-W_ROT-l).clamp_min(0)
                edge_dist = self.edge_dist_for_proj_sample * w_interval
                l_sample = get_interval_sample(l+edge_dist, w_interval-2*edge_dist, generator=self.generator)
                l_sample = l_sample.round().to(torch.int32)
                h_interval = (b-H_ROT-t).clamp_min(0)
                edge_dist = self.edge_dist_for_proj_sample * h_interval
                t_sample = get_interval_sample(t+edge_dist, h_interval-2*edge_dist, generator=self.generator)
                t_sample = t_sample.round().to(torch.int32)
                # NOTE this enforces staying vertically within the projs frustrum... maybe good, maybe not?
                t_sample = torch.zeros_like(t_sample)
                # t_sample = t_sample.clamp(0, H-H_ROT).round().long()
                
                projs_new_cropped = crop_intrinsics(projs_new, l_sample, t_sample, unnormalized_W=W, unnormalized_H=H, new_unnormalized_W=W_ROT, new_unnormalized_H=H_ROT)
                
                debug_vars = {"l": l, "t": t, "r": r, "b": b, "l_sample": l_sample, "t_sample": t_sample}
            case _:
                raise ValueError()
            
        return projs_new_cropped, debug_vars


class OrbitCameraSampler(CameraSampler):
    """Samples cameras relative that are orbited around a vertical axis in the input camera's Z-direction."""
    def __init__(self, x_lims=[0., 0.], y_lims=[0., 0.], z_dist_lims=[0., 0.], orbit_policy: str = "cam_frame", *args, **kwargs) -> None:
        super().__init__(*args, **kwargs)
        self.x_min, self.x_max = x_lims
        self.y_min, self.y_max = y_lims
        self.z_dist_min, self.z_dist_max = z_dist_lims
        self.orbit_policy = orbit_policy

    def _sample_poses(self, poses: torch.Tensor, **kwargs) -> torch.Tensor:
        B, _, _, _ = poses.shape
        NV = self.nv
        device = poses.device
        dtype = poses.dtype
        
        x_degs = torch.rand(B*NV, generator=self.generator, device=device, dtype=dtype) * (self.x_max-self.x_min) + self.x_min
        y_degs = torch.rand(B*NV, generator=self.generator, device=device, dtype=dtype) * (self.y_max-self.y_min) + self.y_min
        poses_new = []
        
        match self.orbit_policy:
            case "cam_frame":
                z_dists = torch.rand(B*NV, generator=self.generator, device=device, dtype=dtype) * (self.z_dist_max-self.z_dist_min) + self.z_dist_min
                for i in range(B):
                    poses_new.append(
                        orbit_poses_about_vert(poses[i:i+1].repeat(1, NV, 1, 1), x_degs[i*NV:(i+1)*NV], y_degs[i*NV:(i+1)*NV], z_dists[i*NV:(i+1)*NV])
                    )
            case "about_world_origin":
                try:
                    poses_c2w = kwargs["poses_c2w"]
                except KeyError:
                    raise ValueError("Camera sampler kwargs needs to contain 'poses_c2w'.")
                for i in range(B):
                    poses_new.append(
                        orbit_poses_about(
                            poses[i:i+1].repeat(1, NV, 1, 1), 
                            poses_c2w[i:i+1].repeat(1, NV, 1, 1), 
                            x_degs[i*NV:(i+1)*NV], y_degs[i*NV:(i+1)*NV]
                        )
                    )
            case _:
                raise ValueError(f"Orbiting policy {self.orbit_policy} is unavailable.")
        
        return torch.concat(poses_new)


class ShiftCameraSampler(CameraSampler):
    """Samples shifted cameras relative to the input camera."""
    def __init__(self, x_lims=[0., 0.], y_lims=[0., 0.], z_lims=[0., 0.], *args, **kwargs) -> None:
        super().__init__(*args, **kwargs)
        self.x_min, self.x_max = x_lims
        self.y_min, self.y_max = y_lims
        self.z_min, self.z_max = z_lims
        
    def _sample_poses(self, poses: torch.Tensor, **kwargs) -> Tuple[torch.Tensor, torch.Tensor, Dict[str, Any]]:
        B, _, _, _ = poses.shape
        NV = self.nv
        
        device = poses.device
        dtype = poses.dtype
        
        xd = torch.rand(B*NV, generator=self.generator, device=device, dtype=dtype) * (self.x_max-self.x_min) + self.x_min
        yd = torch.rand(B*NV, generator=self.generator, device=device, dtype=dtype) * (self.y_max-self.y_min) + self.y_min
        zd = torch.rand(B*NV, generator=self.generator, device=device, dtype=dtype) * (self.z_max-self.z_min) + self.z_min
        
        poses_new = []
        for i in range(B):
            p = shift_poses(poses[i:i+1].repeat(1, NV, 1, 1), xd[i*NV:(i+1)*NV], yd[i*NV:(i+1)*NV], zd[i*NV:(i+1)*NV])
            poses_new.append(p)
        
        return torch.concat(poses_new)
    
    
class ShiftRotCameraSampler(CameraSampler):
    """Samples shifted and rotated cameras relative to the input camera."""
    def __init__(self, x_shift_lims, y_shift_lims, z_shift_lims, x_rot_lims, y_rot_lims, z_rot_lims, *args, **kwargs) -> None:
        super().__init__(*args, **kwargs)
        self.x_shift_min, self.x_shift_max = x_shift_lims
        self.y_shift_min, self.y_shift_max = y_shift_lims
        self.z_shift_min, self.z_shift_max = z_shift_lims
        self.x_rot_min, self.x_rot_max = x_rot_lims
        self.y_rot_min, self.y_rot_max = y_rot_lims
        self.z_rot_min, self.z_rot_max = z_rot_lims
        
    def _sample_poses(self, poses: torch.Tensor, **kwargs) -> torch.Tensor:
        B, _, _, _ = poses.shape
        NV = self.nv
        
        device = poses.device
        dtype = poses.dtype
        
        xd = torch.rand(B*NV, generator=self.generator, device=device, dtype=dtype) * (self.x_shift_max-self.x_shift_min) + self.x_shift_min
        yd = torch.rand(B*NV, generator=self.generator, device=device, dtype=dtype) * (self.y_shift_max-self.y_shift_min) + self.y_shift_min
        zd = torch.rand(B*NV, generator=self.generator, device=device, dtype=dtype) * (self.z_shift_max-self.z_shift_min) + self.z_shift_min
        
        xr = torch.rand(B*NV, generator=self.generator, device=device, dtype=dtype) * (self.x_rot_max-self.x_rot_min) + self.x_rot_min
        yr = torch.rand(B*NV, generator=self.generator, device=device, dtype=dtype) * (self.y_rot_max-self.y_rot_min) + self.y_rot_min
        zr = torch.rand(B*NV, generator=self.generator, device=device, dtype=dtype) * (self.z_rot_max-self.z_rot_min) + self.z_rot_min
        
        poses_new = []
        for i in range(B):
            p = poses[i:i+1].repeat(1, NV, 1, 1)
            p = shift_poses(p, xd[i*NV:(i+1)*NV], yd[i*NV:(i+1)*NV], zd[i*NV:(i+1)*NV])
            p = orientate_poses(p, xr[i*NV:(i+1)*NV], yr[i*NV:(i+1)*NV], zr[i*NV:(i+1)*NV])
            poses_new.append(p)
        
        return torch.concat(poses_new)


class RigCameraSampler(CameraSampler):
    """Samples cameras from a list of cameras with specified positions and orientations."""
    def __init__(self, cams_xyz: Tuple[Tuple[int, int, int]], cams_alpha_beta_gamma: Tuple[Tuple[int, int, int]], *args, **kwargs) -> None:
        super().__init__(*args, **kwargs)
        # Register as non-persistent buffers so they are pushed to the device with the module.
        self.register_buffer("cams_xyz", torch.tensor(cams_xyz), persistent=False)
        self.register_buffer("cams_alpha_beta_gamma", torch.tensor(cams_alpha_beta_gamma), persistent=False)
        assert len(cams_xyz) == len(cams_alpha_beta_gamma), "CAMERA ORIGINS AND ORIENTATIONS HAVE DIFFERENT LENGTHS"
        self.num_cams = len(cams_xyz)
        assert self.num_cams >= self.nv, "NOT ENOUGH CAMERAS IN THE RIG FOR THE REQUESTED NUMBER OF NOVEL VIEWS"
        
    def _sample_poses(self, poses: torch.Tensor, **kwargs) -> Tuple[torch.Tensor, torch.Tensor, Dict[str, Any]]:
        B, _, _, _ = poses.shape
        NV = self.nv
        
        # Sample NV random cameras from the rig.
        idxs = torch.randperm(self.num_cams, generator=self.generator, device=poses.device, dtype=torch.long)[:NV]
        
        xyz = self.cams_xyz.to(poses)[idxs]
        xyz = xyz.repeat(B, 1)
        xd = xyz[:, 0]
        yd = xyz[:, 1]
        zd = xyz[:, 2]
        
        alpha_beta_gamma = self.cams_alpha_beta_gamma.to(poses)[idxs]
        alpha_beta_gamma = alpha_beta_gamma.repeat(B, 1)
        xr = alpha_beta_gamma[:, 0]
        yr = alpha_beta_gamma[:, 1]
        zr = alpha_beta_gamma[:, 2]
        
        poses_new = []
        for i in range(B):
            p = poses[i:i+1].repeat(1, NV, 1, 1)
            p = shift_poses(p, xd[i*NV:(i+1)*NV], yd[i*NV:(i+1)*NV], zd[i*NV:(i+1)*NV])
            p = orientate_poses(p, xr[i*NV:(i+1)*NV], yr[i*NV:(i+1)*NV], zr[i*NV:(i+1)*NV])
            poses_new.append(p)
        
        return torch.concat(poses_new)


class MovementCameraSampler(CameraSampler):
    """Samples according to SE(4) transforms from a file relative to the input camera."""
    def __init__(self, trajectory_file_name: str = "simple_movement.npy", scale: float = 1.0, skip: int = 1, *args, **kwargs) -> None:
        super().__init__(*args, **kwargs)
        assert scale > 0., "SCALE MUST BE LARGER THAN 0."
        assert isinstance(skip, int) and skip >= 1, "SKIP MUST BE INTEGER GREATER THAN 0."

        path = os.path.join("bts", "gt_synthesis", "trajectories", trajectory_file_name)
        
        self.cam_traj = torch.tensor(np.load(path))
        self.cam_traj[:, :3, 3] *= scale
        self.cam_traj = self.cam_traj[::skip, :, :]
        self.num_poses = len(self.cam_traj)
        
    def _sample_poses(self, poses: torch.Tensor, t: Optional[float] = None, **kwargs) -> torch.Tensor:
        B, _, _, _ = poses.shape
        NV = self.nv
        device = poses.device
        dtype = poses.dtype
        self.cam_traj = self.cam_traj.to(device, dtype)
        
        if t is None:        
            t = torch.rand(B*NV, generator=self.generator, device=device, dtype=dtype)
        else:
            t = torch.ones((B*NV,), device=device, dtype=dtype) * t

        idxs = (t * (self.num_poses - 1)).round().long()
            
        poses_new = poses.squeeze(1).repeat_interleave(NV, dim=0)       # [B*NV, 4, 4]
        poses_new = self.cam_traj[idxs] @ poses_new
        poses_new = poses_new.view(B, NV, 4, 4)
        
        return poses_new


class TrajectoryCameraSampler(CameraSampler):
    """Samples a camera from a consecutive, pre-defined trajectory. Useful for creating videos."""
    def __init__(self, policy: str = "left_right", num_steps: int = 10, sampler_init_kwargs: Dict = {}, *args, **kwargs) -> None:
        super().__init__(*args, **kwargs)
        self._step = 0
        self._num_steps = num_steps
        self.policy = policy
        step_to_time = torch.linspace(0, 1, self._num_steps)
        match self.policy:
            case "left_right":
                x_start = -4
                x_end = +4
                def sample(step, **kwargs):
                    t = step_to_time[step]
                    x = t * x_end + (1 - t) * x_start
                    s = ShiftCameraSampler(x_lims=[x, x], **sampler_init_kwargs)
                    return s._sample_poses(**kwargs)
            case "center_left_right_center":
                x_center = 0
                x_left = -3
                x_right = +3
                
                def lerp(a, b, u):
                    return (1 - u) * a + u * b

                def sample(step, **kwargs):
                    t = step_to_time[step]
                    one_third = t.new_tensor(1.0 / 3.0)
                    two_thirds = t.new_tensor(2.0 / 3.0)

                    # piecewise linear path: C (t=0) -> L (t=1/3) -> R (t=2/3) -> C (t=1)
                    if torch.le(t, one_third):
                        # C -> L
                        u = t / one_third
                        x = lerp(t.new_tensor(x_center), t.new_tensor(x_left), u)
                    elif torch.le(t, two_thirds):
                        # L -> R
                        u = (t - one_third) / one_third
                        x = lerp(t.new_tensor(x_left), t.new_tensor(x_right), u)
                    else:
                        # R -> C
                        u = (t - two_thirds) / one_third
                        x = lerp(t.new_tensor(x_right), t.new_tensor(x_center), u)

                    s = ShiftCameraSampler(x_lims=[x, x], **sampler_init_kwargs)
                    return s._sample_poses(**kwargs)
            case "orbit":
                y_start = 5
                y_end = -5
                z_dist = 2.0
                # z_dist = 5
                def sample(step, **kwargs):
                    t = step_to_time[step]
                    x = t * y_end + (1 - t) * y_start
                    s = OrbitCameraSampler(y_lims=[x, x], z_dist_lims=[z_dist, z_dist], **sampler_init_kwargs)
                    return s._sample_poses(**kwargs)
            case "simple_movement":
                name = "simple_movement.npy"
                scale = 2.5
                def sample(step, **kwargs):
                    t = step_to_time[step]
                    s = MovementCameraSampler(name, scale=scale, **sampler_init_kwargs)
                    return s._sample_poses(t=t, **kwargs)
            case _:
                raise ValueError(f"Trajectory sampling policy {policy} is not available.")
            
        self.trajectory_sampler = sample
        
    def reset(self):
        self._step = 0
        
    def step(self):
        self._step += 1
        
    def _sample_poses(self, poses: torch.Tensor, **kwargs) -> torch.Tensor:
        
        assert self.nv == 1, "Camera trajectories require nv=1."
        
        if self._step >= self._num_steps:
            self.reset()
            
        poses_new = self.trajectory_sampler(self._step, poses=poses, **kwargs)
        self.step()
        
        return poses_new
    

class RandomChoiceCameraSampler(CameraSampler):
    """
    Contains a list of camera samplers and samples poses from a randomly chosen sampler.
    """
    def __init__(self, camera_sampler_configs: List[CameraSamplerConfig], *args, **kwargs) -> None:
        super().__init__(*args, **kwargs)
        
        self.camera_samplers = []
        for cfg in camera_sampler_configs:
            cls = globals()[cfg["TYPE"]]
            # Override shared camera sampler parameters (number of novel views, etc.)
            cfg.update(kwargs)
            camera_sampler = cls.from_conf(cfg)
            self.camera_samplers.append(camera_sampler)
        
    def _sample_poses(self, poses: torch.Tensor, **kwargs) -> torch.Tensor:
        
        # Pick a random camera sampler.
        idx = torch.randint(0, len(self.camera_samplers), (1,), generator=self.generator, device=poses.device).cpu().item()
        sampler = self.camera_samplers[idx]
        # This is needed since `nv` and `generator` are only cached for `self` and not the list of camera samplers.
        sampler.nv = self.nv
        sampler.generator = self.generator
        # Sample all novel view poses using the chosen sampler.
        poses_new = sampler._sample_poses(poses, **kwargs)
        
        return poses_new
    

class ExplorationCameraSampler(CameraSampler):
    """
    Treats the camera position sampling as an optimization problem.
    
    Goal: Maximize the visible occluded region in the rendered camera poses, 
    given multiple constraints (on depth, etc.).
    """
    def __init__(
        self, 
        num_proposals: int = 8, 
        num_steps: int = 100, 
        visualize: bool = False, 
        visualization_path: str = "exploration_vis", 
        visualization_no_text: bool = False,
        visualization_steps: List[int] | None = None,
        visualize_gif: bool = False,
        init_policy: Literal["random", "stratified"] = "random",
        resample_loss_threshold: float = 1.0,
        max_occlusion_value: float = 0.2,
        min_depth: float = 6.0,
        prune_distance_thresh: float = 0.4,
        prune_orientation_thresh: float = 0.1,
        imgs_right: bool = True,
        xlims: Tuple[float, float] = [-10., 10.],
        zlims: Tuple[float, float] = [0., 20.],
        beta_lims: Tuple[float, float] = [-45, 45],
        lambda_xz_bounds: float = 1.,
        lambda_beta_bounds: float = 1.,
        lambda_occlusion: float = 1.,
        lambda_occlusion_center: float = 1.,
        lambda_occlusion_edges: float = 1.,
        lambda_depth: float = 0.1,
        lambda_sigmas: float = 0.,
        lambda_pose_sim: float = 0.,
        lambda_weights: float = 1.0,
        *args, 
        **kwargs,
    ) -> None:
        
        super().__init__(*args, **kwargs)
        
        # We need these settings for this sampler to work.
        self.sample_projs_first = True
        self.make_projs_policy = "center_crop"
        # We do not want to adjust the camera inclincation outside of `_sample_poses`.
        self.apply_cam_incl_adjust = False
        # Visualization args
        self.visualize = visualize
        self.visualization_path = visualization_path
        self.visualization_no_text = visualization_no_text
        self.visualization_frames_path = os.path.join(visualization_path, "frames")
        self.visualization_steps = visualization_steps
        self.visualize_gif = visualize_gif
        # Optimization args
        self.xlims = xlims
        self.zlims = zlims
        self.beta_lims = beta_lims
        self.beta_lims = [_ * math.pi / 180 for _ in self.beta_lims]
        self.num_proposals = num_proposals
        self.num_steps = num_steps
        self.init_policy = init_policy
        self.resample_loss_threshold = resample_loss_threshold
        self.max_occlusion_value = max_occlusion_value
        self.min_depth = min_depth
        self.prune_distance_thresh = prune_distance_thresh
        self.prune_orientation_thresh = prune_orientation_thresh
        self.imgs_right = imgs_right
        
        self.lambda_xz_bounds =lambda_xz_bounds
        self.lambda_beta_bounds =lambda_beta_bounds
        self.lambda_occlusion =lambda_occlusion
        self.lambda_occlusion_center =lambda_occlusion_center
        self.lambda_occlusion_edges =lambda_occlusion_edges
        self.lambda_depth = lambda_depth 
        self.lambda_sigmas =lambda_sigmas
        self.lambda_pose_sim =lambda_pose_sim
        self.lambda_weights = lambda_weights 
        
    def _sample_poses(self, poses: torch.Tensor, projs: torch.Tensor, depths: torch.Tensor, projs_new: torch.Tensor, hw_rot: Tuple[int, int], **kwargs) -> torch.Tensor:
        depths_enc = kwargs["depths_enc"]
        poses_enc = kwargs["poses_enc"]
        projs_enc = kwargs["projs_enc"]
        masks_enc = kwargs["masks_enc"]
        num_input_anchors = kwargs.get("num_input_anchors", 1)
        
        device = depths_enc.device
        dtype = depths_enc.dtype
        b, _, _, h, w = depths_enc.shape
        
        cam_incl_adjust = kwargs["cam_incl_adjust"] if "cam_incl_adjust" in kwargs else self.cam_incl_adjust
        assert cam_incl_adjust is not None, f"{self} needs input camera incline adjustment if the input camera is not level (e.g. in KITTI-360)."
        cam_incl_adjust = cam_incl_adjust.to(dtype=dtype, device=device)
            
        z_near = 2.0
        z_far = float(depths_enc.max()) * 1.5      # We want some samples beyond the max depth
        
        height = hw_rot[0] // 6
        width = hw_rot[1] // 6
        time_start = time.time()
        with torch.enable_grad():
            renderer = NeRFRenderer(
                # n_coarse=128, n_fine=64, n_fine_depth=16, depth_std=1.0,
                n_coarse=256, n_fine=128, n_fine_depth=64, depth_std=1.0,
                eval_batch_size=10_000_000,
            )
            vol = PseudoVolume(
                color_sampling_mode="mean_valid_surface", 
                transition_sharpness=100,
                density_value=100.,
                use_optimal_diff_ops=True,
                num_input_anchors=num_input_anchors,
            )
            ray_sampler = ImageRaySampler(z_near=z_near, z_far=z_far, height=height, width=width, channels=2, norm_dir=True)
            
            # We want to find X and Z in a fixed XZ-plane
            # and the rotation in the Y-plane (about the Y-axis).
            match self.init_policy:
                case "random":
                    position_params = torch.rand((b, self.num_proposals, 2), device=device, dtype=dtype, generator=self.generator) * 2. - 1.
                    orientation_params = torch.rand((b, self.num_proposals), device=device, dtype=dtype, generator=self.generator) * 2. - 1.
                    num_proposals = self.num_proposals
                case "stratified":
                    aspect_ratio = (self.xlims[1] - self.xlims[0]) / (self.zlims[1] - self.zlims[0])
                    grid_x_size = int(math.sqrt(self.num_proposals * aspect_ratio))
                    grid_z_size = int(self.num_proposals / grid_x_size)
                    
                    if grid_x_size * grid_z_size != self.num_proposals:
                        grid_x_size = round(math.sqrt(self.num_proposals / aspect_ratio))
                        grid_z_size = int(self.num_proposals / grid_x_size)
                    
                    num_proposals = int(grid_x_size * grid_z_size)
                    if num_proposals != self.num_proposals:
                        logger.warn(f"Number of grid proposals {num_proposals} does not match the number specified {self.num_proposals}.")
                    
                    if num_proposals < self.nv:
                        raise ValueError(f"Invalid value for the number of proposals {self.num_proposals} (reduced to {num_proposals} in the grid) given the number of output novel views {self.nv}.")
                    
                    delta_x = 2.0 / grid_x_size
                    delta_z = 2.0 / grid_z_size
                    grid_x = torch.linspace(-1+delta_x/2, 1-delta_x/2, grid_x_size, dtype=dtype, device=device)
                    grid_z = torch.linspace(-1+delta_z/2, 1-delta_z/2, grid_z_size, dtype=dtype, device=device)
                    grid_x, grid_z = torch.meshgrid(grid_x, grid_z, indexing='ij')
                    grid_x = grid_x.flatten().unsqueeze(0) + torch.rand((b, num_proposals), device=device, dtype=dtype) * delta_x - delta_x / 2
                    grid_z = grid_z.flatten().unsqueeze(0) + torch.rand((b, num_proposals), device=device, dtype=dtype) * delta_z - delta_z / 2
                    
                    position_params = torch.stack([grid_x, grid_z], dim=-1)
                    orientation_params = torch.rand((b, num_proposals), device=device, dtype=dtype, generator=self.generator) * 2. - 1.
                case _:
                    raise ValueError(f"Init policy {self.init_policy} is not available.")
            
            projs_new = projs_new[:, 0:1].repeat(1, num_proposals, 1, 1)
            
            lr = 0.05
            position_params.requires_grad_()
            orientation_params.requires_grad_()
            optim = Adam(params=[position_params, orientation_params], lr=lr)
            # scheduler = MultiStepLR(optim, gamma=0.9, milestones=[20, 40, 50, 60, 70, 80, 90])
            # scheduler = StepLR(optim, gamma=0.9, step_size=20, last_epoch=1)
            # scheduler = MultiStepLR(optim, gamma=0.1, milestones=[20000])
            # https://github.com/bckenstler/CLR
            scheduler = CyclicLR(optim, base_lr=0.01, max_lr=0.05, step_size_up=10, mode="triangular2", cycle_momentum=False)
            # scheduler = PolynomialLR(optim, total_iters=num_steps, power=1)     # Linear
            # scheduler = MultiStepLR(optim, gamma=0.9, milestones=[100, 200, 300, 400, 500, 600, 700, 800, 900])
            
            all_losses = defaultdict(list)

            occlusions = sobel_filter(depths_enc.view(-1, 1, h, w), thresh=0.1).view(depths_enc.shape)
            
            features = torch.concat([occlusions, masks_enc], dim=-3)
            vol.encode(depths_enc, None, projs_enc, poses_enc, features=features)
            
            for i in range(self.num_steps):
                t_step_start = time.time()
                
                poses_new = torch.eye(4, device=device, dtype=dtype)[None, None, :, :].repeat(b, num_proposals, 1, 1)
                # Map unbounded parameters to bounded orientation parameters
                # Tanh maps (-inf, inf) to (-1, 1). Then scale and shift to the desired range.
                scale_xz = torch.tensor([self.xlims[1]-self.xlims[0], self.zlims[1]-self.zlims[0]], device=device, dtype=dtype)
                shift_xz = torch.tensor([self.xlims[0], self.zlims[0]], device=device, dtype=dtype)
                scale_beta = self.beta_lims[1]-self.beta_lims[0]
                shift_beta = self.beta_lims[0]
                
                position = (torch.tanh(position_params) * 0.5 + 0.5) * scale_xz + shift_xz
                orientation = (torch.tanh(orientation_params) * 0.5 + 0.5) * scale_beta + shift_beta

                poses_new[:, :, 0, 3:] = position[:, :, 0:1]
                poses_new[:, :, 2, 3:] = position[:, :, 1:2]
                poses_new[:, :, 0, 0] = torch.cos(orientation)
                poses_new[:, :, 0, 2] = torch.sin(orientation)
                poses_new[:, :, 2, 0] = -torch.sin(orientation)
                poses_new[:, :, 2, 2] = torch.cos(orientation)
                
                poses_new = cam_incl_adjust @ poses_new
                
                t_render = time.time()
                all_rays, _ = ray_sampler.sample(None, poses_new, projs_new)      # [n*nv, n_pts, 8]
                render_dict = renderer(vol, all_rays, want_weights=True)
                t_render = time.time() - t_render
                
                if "fine" not in render_dict:
                    render_dict["fine"] = dict(render_dict["coarse"])
                    
                render_dict = ray_sampler.reconstruct(render_dict)
                rgb = render_dict["fine"]["rgb"].squeeze(4).permute(0, 1, 4, 2, 3)                                                 # [B, 1, 3, H, W]
                rgb = rgb[:, :, :1]
                depths_pred = render_dict["fine"]["depth"].unsqueeze(2)                                                                    # [B, 1, 1, H, W]
                depths_pred = distance_to_z(depths_pred.squeeze(-3), projs_new).unsqueeze(-3).to(depths_pred.dtype)
                
                # A pixel is invalid if all weights are "0".
                invalid = (~render_dict["fine"]["weights"].any(dim=-1)).float().unsqueeze(-3)
                
                losses = {}
                
                rgb = morphological_op(rgb.view(-1, 1, height, width), "closing", kernel_size=3).view(b, num_proposals, 1, height, width)
                
                if self.lambda_weights > 0:
                    weights_sum = render_dict["coarse"]["weights"].sum(dim=-1)
                    # weights_sum = morphological_op(weights_sum.view(-1, 1, height, width), "closing", kernel_size=3).view(b, num_proposals, height, width)
                    losses["loss_weights"] = torch.pow(weights_sum - torch.ones_like(weights_sum), 2).mean(dim=[-1, -2]) * self.lambda_weights
                if self.lambda_sigmas > 0:
                    max_radius = 1.5
                    # Signed distance value of the camera positions.
                    # ray_orig = all_rays[:, :, :3].view(b, num_proposals, height, width, 3)
                    # ray_dirs = all_rays[:, :, 3:6].view(b, num_proposals, height, width, 3)
                    # n_samples = 5
                    # ray_samp = (
                    #     ray_orig.unsqueeze(-1) + ray_dirs.unsqueeze(-1) * torch.linspace(max_radius, max_radius+n_samples, n_samples, device=device)
                    # ).permute(0, 1, 2, 3, 5, 4).reshape(b, num_proposals*height*width*n_samples, 3)
                    circles = []
                    for r in torch.linspace(0, max_radius, 4):
                        phi = torch.linspace(0., 2*torch.pi, int(10 * r + 1), device=device, dtype=dtype)
                        circle = torch.stack([torch.cos(phi) * r, torch.zeros_like(phi), torch.sin(phi) * r], dim=1)
                        circles.append(circle)
                    circles = torch.concat(circles)     # [pts, 3]
                    circles = (
                        circles[None, None, :, :].expand(b, num_proposals, len(circles), 3) +#.repeat_interleave(num_proposals, dim=1) + 
                        poses_new[:, :, :3, 3:].view(b, num_proposals, 1, 3)#.view(b, num_proposals, 3).repeat_interleave(len(circles), dim=1)
                    ).view(b, -1, 3)
                    _, _, sigmas, _ = vol(circles)
                    losses["loss_sigmas"] = F.mse_loss(sigmas, torch.ones_like(sigmas)*-1*vol._density_value) * self.lambda_sigmas
                if self.lambda_occlusion > 0:
                    valid_mask = (~invalid.bool()).float()
                    mean_occlusion = (rgb * valid_mask).sum(dim=[-3, -2, -1]) / valid_mask.sum(dim=[-3, -2, -1]).clamp_min(1)
                    occlusion_loss_above = F.relu((mean_occlusion - self.max_occlusion_value) * 10).pow(2)
                    losses["loss_occlusion_max"] = occlusion_loss_above * self.lambda_occlusion
                if self.lambda_xz_bounds > 0:
                    losses["err_xz_bounds"] = F.relu(torch.sqrt(torch.pow(position_params[:, :, 0], 2) + torch.pow(position_params[:, :, 1], 2)) - 1.0) * self.lambda_xz_bounds
                if self.lambda_beta_bounds > 0:
                    losses["err_beta_bounds"] = F.relu(orientation_params.abs() - 1.0) * self.lambda_beta_bounds
                if self.lambda_occlusion_center > 0:
                    loss_center = compute_occlusion_center_error(rgb, torch.ones_like(rgb), invalid)
                    losses["loss_occlusion_center"] = loss_center * self.lambda_occlusion_center
                if self.lambda_occlusion_edges > 0:
                    loss_edges = compute_occlusion_edges_error(rgb, torch.zeros_like(rgb), invalid)
                    losses["loss_occlusion_edges"] = loss_edges * self.lambda_occlusion_edges
                if self.lambda_depth > 0:
                    # Penalize depths that are smaller than the min depth.
                    bottom_invalid = torch.zeros_like(invalid)
                    bottom_invalid[..., -(height//5):, :] = 1.
                    losses["loss_depth"] = torch.pow(F.relu(self.min_depth - depths_pred) * (1-invalid) * (1-bottom_invalid), 2).mean(dim=[-3, -2, -1]) * self.lambda_depth
                if self.lambda_pose_sim > 0:
                    # Compute pairwise Euclidean distances between all position vectors
                    # https://chatgpt.com/share/7c95d252-3252-475b-961f-d0c74309c8ca
                    pos_diff = torch.norm(
                        poses_new[:, :, :3, 3].unsqueeze(2) - poses_new[:, :, :3, 3].unsqueeze(1), 
                        dim=3
                    )
                    # Expanding orientation matrices for pairwise comparison
                    R_i = poses_new[:, :, :3, :3].unsqueeze(2)  # [b, num_proposals, 1, 3, 3]
                    R_j = poses_new[:, :, :3, :3].unsqueeze(1)  # [b, 1, num_proposals, 3, 3]
                    ori_diff = torch.matmul(R_i, R_j.transpose(-2, -1))  # [b, num_proposals, num_proposals, 3, 3]
                    # Compute the trace of the relative rotation matrices
                    trace = ori_diff[..., 0, 0] + ori_diff[..., 1, 1] + ori_diff[..., 2, 2]
                    trace = (trace - 1) / 2.0
                    # The acos function is differentiable at all points in the open interval (−1,1), 
                    # but at exactly -1 and 1, the derivative is undefined (or goes to infinity), 
                    # which can lead to NaN values in gradients. Clamp to avoid numerical issues.
                    trace = torch.clamp(trace, -0.99999, 0.99999)  
                    ori_diff = torch.acos(trace)
                    # Use exponential decay on distances and angular differences to penalize similarity
                    pos_penalty = torch.exp(-pos_diff)
                    ori_penalty = torch.exp(-ori_diff)
                    # Sum penalties across all pairs; exclude self-comparisons by masking diagonal
                    mask = torch.eye(pos_penalty.size(1), device=pos_penalty.device).bool()
                    pos_penalty = pos_penalty.masked_fill(mask.unsqueeze(0), 0.)
                    ori_penalty = ori_penalty.masked_fill(mask.unsqueeze(0), 0.)
                    # Sum up the penalties
                    losses["total_penalty"] = (pos_penalty * ori_penalty).mean() * self.lambda_pose_sim
                
                t_loss_back = time.time()
                loss = sum([l for l in losses.values()])
                loss_mean = loss.mean()
                optim.zero_grad()
                loss_mean.backward()
                t_loss_back = time.time() - t_loss_back
                
                with torch.no_grad():
                    
                    # Calculate pairwise distance between position and orientation params
                    position_diff = torch.cdist(position_params, position_params)
                    orientations = torch.abs(orientation_params.unsqueeze(2) - orientation_params.unsqueeze(1)) # [b, num_proposals, num_proposals]
                    # Identify similar poses
                    similar_mask = (position_diff < self.prune_distance_thresh) & (orientations < self.prune_orientation_thresh)
                    similar_mask.diagonal(dim1=-1, dim2=-2).zero_()     # Set the self-similarities to False
                    # Choose the best pose among similar ones based on loss
                    losses_repeated = loss.unsqueeze(-1).repeat(1, 1, num_proposals)
                    better_pose_mask = losses_repeated > losses_repeated.transpose(1, 2)  # compare each pair of losses
                    # Final mask to prune: poses are similar and current pose has worse loss
                    prune_mask = similar_mask & better_pose_mask
                    
                    # best_k_losses = losses["loss_occlusion_center"] + losses["loss_occlusion_edges"] + losses["loss_depth"]
                    best_k_losses = losses.get("loss_occlusion_center", 0.) + losses.get("loss_occlusion_max", 0.) + losses.get("loss_depth", 0.)
                    best_k_losses = best_k_losses.masked_fill(prune_mask.any(dim=2), 999.)
                        
                    if self.visualize:
                        t_vis = time.time()
                        
                        print(f"[{i}] loss={loss_mean.detach().cpu().item():.3f} | {' '.join([k + '=' + f'{v.detach().mean().cpu().item():.3f}' for k, v in losses.items()])}")
                        
                        if self.visualization_steps is None or (self.visualization_steps and i in self.visualization_steps):
                            self._visualize(
                                vol, poses_new, projs_new, 
                                occl_imgs=rgb,
                                depth_imgs=(1 / depths_pred.clamp_min(z_near) - 1 / z_near) / (1/z_far - 1/z_near),
                                inv_imgs=invalid >= 0.5,
                                vis_volume=kwargs["vis_volume"],
                                # circles=torch.concat([circles, ray_samp], dim=1),
                                dxdy=(torch.tanh(position_params-position_params.grad)-torch.tanh(position_params)) * scale_xz * scheduler.get_last_lr()[0],
                                dg=(torch.tanh(orientation_params-orientation_params.grad)-torch.tanh(orientation_params)) * scale_beta * scheduler.get_last_lr()[0],
                                step=i,
                                losses=best_k_losses,
                                top_indices=gather_bottom_k_error(best_k_losses, poses_new, self.nv)[1],
                                imgs_right=self.imgs_right,
                            )
                        all_losses["total_loss"].append(loss_mean.detach().cpu().item())
                        all_losses["lr"].append(scheduler.get_last_lr()[0])
                        for loss_name, loss_val in losses.items():
                            all_losses[loss_name].append(loss_val.detach().mean().cpu().item())
                    
                    optim.step()
                    scheduler.step()
                    
                    resample_mask = loss > self.resample_loss_threshold
                    if resample_mask.any():
                        position_params[resample_mask] = torch.rand_like(position_params)[resample_mask] * 2 - 1
                        orientation_params[resample_mask] = torch.rand_like(orientation_params)[resample_mask] * 2 - 1
                    
                    if self.visualize:
                        t_vis = time.time() - t_vis
                        print(f"t_step={time.time() - t_step_start:.2f} | t_render={t_render:.2f} t_loss_back={t_loss_back:.2f} t_vis={t_vis:.2f}")
                    
            
        if self.visualize:
            if hasattr(self, "visualization_loss_plot_axis") and self.visualization_loss_plot_axis is not None:
                fig = None
                ax = self.visualization_loss_plot_axis
            else:
                fig, ax = plt.subplots(1, 1, figsize=(PAGE_WIDTH_INCHES, 4))
                
            for loss_name, loss_vals in all_losses.items():
                if loss_name != "lr":
                    label = {
                        "total_loss": "$L_{total}$",
                        "loss_depth": "$L_{d}$",
                        "loss_weights": "$L_{weights}$",
                        "err_xz_bounds": "$L_{xz}$",
                        "err_beta_bounds": "$L_{\\beta}$",
                        "loss_occlusion_center": "$L_{occl-center}$",
                        "loss_occlusion_edges": "$L_{occl-edges}$",
                        "loss_occlusion_max": "$L_{occl-max}$",
                    }.get(loss_name, loss_name)
                    
                    ax.plot(loss_vals, label=label, linewidth=0.5)
            ax.set(xlabel="Iteration", ylabel="Loss")
            ax_ = ax.twinx()
            ax_.plot(all_losses["lr"], label="LR", color="black", linestyle="dashed", linewidth=0.5)
            ax_.set_ylabel('LR', color="black")
            ax_.tick_params(axis='y', labelcolor="black")
            ax.grid(True)
            ax_.grid(False)
            ax.legend(loc="upper right", ncols=2)
            if fig is not None:
                fig.savefig(os.path.join(self.visualization_path, "losses.png"), bbox_inches='tight', pad_inches=0)

            image_files = [img for img in os.listdir(self.visualization_frames_path) if ".png" in img]
            image_files.sort(key=lambda s: int(re.findall(r'\d+', s)[0]))
            clip = moviepy.video.io.ImageSequenceClip.ImageSequenceClip(
                [os.path.join(self.visualization_frames_path, img) for img in image_files], fps=3
            )
            # Crop to an even resolution, since .mp4 is not playable not Mac otherwise.
            w, h = clip.size
            clip = crop(clip, width=w-w%2, height=h-h%2, x_center=w/2, y_center=h/2)
            clip.write_videofile(os.path.join(self.visualization_path, 'vid.mp4'))
            if self.visualize_gif:
                clip.write_gif(os.path.join(self.visualization_path, "vid.gif"))
            clip.close()
            print(f"Time: {time.time()-time_start:.2f}s")
            
        # Keep the best `self.nv` poses.
        poses_new, _ = gather_bottom_k_error(best_k_losses, poses_new, self.nv)

        return poses_new

    def _visualize(
        self, 
        net, poses, projs, 
        vis_volume: VisVolume, 
        depth_imgs=None, occl_imgs=None, inv_imgs=None,
        circles=None, dxdy=None, dg=None, step=None, top_indices=None, losses=None,
        imgs_right=True,
        **kwargs,
        ):
        set_thesis_rcparams()
        
        b, num_proposals, _, h, w = depth_imgs.shape
        n_rows = n_cols = int(math.sqrt(b))
        fig, axes = plt.subplots(n_rows, n_cols, figsize=(10, 10))
        
        depth_imgs = color_tensor(depth_imgs.squeeze(-3).cpu().float(), cmap=cmap_magma, norm=True)
        occl_imgs = occl_imgs.squeeze(-3).cpu().float().clamp(0, 1).unsqueeze(-1) * torch.tensor([0, 1, 0])
        occl_imgs[inv_imgs.squeeze(-3).cpu()] = torch.tensor([1, 1, 0]).float()
        
        profiles = render_profile(net, vis_volume, cam_incl_adjust=self.cam_incl_adjust.to(poses), mask_channel=-2, mode="mask")#mode="mask")
        # Loop over batch
        for i, profile in enumerate(profiles):
            ax = axes if isinstance(axes, plt.Axes) else axes[int(i / n_rows), int(i % n_rows)]
            plot_profile(
                ax, 
                profile.cpu().float(), 
                vis_volume, 
                poses=poses[i].cpu().float(), 
                projs=projs[i].cpu().float(),
                flip=True,
                linewidths=[4. if _ in top_indices[i] else 3. for _ in range(num_proposals)],
                linecolors=["cyan" if _ in top_indices[i] else "red" for _ in range(num_proposals)],
            )
            
            for j in range(num_proposals):
                x = poses[i, j, 0, 3].cpu().item()
                z = poses[i, j, 2, 3].cpu().item()
                ax.annotate(f"{losses[i, j].cpu().item():.3f}", xy=[x, z])
            
            ax.set(title=f"Sample {i} | Step={step}", xlabel="X [m]", ylabel="Z [m]", xlim=vis_volume.X_RANGE, ylim=vis_volume.Z_RANGE)
            if self.visualization_no_text:
                ax.set(title=None, xlabel=None, ylabel=None, xticks=[], yticks=[])
                set_spines(ax, visible=False)
            
            if imgs_right:
                depth_and_occl = torch.concat([
                    depth_imgs[i].view(-1, w, 3), occl_imgs[i].view(-1, w, 3)
                ], dim=1)
                ax_inset = ax.inset_axes([1.005, 0, 0.5, 1.])     # [x0, y0, w, h]
            else:
                depth_and_occl = torch.concat([
                    depth_imgs[i].permute(1, 0, 2, 3).reshape(h, -1, 3), occl_imgs[i].permute(1, 0, 2, 3).reshape(h, -1, 3)
                ], dim=0)
                ax_inset = ax.inset_axes([0, -.305, 1., 0.3])     # [x0, y0, w, h]
            ax_inset.imshow(depth_and_occl, interpolation="none")
            ax_inset.set(xticks=[], yticks=[], title=None if self.visualization_no_text else "Proposal\nDepths and Occlusions")
            set_spines(ax_inset, visible=False)
            if imgs_right:
                # Top proposal highlighting is only implemented for this mode.
                for top_idx in top_indices[i].unique().cpu():
                    ax_inset.add_patch(Rectangle(xy=(0, top_idx*h+3), width=3, height=h-4-3, edgecolor="none", facecolor="cyan", linewidth=2))

            if circles is not None:
                ax.scatter(x=circles[i, :, 0].cpu(), y=circles[i, :, 2].cpu(), color="blue", s=1.)
            
            if poses is not None and projs is not None:
                DXY_ARROW_SCALE = 10.0
                DG_ARROW_SCALE = 10.0
                for j in range(poses.size(1)):
                    x = poses[i, j, 0, 3].cpu().item()
                    y = poses[i, j, 2, 3].cpu().item()
                    if dxdy is not None:
                        dx = dxdy[i, j, 0].cpu().item() * DXY_ARROW_SCALE
                        dy = dxdy[i, j, 1].cpu().item() * DXY_ARROW_SCALE
                        if dx != 0. or dy != 0.:
                            ax.arrow(x=x, y=y, dx=dx, dy=dy, width=0.1, alpha=0.5, color="black")
                            
                    if dg is not None:
                        dg_ = dg[i, j].cpu().item()
                        if dg_ != 0.:
                            left = dg_ > 0
                            diameter = 2.0
                            angle = abs(dg_ * 180 / math.pi) * DG_ARROW_SCALE
                            startangle = 0. if left else angle 
                            startarrow = not left#False
                            endarrow = left#True
                            linewidth = 1.

                            # https://stackoverflow.com/questions/44526103/draw-curved-arrow-that-looks-just-like-pyplot-arrow
                            arc = Arc([x, y], diameter, diameter, angle=0., theta1=0,theta2=angle,linestyle="-",color=kwargs.get("color","black"), linewidth=linewidth)
                            ax.add_patch(arc)
                            
                            head_length = 1.5*3*linewidth
                            if startarrow:
                                startX=diameter/2*np.cos(np.radians(startangle)) + x
                                startY=diameter/2*np.sin(np.radians(startangle)) + y
                                startDX=+.000001*diameter/2*np.sin(np.radians(startangle) + head_length)
                                startDY=-.000001*diameter/2*np.cos(np.radians(startangle) + head_length)
                                ax.arrow(startX-startDX, startY-startDY, startDX, startDY, width=0.1, alpha=0.5, color="black")

                            if endarrow:
                                endX=diameter/2*np.cos(np.radians(startangle+angle)) + x
                                endY=diameter/2*np.sin(np.radians(startangle+angle)) + y
                                endDX=-.000001*diameter/2*np.sin(np.radians(startangle+angle) - head_length)
                                endDY=+.000001*diameter/2*np.cos(np.radians(startangle+angle) - head_length)
                                ax.arrow(endX-endDX, endY-endDY, endDX, endDY, width=0.1, alpha=0.5, color="black")
                
        # fig.tight_layout()
        fig.canvas.draw()
        
        filepath = os.path.join(self.visualization_frames_path, f"{step:04d}.png")
        if not os.path.exists(os.path.dirname(filepath)):
            os.makedirs(os.path.dirname(filepath), exist_ok=True)
        # The saving op is the most time-expensive here
        fig.savefig(filepath, bbox_inches='tight', pad_inches=0)
    
        plt.close(fig)