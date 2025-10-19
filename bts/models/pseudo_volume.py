"""
Main model implementation
"""

from typing import Literal

import torch
import torch.autograd.profiler as profiler
import torch.nn.functional as F
from .base_model import BaseModel
from utils.cameras.pinhole import (
    outside_frustum_diff,
    project_to_image,
    pts_into_camera_or_world,
)


class PseudoVolume(BaseModel):
    def __init__(
        self, 
        color_sampling_mode: Literal["mean", "mean_valid", "mean_valid_surface"] = "mean",
        surface_thresh: float = 1.0,
        density_value: float = 100.0,
        transition_sharpness: float = 100_000.,
        color_interpolation: str = "bilinear",
        use_optimal_diff_ops: bool = False,
        num_input_anchors: int = 1,
        set_only_occlusions_empty: bool = True,
        density_downscale_factor: float = 1.0,
        ):
        super().__init__()

        self._color_interpolation = color_interpolation
        self._color_sampling_mode = color_sampling_mode
        self._surface_thresh = surface_thresh
        self._density_value = density_value
        self._num_input_imgs = num_input_anchors
        self._scale = 0
        self._transition_sharpness = transition_sharpness
        self.use_optimal_diff_ops = use_optimal_diff_ops
        self.density_downscale_factor = density_downscale_factor    
        
        self.set_only_occlusions_empty = set_only_occlusions_empty

    def set_scale(self, scale):
        self._scale = scale

    def get_scale(self):
        return self._scale
    
    def compute_grid_transforms(self, *args, **kwargs):
        pass

    def encode(self, depths, imgs, Ks, poses_c2w, features=None, *args, **kwargs):
        """
        Args:
            images: [n, nv, 3, h, w]
            depths: [n, nv, 1, h, w] along Z axis.
        """

        # TODO keep in mind if a clone is needed
        imgs_drgbf = depths
        if imgs is not None:
            imgs_drgbf = torch.concat([imgs_drgbf, imgs], dim=2)
        if features is not None:
            imgs_drgbf = torch.concat([imgs_drgbf, features], dim=2)
        
        self.drgbf = imgs_drgbf
        self.Ks = Ks
        # Needs to be float32 in case of autocast    # TODO check
        self.poses_w2c = torch.inverse(poses_c2w.float()).to(poses_c2w.dtype)

    def sample_colors_and_density(self, xyz):
        """
        Args:
            xyz: [batch, num points, 3]
        """
        n, n_pts, _ = xyz.shape
        n, nv, c, h, w = self.drgbf.shape
        
        self.xyz = xyz
        
        xyz_projected = pts_into_camera_or_world(
            xyz, self.poses_w2c
        )  # [B, n_views, n_pts, 3]
        xy, z = project_to_image(xyz_projected, self.Ks)
        invalid = outside_frustum_diff(xy, z, transition_sharpness=self._transition_sharpness)
        xy = xy.clamp(-2, 2)  # For numerical stability with AMP

        sampled_drgbf = F.grid_sample(
            self.drgbf.view(n * nv, c, h, w), 
            xy.reshape(n * nv, 1, -1, 2), 
            mode="bilinear", 
            # *Might* not make a difference if this is "zeros" or "border", since
            # we force all densities outside any frustum to 0.
            padding_mode="border",
            align_corners=True,
        ).view(n, nv, c, n_pts).permute(0, 1, 3, 2)  # [n, nv, pts, c]

        # Negative = behind the surface, positive = in front of the surface
        sdf = sampled_drgbf[..., :1] - z
        is_empty_prob = F.sigmoid(self._transition_sharpness * sdf)
        
        # TODO find better place for this
        set_only_occlusions_empty = self.set_only_occlusions_empty and c > 1
        
        if set_only_occlusions_empty:
            # Expecting the occlusion mask to be the last feature channel
            occl_mask_c = -1
            occl_thresh = .5
            # Occluded pixels are valid
            is_occluded = F.sigmoid(self._transition_sharpness * (sampled_drgbf[..., occl_mask_c:] - occl_thresh))
            # Inputs pixels are always valid
            is_input = torch.concat([
                torch.ones_like(sampled_drgbf[:, :self._num_input_imgs, :, :1]), 
                torch.zeros_like(sampled_drgbf[:, :nv-self._num_input_imgs, :, :1]),
            ], dim=1)
            is_input_or_occluded = torch.maximum(is_occluded, is_input)
            is_empty_prob = is_empty_prob * is_input_or_occluded
            
        if not self.use_optimal_diff_ops:
            # A soft invalid ensures that samples on the decision boundary
            # (where invalid equals 0.5) are treated as valid, i.e. invalid < 0.5.
            # When setting this threshold to .51, some rendered pixels at the frustum 
            # edges lack content, but are not invalid. Thus, the threshold is set to .499
            # to "keep the boundary-to-invalid inside the frustum".
            soft_invalid = F.sigmoid(self._transition_sharpness * (invalid - .499))
        else:
            soft_invalid = invalid
        # A point is occupied if no camera can see it.
        is_empty_valid_prob = is_empty_prob * (1 - soft_invalid)
        is_occupied_valid_prob = 1 - is_empty_valid_prob.max(dim=1)[0]
        # Points in invalid regions (not in any frustum) should be empty.
        is_occupied_valid_prob = is_occupied_valid_prob * (1 - soft_invalid[:, :self._num_input_imgs].min(dim=1)[0])

        # TODO add some arg for this
        cs_idx_end = c
        
        # TODO use mean of valid and non-occluded views
        match self._color_sampling_mode:
            case "mean":
                # Mean across all cameras. 
                # This leads to "edge-bleeding" artefacts for cameras with non-overlapping frustums.
                sampled_rgb = sampled_drgbf[..., 1:cs_idx_end].mean(dim=1)
            case "mean_valid":
                # Mean across all valid cameras.
                # valid_mask = ~invalid         # [n, nv, pts, 1]
                valid_mask = 1 - invalid         # [n, nv, pts, 1]
                if set_only_occlusions_empty:
                    valid_mask = valid_mask * is_input_or_occluded
                valid_rgb = sampled_drgbf[..., 1:cs_idx_end] * valid_mask   # Zero out invalid samples
                sampled_rgb = valid_rgb.sum(dim=1) / valid_mask.sum(dim=1).clamp_min(1)
                # TODO colors for samples that are invalid in all cameras are 0 right now
                # Maybe use the mean accross all cameras fro them
                
            case "mean_valid_surface":
                # Color of an empty point is the mean of all unoccluded cameras seeing it.
                valid_mask = 1 - invalid         # [n, nv, pts, 1]
                if set_only_occlusions_empty:
                    valid_mask = valid_mask * is_input_or_occluded
                valid_rgb = sampled_drgbf[..., 1:cs_idx_end] * valid_mask         # [n, nv, pts, 3]
                valid_rgb = valid_rgb.sum(dim=1) / valid_mask.sum(dim=1).clamp_min(1)         # [n, pts, 3]
                
                # Color of occluded points is the mean of the N cameras with the smallest occlusion depth.
                # Determine distance behind the depth surface for fully occluded points
                # and ignore the respective invalid camera frustums 
                dist_to_surface = (z - sampled_drgbf[..., :1])         # [n, nv, pts, 1]
                surface_valid_mask = torch.logical_and(dist_to_surface >= 0, dist_to_surface < self._surface_thresh) * valid_mask
                surface_valid_rgb = sampled_drgbf[..., 1:cs_idx_end] * surface_valid_mask
                surface_valid_rgb = (surface_valid_rgb.sum(dim=1) / surface_valid_mask.sum(dim=1).clamp_min(1))
                
                input_nv_mask = torch.tensor(
                    [True] * self._num_input_imgs + [False] * (nv-self._num_input_imgs), dtype=xyz.dtype, device=xyz.device
                )[None, :, None, None].expand(sampled_drgbf[..., 1:cs_idx_end].shape)
                input_nv_mask = torch.logical_and(input_nv_mask, surface_valid_mask)
                input_nv_rgb = sampled_drgbf[..., 1:cs_idx_end] * input_nv_mask
                input_nv_rgb = input_nv_rgb.sum(dim=1) / input_nv_mask.sum(dim=1).clamp_min(1)
                
                sampled_rgb = (
                    # All points a valid mean
                    valid_rgb + 
                    # Update points near the surface
                    surface_valid_mask.any(dim=1) * (surface_valid_rgb - valid_rgb)
                    
                )
                # Update points visible in input imgs
                sampled_rgb += input_nv_mask.any(dim=1) * (input_nv_rgb - sampled_rgb)
            case _:
                raise ValueError(f"Color sampling mode {self._color_sampling_mode} is not available.")
            
        # If xyz is invalid in all cameras.
        invalid = (invalid > 0.5).all(dim=1)
        sampled_densities = torch.ones((n, n_pts, 1), device=xyz.device, dtype=xyz.dtype) * -self._density_value
        sampled_densities = sampled_densities + is_occupied_valid_prob * self._density_value * 2

        sdf_in = None
        if self._num_input_imgs > 0:
            sdf_in = sdf[:, :self._num_input_imgs].max(dim=1).values
        sdf_nv = None
        if nv > self._num_input_imgs:
            sdf_nv = sdf[:, self._num_input_imgs:].max(dim=1).values

        return sampled_rgb, sampled_densities, invalid, sdf_in, sdf_nv

    def forward(self, xyz, downscale_sigmas: bool = False, **kwargs):
        """
        Predict (r, g, b, sigma) at world space points xyz.
        Please call encode first!
        :param xyz (B, 3) B is batch of points (in rays)
        :param scale_sigma: whether to scale sigma by the density_scale_factor
        :return (B, 4) r g b sigma
        """

        with profiler.record_function("pseudo_volume_inference"):
            rgb, density, invalid, sdf_in, sdf_nv = self.sample_colors_and_density(xyz)  # (n, nv, pts, 3)

            # TODO remember I added this and check if it causes problems
            # https://pytorch.org/docs/stable/generated/torch.nn.Softplus.html
            sigma = F.softplus(density).to(density.dtype)

            if downscale_sigmas:
                # Downscale the densities for example for use as supervision in a loss.
                sigma = sigma / self.density_downscale_factor

        state_dict = {}
        if sdf_in is not None:
            state_dict["sdf_in"] = sdf_in
        if sdf_nv is not None:
            state_dict["sdf_nv"] = sdf_nv

        return rgb, invalid, sigma, state_dict
