import torch
import torch.nn as nn
import torch.autograd.profiler as profiler
from utils.utils import asdict_lowercase_keys_override

from utils.cameras.pinhole import (
    outside_frustum,
    project_to_image,
    pts_into_camera_or_world,
    EPS
)
from bts.models.bts import BTSNet

import torch.nn.functional as F


class BTSDirect(BTSNet):
    def __init__(
        self,
        conf,
        # encoder: nn.Module,
        # code_xyz,
        # heads,
    ):
        super().__init__(conf, 
                        #  encoder, code_xyz, heads
                         )
        # FIXME do proper init
        self.heads = None
        self.sampled_density_lambda = conf.get("SAMPLED_DENSITY_LAMBDA", 1.0)
    
    @classmethod
    def from_conf(cls, conf, **kwargs):
        return cls(
            **asdict_lowercase_keys_override(conf),
            **kwargs
        )

    def sample_density(
        self,
        xyz,   
        # use_single_featuremap=True
    ):
        ## Get the shape of the input point cloud and the feature grid (n, pts, spatial_coordinate == 3)
        B, n_pts, _ = xyz.shape
        B, n_views, c_, h_, w_ = self.grid_f_features[
            self._scale
        ].shape  # [B, n_views, C, H, W]

        channels = 1
        if self.config.get("OUTPUT_UNCERTAINTY", False):
            channels = 2
            c_ //= 2

        # --- Map xyz to frustum coordinates ---
        xyz_projected = pts_into_camera_or_world(
            xyz, self.grid_f_poses_w2c
        )  # [B, n_views, n_pts, 3]
        # `torch.norm` upcasts to FP32 with AMP if dtype is not specified.
        distance = torch.norm(xyz_projected, dim=-2, keepdim=True, dtype=xyz_projected.dtype)
        xy, z = project_to_image(xyz_projected, self.grid_f_Ks)
        invalid = outside_frustum(xy, z)
        # Restrict to [-2, 2] for numerical stability when using AMP
        xyz = self.encoding_mode(xy, z, distance).view(B, n_views, n_pts, -1).clamp(-2, 2)

        # --- Sample densities from the predicted grid ---
        # These samples are from different scales
        sampled_density = F.grid_sample(
                self.grid_f_features[self._scale].view(B * n_views, channels, c_, h_, w_),
                xyz.view(B * n_views, 1, 1, -1, 3),
                mode="bilinear",
                padding_mode="border",      # TODO try padding mode 0
                # The extrema (-1 and 1) are the center points of the corner pixels.
                align_corners=False,
            ).view(B, n_views, channels, n_pts).permute(0, 1, 3, 2)
        
        # padding_space = outside_frustum(xy, z, limit_z=self.d_min)
        # sampled_density[padding_space] = -10    # padding val
        
        # if torch.isnan(sampled_density).any():
        #     print("NaN in sampled_density")
        #     pass

        return (
            sampled_density.permute(0, 2, 1, 3),   # (n, n_pts, nv, channels)
            invalid[..., 0].permute(0, 2, 1),
        )

    def forward(self, xyz: torch.Tensor, **kwargs):
        # context manager that helps to measure the execution time of the code block inside it. i.e. used to profile the execution time of the forward pass of the model during inference for performance analysis and optimization purposes. ## to analyze the performance of the code block, helping developers identify bottlenecks and optimize their code.
        with profiler.record_function(
            "model_inference"
        ):  ## create object with the name "model_inference". ## stop the timer when exiting the block

            # TODO: figure out state dict fusion, probably collate fn
            state_dict = {}

            only_density = kwargs.get("only_density", False)
            n_, n_pts, _ = xyz.shape  ## n_ := Batch_size, n_pts == M
            nv_ = self.grid_c_imgs.shape[1]  ## 4 == (stereo 2 + side fish eye cam 2)

            if self.grid_c_combine is not None:
                nv_ = len(self.grid_c_combine)

            sampled_density_and_uncertainty, invalid_features = self.sample_density(xyz)

            sampled_density = sampled_density_and_uncertainty[:, :, :, 0:1].reshape(n_, n_pts, 1)
            # `softplus` upcasts to FP32 with AMP to avoid overflows in the exponential operator.
            sigma = F.softplus(sampled_density * self.sampled_density_lambda).to(sampled_density.dtype)

            if self.config.get("OUTPUT_UNCERTAINTY", False):
                sampled_uncertainty = sampled_density_and_uncertainty[:, :, :, 1:2].reshape(n_, n_pts, 1)
                sampled_uncertainty = F.softplus(sampled_uncertainty).to(sampled_density.dtype)
                state_dict["uncertainties"] = sampled_uncertainty
            if self.config.get("OUTPUT_RAW", False):
                state_dict["densities_raw"] = sampled_density

            rgb, invalid_colors = self.sample_colors(xyz)  # (n, nv_, pts, 3)

            """Combine RGB colors and invalid colors"""
            if not only_density:
                _, _, _, c_ = rgb.shape
                rgb = rgb.permute(0, 2, 1, 3).reshape(
                    n_, n_pts, nv_ * c_
                )  # (n, pts, nv * 3)
                invalid_colors = invalid_colors.permute(0, 2, 1, 3).reshape(
                    n_, n_pts, nv_
                )

                invalid = (
                    invalid_colors | torch.all(invalid_features, dim=-1)[..., None]
                )
                invalid = invalid.to(rgb.dtype)
            else:
                rgb = torch.zeros((n_, n_pts, nv_ * 3), device=sigma.device)
                invalid = invalid_features.to(sigma.dtype)

            state_dict["invalid_features"] = invalid_features.flatten(0, 1)[None]

        return rgb, invalid, sigma, state_dict