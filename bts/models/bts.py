import torch
import torch.nn as nn
import torch.autograd.profiler as profiler
from torch.cuda.amp import autocast

from utils.cameras.pinhole import (
    outside_frustum,
    project_to_image,
    pts_into_camera_or_world,
)
from utils.cameras.pinhole import EPS
from bts.common.positional_encoding import encoding_mode
from bts.models.base_model import BaseModel
from bts.models.backbones.backbone_util import make_backbone
from bts.common.positional_encoding import PositionalEncoding
from bts.models.prediction_heads import make_head, make_mlp

"""
Main model implementation
"""

import torch.autograd.profiler as profiler
import torch.nn.functional as F
from torch import nn


EPS = 1e-3




class BTSNet(BaseModel):
    def __init__(
        self,
        conf,
        final_pred_head: str | None = None,
        ren_nc=None,
    ):
        super().__init__()
        
        self.config = conf
        d_out = 1 if conf.get("sample_color", True) else 4
        self.encoder = make_backbone(conf["encoder"])
        self.code_xyz = PositionalEncoding.from_conf(conf["code"], d_in=3)
        d_in = self.encoder.latent_size + self.code_xyz.d_out
        heads = {
                head_conf["name"]: make_head(head_conf, d_in, d_out)
                for head_conf in conf["DECODER_HEADS"]
        }
        self.heads = nn.ModuleDict(heads)
        
        # self.mlp_coarse = make_mlp(conf["mlp_coarse"], d_in, d_out=d_out)
        # self.mlp_fine = make_mlp(conf["mlp_fine"], d_in, d_out=d_out, allow_empty=True)
        
        if final_pred_head:
            self.final_pred_head = final_pred_head
        else:
            self.final_pred_head = list(self.heads.keys())[0]

        self.requires_bottleneck_feats = False

        # for _, head in self.heads.items():
        #     if hasattr(head, "require_bottleneck_feats"):
        #         if head.require_bottleneck_feats and (
        #             head.independent_token_net.__class__.__name__
        #             == "NeuRayIndependentToken"
        #         ):  ## For read out token type: "NeuRayIndependentToken"
        #             self.requires_bottleneck_feats = True
        #             break
        self.use_viewdirs = conf.get("use_viewdirs", True)

        # TODO: remove hardcoded value
        self.n_coarse = 64

        # TODO: figure out how to pass z_near and z_far to the model, probably outside with the positional encoding
        self.d_min, self.d_max = conf.get("z_near", 3), conf.get("z_far", 80)
        self.learn_empty, self.empty_empty, self.inv_z = (
            conf.get("learn_empty", True),
            conf.get("empty_empty", False),
            conf.get("inv_z", True),
        )
        self.color_interpolation = conf.get("color_interpolation", "bilinear")

        # TODO: rethink encoding mode
        self.encoding_mode = encoding_mode(
            conf.get("code_mode", "z"), self.d_min, self.d_max, self.inv_z, EPS
        )

        self.flip_augmentation = conf.get("flip_augmentation", False)
        self.return_sample_depth = conf.get("return_sample_depth", False)
        self.sample_color = conf.get("sample_color", True)

        # TODO: manage _d_out in another way
        d_in = self.encoder.latent_size + self.code_xyz.d_out  ### 64 + 39
        d_out = 1 if self.sample_color else 4

        self._d_in, self._d_out = d_in, d_out

        if self.learn_empty:
            self.empty_feature = nn.Parameter(
                torch.randn((self.encoder.latent_size,), requires_grad=True)
            )
        self._scale = 0  ## set spatial resolution size accoridng to the scale of output feature map from the encoder

    def set_scale(self, scale):
        self._scale = scale

    def get_scale(self):
        return self._scale

    def compute_grid_transforms(self, *args, **kwargs):
        pass

    def encode(
        self,
        images,
        Ks,
        poses_c2w,
        ids_encoder=None,
        ids_render=None,
        images_alt=None,
        combine_ids=None,
    ):
        poses_w2c = torch.inverse(poses_c2w.float()).to(poses_c2w.dtype)

        if ids_encoder is None:
            images_encoder = images
            Ks_encoder = Ks
            poses_w2c_encoder = poses_w2c
            ids_encoder = list(range(len(images)))
        else:
            images_encoder = images[:, ids_encoder]
            Ks_encoder = Ks[:, ids_encoder]
            poses_w2c_encoder = poses_w2c[:, ids_encoder]

        # TODO: Why?
        if images_alt is not None:
            images = images_alt
        else:
            images = images * 0.5 + 0.5

        if ids_render is None:
            images_render = images
            Ks_render = Ks
            poses_w2c_render = poses_w2c
            ids_render = list(range(len(images)))
        else:
            images_render = images[:, ids_render]
            Ks_render = Ks[:, ids_render]
            poses_w2c_render = poses_w2c[:, ids_render]

        if combine_ids is not None:
            combine_ids = list(list(group) for group in combine_ids)
            get_combined = set(sum(combine_ids, []))
            for i in range(images.shape[1]):
                if i not in get_combined:
                    combine_ids.append((i,))
            remap_encoder = {v: i for i, v in enumerate(ids_encoder)}
            remap_render = {v: i for i, v in enumerate(ids_render)}
            comb_encoder = [
                [remap_encoder[i] for i in group if i in ids_encoder]
                for group in combine_ids
            ]
            comb_render = [
                [remap_render[i] for i in group if i in ids_render]
                for group in combine_ids
            ]
            comb_encoder = [group for group in comb_encoder if len(group) > 0]
            comb_render = [group for group in comb_render if len(group) > 0]
        else:
            comb_encoder = None
            comb_render = None
        ## Note: This is yet to be feature map before passing img to encoder
        n_, nv_, c_, h_, w_ = images_encoder.shape  ### [n_, nv_, 3:=RGB, 192, 640]
        c_l = self.encoder.latent_size  ### 64 c.f. paper D.1.

        if self.flip_augmentation and self.training:  ## data augmentation for color
            do_flip = (torch.rand(1) > 0.5).item()
        else:
            do_flip = False

        if do_flip:
            images_encoder = torch.flip(images_encoder, dims=(-1,))

        image_latents_ms = self.encoder(images_encoder.view(n_ * nv_, c_, h_, w_))
        
        if any([torch.isnan(_).any() for _ in image_latents_ms]):
            print(f"NaN in encoder output ({image_latents_ms[0].dtype}).")
        if any([torch.isinf(_).any() for _ in image_latents_ms]):
            print(f"inf in image_latents_ms ({image_latents_ms[0].dtype})")
            if image_latents_ms[0].dtype == torch.float16:
                image_latents_ms = [_.clamp(torch.finfo(torch.float16).min, torch.finfo(torch.float16).max) for _ in image_latents_ms]

        if do_flip:
            image_latents_ms = [torch.flip(il, dims=(-1,)) for il in image_latents_ms]

        _, _, h_, w_ = image_latents_ms[
            0
        ].shape  ## get spatial resol from 1st layer out of 4 from feature maps generated by Enc
        image_latents_ms = [
            nn.functional.interpolate(image_latents, size=(h_, w_)).view(
                n_, nv_, c_l, h_, w_
            )
            for image_latents in image_latents_ms
        ]  ## upsampling the feature maps from down-sampled 4 layers to the same spatial resolution of 1st layer
        # img_feat_ms = [nn.functional.interpolate(feat_latents, size=(h_, w_)).view(n_, nv_, img_feat_ms[-1].shape[1], h_, w_) for feat_latents in img_feat_ms]    ## upsampling the feature maps from down-sampled 4 layers to the same spatial resolution of 1st layer

        ## feature
        self.grid_f_features = image_latents_ms
        self.grid_f_Ks = Ks_encoder
        self.grid_f_poses_w2c = poses_w2c_encoder
        self.grid_f_combine = comb_encoder

        ## color
        self.grid_c_imgs = images_render
        self.grid_c_Ks = Ks_render
        self.grid_c_poses_w2c = poses_w2c_render
        self.grid_c_combine = comb_render

    def sample_features(
        self,
        xyz,
        # use_single_featuremap=True
    ):
        ## Get the shape of the input point cloud and the feature grid (n, pts, spatial_coordinate == 3)
        B, n_pts, _ = xyz.shape
        B, n_views, c_, h_, w_ = self.grid_f_features[
            self._scale
        ].shape  # [B, n_views, C, H, W]

        # with autocast(enabled=False):
        xyz_projected = pts_into_camera_or_world(
            xyz.float(), self.grid_f_poses_w2c.float()
        )  # [B, n_views, n_pts, 3]
        distance = torch.norm(xyz_projected, dim=-2, keepdim=True)
        xy, z = project_to_image(xyz_projected, self.grid_f_Ks)
        invalid = outside_frustum(xy, z)

        # For numerical stability with AMP. Should not affect training outcome
        xy = xy.clamp(-2, 2)

        """given a vector p = (x, y, z) this is the difference of normalizing either:z ||p|| = sqrt(x^2 + y^2 + z^2). So you either give the network (x, y, z_normalized) or (x, y, ||p||_normalized) as input. It is just different parameterizations of the same point."""
        xyz_code = self.code_xyz(
            self.encoding_mode(xy, z, distance).view(B * n_views * n_pts, -1)
        ).view(B, n_views, n_pts, -1)

        # These samples are from different scales
        sampled_features = (
            nn.functional.grid_sample(
                self.grid_f_features[self._scale].view(B * n_views, c_, h_, w_),
                xy.view(B * n_views, 1, -1, 2),
                mode="bilinear",
                padding_mode="border",
                align_corners=False,
            )
            .view(B, n_views, c_, n_pts)
            .permute(0, 1, 3, 2)
        )  ## set x,y coordinates as grid feature

        if self.learn_empty:
            ## "empty space" can refer to areas in a scene where there is no object, or it could also refer to areas that are not observed or are beyond the range of the sensor. This allows the model to have a distinct learned representation for "empty" space, which can be beneficial in tasks like 3D reconstruction where understanding both the objects in a scene and the empty space between them is important.
            ## Replace invalid features in the sampled features tensor with the corresponding features from the expanded empty feature
            empty_feature_expanded = self.empty_feature.view(1, 1, 1, c_).expand(
                B, n_views, n_pts, c_
            )  ## trainable parameter, initialized with random features
            sampled_features[invalid.expand(-1, -1, -1, c_)] = empty_feature_expanded[
                invalid.expand(-1, -1, -1, c_)
            ]  ## broadcasting and make it fit to feature map

        sampled_features = torch.cat(
            (sampled_features, xyz_code), dim=-1
        )  # [B, n_views, n_pts, C+C_pos_emb]

        return (
            sampled_features.permute(0, 2, 1, 3),
            invalid[..., 0].permute(0, 2, 1),
        )

    def sample_colors(self, xyz):
        n_, n_pts, _ = xyz.shape  ## n := batch size, n_pts := #_points in world coord.
        n_, nv_, c_, h_, w_ = self.grid_c_imgs.shape  ## nv_ := #_views

        xyz_projected = pts_into_camera_or_world(
            xyz, self.grid_c_poses_w2c
        )  # [B, n_views, n_pts, 3]
        distance = torch.norm(xyz_projected, dim=-2, keepdim=True)

        xy, z = project_to_image(xyz_projected, self.grid_c_Ks)

        # For numerical stability with AMP. Should not affect training outcome.
        xy = xy.clamp(-2, 2)

        invalid = outside_frustum(xy, z)

        sampled_colors = (
            nn.functional.grid_sample(
                self.grid_c_imgs.view(n_ * nv_, c_, h_, w_),
                xy.view(n_ * nv_, 1, -1, 2),
                mode=self.color_interpolation,
                padding_mode="border",
                align_corners=False,
            )
            .view(n_, nv_, c_, n_pts)
            .permute(0, 1, 3, 2)
        )  ## Sample colors from the grid using the projected world coordinates.

        assert not torch.any(
            torch.isnan(sampled_colors)
        )  ## Check that there are no NaN values in the sampled colors tensor.

        if (
            self.grid_c_combine is not None
        ):  ## If self.grid_c_combine is not None, combine colors from multiple points in the same group.
            invalid_groups, sampled_colors_groups = [], []

            for (
                group
            ) in (
                self.grid_c_combine
            ):  ## group:=list of indices that correspond to a subset of the total set of points in the point cloud. These subsets are combined to create a single image of the entire point cloud from multiple views.
                if (
                    len(group) == 1
                ):  ## If the group contains only one point, append the corresponding invalid tensor and sampled colors tensor to the respective lists.
                    invalid_groups.append(invalid[:, group])
                    sampled_colors_groups.append(sampled_colors[:, group])
                    continue

                invalid_to_combine = invalid[
                    :, group
                ]  ## Otherwise, combine colors from the group by picking the color of the first valid point in the group.
                colors_to_combine = sampled_colors[:, group]

                indices = torch.min(invalid_to_combine, dim=1, keepdim=True)[
                    1
                ]  ## Get the index of the first valid point in the group.
                invalid_picked = torch.gather(
                    invalid_to_combine, dim=1, index=indices
                )  ## Pick the invalid tensor and sampled colors tensor corresponding to the first valid point in the group.
                colors_picked = torch.gather(
                    colors_to_combine,
                    dim=1,
                    index=indices.expand(-1, -1, -1, colors_to_combine.shape[-1]),
                )

                invalid_groups.append(
                    invalid_picked
                )  ## Append the picked invalid tensor and sampled colors tensor to the respective lists.
                sampled_colors_groups.append(colors_picked)

            invalid = torch.cat(
                invalid_groups, dim=1
            )  ## Concatenate the invalid tensors and sampled colors tensors along the second dimension.
            sampled_colors = torch.cat(sampled_colors_groups, dim=1)

        if (
            self.return_sample_depth
        ):  ## If self.return_sample_depth is True, concatenate the sample depth to the sampled colors tensor.
            distance = distance.view(n_, nv_, n_pts, 1)
            sampled_colors = torch.cat(
                (sampled_colors, distance), dim=-1
            )  ## cat along the last elem (c.f. paper pipeline)

        return (
            sampled_colors,
            invalid,
        )  ## Return the sampled colors tensor and the invalid tensor.

    def forward(self, xyz: torch.Tensor, **kwargs):
        # context manager that helps to measure the execution time of the code block inside it. i.e. used to profile the execution time of the forward pass of the model during inference for performance analysis and optimization purposes. ## to analyze the performance of the code block, helping developers identify bottlenecks and optimize their code.
        with profiler.record_function(
            "model_inference"
        ):  ## create object with the name "model_inference". ## stop the timer when exiting the block
            only_density = kwargs.get("only_density", False)
            n_, n_pts, _ = xyz.shape  ## n_ := Batch_size, n_pts == M
            nv_ = self.grid_c_imgs.shape[1]  ## 4 == (stereo 2 + side fish eye cam 2)

            if self.grid_c_combine is not None:
                nv_ = len(self.grid_c_combine)

            (
                sampled_features,
                invalid_features,
            ) = self.sample_features(
                xyz,
                # use_single_featuremap=False,
            )

            mlp_input = sampled_features.flatten(0, 1)  # (B * n_pts, n_views, C)

            # Camera frustum culling stuff, currently disabled
            combine_index, dim_size = None, None

            kwargs = {
                "invalid_features": invalid_features.flatten(
                    0, 1
                ),  # (B* n_pts, n_views)
                "combine_inner_dims": (n_pts,),
                "combine_index": combine_index,
                "dim_size": dim_size,
            }

            head_outputs = {
                name: head(mlp_input, **{**kwargs, "head_name": name}).reshape(
                    n_, -1, self._d_out
                )
                for name, head in self.heads.items()
            }

            mlp_output = head_outputs[self.final_pred_head]
            # mlp_output = self.mlp_coarse(mlp_input)

            if self.sample_color:
                sigma = mlp_output[..., :1]
                sigma = nn.functional.softplus(sigma)
                rgb, invalid_colors = self.sample_colors(xyz)  # (n, nv_, pts, 3)
            else:  ## RGB colors and invalid colors are computed directly from the mlp_output tensor. i.e. w/o calling sample_colors(xyz)
                sigma = mlp_output[..., :1]
                sigma = nn.functional.relu(sigma)
                rgb = mlp_output[..., 1:4].reshape(n_, 1, n_pts, 3)
                rgb = nn.functional.sigmoid(rgb)
                invalid_colors = invalid_features.unsqueeze(-2)
                nv_ = 1

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

            state_dict = {
                "invalid_features": invalid_features.flatten(0, 1)[None],
                # TODO: figure out state dict fusion, probably collate fn
                # "sigmas": head_outputs,
            }
        return rgb, invalid, sigma, state_dict