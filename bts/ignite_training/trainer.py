import math
import logging

import lpips
import skimage.metrics
import torch
from torch import nn, optim, profiler
from torch.nn import functional as F
from torch.utils.data import DataLoader
import ignite.distributed as idist
from torch.utils.data import Subset

from datasets.data_util import make_datasets
from datasets.kitti_odom.kitti_odometry_dataset import KittiOdometryDataset
from datasets.kitti_raw.kitti_raw_dataset import KittiRawDataset
from bts.common.scheduler import make_scheduler
from bts.renderer import NeRFRenderer
from bts.common.image_processor import make_image_processor, RGBProcessor, DepthProcessor
from bts.losses.depth_loss import DepthReconstructionLoss
from bts.losses import make_losses
from bts.common.ray_sampler import ImageRaySampler, PatchRaySampler, RandomRaySampler
from bts.models import BTSNet, BTSDirect
from bts.gt_synthesis.gt_synthesis import GTSynthesisWrapper, Outputs, Data
from bts.ignite_training.base_trainer import base_training
from utils.metrics import MeanMetric
from bts.losses.utils import compute_depth_metrics
from utils.projection_ops import distance_to_z
from configs.structured_configs.main_config import MainConfig
from configs.structured_configs.bts_config import BTSConfig
from utils.plotting import render_profile
from bts.ignite_training.visualize import visualize
from bts.losses.density_grid_regul import DensityGridRegularizationLoss
from bts.losses.density_grid_loss import DensityGridLoss
from bts.losses.depth_loss import DepthReconstructionLoss
from bts.losses.invalid_policy import InvalidPolicy
from utils.cameras.pinhole import unproject_from_image
from bts.gt_synthesis.cascade_wrapper import CascadeWrapper
from utils.utils import asdict_lowercase_keys_override
from bts.models import make_model

class BTSWrapper(nn.Module):
    def __init__(self, renderer: NeRFRenderer, gt_synthesizer_or_cascade_wrapper: GTSynthesisWrapper | CascadeWrapper, config: MainConfig, eval_nvs: bool = False) -> None:      
        super().__init__()
        
        self.config = config
        # self.config = asdict(config.BTS)["MODEL_CONF"]

        self.renderer = renderer
        
        self.synthetic_gt = True
        
        self.gt_synthesizer = None
        self.cascade_wrapper = None
        if isinstance(gt_synthesizer_or_cascade_wrapper, GTSynthesisWrapper):
            self.gt_synthesizer = gt_synthesizer_or_cascade_wrapper
        elif isinstance(gt_synthesizer_or_cascade_wrapper, CascadeWrapper):
            self.cascade_wrapper = gt_synthesizer_or_cascade_wrapper
        else:
            raise ValueError("gt_synthesizer_or_cascade_wrapper must be an instance of GTSynthesisWrapper or CascadeWrapper")

        self.z_near = config.BTS.MODEL_CONF.z_near
        self.z_far = config.BTS.MODEL_CONF.z_far
        self.ray_batch_size = config.BTS.MODEL_CONF.ray_batch_size
        frames_render = config.BTS.MODEL_CONF.n_frames_render#", 2)
        self.frame_sample_mode = config.BTS.MODEL_CONF.frame_sample_mode#", "default")
        self.loss_from_single_img = False#config.BTS.MODEL_CONF.loss_from_single_img#", False)
        
        self.sample_mode = config.BTS.MODEL_CONF.sample_mode#", "random")
        self.patch_size = config.BTS.MODEL_CONF.patch_size#", 16)
        self.use_scales = False#config.BTS.MODEL_CONF.use_scales#", False)
        self.use_automasking = False#config.BTS.MODEL_CONF.use_automasking#", False)

        self.prediction_mode = config.BTS.MODEL_CONF.prediction_mode#", "multiscale")

        self.alternating_ratio = None#config.BTS.MODEL_CONF.alternating_ratio#", None)

        cfg_ip = {}#config.BTS.MODEL_CONF.image_processor#", {})
        self.train_image_processor = make_image_processor(cfg_ip)
        self.val_image_processor = RGBProcessor()
        self.depth_image_processor = DepthProcessor()       # TODO needed?

        if type(frames_render) == int:
            self.frames_render = list(range(frames_render))
        else:
            self.frames_render = frames_render
        self.frames = self.frames_render

        if self.sample_mode == "random":
            self.train_sampler = RandomRaySampler(self.ray_batch_size, self.z_near, self.z_far, channels=self.train_image_processor.channels)
        elif self.sample_mode == "patch":
            self.train_sampler = PatchRaySampler(self.ray_batch_size, self.z_near, self.z_far, self.patch_size, channels=self.train_image_processor.channels)
        elif self.sample_mode == "image":
            # self.train_sampler = ImageRaySampler(self.z_near, self.z_far, channels=self.train_image_processor.channels)
            self.train_sampler = ImageRaySampler(self.z_near, self.z_far, channels=1)       # TODO add arg
        else:
            raise NotImplementedError
        
        self.train_sampler_in = ImageRaySampler(self.z_near, self.z_far, channels=1)        # TOFO arg

        if self.use_automasking:
            self.train_sampler.channels += 1

        # self.val_sampler = ImageRaySampler(self.z_near, self.z_far)
        self.val_sampler = ImageRaySampler(self.z_near, self.z_far, channels=1)
        self.val_sampler_in = ImageRaySampler(self.z_near, self.z_far, channels=1)

        self.eval_nvs = eval_nvs
        if self.eval_nvs:
            self.lpips = lpips.LPIPS(net="alex")

        self._counter = 0
        
        self.save_predicted_density = False
        self.save_synthetic_density = False
        self.save_rgbd_images = False
        for _, loss_cfg in self.config.BTS.LOSSES.items():
            if not hasattr(loss_cfg, "TYPE"):
                # TODO fix this problem: if a child config removes a loss via override, but the parent sets fields, this exists
                continue
            match loss_cfg.TYPE:
                case DensityGridRegularizationLoss.__name__:
                    self.save_predicted_density = True
                case DensityGridLoss.__name__:
                    self.save_predicted_density = True
                    self.save_synthetic_density = True
                case DepthReconstructionLoss.__name__:
                    # TODO possibly save only input images and not novel views.
                    self.save_rgbd_images = True
        
        
    def to(self, *args, **kwargs):
        """Invokes .to() of submodules."""
        super().to(*args, **kwargs)
        if self.gt_synthesizer is not None:
            self.gt_synthesizer.to(*args, **kwargs)
        if self.cascade_wrapper is not None:
            self.cascade_wrapper.to(*args, **kwargs)
        return self

    # @staticmethod
    # def get_loss_metric_names():
    #     return ["loss", "loss_l2", "loss_mask", "loss_temporal"]
    
    def forward(self, data):
        data = dict(data)
        images = torch.stack(data["imgs"], dim=1)                           # n, v, c, h, w
        n, v, c, h, w = images.shape
        poses = torch.stack(data["poses"], dim=1)                           # n, v, 4, 4 w2c           # TODO isnt this c2w?
        projs = torch.stack(data["projs"], dim=1)                           # n, v, 4, 4 (-1, 1)
        hw_unnorm = data.get("hw_unnorm", [[h, w]])[0]
        
        if v > 1:
            to_base_pose = torch.inverse(poses[:, :1, :, :])    # Use first frame as keyframe
            poses = to_base_pose.expand(-1, v, -1, -1) @ poses
        else:
            poses = torch.eye(4, device=poses.device, dtype=poses.dtype)[None, None, :, :].expand(n, v, -1, -1)
            poses = poses @ poses  # Hacky way to allow cast to fp16
        
        images = images.to(poses.dtype)
        projs = projs.to(poses.dtype)
        
        USE_SYNTHETIC_GT = True#self.gt_synthesizer is not None or self.cascade_wrapper is not None
        idxs = data["idxs"][:, 0].tolist()
        keys = [f"{'train' if self.training else 'val'}_{idx}" for idx in idxs]
        is_cached_data = False
        if USE_SYNTHETIC_GT and self.config.BTS.CACHE_SYNTHETIC_GT and "xyz" in data and len(data["xyz"]) == n:
            poses_gt = data["poses_gt"]
            projs_gt = data["projs_gt"]
            gt_invalid = data["gt_invalid"]
            depths = data["depths_in"]
            depths_gt = data["depths_gt"]
            is_cached_data = True
        else:
            with torch.no_grad(), profiler.record_function("trainer-synthesize_gt"):
                if self.gt_synthesizer is not None:
                    gs_output: Outputs
                    gs_data: Data
                    gs_output, gs_data, _ = self.gt_synthesizer(images, poses, projs, hw_unnorm=hw_unnorm, output_in_nv=True, refine_output=True)
                    
                    poses_gt = gs_output.POSES_OUT
                    projs_gt = gs_output.PROJS_OUT
                    gt_invalid = gs_output.INVALID_SYNTH
                    depths = gs_output.DEPTHS_IN
                    depths_gt = gs_output.DEPTHS_SYNTH[0] if gs_output.DEPTHS_SYNTH is not None else None
                    
                    if not self.training:
                        # Store results for validation and visualization
                        data["images_synthetic_cond"] = gs_output.IMGS_COND
                        data["images_synthetic_gt"] = gs_output.IMGS_SYNTH[0] if gs_output.IMGS_SYNTH is not None else None
                        data["depths_synthetic_gt"] = depths_gt
                        data["depths_in"] = depths
                        data["depths_synthetic_reproj"] = gs_data.DEPTHS_NV
                    del(_, gs_output, gs_data)

                elif self.cascade_wrapper is not None:
                    outputs: list[Outputs]
                    datas: list[Data]
                    outputs, datas, _ = self.cascade_wrapper(
                        images, poses, projs, return_profile=False, anchor_seed=None, cascade_seed=None, debug=False
                    )
                    
                    poses_gt = outputs[0].POSES_OUT
                    projs_gt = outputs[0].PROJS_OUT
                    gt_invalid = outputs[0].INVALID_SYNTH
                    depths = outputs[0].DEPTHS_IN
                    depths_gt = outputs[0].DEPTHS_SYNTH[0] if outputs[0].DEPTHS_SYNTH is not None else None

                    if not self.training:
                        # Store results for validation and visualization
                        data["images_synthetic_cond"] = outputs[0].IMGS_COND
                        data["images_synthetic_gt"] = outputs[0].IMGS_SYNTH[0] if outputs[0].IMGS_SYNTH is not None else None
                        data["depths_synthetic_gt"] = depths_gt
                        data["depths_in"] = depths
                        data["depths_synthetic_reproj"] = datas[0].DEPTHS_NV
                        # For debugging
                        # data["outputs"] = outputs
                        # data["datas"] = datas
                        # data["debug_dict"] = _
                    del(_, outputs, datas)

                    if self.save_synthetic_density:
                        # Unproject the voxel grid from the keyframe input camera.

                        # xyz = unproj_grid(
                        #     w, h, self.config.BTS.MODEL_CONF.encoder.d_out, self.z_near, self.z_far, projs[:1], inv_z=self.config.BTS.MODEL_CONF.inv_z, device=images.device, dtype=images.dtype
                        #     ) # [B, H, W, D, 3]
                        # n_pts = w * h * self.config.BTS.MODEL_CONF.encoder.d_out
                        # xyz = xyz.permute(0, 2, 1, 3, 4).reshape(1, -1, 3).expand(n, n_pts, 3)  # [B, n_pts, 3]

                        # z = torch.linspace(self.z_near, self.z_far, self.config.BTS.MODEL_CONF.encoder.d_out + 1, device=images.device, dtype=images.dtype)
                        # TODO use conf arg
                        x_res = w // 2
                        y_res = h // 2
                        z_res = self.config.BTS.MODEL_CONF.encoder.d_out // 2
                        z = torch.linspace(1/self.z_near, 1/self.z_far, z_res + 1, device=images.device, dtype=images.dtype)
                        z = (1 / z.clamp_min(1e-6)).to(images)

                        x = torch.linspace(-1, 1, x_res + 1, device=images.device, dtype=images.dtype)
                        y = torch.linspace(-1, 1, y_res + 1, device=images.device, dtype=images.dtype)
                        xyz = torch.stack(torch.meshgrid(x, y, z), dim=-1)   # [W+1, H+1, D+1, 3]
                        xyz_deltas = xyz[1:, 1:, 1:] - xyz[:-1, :-1, :-1]
                        # Apply to all but the last element
                        xyz = xyz[:-1, :-1, :-1] + torch.rand_like(xyz_deltas) * xyz_deltas
                        xyz = xyz.reshape(1, -1, 3).expand(n, -1, 3)  # [B, n_pts, 3]
                        xyz = unproject_from_image(uv=xyz[..., :2], Ks=projs[:, 0], z=xyz[..., 2:3])

                        _, invalid, sigmas, sdf = self.cascade_wrapper.gt_synthesizer.renderer.net(xyz, only_density=True, downscale_sigmas=True)
                        
                        # FIXME check why this is not FP16????
                        sigmas = sigmas.to(images)
                        data["sampled_densities_gt"] = sigmas.view(n, 1, x_res, y_res, z_res) # [B, n_views, C, H, W]
                        data["sampled_densities_invalid"] = invalid.view(n, 1, x_res, y_res, z_res)
                        if "sdf_in" in sdf and "sdf_nv" in sdf:
                            occl_and_empty = torch.logical_and(sdf["sdf_in"] < 0, sdf["sdf_nv"] > 0)
                            data["sampled_densities_occluded_and_empty"] = occl_and_empty.view(n, 1, x_res, y_res, z_res)
                        # data["density_grid_pseudo_vol"] = sigmas.view(n, 1, x_res, y_res, z_res) # [B, n_views, C, H, W]
                        # data["xyz"] = xyz.view(n, 1, x_res, y_res, z_res, 3) # [B, n_views, H, W, D, 3]
                        data["xyz"] = xyz.view(n, 1, x_res, y_res, z_res, 3) # [B, n_views, H, W, D, 3]
                if USE_SYNTHETIC_GT:
                    downscale = self.config.BTS.DATA_DOWNSCALE_FACTOR
                    if downscale > 1:
                        _, nv, _, h_nv, w_nv = gt_invalid.shape
                        gt_invalid = F.interpolate(gt_invalid.view(-1, 1, h_nv, w_nv).float(), scale_factor=1/downscale, mode="nearest").bool().view(n, nv, 1, h_nv // downscale, w_nv // downscale)
                        depths = F.interpolate(depths.view(-1, 1, h, w), scale_factor=1/downscale, mode="bilinear", align_corners=False).view(n, v, 1, h // downscale, w // downscale)
                        depths_gt = F.interpolate(depths_gt.view(-1, 1, h_nv, w_nv), scale_factor=1/downscale, mode="bilinear", align_corners=False).view(n, nv, 1, h_nv // downscale, w_nv // downscale)
                        if not self.training:
                            data["depths_synthetic_gt"] = depths_gt
                            data["depths_in"] = depths
                            data["images_synthetic_cond"] = F.interpolate(data["images_synthetic_cond"].view(-1, data["images_synthetic_cond"].size(2), h_nv, w_nv), scale_factor=1/downscale, mode="bilinear", align_corners=False).view(n, nv, data["images_synthetic_cond"].size(2), h_nv // downscale, w_nv // downscale)
                            data["images_synthetic_gt"] = F.interpolate(data["images_synthetic_gt"].view(-1, 3, h_nv, w_nv), scale_factor=1/downscale, mode="bilinear", align_corners=False).view(n, nv, 3, h_nv // downscale, w_nv // downscale)
                            data["depths_synthetic_reproj"] = F.interpolate(data["depths_synthetic_reproj"].view(-1, 1, h_nv, w_nv), scale_factor=1/downscale, mode="bilinear", align_corners=False).view(n, nv, 1, h_nv // downscale, w_nv // downscale)

                    if self.config.BTS.CACHE_SYNTHETIC_GT:
                        data["poses_gt"] = poses_gt
                        data["projs_gt"] = projs_gt
                        data["gt_invalid"] = gt_invalid
                        data["depths_in"] = depths
                        data["depths_gt"] = depths_gt

        if self.training and self.alternating_ratio is not None:
            step = self._counter % (self.alternating_ratio + 1)
            if step < self.alternating_ratio:
                for params in self.renderer.net.encoder.parameters(True):
                    params.requires_grad_(True)
                for params in self.renderer.net.mlp_coarse.parameters(True):
                    params.requires_grad_(False)
            else:
                for params in self.renderer.net.encoder.parameters(True):
                    params.requires_grad_(False)
                for params in self.renderer.net.mlp_coarse.parameters(True):
                    params.requires_grad_(True)

        # Frames used to encode the neural volume
        ids_encoder = [0]
        # Frames used for sampling color (max `self.frames_render` random samples)
        ids_render = [0]
        # Frames used for loss computation
        ids_loss = list(range(v))

        combine_ids = None

        if self.loss_from_single_img:
            ids_loss = ids_loss[:1]

        ip = self.train_image_processor if self.training else self.val_image_processor

        if USE_SYNTHETIC_GT:
            images_ip = images[:, ids_loss].mean(dim=2, keepdim=True)#[:, :, :1])  
        else:
            images_ip = ip(images)

        with profiler.record_function("trainer_encode-grid"):
            self.renderer.net.compute_grid_transforms(projs[:, ids_encoder], poses[:, ids_encoder])
            self.renderer.net.encode(images, projs, poses, ids_encoder=ids_encoder, ids_render=ids_render, images_alt=images_ip, combine_ids=combine_ids)
            if self.save_predicted_density:
                # Values used for optional regularization
                xyz = data["xyz"].view(n, -1, 3)
                _, invalid, sigmas, state_dict = self.renderer.net(xyz, only_density=True)
                if state_dict is not None and "uncertainties" in state_dict:
                    data["sampled_uncertainties_pred"] = state_dict["uncertainties"].view(data["xyz"].shape[:-1])
                if state_dict is not None and "densities_raw" in state_dict:
                    data["sampled_densities_pred"] = state_dict["densities_raw"].view(data["xyz"].shape[:-1])
                else:
                    data["sampled_densities_pred"] = sigmas.view(data["xyz"].shape[:-1])
                invalid_pred = invalid.view(data["xyz"].shape[:-1])
                if "sampled_densities_invalid" in data:
                    data["sampled_densities_invalid_comb"] = torch.logical_or(data["sampled_densities_invalid"], invalid_pred)
                else:
                    data["sampled_densities_invalid_comb"] = invalid_pred

        sampler = self.train_sampler if self.training else self.val_sampler
        sampler_in = self.train_sampler_in if self.training else self.val_sampler_in

        data["z_near"] = torch.tensor(self.z_near, device=images.device)
        data["z_far"] = torch.tensor(self.z_far, device=images.device)
        # TODO potentially add more flags
        if self.save_rgbd_images:
            with profiler.record_function("trainer_sample-rays"):
                if USE_SYNTHETIC_GT:
                    all_rays, all_rgb_gt = sampler.sample(depths_gt, poses_gt, projs_gt)
                    all_rays_2, all_rgb_gt_2 = sampler_in.sample(depths[:, ids_loss], poses[:, ids_loss], projs[:, ids_loss])
                else:
                    all_rays, all_rgb_gt = sampler.sample(images_ip[:, ids_loss], poses[:, ids_loss], projs[:, ids_loss])
                
            render_data = self.render_and_reconstruct(sampler, all_rays, all_rgb_gt, gt_invalid)
            data.update(render_data)
            
            if USE_SYNTHETIC_GT:
                gt_invalid_in = torch.logical_or(self.z_near > depths, depths > self.z_far) # TODO check if this is redundant
                render_data = self.render_and_reconstruct(sampler_in, all_rays_2, all_rgb_gt_2, gt_invalid_in)
                data.update({f"{k}_in": v for k, v in render_data.items()})

            
            for downscale in range(len(data["coarse"])):
                data["coarse"][downscale]["depth"] = distance_to_z(data["coarse"][downscale]["depth"], projs_gt)
                data["fine"][downscale]["depth"] = distance_to_z(data["fine"][downscale]["depth"], projs_gt)
                data["coarse_in"][downscale]["depth"] = distance_to_z(data["coarse_in"][downscale]["depth"], projs[:, ids_loss])
                data["fine_in"][downscale]["depth"] = distance_to_z(data["fine_in"][downscale]["depth"], projs[:, ids_loss])
        
        if not self.training and self.save_rgbd_images:
            # TODO add eval if self.save_rgbd_images

            # Depth along camera rays.
            depth_pred_in = data["fine_in"][0]["depth"][:, 0].unsqueeze(1)     # [B, 1, H, W]          # TODO val this

            if len(data["depths"]) > 0:
                data.update(
                    # TODO Why are values float32 and float64 here?
                    self.compute_depth_metrics(
                        # All input views
                        depth_pred=depth_pred_in,
                        # Lidar depth
                        depth_gt=torch.stack(data["depths"], dim=1)[:, 0],
                        prefix="depth_lidar",       # TODO add const str
                    )
                )
            if self.eval_nvs:
                data.update(self.compute_nvs_metrics(data))

            if depths_gt is not None:        # TODO add arg
                data.update(
                    self.compute_depth_metrics(
                        # All novel views
                        depth_pred=data["fine"][0]["depth"].view(-1, 1, *depths_gt.shape[-2:]),
                        depth_gt=depths_gt.view(-1, 1, *depths_gt.shape[-2:]), 
                        prefix="depth_nv",
                    )
                )
            
        if not self.training:
            cam_incl_adjust = self.config.BTS.DATA.CAM_INCL_ADJUST.to(dtype=images.dtype, device=images.device).view(1, 4, 4)
            vis_volume = self.config.BTS.DATA.VIS_VOLUME

            # Debugging visualization
            # import matplotlib.pyplot as plt
            # profile = render_profile(self.renderer.net, vis_volume, cam_incl_adjust, thresh=10)
            # plt.imshow(profile[0].cpu())
            # plt.savefig("temp.png")

            data["profiles"] = render_profile(
                self.renderer.net, vis_volume, cam_incl_adjust)
            if not is_cached_data:
                if self.gt_synthesizer is not None:
                    data["profiles_pseudo"] = render_profile(self.gt_synthesizer.renderer.net, vis_volume, cam_incl_adjust)
                elif self.cascade_wrapper is not None:
                    data["profiles_pseudo"] = render_profile(self.cascade_wrapper.gt_synthesizer.renderer.net, vis_volume, cam_incl_adjust)

        if self.training:
            self._counter += 1

        return data
    
    def render_and_reconstruct(self, sampler, rays, rgb_gt, invalid_gt=None):
        data = {
            "fine": [],
            "coarse": [],
        }

        if self.prediction_mode == "multiscale":
            for scale in self.renderer.net.encoder.scales:
                self.renderer.net.set_scale(scale)

                using_fine = self.renderer.renderer.using_fine
                if scale != 0 and using_fine:
                    self.renderer.renderer.using_fine = False
                    
                with profiler.record_function("trainer_render"):
                    # TODO only want weights if loss invalid policy is weight_guided
                    render_dict = self.renderer(rays, want_weights=True, want_alphas=True, want_rgb_samps=False)       # TODO maybe add a cfg for this depending on the loss invalid policy
                
                if scale != 0 and using_fine:
                    self.renderer.renderer.using_fine = True

                if "fine" not in render_dict:
                    render_dict["fine"] = dict(render_dict["coarse"])
                
                render_dict["rgb_gt"] = rgb_gt
                render_dict["rays"] = rays
                
                with profiler.record_function("trainer_reconstruct"):
                    render_dict = sampler.reconstruct(render_dict)

                data["fine"].append(render_dict["fine"])
                data["coarse"].append(render_dict["coarse"])
                data["rgb_gt"] = render_dict["rgb_gt"]
                data["rays"] = render_dict["rays"]
        else:
            with profiler.record_function("trainer_render"):
                render_dict = self.renderer(rays, want_weights=True, want_alphas=False, want_rgb_samps=False)

            if "fine" not in render_dict:
                render_dict["fine"] = dict(render_dict["coarse"])

            render_dict["rgb_gt"] = rgb_gt
            render_dict["rays"] = rays
            
            # TODO add gt invalid here

            with profiler.record_function("trainer_reconstruct"):
                # render_dict = sampler.reconstruct(render_dict, channels=3)
                render_dict = sampler.reconstruct(render_dict, channels=1)
            
            self.invalid_policies = {}
            for _, loss_cfg in self.config.BTS.LOSSES.items():
                if hasattr(loss_cfg, "invalid_policy"):
                    self.invalid_policies[loss_cfg.invalid_policy] = InvalidPolicy(loss_cfg.invalid_policy)
            
            for key in ["coarse", "fine"]:
                for invalid_policy_name, invalid_policy in self.invalid_policies.items():
                    kwargs = {"invalids": render_dict[key]["invalid"]}
                    if "weights" in render_dict[key]:
                        kwargs["weights"] = render_dict[key]["weights"]
                    if "rgb_samps" in render_dict[key]:
                        kwargs["rgb_samps"] = render_dict[key]["rgb_samps"]
                    invalid_key = f"invalid_{invalid_policy_name}"
                    render_dict[key][invalid_key] = invalid_policy(**kwargs) 
                    del render_dict[key]["invalid"]
            
                    # TODO make better and add to multi scale
                    if invalid_gt is not None:
                        render_dict[key][invalid_key] = torch.logical_or(render_dict[key][invalid_key], invalid_gt.squeeze(2).unsqueeze(-1))
                
            data["fine"].append(render_dict["fine"])
            data["coarse"].append(render_dict["coarse"])
            data["rgb_gt"] = render_dict["rgb_gt"]
            data["rays"] = render_dict["rays"]
            
        return data

    def compute_depth_metrics(self, depth_pred: torch.Tensor, depth_gt: torch.Tensor, prefix: str = None):
        """Inputs must be [B, 1, H, W]"""
        depth_pred = torch.clamp(depth_pred, 1e-3, 80)
        metrics_dict = compute_depth_metrics(depth_pred, depth_gt)

        if prefix:
            return {f"{prefix}/{k}": v for k, v in metrics_dict.items()}
        
        return metrics_dict

    def compute_nvs_metrics(self, data):
        # TODO: This is only correct for batchsize 1!
        # Following tucker et al. and others, we crop 5% on all sides

        # idx of stereo frame (the target frame is always the "stereo" frame).
        sf_id = data["rgb_gt"].shape[1] // 2

        imgs_gt = data["rgb_gt"][:1, sf_id:sf_id+1]
        imgs_pred = data["fine"][0]["rgb"][:1, sf_id:sf_id+1]

        imgs_gt = imgs_gt.squeeze(0).permute(0, 3, 1, 2)
        imgs_pred = imgs_pred.squeeze(0).squeeze(-2).permute(0, 3, 1, 2)

        n, c, h, w = imgs_gt.shape
        y0 = int(math.ceil(0.05 * h))
        y1 = int(math.floor(0.95 * h))
        x0 = int(math.ceil(0.05 * w))
        x1 = int(math.floor(0.95 * w))

        imgs_gt = imgs_gt[:, :, y0:y1, x0:x1]
        imgs_pred = imgs_pred[:, :, y0:y1, x0:x1]

        imgs_gt_np = imgs_gt.detach().squeeze().permute(1, 2, 0).cpu().numpy()
        imgs_pred_np = imgs_pred.detach().squeeze().permute(1, 2, 0).cpu().numpy()

        ssim_score = skimage.metrics.structural_similarity(imgs_pred_np, imgs_gt_np, multichannel=True, data_range=1)
        psnr_score = skimage.metrics.peak_signal_noise_ratio(imgs_pred_np, imgs_gt_np, data_range=1)
        lpips_score = self.lpips(imgs_pred, imgs_gt, normalize=False).mean()

        metrics_dict = {
            "ssim": torch.tensor([ssim_score], device=imgs_gt.device),
            "psnr": torch.tensor([psnr_score], device=imgs_gt.device),
            "lpips": torch.tensor([lpips_score], device=imgs_gt.device)
        }
        return metrics_dict


def training(local_rank: int, config: MainConfig):
    return base_training(local_rank, config, get_dataflow, initialize, get_metrics, visualize)



def get_dataflow(
    config: BTSConfig, 
    nproc_per_node: int | None = None, 
    logger: logging.Logger | None = None, 
    rank_0_barrier: bool = False
    ) -> tuple[DataLoader, DataLoader, DataLoader]:
    """Returns train, test and visualization datasets."""
    if rank_0_barrier and idist.get_local_rank() > 0:
        # Ensure that only local rank 0 download the dataset. Thus each node will download a copy of the dataset.
        # This causes timeout in rank>0 processes if `get_dataflow` is called during training. 
        idist.barrier()

    if logger:
        logger.info("Making datasets...")

    train_dataset, val_dataset, _ = make_datasets(config.DATA)
    
    val_dataset.dataset._left_offset = 0
    val_dataset.dataset.return_depth = True
    
    num_scenes = min(
        len(val_dataset), 
        int(config.VISUALIZE.NUM_SCENES*nproc_per_node if nproc_per_node else config.VISUALIZE.NUM_SCENES)
    )
    match config.VISUALIZE.POLICY:
        case "random":
            idxs = torch.randperm(len(val_dataset))[:num_scenes].tolist()
        case "eqspaced":
            idxs = torch.linspace(0, len(val_dataset)-1, steps=num_scenes, dtype=torch.int).tolist()
        case _:
            raise ValueError(f"Unknown policy {config.VISUALIZE.POLICY}")
    vis_dataset = Subset(
        val_dataset,
        idxs,
    )

    # Change visualisation dataset
    vis_dataset.dataset._skip = 12 if isinstance(train_dataset, KittiRawDataset) or isinstance(train_dataset, KittiOdometryDataset) else 50
    vis_dataset.dataset.return_depth = True

    # Setup data loader also adapted to distributed config: nccl, gloo, xla-tpu
    eval_batch_size = max(1, config.BATCH_SIZE//2)      # Avoid memory spike during eval
    train_loader = idist.auto_dataloader(train_dataset, batch_size=config.BATCH_SIZE, num_workers=config.NUM_WORKERS, shuffle=True, drop_last=True)
    val_loader = idist.auto_dataloader(val_dataset, batch_size=eval_batch_size, num_workers=config.NUM_WORKERS, shuffle=False)
    vis_loader = idist.auto_dataloader(vis_dataset, batch_size=eval_batch_size, num_workers=config.NUM_WORKERS, shuffle=False)

    if hasattr(train_loader, "dataset") and logger:
        train_dataset_len = len(train_loader.dataset)
        val_dataset_len = len(val_loader.dataset)
        vis_dataset_len = len(vis_loader.dataset)
        logger.info(f"Dataset length: train={train_dataset_len}, val={val_dataset_len}, vis={vis_dataset_len}")

    return train_loader, val_loader, vis_loader


def get_metrics(config: BTSConfig, device):
    if config.SUPERVISION_FROM_CASCADE:
        return {}
    metric_names = ["abs_rel", "sq_rel", "rmse", "rmse_log", "a1", "a2", "a3", "ssim"]
    names = ["depth_nv/" + n for n in metric_names]
    if config.MODE == "nvs":
        names += ["ssim", "psnr", "lpips"]

    metrics = {name: MeanMetric((lambda n: lambda x: x["output"][n])(name), device) for name in names}
    return metrics


def initialize(config: MainConfig, logger=None):
    
    net = make_model(config.BTS.MODEL_CONF)
    renderer = NeRFRenderer.from_conf(config.BTS.RENDERER)
    renderer = renderer.bind_parallel(net, gpus=None).eval()
    
    with torch.no_grad():
        gt_synthesizer = GTSynthesisWrapper.from_conf(config, cam_incl_adjust=config.BTS.DATA.CAM_INCL_ADJUST, refiner=True)
        gt_synthesizer = gt_synthesizer.eval()
        gt_synthesizer.requires_grad_(False)

        cascade_model = None
        if config.BTS.SUPERVISION_FROM_CASCADE:
            cascade_model = CascadeWrapper.from_conf(
                config=config,
                gt_synthesizer=gt_synthesizer,
                cam_incl_adjust=config.BTS.DATA.CAM_INCL_ADJUST,
                **asdict_lowercase_keys_override(config.SYNTHETIC_GT.CASCADE),
            )

    model = BTSWrapper(
        renderer,
        cascade_model if cascade_model is not None else gt_synthesizer,
        config,
        eval_nvs=config.BTS.MODE == "nvs",
    )

    if idist.get_world_size() == 1:
        model = idist.auto_model(model)        # TODO fix this
    else:
        model = idist.auto_model(model, find_unused_parameters=True)        # TODO fix this
    
    def get_model_parameters(model):
        if hasattr(model, 'module'):
            # If DP or DDP wrapped
            return model.module.renderer.net.parameters()
        else:
            return model.renderer.net.parameters()

    # kwargs = {"eps": torch.finfo(torch.float16).eps} if config.BTS.WITH_AMP else {}
    kwargs = {}
    optimizer = optim.Adam(get_model_parameters(model), lr=config.BTS.LR, weight_decay=config.BTS.WEIGHT_DECAY, **kwargs)        # TODO make sure these are only the parameters we care about
    
    optimizer = idist.auto_optim(optimizer)

    lr_scheduler = make_scheduler(config.BTS.SCHEDULER, optimizer)

    criterions = make_losses(config.BTS.LOSSES, use_automasking=config.BTS.MODEL_CONF.USE_AUTOMASKING)

    return model, optimizer, criterions, lr_scheduler


