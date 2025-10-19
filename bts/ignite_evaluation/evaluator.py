from dataclasses import asdict
import os
import torch
from ignite.contrib.handlers import TensorboardLogger
from ignite.engine import Engine
from torch import nn
from torch.nn import functional as F
from torch.utils.data import DataLoader

from datasets.data_util import make_datasets
from bts.renderer import NeRFRenderer
from bts.common.ray_sampler import ImageRaySampler
from bts.ignite_evaluation.base_evaluator import base_evaluation
from utils.metrics import MeanMetric
from utils.projection_ops import distance_to_z
from bts.losses.utils import compute_depth_metrics, compute_rgb_metrics
from bts.gt_synthesis.gt_synthesis import GTSynthesisWrapper
from bts.gt_synthesis.cascade_wrapper import CascadeWrapper
from configs.structured_configs.main_config import MainConfig
from logging import Logger
import lpips
from vcm.utils.model_utils import get_accelerator_checkpoint_path
from bts.models import BTSNet, BTSDirect, BTSNetOld
from utils.utils import asdict_lowercase_keys_override
from bts.common.train_eval_utils import get_checkpoint_path

IDX = 0


class InferenceWrapper(nn.Module):
    def __init__(self, config: MainConfig, renderer: NeRFRenderer, gt_synthesizer: GTSynthesisWrapper, cascade_model: CascadeWrapper, refine_output: bool = False, **kwargs) -> None:
        super().__init__()

        self.cfg = config
        self.renderer = renderer
        self.gt_synthesizer = gt_synthesizer
        self.cascade_model = cascade_model
        self.refine_output = refine_output

        self.z_near = config.BTS.MODEL_CONF.z_near
        self.z_far = config.BTS.MODEL_CONF.z_far
        self.sampler = ImageRaySampler(self.z_near, self.z_far)
        self.sampler_in = ImageRaySampler(self.z_near, self.z_far)
        
        # Should not be in state_dict
        object.__setattr__(self, 'lpips_net', lpips.LPIPS(net="vgg"))

        self.depth_scaling = None   # TODO add param
        # Useful for logging
        self.checkpoint_fp = None
        
    def to(self, *args, **kwargs):
        """Moves all submodules."""
        super().to(*args, **kwargs)
        self.lpips_net.to(*args, **kwargs)
        if self.gt_synthesizer is not None:
            self.gt_synthesizer.to(*args, **kwargs)
        if self.cascade_model is not None:
            self.cascade_model.to(*args, **kwargs)
        return self
    
    @classmethod
    def from_conf(cls, config: MainConfig, refine_output=False, **kwargs):
        
        if not config.CONTROLNET.MODEL.CONTROLNET_MODEL_NAME_OR_PATH and refine_output:
            # Load the latest checkpoint for this experiment
            try:
                path = get_accelerator_checkpoint_path(config, use_latest=True)
                config.CONTROLNET.MODEL.CONTROLNET_MODEL_NAME_OR_PATH = os.path.join(path, "controlnet")
            except Exception as exc:
                print(f"Controlnet checkpoint not found ({exc}). Continuing.")
        
        gt_synthesizer = GTSynthesisWrapper.from_conf(config=config, cam_incl_adjust=config.BTS.DATA.CAM_INCL_ADJUST, refiner=refine_output, depth_pred=True)
        
        cascade_model = CascadeWrapper.from_conf(
            config=config,
            gt_synthesizer=gt_synthesizer,
            cam_incl_adjust=config.BTS.DATA.CAM_INCL_ADJUST,
            **asdict_lowercase_keys_override(config.SYNTHETIC_GT.CASCADE, **kwargs),
        )

        net = globals()[config.BTS.MODEL_CONF.ARCH](asdict(config.BTS.MODEL_CONF))
        
        renderer = NeRFRenderer.from_conf(config.BTS.RENDERER)
        renderer = renderer.bind_parallel(net, gpus=None).eval()
        
        wrapper = cls(
            config=config,
            renderer=renderer,
            gt_synthesizer=gt_synthesizer,
            cascade_model=cascade_model,
            refine_output=refine_output,
            **asdict_lowercase_keys_override(config.EVAL_OCCUPANCY, **kwargs),
        )
        
        match config.BTS.RESUME_FROM:
            case "latest":
                checkpoint_fp = get_checkpoint_path("latest", config.RECON_EXP_DIR)
            case _:
                checkpoint_fp = get_checkpoint_path(config.BTS.RESUME_FROM)

        if checkpoint_fp is not None and checkpoint_fp.exists():
            print(f"Loading reconstructor checkpoint: {checkpoint_fp.as_posix()}")
            checkpoint = torch.load(checkpoint_fp.as_posix())
            if "model" in checkpoint:
                # If the checkpoint contains ['trainer', 'model', 'optimizer', 'lr_scheduler']
                wrapper.load_state_dict(checkpoint["model"], strict=False)
            else:
                wrapper.load_state_dict(checkpoint, strict=False)

            # For logging
            wrapper.checkpoint_fp = checkpoint_fp
        
        return wrapper
    
    def forward(self, data, forward_type: str = "controlnet_novel_view", seed: int | None = None):
        
        data = dict(data)
        images = torch.stack(data["imgs"], dim=1)                           # n, v, c, h, w
        poses_c2w = torch.stack(data["poses"], dim=1)                       # n, v, 4, 4 c2w
        projs = torch.stack(data["projs"], dim=1)                           # n, v, 4, 4 (-1, 1)
        hw_unnorm = data["hw_unnorm"][0]
        cam_incl_adjust = data.get("cam_incl_adjusts", None)
        cam_incl_adjust = cam_incl_adjust[0].to(images) if cam_incl_adjust is not None else cam_incl_adjust

        n, v, _, _, _ = images.shape
        dtype = (poses_c2w @ torch.eye(4).to(poses_c2w.device)).dtype
        # Use first frame as keyframe
        to_base_pose = torch.inverse(poses_c2w[:, :1, :, :])
        poses = to_base_pose.expand(-1, v, -1, -1).double() @ poses_c2w.double()
        
        poses = poses.to(dtype)
        images = images.to(dtype)
        projs = projs.to(dtype)
        
        camera_sampler_kwargs = {"poses_c2w": poses_c2w, "cam_incl_adjust": cam_incl_adjust, "vis_volume": self.cfg.BTS.DATA.VIS_VOLUME}
        if seed is None:
            seed = self.cfg.SEED + int(data["idxs"][0]) if self.cfg.SEED is not None else None
        
        match forward_type:
            case "controlnet_novel_view":
                assert self.gt_synthesizer is not None
                
                out_synth, data_synth, data_debug = self.gt_synthesizer(
                    images, poses, projs, 
                    output_in_nv=True, 
                    refine_output=self.refine_output, 
                    seed=seed, 
                    hw_unnorm=hw_unnorm,
                    camera_sampler_kwargs=camera_sampler_kwargs,
                    debug=True, 
                )
                data["out_synth"] = out_synth
                data["data_synth"] = data_synth
                data["data_debug"] = data_debug
                
            case "controlnet_input_view":
                assert self.gt_synthesizer is not None
                
                out_synth, data_synth, data_debug = self.gt_synthesizer(
                    images, poses, projs, 
                    output_in_nv=False, 
                    refine_output=False, 
                    seed=seed,
                    hw_unnorm=hw_unnorm,
                    camera_sampler_kwargs=camera_sampler_kwargs,
                    debug=True, 
                )
                data["out_synth"] = out_synth
                data["data_synth"] = data_synth
                data["data_debug"] = data_debug
                
            case "controlnet_cascade":
                assert self.cascade_model is not None
                
                out_synth, data_synth, data_debug = self.cascade_model(
                    images, poses, projs,
                    anchor_seed=seed,
                    cascade_seed=seed,
                    return_profile=True,
                    camera_sampler_kwargs=camera_sampler_kwargs,
                )
                data["out_synth_l"] = out_synth
                data["data_synth_l"] = data_synth
                data["data_debug"] = data_debug
                
            case "recon":
                assert self.renderer is not None
                
                ids_encoder = [0]
                self.renderer.net.compute_grid_transforms(projs[:, ids_encoder], poses[:, ids_encoder])
                self.renderer.net.encode(images, projs, poses, ids_encoder=ids_encoder, ids_render=ids_encoder, images_alt=images * .5 + .5)
                self.renderer.net.set_scale(0)

                all_rays, all_rgb_gt = self.sampler.sample(images * .5 + .5, poses, projs)
                render_data = self.render_and_reconstruct(all_rays, all_rgb_gt, projs)
                data.update(render_data)

        globals()["IDX"] += 1
        
        data["vis_volume"] = self.cfg.BTS.DATA.VIS_VOLUME

        return data

    def render_and_reconstruct(self, rays: torch.Tensor, rgb_gt: torch.Tensor, projs: torch.Tensor):
        """Renders given camera rays and reconstructs them into images."""
        data = {
            "fine": [],
            "coarse": [],
        }

        self.renderer.net.set_scale(0)
        render_dict = self.renderer(rays, want_weights=True, want_alphas=True)

        if "fine" not in render_dict:
            render_dict["fine"] = dict(render_dict["coarse"])

        render_dict["rgb_gt"] = rgb_gt
        render_dict["rays"] = rays

        render_dict = self.sampler.reconstruct(render_dict) # TODO channels=1

        render_dict["coarse"]["depth"] = distance_to_z(render_dict["coarse"]["depth"], projs)
        render_dict["fine"]["depth"] = distance_to_z(render_dict["fine"]["depth"], projs)

        data["fine"].append(render_dict["fine"])
        data["coarse"].append(render_dict["coarse"])
        data["rgb_gt"] = render_dict["rgb_gt"]
        data["rays"] = render_dict["rays"]

        data["z_near"] = torch.tensor(self.z_near, device=rays.device)
        data["z_far"] = torch.tensor(self.z_far, device=rays.device)

        return data

    def compute_depth_metrics(self, data: dict):
        # TODO: This is only correct for batchsize 1!
        depth_gt = data["depths"][0]
        depth_pred = data["fine"][0]["depth"][:, :1]

        depth_pred = F.interpolate(depth_pred, depth_gt.shape[-2:])

        match self.depth_scaling:
            case "median":
                mask = depth_gt > 0
                scaling = torch.median(depth_gt[mask]) / torch.median(depth_pred[mask])
                depth_pred = scaling * depth_pred
            case "l2":
                mask = depth_gt > 0
                depth_pred = depth_pred
                depth_gt_ = depth_gt[mask]
                depth_pred_ = depth_pred[mask]
                depth_pred_ = torch.stack((depth_pred_, torch.ones_like(depth_pred_)), dim=-1)
                x = torch.linalg.lstsq(depth_pred_.to(torch.float32), depth_gt_.unsqueeze(-1).to(torch.float32)).solution.squeeze()
                depth_pred = depth_pred * x[0] + x[1]

        depth_pred = torch.clamp(depth_pred, 1e-3, 80)
        return compute_depth_metrics(depth_pred, depth_gt)
    
    def compute_nvs_metrics(self, data: dict):
        # TODO: This is only correct for batchsize 1!

        # idx of stereo frame (the target frame is always the "stereo" frame).
        sf_id = data["rgb_gt"].shape[1] // 2

        imgs_gt = data["rgb_gt"][:, sf_id:sf_id+1]
        imgs_pred = data["fine"][0]["rgb"][:, sf_id:sf_id+1]

        imgs_gt = imgs_gt.squeeze(0).permute(0, 3, 1, 2)
        imgs_pred = imgs_pred.squeeze(0).squeeze(-2).permute(0, 3, 1, 2)

        return compute_rgb_metrics(imgs_pred, imgs_gt, crop_5_percent_on_all_sides=True)


def evaluation(local_rank: int, config: MainConfig):
    return base_evaluation(local_rank, config, get_dataflow, initialize, get_metrics, visualize=visualize)


def get_dataflow(config, num_workers: int = 2):
    _, _, test_dataset = make_datasets(config)
    test_loader = DataLoader(test_dataset, batch_size=1, num_workers=num_workers, shuffle=False, drop_last=False)
    return test_loader


def get_metrics(config, device):
    names = ["abs_rel", "sq_rel", "rmse", "rmse_log", "a1", "a2", "a3"]
    metrics = {name: MeanMetric((lambda n: lambda x: x["output"][n])(name), device) for name in names}
    return metrics


def initialize(config: MainConfig, logger: Logger = None, refine_output: bool = True):
    model = InferenceWrapper.from_conf(config, refine_output)
    return model


def visualize(engine: Engine, logger: TensorboardLogger, step: int, tag: str):
    pass
