import os
import matplotlib.pyplot as plt

import torch
from ignite.engine import Engine
from ignite.contrib.handlers import TensorboardLogger

from bts.ignite_evaluation.base_evaluator import base_evaluation
from utils.metrics import MeanMetric
from bts.losses.utils import compute_depth_metrics, compute_rgb_metrics
from bts.gt_synthesis.gt_synthesis import Outputs, Data
from configs.structured_configs.main_config import MainConfig
from bts.ignite_evaluation.evaluator import InferenceWrapper, get_dataflow
from vcm.utils.eval_utils import format_conditioning_imgs
from utils.plotting import render_profile, plot_profile
from bts.common.ray_sampler import ImageRaySampler
from utils.occlusion_ops import comp_occlusion_map, DEFINITIONS
from utils.plotting import color_tensor, cmap_magma, set_thesis_rcparams, set_spines, PAGE_WIDTH_INCHES
from utils.utils import invert_depth


class NovelViewInferenceWrapper(InferenceWrapper):
    def __init__(self, *args, **kwargs) -> None:
        super().__init__(*args, **kwargs)

    def forward(self, data):
        data = super().forward(data, forward_type="controlnet_novel_view")
        
        out: Outputs = data["out_synth"]
        synth_data: Data = data["data_synth"]
        device = out.IMGS_IN.device
        
        profiles = render_profile(self.gt_synthesizer.renderer.net, self.cfg.CONTROLNET.DATA.VIS_VOLUME, self.cfg.CONTROLNET.DATA.CAM_INCL_ADJUST.to(device)).float()
        
        refine_results = {}
        imgs_reren, depths_reren, invalids_reren = [], [], []
        if self.refine_output:
            num_v, b, nv, _, h, w = out.IMGS_SYNTH.shape
            _, _, _, h_gt, w_gt = out.IMGS_IN.shape
            
            for i in range(num_v):
                # self.gt_synthesizer.renderer.net._num_input_imgs = nv
                
                # Rerender the input from the refined novel view.
                
                i_reren, d_reren, inv_reren = self.gt_synthesizer.encode_and_render(
                    ImageRaySampler(self.gt_synthesizer.z_near, self.gt_synthesizer.z_far, *synth_data.HW_DATA, norm_dir=True),
                    out.IMGS_SYNTH[i],
                    out.DEPTHS_SYNTH[i],
                    projs_enc=out.PROJS_OUT,
                    poses_enc=out.POSES_OUT,
                    projs_rend=out.PROJS_IN[:, :1],
                    poses_rend=out.POSES_IN[:, :1],
                )
                imgs_reren.append(i_reren)
                depths_reren.append(d_reren)
                invalids_reren.append(inv_reren)
                # TODO maybe render the nv-conditioned profile here?
                        
            # --- Aggregate results ---
            
            # Shape [num_versions, b, nv, c, h, w]
            imgs_reren = torch.stack(imgs_reren).float()
            depths_reren = torch.stack(depths_reren).float()
            invalids_reren = torch.stack(invalids_reren).float()
            
            # Metrics in the novel view
            # Shape -> [num_versions * b * nv, c, h, w]
            imgs_synth_b = out.IMGS_SYNTH.view(-1, 3, h, w)
            # perceptual_metrics_dict = perceptual_metrics(imgs_synth_b)      # TODO add input img to compute FID
            
            # Metrics in the input view
            # Shape -> [num_versions * b * nv, c, h, w]
            imgs_reren_b = imgs_reren.view(-1, 3, h_gt, w_gt)
            depths_reren_b = depths_reren.view(-1, 1, h_gt, w_gt)
            imgs_gt_b = out.IMGS_IN[:, :1].unsqueeze(0).expand_as(imgs_reren).view(-1, 3, h_gt, w_gt)
            depths_gt_b = out.DEPTHS_IN[:, :1].unsqueeze(0).expand_as(depths_reren).view(-1, 1, h_gt, w_gt)

            # Exclude pixels that are outside the novel view frustum 
            # or occluded in the novel view.
            valid_mask_b = invalids_reren.view(-1, 1, h_gt, w_gt) < .5
            if nv == 1:
                input_occlusion_mask = comp_occlusion_map(
                    self.cfg.SYNTHETIC_GT.OCCLUSIONS, 
                    pose1=synth_data.POSES_NV[:, 0], 
                    pose2=out.POSES_IN[:, 0],
                    proj1=synth_data.PROJS_NV[:, 0], 
                    proj2=out.PROJS_IN[:, 0],
                    depth1=synth_data.DEPTHS_NV[:, 0], 
                    depth2=out.DEPTHS_IN[:, 0],
                )["occlusions_full"]
                valid_mask_b = torch.logical_and(valid_mask_b, input_occlusion_mask == DEFINITIONS.IS_VISIBLE)
            
            rgb_metrics_masked = compute_rgb_metrics(imgs_reren_b, imgs_gt_b, valid_mask=valid_mask_b.repeat(1, 3, 1, 1))
            depth_metrics_masked = compute_depth_metrics(depths_reren_b, depths_gt_b, valid_mask=valid_mask_b)
                
            depth_var = out.DEPTHS_SYNTH.var(dim=0)
            refine_results = {
                "imgs_reren": imgs_reren,
                "depths_reren": depths_reren,
                "depths_var_pixel": depth_var,
                # Scalar metrics
                "scale": float(out.SCALES.mean()) if out.SCALES is not None else 0.,
                "shift": float(out.SHIFTS.mean()) if out.SHIFTS is not None else 0.,
                "depth_mean": float(out.DEPTHS_SYNTH.mean()),
                "depth_var": 0. if torch.isnan(depth_var).any() else float(depth_var.mean()),
                **{f"rgb_{k}_masked": float(v) for k, v in rgb_metrics_masked.items()},
                **{f"depth_{k}_masked": float(v) for k, v in depth_metrics_masked.items()},
                # **{f"synth_view_{k}": float(v) for k, v in perceptual_metrics_dict.items()},
                # Outputs for computing perceptual metrics (FID and IS)
                "imgs_synth_b": imgs_synth_b,
                "imgs_in_b": imgs_gt_b.repeat(nv, 1, 1, 1),       
                "valid_mask_b": valid_mask_b,#.repeat(nv, 1, 1, 1),       
            }
        
        data.update({
            "imgs_cond_formatted": format_conditioning_imgs(synth_data.IMGS_CONDITIONING).float(),
            "fraction_invalid": float(synth_data.INVALID_NV.float().mean()),
            "fraction_occluded": float((synth_data.MASKS_NV == DEFINITIONS.IS_OCCLUDED).float().mean()),
            "prompts": synth_data.CAPTION_STRS,
            "z_near": self.z_near,
            "z_far": self.z_far,
            "profiles": profiles,
            **refine_results,
        })
            
        return data


def evaluation(local_rank: int, config: MainConfig):
    return base_evaluation(local_rank, config, get_dataflow, initialize, get_metrics, visualize)


def get_metrics(config, device):
    rgb_names = ["rmse", "psnr", "ssim", "lpips"]
    depth_names = ["abs_rel", "sq_rel", "rmse", "rmse_log", "a1", "a2", "a3"]
    # perceptual_names = ["is"]
    names = (
        [f"rgb_{_}_masked" for _ in rgb_names[:-2]] + 
        [f"depth_{_}_masked" for _ in depth_names] + 
        ["scale", "shift", "depth_mean", "depth_var"] + 
        # [f"synth_view_{_}" for _ in perceptual_names] +
        ["fraction_invalid", "fraction_occluded"]
    )
    metrics = {name: MeanMetric((lambda n: lambda x: x["output"][n])(name), device) for name in names}
    
    # When using less than at least 2048 samples in one of the datasets
    # and using FID, I get: 'ValueError: Imaginary component 5.029033962366753e+75'
    # See: https://github.com/mseitzer/pytorch-fid/issues/13
    # This is a fix: https://github.com/ahmadki/mlperf_sd_inference/issues/4#issuecomment-1806149386
    # metrics["fid"] = FID(device=device, output_transform=lambda output: [output['output']["imgs_synth_b"], output['output']["imgs_in_b"]])
    # metrics["inception_score"] = InceptionScore(device=device, output_transform=lambda output: output['output']["imgs_synth_b"])
    
    return metrics


def initialize(cfg: MainConfig, refine_output=True, **kwargs):
    return NovelViewInferenceWrapper.from_conf(cfg, refine_output=refine_output)


def visualize(engine: Engine, logger: TensorboardLogger, output_dir: str, step: int):
    
    set_thesis_rcparams()
    
    data = engine.state.output["output"]
    writer = logger.writer
    
    out_synth: Outputs = data["out_synth"]
    data_synth: Data = data["data_synth"]
    
    z_near, z_far = data["z_near"], data["z_far"]
    idxs = data["idxs"].flatten().tolist()
    num_v, b, nv, _, h, w = out_synth.IMGS_SYNTH.shape
    
    imgs_in = out_synth.IMGS_IN.permute(0, 1, 3, 4, 2).cpu().float()
    depths_in = out_synth.DEPTHS_IN[:, 0, 0].cpu().float()
    
    assert nv == 1 or num_v == 1
    
    imgs_synth = out_synth.IMGS_SYNTH.permute(1, 4, 0, 2, 5, 3).reshape(b, h, w*nv*num_v, 3).cpu().float()
    depths_synth = out_synth.DEPTHS_SYNTH.permute(1, 4, 0, 2, 5, 3).reshape(b, h, w*nv*num_v).cpu().float()
    depths_nv = data_synth.DEPTHS_NV.permute(0, 2, 3, 1, 4).reshape(b, h, w*nv).repeat(1, 1, num_v).cpu().float()
    d_diff = (depths_synth - depths_nv).abs()
    # d_diff = d_diff / d_diff.max().clamp_min(1.)
    d_diff = d_diff / depths_nv.clamp_min(1.)
    
    imgs_cond = data["imgs_cond_formatted"].permute(0, 3, 1, 4, 2).reshape(b, h, data["imgs_cond_formatted"].size(-1)*nv, 3).cpu().float()
    depths_synth = color_tensor(invert_depth(depths_synth, z_near, z_far), cmap_magma, norm=False) 
    d_diff = color_tensor(d_diff, cmap_magma, norm=False) 
    
    # Vstack re-rendered versions,
    h_, w_ = data["imgs_reren"].shape[-2:]
    imgs_reren = data["imgs_reren"].permute(1, 2, 0, 4, 5, 3).reshape(b, 1, h_*num_v, w_, 3).cpu().clamp(0, 1)
    depths_reren = data["depths_reren"].permute(1, 2, 0, 4, 5, 3).reshape(b, 1, h_*num_v, w_, 1).cpu()
    
    profiles = data["profiles"].cpu()
    
    # nv_idx = 0
    
    for i in range(b):
        
        # Input img
        fig, ax = plt.subplots(1, 1, layout="tight")
        ax.imshow(imgs_in[i, 0])
        ax.set(xticks=[], yticks=[])
        set_spines(ax, visible=False)
        fig.savefig(
            os.path.join(output_dir, f"{idxs[i]:04d}_input.pdf"),
            bbox_inches='tight', 
            pad_inches=0,
        )
        plt.close(fig)
        
        # Reproj error debug img
        error_img = torch.concat([
            torch.concat([imgs_in, torch.zeros_like(imgs_in)], dim=-2)[i, 0],
            torch.concat([imgs_reren[i, 0], data["valid_mask_b"].reshape(-1, w_, 1).repeat(1, 1, 3).cpu()], dim=1),
        ], dim=0)
        fig, ax = plt.subplots(1, 1, layout="tight")
        ax.imshow(error_img)
        ax.set(xticks=[], yticks=[])
        set_spines(ax, visible=False)
        fig.savefig(
            os.path.join(output_dir, f"{idxs[i]:04d}_error.pdf"),
            bbox_inches='tight', 
            pad_inches=0,
        )
        plt.close(fig)  
        
        # Thesis plot img
        fig, ax = plt.subplots(4, 1, figsize=(PAGE_WIDTH_INCHES, 3), layout="tight", gridspec_kw={"hspace": 0.05})
        
        ax[0].imshow(imgs_cond[i])
        ax[0].set(xticks=[], yticks=[], ylabel="$I_{cond}$")
        set_spines(ax[0], visible=True, lw=0.1)
        
        ax[1].imshow(imgs_synth[i])
        ax[1].set(xticks=[], yticks=[], ylabel="$I_{pred}$")
        set_spines(ax[1], visible=False)

        ax[2].imshow(depths_synth[i])
        ax[2].set(xticks=[], yticks=[], ylabel="$inv(d_{pred})$")
        set_spines(ax[2], visible=False)         

        ax[3].imshow(d_diff[i])
        ax[3].set(xticks=[], yticks=[], ylabel="Rel. Err.")
        set_spines(ax[3], visible=False)         
        
        fig.savefig(
            os.path.join(output_dir, f"{idxs[i]:04d}_compact.pdf"),
            bbox_inches='tight', 
            pad_inches=0,
        )
        plt.close(fig)  
        
        
        # Full Overview img
        rows, cols = 4, 2
        fig = plt.figure(figsize=(PAGE_WIDTH_INCHES, rows))
        gs = fig.add_gridspec(rows, cols, width_ratios=[2, num_v*nv], hspace=0., wspace=0.05)
        
        ax = fig.add_subplot(gs[0, 0])
        ax.imshow(imgs_in[i, 0])
        ax.set(xticks=[], yticks=[], ylabel="$I_{in}$")
        set_spines(ax, visible=False)

        ax = fig.add_subplot(gs[0, 1])
        ax.imshow(imgs_cond[i])
        ax.set(xticks=[], yticks=[], ylabel="$I_{cond}$")
        set_spines(ax, visible=False)
        
        ax = fig.add_subplot(gs[1, 1])
        ax.imshow(imgs_synth[i])
        ax.set(xticks=[], yticks=[], ylabel="$I_{pred}$")
        set_spines(ax, visible=False)

        ax = fig.add_subplot(gs[2, 1])
        ax.imshow(depths_synth[i])
        ax.set(xticks=[], yticks=[], ylabel="$inv(d_{pred})$")
        set_spines(ax, visible=False)

        ax = fig.add_subplot(gs[3, 1])
        ax.imshow(d_diff[i])
        ax.set(xticks=[], yticks=[], ylabel="$|d_{nv}-d_{pred}|/d_{nv}$")
        set_spines(ax, visible=False)

        ax = fig.add_subplot(gs[1:3, 0])
        profile = torch.flip(profiles[i], dims=(-3,)).squeeze().cpu().float()
        plot_profile(ax, profile, data["vis_volume"], data_synth.POSES_NV[i].cpu().float(), data_synth.PROJS_NV[i].cpu().float(), frustum_length=3.0)
        ax.set(xticks=[], yticks=[], xlabel="BEV density field\nwith novel views", ylabel=None, title=None)
        set_spines(ax, visible=False)

        fig.savefig(
            os.path.join(output_dir, f"{idxs[i]:04d}.pdf"),
            bbox_inches='tight', 
            pad_inches=0,
        )
        
        writer.add_figure(f"eval/novel_views_{step}", fig, global_step=step, close=True)
        
        plt.close(fig)
