import os
import time

import torch
from ignite.contrib.handlers import TensorboardLogger
from ignite.engine import Engine
from torchvision.transforms.functional import InterpolationMode, resize
import matplotlib.pyplot as plt

from bts.ignite_evaluation.base_evaluator import base_evaluation
from utils.metrics import MeanMetric
from bts.losses.utils import compute_depth_metrics, compute_rgb_metrics
from bts.gt_synthesis.gt_synthesis import Outputs, Data
from configs.structured_configs.main_config import MainConfig
from bts.ignite_evaluation.evaluator import InferenceWrapper, get_dataflow
from vcm.utils.eval_utils import format_conditioning_imgs
from utils.occlusion_ops import DEFINITIONS
from utils.utils import invert_depth
from utils.plotting import color_tensor, cmap_magma, cmap_jet, cmap_spectral, set_thesis_rcparams, set_spines, PAGE_WIDTH_INCHES
import matplotlib.colors as mcolors
from mpl_toolkits.axes_grid1 import make_axes_locatable


class InputViewInferenceWrapper(InferenceWrapper):
    def __init__(self, *args, **kwargs) -> None:
        super().__init__(*args, **kwargs)

    def forward(self, data):
        data = super().forward(data, forward_type="controlnet_input_view")

        out: Outputs = data["out_synth"]
        synth_data: Data = data["data_synth"]

        results = {}
        if self.refine_output:
            time_start = time.time()
            imgs_versions = []
            depths_versions = []
            scales, shifts = [], []
            for _ in range(self.gt_synthesizer.num_synthetic_versions):
                synth_data.GENERATOR = None
                imgs_refined, depths_out, imgs_conditioning_out, hw_out_new = self.gt_synthesizer.refine_and_predict_depth(synth_data)
                depths_out, scale, shift = self.gt_synthesizer._align_depth(
                    depths_out, synth_data.DEPTHS_REREN, synth_data.MASKS_REREN == DEFINITIONS.IS_VISIBLE
                )
                scales.append(scale), shifts.append(shift)
                imgs_versions.append(imgs_refined)
                depths_versions.append(depths_out)
                    
            imgs_gt = resize(out.IMGS_GT, hw_out_new, interpolation=InterpolationMode.BILINEAR, antialias=None).float()
            imgs_gt = imgs_gt * 0.5 + 0.5
            depths_gt = resize(out.DEPTHS_GT, hw_out_new, interpolation=InterpolationMode.BILINEAR, antialias=None).float()
            imgs_cond = format_conditioning_imgs(imgs_conditioning_out.view(-1, *imgs_conditioning_out.shape[-3:])).permute(0, 2, 3, 1).float()
            
            torch.cuda.synchronize()
            per_sample_time = (time.time() - time_start) / (synth_data.B * self.gt_synthesizer.num_synthetic_versions)      # NV should be 1, so we can ignore it.
            
            imgs_synth = torch.stack(imgs_versions).float()         
            depths_synth = torch.stack(depths_versions).float()
            valid_mask = (synth_data.MASKS_REREN == DEFINITIONS.IS_VISIBLE)
            
            # Shape [num_versions, b, nv, c, h, w] -> [num_versions * b * nv, c, h, w]
            _, _, _, _, h, w = imgs_synth.shape
            imgs_synth_b = imgs_synth.view(-1, 3, h, w)
            imgs_gt_b = imgs_gt.unsqueeze(0).unsqueeze(0).expand_as(imgs_synth).view(-1, 3, h, w)
            depths_synth_b = depths_synth.view(-1, 1, h, w)
            depths_gt_b = depths_gt.unsqueeze(0).unsqueeze(0).expand_as(depths_synth).view(-1, 1, h, w)
            valid_mask_b = valid_mask.unsqueeze(-3).expand_as(depths_synth).view(-1, 1, h, w)
            
            rgb_metrics = compute_rgb_metrics(imgs_synth_b, imgs_gt_b, lpips_net=self.lpips_net)
            depth_metrics = compute_depth_metrics(depths_synth_b, depths_gt_b)
            rgb_metrics_masked = compute_rgb_metrics(imgs_synth_b, imgs_gt_b, valid_mask=valid_mask_b.repeat(1, 3, 1, 1))
            depth_metrics_masked = compute_depth_metrics(depths_synth_b, depths_gt_b, valid_mask=valid_mask_b)
            
            depth_var = depths_synth.var(dim=0)
            results = {
                "imgs_synth": imgs_synth, 
                "depths_synth": depths_synth, 
                "depth_var_pixel": depth_var,
                "t_denoising": per_sample_time,
                "depth_mean": float(depths_synth.mean()),
                "depth_var": 0. if torch.isnan(depth_var).any() else float(depth_var.mean()),
                "scale": float(torch.stack(scales).mean()),
                "shift": float(torch.stack(shifts).mean()),
                **{f"rgb_{k}": float(v) for k, v in rgb_metrics.items()},
                **{f"rgb_{k}_masked": float(v) for k, v in rgb_metrics_masked.items()},
                **{f"depth_{k}": float(v) for k, v in depth_metrics.items()},
                **{f"depth_{k}_masked": float(v) for k, v in depth_metrics_masked.items()},
            }
        else:
            imgs_gt = out.IMGS_GT
            depths_gt = out.DEPTHS_GT
            imgs_cond = format_conditioning_imgs(out.IMGS_COND.view(-1, *out.IMGS_COND.shape[-3:])).permute(0, 2, 3, 1).float()
            
        data.update({
            "prompts": synth_data.CAPTION_STRS,
            "z_near": self.z_near,
            "z_far": self.z_far,
            # "fraction_invalid": float((synth_data.MASKS_REREN == DEFINITIONS.IS_INVALID).float().mean()),
            "fraction_invalid": float(synth_data.INVALID_REREN.float().mean()),
            "fraction_occluded": float((synth_data.MASKS_REREN == DEFINITIONS.IS_OCCLUDED).float().mean()),
            # Maps
            "imgs_in": imgs_gt, 
            "depths_in": depths_gt, 
            "imgs_cond": imgs_cond,
            # Refinement metrics
            **results
        })

        return data


def evaluation(local_rank: int, config: MainConfig):
    return base_evaluation(local_rank, config, get_dataflow, initialize, get_metrics, visualize)


def get_metrics(config, device):
    rgb_names = ["rmse", "psnr", "ssim", "lpips"]
    depth_names = ["abs_rel", "sq_rel", "rmse", "rmse_log", "a1", "a2", "a3"]
    names = (
        [f"rgb_{_}" for _ in rgb_names] + 
        [f"rgb_{_}_masked" for _ in rgb_names[:-2]] + 
        [f"depth_{_}" for _ in depth_names] + 
        [f"depth_{_}_masked" for _ in depth_names] + 
        ["t_denoising", "scale", "shift", "depth_mean", "depth_var"] +
        ["fraction_invalid", "fraction_occluded"]
    )
    metrics = {name: MeanMetric((lambda n: lambda x: x["output"][n])(name), device) for name in names}
    return metrics


def initialize(cfg: MainConfig, refine_output=True, **kwargs):
    return InputViewInferenceWrapper.from_conf(cfg, refine_output)


def visualize(engine: Engine, logger: TensorboardLogger, output_dir: str, step: int):

    set_thesis_rcparams()
    
    data = engine.state.output["output"]
    writer = logger.writer
    
    z_near, z_far = data["z_near"], data["z_far"]
    idxs = data["idxs"].flatten().tolist()
    # Extract and stitch image versions together.
    num_v, b, nv, _, h, w = data["imgs_synth"].shape
    assert nv == 1
    imgs_synth = data["imgs_synth"].permute(1, 2, 4, 0, 5, 3).reshape(b, h, w*num_v, 3).cpu()
    imgs_in = data["imgs_in"].permute(0, 2, 3, 1).cpu()
    depths_synth = data["depths_synth"].permute(1, 2, 4, 0, 5, 3).reshape(b, h, w*num_v).cpu()
    depths_in = data["depths_in"].squeeze(1).cpu()
    imgs_cond = data["imgs_cond"].cpu()
    # Stack images together, invert and color depth images.
    rgb = torch.concatenate([imgs_cond, imgs_synth], dim=2)
    
    cmap_inv = cmap_magma
    cmap_direct = cmap_spectral
    cmap_diff = cmap_jet
    
    d_direct = depths_synth
    d_inv = invert_depth(d_direct, z_near, z_far)
    d_inv = color_tensor(d_inv, cmap_inv, norm=False)
    
    norm_direct = mcolors.Normalize(vmin=d_direct.min(), vmax=d_direct.max())
    d_direct = color_tensor(torch.tensor(norm_direct(d_direct)), cmap_direct, norm=False)
    
    d_diff = torch.abs(depths_in.repeat(1, 1, num_v) - depths_synth)
    norm_diff = mcolors.Normalize(vmin=0., vmax=d_diff.max())
    d_diff = color_tensor(torch.tensor(norm_diff(d_diff)), cmap_diff, norm=False)

    d_gt = color_tensor(torch.tensor(norm_direct(depths_in)), cmap_direct, norm=False)    
    d_gt_inv = color_tensor(invert_depth(depths_in, z_near, z_far), cmap_inv, norm=False)    
    
    for i in range(b):
        # Thesis plot img
        cols, rows = 2, 3
        fig = plt.figure(figsize=(PAGE_WIDTH_INCHES/2, 1.5), layout="constrained")
        gs = fig.add_gridspec(rows, cols, height_ratios=[1]*rows, width_ratios=[3, 1], hspace=0.05, wspace=0.)
        
        ax = fig.add_subplot(gs[0, 0])
        ax.imshow(imgs_cond[i])
        ax.set(xticks=[], yticks=[], ylabel="$I_{cond}$")
        set_spines(ax, visible=True, lw=0.5)
        
        ax = fig.add_subplot(gs[0, 1])
        ax.yaxis.tick_right()
        ax.yaxis.set_label_position("right")
        ax.set(xticks=[], yticks=[], ylabel="$inv(d)$")
        ax.imshow(d_gt_inv[i])
        set_spines(ax, visible=False)
        
        ax = fig.add_subplot(gs[1, :])
        ax.imshow(imgs_synth[i])
        ax.set(xticks=[], yticks=[], ylabel="$I_{pred}$")
        set_spines(ax, visible=False)

        ax = fig.add_subplot(gs[2, :])
        ax.imshow(d_inv[i])
        ax.set(xticks=[], yticks=[], ylabel="$inv(d_{pred})$")
        set_spines(ax, visible=False)
        # SQUARE_IMG = h == w
        # cols, rows = 2, 2 if SQUARE_IMG else 3
        # fig = plt.figure(figsize=(PAGE_WIDTH_INCHES, 3.), layout="constrained")
        # gs = fig.add_gridspec(rows, cols, width_ratios=[1, num_v], height_ratios=[1]*rows, hspace=0.05, wspace=0.05)
        
        # ax = fig.add_subplot(gs[0, 0] if SQUARE_IMG else gs[0, 1])
        # ax.imshow(imgs_cond[i])
        # ax.set(xticks=[], yticks=[], ylabel="$I_{cond}$")
        # set_spines(ax, visible=True, lw=0.5)
        
        # ax = fig.add_subplot(gs[1, 0] if SQUARE_IMG else gs[2, 0])
        # ax.set(xticks=[], yticks=[], ylabel="$inv(d)$")
        # ax.imshow(d_gt_inv[i])
        # set_spines(ax, visible=False)
        
        # ax = fig.add_subplot(gs[0, 1] if SQUARE_IMG else gs[1, 1])
        # ax.imshow(imgs_synth[i])
        # ax.set(xticks=[], yticks=[], ylabel="$I_{pred}$")
        # set_spines(ax, visible=False)

        # ax = fig.add_subplot(gs[1, 1] if SQUARE_IMG else gs[2, 1])
        # ax.imshow(d_inv[i])
        # ax.set(xticks=[], yticks=[], ylabel="$inv(d_{pred})$")
        # set_spines(ax, visible=False)
        
        fig.savefig(
            os.path.join(output_dir, f"{idxs[i]:04d}_compact.pdf"),
            bbox_inches='tight', 
            pad_inches=0,
        )
        plt.close(fig)  
                
        # Main plot
        cols, rows = 2, 4
        fig = plt.figure(figsize=(PAGE_WIDTH_INCHES, 3.5), layout="constrained")
        gs = fig.add_gridspec(rows, cols, width_ratios=[1, num_v], hspace=0., wspace=0.05)
        
        ax = fig.add_subplot(gs[0, 0])
        ax.imshow(imgs_cond[i])
        ax.set(xticks=[], yticks=[], ylabel="$I_{cond}$")
        set_spines(ax, visible=False)
        ax = fig.add_subplot(gs[1, 0])
        ax.imshow(d_gt_inv[i])
        ax.set(xticks=[], yticks=[], ylabel="$inv(d_{GT})$")
        set_spines(ax, visible=False)
        ax = fig.add_subplot(gs[2, 0])
        ax.imshow(d_gt[i])
        ax.set(xticks=[], yticks=[], ylabel="$d_{GT}$")
        set_spines(ax, visible=False)
        
        # Input img + Synthetic image re-projected to the input view
        ax = fig.add_subplot(gs[0, 1:])
        ax.set(xticks=[], yticks=[], ylabel="$I_{pred}$")
        ax.imshow(imgs_synth[i])
        set_spines(ax, visible=False)
        ax = fig.add_subplot(gs[1, 1:])
        ax.set(xticks=[], yticks=[], ylabel="$inv(d)$")
        ax.imshow(d_inv[i])
        set_spines(ax, visible=False)
        
        ax = fig.add_subplot(gs[2, 1:])
        ax.imshow(d_direct[i])
        divider = make_axes_locatable(ax)
        fig.colorbar(
            plt.cm.ScalarMappable(norm=norm_direct, cmap=cmap_direct), 
            cax=divider.append_axes('right', size='2%', pad=0.05), 
            label="[m]"
        )
        ax.set(xticks=[], yticks=[], ylabel="$d$")
        set_spines(ax, visible=False)
        
        ax = fig.add_subplot(gs[3, 1:])
        ax.imshow(abs(d_diff[i]))
        divider = make_axes_locatable(ax)
        fig.colorbar(
            plt.cm.ScalarMappable(norm=norm_diff, cmap=cmap_diff),
            cax=divider.append_axes('right', size='2%', pad=0.05), 
            label="[m]"
        )
        ax.set(xticks=[], yticks=[], ylabel="$|d_{GT}-d|$")
        set_spines(ax, visible=False)

        fig.savefig(
            os.path.join(output_dir, f"{idxs[i]:04d}.pdf"),
            bbox_inches='tight', 
            pad_inches=0,
        )
        
        writer.add_figure(f"eval/input_views_{step}", fig, global_step=step, close=True)
        
        plt.close(fig)
