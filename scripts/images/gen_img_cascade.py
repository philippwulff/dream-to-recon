from typing import List
import sys
sys.path.append(".")
import shutil

import torch
from scripts.inference_setup import *
from utils.array_ops import to_tensor_unsqueeze, to
from utils.plotting import set_spines, PAGE_WIDTH_INCHES, set_thesis_rcparams, color_tensor, cmap_magma
from bts.gt_synthesis.gt_synthesis import Outputs, Data
from bts.ignite_evaluation.evaluator import InferenceWrapper
from configs.structured_configs.synthetic_gt_config import ExplorationCameraSamplerConfig
import matplotlib.pyplot as plt
from utils.utils import invert_depth
from vcm.utils.eval_utils import format_conditioning_imgs

set_thesis_rcparams()

def main():
    
    PLOT_SINGLE_SAMPLER_OVER_TIME = False
    LOSS_MOD = False
    MAKE_FULL_VID = True
    NUM_CASC_STEPS = 4
    if MAKE_FULL_VID:
        PLOT_SINGLE_SAMPLER_OVER_TIME = False
        NUM_CASC_STEPS = 1

    indices = [0]
    
    config = load_and_setup_config(
        config_name="eval_controlnet_cascade_base", 
        # config_name="eval_controlnet_cascade", 
        # data_config_name="co3d"
    )
    config.BTS.DATA.CATEGORY_NAME = "hydrant"
    dataset, out_path = setup_task(config, "figures/cascade")
    config.SYNTHETIC_GT.INFERENCE_OCCLUSIONS.POSTPROCESSING_OPS_DEPTH_GRADS=[['opening',3],['closing',15]]
    config.SYNTHETIC_GT.PSEUDO_VOLUME.SET_ONLY_OCCLUSIONS_EMPTY = True
    
    out_path_explore = os.path.join(out_path, "exploration_sampler")
    # Cleanup from previous run
    if os.path.exists(out_path_explore):
        shutil.rmtree(out_path_explore)

    config.SYNTHETIC_GT.CASCADE.TRAJECTORY_SAMPLER_CONFIG.NUM_STEPS = 1
    config.SYNTHETIC_GT.CASCADE.SHOW_PROGRESS_BAR = True
    config.SYNTHETIC_GT.CASCADE.CHOSEN_ANCHOR_TYPE = "explore"
    samplers: List[ExplorationCameraSamplerConfig] = config.SYNTHETIC_GT.CASCADE._generate_default_anchor_configs()[:NUM_CASC_STEPS]
    get_sampler_path = lambda i: os.path.join(out_path_explore, str(i))
    for i, sampler in enumerate(samplers):
        sampler.VISUALIZE = True
        sampler.VISUALIZATION_PATH = get_sampler_path(i)
        # config.SYNTHETIC_GT.NV_CAM_SAMPLER.NUM_PROPOSALS = 4 if PLOT_SINGLE_SAMPLER_OVER_TIME else 9
        sampler.NUM_PROPOSALS = 4 if PLOT_SINGLE_SAMPLER_OVER_TIME else 8
        sampler.NUM_NOVEL_VIEWS = 2 if PLOT_SINGLE_SAMPLER_OVER_TIME else 4
        sampler.NUM_STEPS = 100
        sampler.ZLIMS = [0, 10]
        
        if LOSS_MOD:        
            sampler.LAMBDA_XZ_BOUNDS = 1.0
            sampler.LAMBDA_BETA_BOUNDS = 1.0
            sampler.LAMBDA_OCCLUSION = 0.0
            sampler.LAMBDA_OCCLUSION_CENTER = 1.0
            sampler.LAMBDA_OCCLUSION_EDGES = 1.0
            sampler.LAMBDA_DEPTH = 0.
            sampler.LAMBDA_SIGMAS = 0.0
            sampler.LAMBDA_POSE_SIM = 0.0
            sampler.LAMBDA_WEIGHTS = 0.
        
        num_frames_to_show = 5
        linear_points = torch.linspace(0, 1, num_frames_to_show)
        ts = (linear_points ** 2 * (sampler.NUM_STEPS-1)).round().long().tolist()
        sampler.VISUALIZATION_STEPS = ts if PLOT_SINGLE_SAMPLER_OVER_TIME else ts[-1:]
        if MAKE_FULL_VID:
            sampler.ZLIMS = [0, 15]
            sampler.VISUALIZE_GIF = True
            sampler.VISUALIZATION_STEPS = None
    
    if PLOT_SINGLE_SAMPLER_OVER_TIME:
        config.SYNTHETIC_GT.CASCADE.ANCHOR_SAMPLER_CONFIGS = None
        config.SYNTHETIC_GT.NV_CAM_SAMPLER = samplers[0]
    else:
        config.SYNTHETIC_GT.CASCADE.ANCHOR_SAMPLER_CONFIGS = samplers
    
    model = InferenceWrapper.from_conf(config, refine_output=True)
    model.to(device)
    
    for idx in indices:
    
        if PLOT_SINGLE_SAMPLER_OVER_TIME:
            fig = plt.figure(figsize=(PAGE_WIDTH_INCHES, 4), layout="tight")     # width, height
            gs = fig.add_gridspec(2, num_frames_to_show, height_ratios=[1, 2], hspace=0.05, wspace=0.05)
            sampler_step = 0
            loss_ax = fig.add_subplot(gs[0, :])
            # Make the camera sampler plot on the given ax
            if hasattr(model.cascade_model.gt_synthesizer.camera_sampler, "visualization_loss_plot_axis"):
                del model.cascade_model.gt_synthesizer.camera_sampler.visualization_loss_plot_axis
            model.cascade_model.gt_synthesizer.camera_sampler.visualization_loss_plot_axis = loss_ax
            model.cascade_model.gt_synthesizer.camera_sampler.imgs_right = False
    
        data_batch = dataset[idx]
        data_batch = to_tensor_unsqueeze(data_batch)
        data_batch = to(data_batch, device)
        
        with torch.no_grad():
            data_batch = model(data_batch, forward_type="controlnet_cascade", seed=42)
            
        out: List[Outputs] = data_batch["out_synth_l"]
        data: List[Data] = data_batch["data_synth_l"]
        # debug = data_batch["data_debug"]
        
        _, nv, c, h, w = data[0].IMGS_CONDITIONING.shape
        ver_idx = b_idx = 0
        
        if PLOT_SINGLE_SAMPLER_OVER_TIME:
            # ts = torch.linspace(0, samplers[sampler_step].NUM_STEPS-1, num_frames_to_show).long()
            for frame_i in range(num_frames_to_show):
                ax = fig.add_subplot(gs[1, frame_i])
                t = ts[frame_i]
                ax.imshow(plt.imread(os.path.join(get_sampler_path(sampler_step), "frames", f"{t:04d}.png")))
                ax.set(xticks=[], yticks=[], xlabel=f"Iteration {t}")
                set_spines(ax, visible=False)
                if frame_i == 0:
                    ax.set_ylabel(f"Proposals  &  BEV Profile")

            filepath = os.path.join(out_path, "one_sampler_timeseries")
        
        # --- Plot final sampler time of all cascade steps ---
        else:
            num_samplers_to_show = len(samplers)
            fig = plt.figure(figsize=(PAGE_WIDTH_INCHES, num_samplers_to_show*2), layout="tight")     # width, height
            gs = fig.add_gridspec(num_samplers_to_show, 5, width_ratios=[3, 2, 1, 1, 1], hspace=0.01, wspace=0.01)
            
            for step in range(num_samplers_to_show):
                t = samplers[step].NUM_STEPS-1
                
                ax = fig.add_subplot(gs[step, 0])
                ax.imshow(plt.imread(os.path.join(get_sampler_path(step), "frames", f"{t:04d}.png")))
                ax.set(xticks=[], yticks=[], ylabel=f"Cascade Step {step}", title="BEV" if step==0 else None)
                set_spines(ax, visible=False)
                
                if c < 4:                
                    ax = fig.add_subplot(gs[step, 1])
                    img_file_path = os.path.join(get_sampler_path(step), "losses.png")
                    ax.imshow(plt.imread(img_file_path))
                    ax.set(xticks=[], yticks=[], title="Loss" if step==0 else None)
                    set_spines(ax, visible=False)
                
                    ax = fig.add_subplot(gs[step, -3])
                    im = data[step].IMGS_CONDITIONING[b_idx]
                else:
                    # We have a 4 or 5 channel cond img
                    ax = fig.add_subplot(gs[step, 1:-2])
                    im = format_conditioning_imgs(data[step].IMGS_CONDITIONING[b_idx])
                ax.imshow(im.permute(0, 2, 3, 1).reshape(h*nv, w*(c-2), 3).cpu())
                ax.set(xticks=[], yticks=[], title="$I_{cond}$" if step==0 else None)
                set_spines(ax, visible=False)

                ax = fig.add_subplot(gs[step, -2])
                ax.imshow(out[step].IMGS_SYNTH[ver_idx, b_idx].permute(0, 2, 3, 1).reshape(h*nv, w, 3).cpu())
                ax.set(xticks=[], yticks=[], title="$I_{synth}$" if step==0 else None)
                set_spines(ax, visible=False)

                ax = fig.add_subplot(gs[step, -1])
                im = out[step].DEPTHS_SYNTH[ver_idx, b_idx].permute(0, 2, 3, 1).reshape(h*nv, w).cpu()
                im = invert_depth(im, model.cascade_model.gt_synthesizer.z_near, model.cascade_model.gt_synthesizer.z_far)
                im = color_tensor(im, cmap_magma, norm=False)
                ax.imshow(im)
                ax.set(xticks=[], yticks=[], title="$D_{synth}$" if step==0 else None)
                set_spines(ax, visible=False)
            
            filepath = os.path.join(out_path, "cascade_all_steps")
        
        filepath = f"{filepath}{'_only_occl_loss' if LOSS_MOD else ''}_{idx:010d}.pdf"
        print(f"Saving to {filepath}")
        fig.savefig(filepath, bbox_inches='tight', pad_inches=0)
        plt.close(fig)

    print("Completed.")


if __name__ == '__main__':
    main()

