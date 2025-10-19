import sys
sys.path.append(".")

import torch
import matplotlib.pyplot as plt
import matplotlib.gridspec as gridspec
from matplotlib.colors import LinearSegmentedColormap

from scripts.inference_setup import *
from utils.array_ops import to_tensor_unsqueeze, to
from bts.ignite_evaluation.evaluator_controlnet_novel_view import initialize, NovelViewInferenceWrapper
from bts.gt_synthesis.gt_synthesis import Outputs, Data
from utils.plotting import cmap_spectral, set_spines, HALF_PAGE_WIDTH_INCHES, set_thesis_rcparams

cmap = cmap_spectral
set_thesis_rcparams()

# #fcf9ba = rgb(252,249,186) = pale yellow
# #260e46 = rgb(38,14,70) = aubergine

cmap_custom = LinearSegmentedColormap.from_list("custom_green_black", ["#fcf9ba", "#260e46"], N=256)
cmap_custom_green_black = LinearSegmentedColormap.from_list("custom_green_black", ["#39FF14", "black"], N=256)

def main():

    ADD_LABELS = False
    dry_run = False

    indices = [
        0, 
        # 1, 2, 3, 4, 100, 200
    ]

    config = load_and_setup_config(
        # config_name="exp_bts_synthetic_rig", 
        config_name="exp_recon_full", 
        # data_config_name="co3d",
    )
    config.BTS.DATA.CATEGORY_NAME = "hydrant"
    dataset, out_path = setup_task(config, "figures/occlusion_detection", return_dataset="val")
    
    config.SYNTHETIC_GT.INFERENCE_OCCLUSIONS.USE_FLOW = True
    config.SYNTHETIC_GT.INFERENCE_OCCLUSIONS.USE_DEPTH_GRADS = True
    
    from configs.structured_configs.synthetic_gt_config import Rig4CameraSamplerConfig
    config.SYNTHETIC_GT.NV_CAM_SAMPLER = Rig4CameraSamplerConfig(NUM_NOVEL_VIEWS=4)
    # config.SYNTHETIC_GT.NV_CAM_SAMPLER.NUM_NOVEL_VIEWS = 4
    config.SYNTHETIC_GT.TOP_K_NOVEL_VIEWS_TO_KEEP = 4
    
    NVSModel: NovelViewInferenceWrapper = initialize(config, refine_output=False)
    NVSModel.to(device)
    
    with torch.no_grad():
        for idx in indices:
            data = dataset[idx]
            data = to_tensor_unsqueeze(data)
            data = to(data, device)
            data = NVSModel(data)
            
            out: Outputs = data["out_synth"]
            data_synth: Data = data["data_synth"]
            debug_data = data["data_debug"]

            # Setup figure
            B_IDX = 0
            NV_IDXS = [0, 1, 3]
            NV = len(NV_IDXS)
            TOP = 0 # 1
            TOP_PADDING = 0 # 0
            height_ratios=[1] * NV # [1, 0.5] + [1] * NV
            fig = plt.figure(figsize=(HALF_PAGE_WIDTH_INCHES, sum(height_ratios)*0.29))
            gs = gridspec.GridSpec(NV+TOP+TOP_PADDING, 5, height_ratios=height_ratios, hspace=0.05, wspace=0.05)
            for i, v_idx in enumerate(NV_IDXS):
                row = i + TOP + TOP_PADDING
                
                ax = fig.add_subplot(gs[row, 0])
                ax.imshow(debug_data["input_crops_for_optflow"][B_IDX, v_idx].cpu().permute(1, 2, 0))
                ax.set(xticks=[], yticks=[])
                if i == 0 and ADD_LABELS:
                    ax.set_title(r"$I_\text{in}$")
                set_spines(ax, visible=False)

                ax = fig.add_subplot(gs[row, 1])
                ax.set(xticks=[], yticks=[])#, ylabel=f"{v_idx+1}")
                img_nv = data_synth.IMGS_NV[B_IDX, v_idx].cpu().permute(1, 2, 0).clamp(0, 1)      # Slightly larger than 1. for some reason...
                ax.imshow(img_nv)
                set_spines(ax, visible=False)
                if i == 0 and ADD_LABELS:
                    ax.set_title("$I_{nv}$")

                # ax = fig.add_subplot(gs[row, 2])
                # occl_depth_grad = debug_data["occlusion_masks_dict"]["occlusions_depth_grads"].cpu()[v_idx, 0]
                # ax.imshow(occl_depth_grad, interpolation="none", cmap=cmap_custom)
                # ax.set(xticks=[], yticks=[])
                # if v_idx == 0:
                #     ax.set_title(r"$O_\text{grad}$")
                # set_spines(ax, visible=False)

                ax = fig.add_subplot(gs[row, 2])
                occl_depth_grad_post = debug_data["occlusion_masks_dict"]["occlusions_depth_grads_post"].cpu()[v_idx, 0]
                ax.imshow(occl_depth_grad_post, interpolation="none", cmap=cmap_custom)
                ax.set(xticks=[], yticks=[])
                if v_idx == 0 and ADD_LABELS:
                    ax.set_title(r"$O_\text{grad-post}$")
                set_spines(ax, visible=False)

                # occl_flow = debug_data["occlusion_masks_dict"]["occlusions_flow"].cpu()[v_idx, 0]
                # ax = fig.add_subplot(gs[row, 4])
                # ax.imshow(occl_flow, interpolation="none", cmap=cmap_custom)
                # ax.set(xticks=[], yticks=[])
                # if v_idx == 0:
                #     ax.set_title(r"$O_\text{flow}$")
                # set_spines(ax, visible=False)

                occl_flow_post = debug_data["occlusion_masks_dict"]["occlusions_flow_post"].cpu()[v_idx, 0]
                ax = fig.add_subplot(gs[row, 3])
                ax.imshow(occl_flow_post, interpolation="none", cmap=cmap_custom)
                ax.set(xticks=[], yticks=[])
                if i == 0 and ADD_LABELS:
                    ax.set_title(r"$O_\text{flow-post}$")
                set_spines(ax, visible=False)

                ax = fig.add_subplot(gs[row, 4])
                ax.set(xticks=[], yticks=[])
                ax.imshow(debug_data["occlusion_masks_dict"]["occlusions_full"].cpu()[v_idx, 0], interpolation="none", cmap=cmap_custom)
                if i == 0 and ADD_LABELS:
                    ax.set_title("$O_{fused}$")
                set_spines(ax, visible=False)
                
            if not dry_run:
                plt.tight_layout(pad=0, h_pad=0, w_pad=0)
                filepath = os.path.join(out_path, f"occlusions_{idx:03d}_labels={ADD_LABELS}.png")
                print(f"Saving to {filepath}")
                fig.savefig(
                    filepath + ".pdf",
                    # bbox_inches=None, 
                    bbox_inches='tight', 
                    pad_inches=0,
                )
                fig.savefig(
                    filepath + ".png",
                    # bbox_inches=None, 
                    bbox_inches='tight', 
                    pad_inches=0,
                )
                plt.close(fig)

    print("Completed.")


if __name__ == '__main__':
    main()
