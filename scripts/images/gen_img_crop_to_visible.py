import sys
sys.path.append(".")

import torch
import matplotlib.pyplot as plt
import matplotlib.gridspec as gridspec

from scripts.inference_setup import *
from utils.plotting import set_spines, set_thesis_rcparams, PAGE_WIDTH_INCHES
from utils.utils import invert_depth
from utils.plotting import cmap_spectral as cmap
from vcm.utils.eval_utils import format_conditioning_imgs
from utils.array_ops import to_tensor_unsqueeze, to
from bts.gt_synthesis.gt_synthesis import Outputs, Data
from bts.ignite_evaluation.evaluator_controlnet_input_view import initialize, InputViewInferenceWrapper
from configs.structured_configs.data_config import DataConfigCO3D

set_thesis_rcparams()


def main():    
    dry_run = False

    indices = [0, 100]

    config = load_and_setup_config(
        # config_name="controlnet_rgb", 
        config_name="controlnet_full", 
        # config_name="controlnet_co3d_full", 
    )
    config.BTS.DATA.CATEGORY_NAME = "bench"
    config.SYNTHETIC_GT.NV_CAM_SAMPLER.MAKE_PROJS_POLICY = "crop_to_visible"
    config.SYNTHETIC_GT.CLOSE_CONDITIONING_INVALID = True
    dataset, out_path = setup_task(config, "figures/crop_to_visible")
    IS_CO3D = isinstance(config.BTS.DATA, DataConfigCO3D)
    
    # from configs.defaults.synthetic_gt_config import OrbitCameraSamplerConfig
    # config.SYNTHETIC_GT.NV_CAM_SAMPLER = OrbitCameraSamplerConfig(Y_LIMS=[10, 10], Z_DIST_LIMS=[2, 2])
    # config.SYNTHETIC_GT.Z_FAR = 10.
    
    NVSModel: InputViewInferenceWrapper = initialize(config, refine_output=False)
    NVSModel.to(device)
    
    z_near = NVSModel.gt_synthesizer.z_near
    z_far = NVSModel.gt_synthesizer.z_far
    
    with torch.no_grad():
        for idx in indices:

            data = dataset[idx]
            data = to_tensor_unsqueeze(data)
            data = to(data, device)
            data_out = NVSModel(data)
            
            out: Outputs = data_out["out_synth"]
            data: Data = data_out["data_synth"]
            debug_data = data_out["data_debug"]
            
            imgs_conditioning = out.IMGS_COND
            imgs_gt = out.IMGS_GT
            masks = data.MASKS_REREN

            j = 0
            nrows = len(imgs_conditioning)*2
            fig = plt.figure(figsize=(PAGE_WIDTH_INCHES, nrows*1.1), layout="constrained")
            gs = gridspec.GridSpec(nrows, 5, width_ratios=[1 if IS_CO3D else 3, 1, 1, 1, 1], height_ratios=[1]*nrows, wspace=0.05)
            B = len(imgs_conditioning)
            for i in range(B):
                show_left_right_bounds = debug_data["min_left_nv"] is not None
                
                img_in = data.IMGS[i, 0].cpu().permute(1, 2, 0)
            
                ax = fig.add_subplot(gs[i*2+0, 0])
                ax.set(xticks=[], yticks=[], title="Input")
                ax.imshow(img_in.clamp(0, 1))
                ax.axvline(debug_data["min_left_reren"][i, j].cpu(), linewidth=1, c=cmap(j/B))
                ax.axvline(debug_data["max_right_reren"][i, j].cpu(), linewidth=1, c=cmap(j/B))
                if show_left_right_bounds:
                    set_spines(ax, "blue", lw=1)
            
                ax = fig.add_subplot(gs[i*2+1, 0])
                ax.set(xticks=[], yticks=[], title="$I_{nv}$")
                img_nv = data.IMGS_NV[i, j].cpu().permute(1, 2, 0)
                if show_left_right_bounds:
                    l = debug_data["lefts_nv_crop"][i, j].cpu()
                    ax.imshow(img_nv.clamp(0, 1), extent=[
                        l, l+img_nv.shape[1], 0, img_nv.shape[0]
                    ])
                    ax.axvline(debug_data["min_left_nv"][i, j].cpu(), linewidth=1, c="b")
                    ax.axvline(debug_data["max_right_nv"][i, j].cpu(), linewidth=1, c="b")
                    set_spines(ax, cmap(j/B), lw=1)
                else:
                    ax.imshow(img_nv)
            
                ax = fig.add_subplot(gs[i*2, 1])
                im = invert_depth(data.DEPTHS_CROPPED_TO_REREN[i, 0, 0].cpu(), z_near, z_far)
                ax.imshow(im.clamp(0, 1), interpolation="none")
                ax.set(xticks=[], yticks=[], title="$D_{in}$")
                set_spines(ax, visible=False)
                
                ax = fig.add_subplot(gs[i*2, 2])
                im = invert_depth(data.DEPTHS_REREN[i, 0, 0].cpu(), z_near, z_far)
                ax.imshow(im.clamp(0, 1), interpolation="none")
                ax.set(xticks=[], yticks=[], title="$D_{reren}$")
                set_spines(ax, visible=False)

                ax = fig.add_subplot(gs[i*2, 3])
                diff = (data.DEPTHS_REREN - data.DEPTHS_CROPPED_TO_REREN).abs()[i, 0, 0].cpu()
                ax.imshow(diff/diff.max().clamp(0, 1), interpolation="none")
                ax.set(xticks=[], yticks=[], title="Depth Err.")
                set_spines(ax, visible=False)
                
                ax = fig.add_subplot(gs[i*2, -1])
                mask = masks[i, j].cpu().permute(1, 2, 0)
                mask = torch.concat([
                    # debug_data["occlusion_masks_dict"]["occlusions_depth"][i, j],
                    # debug_data["occlusion_masks_dict"]["occlusions_depth_post"][i, j],
                    masks[i, j].squeeze(), 
                ], dim=-1).cpu()
                mask = (mask-mask.min())/(mask.max()-mask.min())
                ax.imshow(mask, interpolation="none")
                ax.set(xticks=[], yticks=[], title="$M_{occl}$")
                set_spines(ax, visible=False)
                
                ax = fig.add_subplot(gs[i*2+1, 1])
                ax.imshow((imgs_gt[i].cpu().permute(1, 2, 0) * 0.5 + 0.5).clamp(0, 1))
                ax.set(xticks=[], yticks=[], title="$I_{gt}$")
                set_spines(ax, visible=False)
                
                im = format_conditioning_imgs(imgs_conditioning[i]).permute(1, 2, 0).cpu()
                cond_ax_idx = slice(2, None) if im.shape[0] != im.shape[1] else 2
                ax = fig.add_subplot(gs[i*2+1, cond_ax_idx])
                ax.imshow(im.clamp(0, 1))
                ax.set(xticks=[], yticks=[], title="$I_{cond}$")
                set_spines(ax, visible=False)

            if not dry_run:
                filepath = os.path.join(out_path, f"crop_to_visible_{idx:010d}.pdf")
                print(f"Saving to {filepath}")
                fig.savefig(
                    filepath,
                    bbox_inches='tight', 
                    pad_inches=0,
                )
                plt.close(fig)

    print("Completed.")


if __name__ == '__main__':
    main()
