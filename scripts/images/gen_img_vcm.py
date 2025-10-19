import sys
sys.path.append(".")

import torch
import matplotlib.pyplot as plt
import matplotlib.gridspec as gridspec

from scripts.inference_setup import *
from utils.plotting import set_spines, set_thesis_rcparams, PAGE_WIDTH_INCHES, color_tensor, cmap_magma
from utils.utils import invert_depth
from utils.plotting import cmap_spectral as cmap
from vcm.utils.eval_utils import format_conditioning_imgs
from utils.array_ops import to_tensor_unsqueeze, to
from bts.gt_synthesis.gt_synthesis import Outputs, Data
from bts.ignite_evaluation.evaluator_controlnet_input_view import initialize, InputViewInferenceWrapper
from configs.structured_configs.data_config import DataConfigCO3D
from matplotlib.colors import LinearSegmentedColormap

set_thesis_rcparams()
cmap_custom = LinearSegmentedColormap.from_list("custom_green_black", ["#fcf9ba", "#260e46"], N=256)


def main():    
    dry_run = False

    indices = [0
            #    , 50, 100, 150, 100, 300, 400
    ]

    config = load_and_setup_config(
        # config_name="controlnet_rgb", 
        # config_name="controlnet_co3d_full", 
        config_name="controlnet_full_512x768",
    )
    # config.BTS.DATA.CATEGORY_NAME = "bench"
    # config.SYNTHETIC_GT.NV_CAM_SAMPLER.MAKE_PROJS_POLICY = "crop_to_visible"
    # config.SYNTHETIC_GT.CLOSE_CONDITIONING_INVALID = True
    dataset, out_path = setup_task(config, "figures/controlnet_training_data")
    
    IS_CO3D = isinstance(config.BTS.DATA, DataConfigCO3D)
    
    from configs.structured_configs.synthetic_gt_config import ShiftCameraSamplerConfig
    config.SYNTHETIC_GT.NV_CAM_SAMPLER = ShiftCameraSamplerConfig(X_LIMS=(2,2))
    
    # config.SYNTHETIC_GT.Z_FAR = 10.

    NVSModel: InputViewInferenceWrapper = initialize(config, refine_output=True)
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
                depths_cropped_to_reren = invert_depth(data.DEPTHS_CROPPED_TO_REREN[i, 0, 0].cpu(), z_near, z_far)
                ax.imshow(depths_cropped_to_reren.clamp(0, 1), interpolation="none")
                ax.set(xticks=[], yticks=[], title="$D_{in}$")
                set_spines(ax, visible=False)
                
                ax = fig.add_subplot(gs[i*2, 2:4])
                depths_reren = invert_depth(data.DEPTHS_REREN[i, 0, 0].cpu(), z_near, z_far)
                diff = (data.DEPTHS_REREN - data.DEPTHS_CROPPED_TO_REREN).abs()[i, 0, 0].cpu()
                ax.imshow(torch.concat([depths_reren, diff/diff.max()], dim=-1).clamp(0, 1), interpolation="none")
                ax.set(xticks=[], yticks=[], title="Depths")
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
                ax.set(xticks=[], yticks=[], title="Occlusion Mask")
                set_spines(ax, visible=False)
                
                ax = fig.add_subplot(gs[i*2+1, 1])
                ax.imshow((imgs_gt[i].cpu().permute(1, 2, 0) * 0.5 + 0.5).clamp(0, 1))
                ax.set(xticks=[], yticks=[], title="$I_{gt}$")
                set_spines(ax, visible=False)
                
                cond = format_conditioning_imgs(imgs_conditioning[i]).permute(1, 2, 0).cpu()
                cond_ax_idx = slice(2, None) if cond.shape[0] != cond.shape[1] else 2
                ax = fig.add_subplot(gs[i*2+1, cond_ax_idx])
                ax.imshow(cond.clamp(0, 1))
                ax.set(xticks=[], yticks=[], title="$I_{cond}$")
                set_spines(ax, visible=False)

                # Save full figure
                filepath = os.path.join(out_path, f"{idx:03d}.pdf")
                print(f"Saving to {filepath}")
                fig.savefig(
                    filepath,
                    bbox_inches='tight', 
                    pad_inches=0,
                    dpi=100,
                )
                plt.close(fig)

                # Save individual figures
                fig, ax = plt.subplots(1, 1)
                ax.imshow(img_nv)
                ax.set(xticks=[], yticks=[])
                set_spines(ax, visible=False)
                fig.savefig(
                    os.path.join(out_path, f"{idx:03d}_rgb_nv.png"),
                    bbox_inches='tight', 
                    pad_inches=0,
                    dpi=100,
                )
                plt.close(fig)

                fig, ax = plt.subplots(1, 1)
                d_nv = data.DEPTHS_NV[i, j, 0].cpu()
                d_nv = color_tensor(invert_depth(d_nv, z_near, z_far), cmap_magma, norm=False)
                ax.imshow(d_nv)
                ax.set(xticks=[], yticks=[])
                set_spines(ax, visible=False)
                fig.savefig(
                    os.path.join(out_path, f"{idx:03d}_d_nv.png"),
                    bbox_inches='tight', 
                    pad_inches=0,
                    dpi=100,
                )
                plt.close(fig)

                fig, ax = plt.subplots(1, 1)
                d_in = color_tensor(depths_cropped_to_reren.clamp(0, 1), cmap_magma, norm=False)
                ax.imshow(d_in)
                ax.set(xticks=[], yticks=[])
                set_spines(ax, visible=False)
                fig.savefig(
                    os.path.join(out_path, f"{idx:03d}_d_in.png"),
                    bbox_inches='tight', 
                    pad_inches=0,
                    dpi=100,
                )
                plt.close(fig)

                fig, ax = plt.subplots(1, 1)
                rgb_in = data.IMGS_CROPPED_TO_REREN[i, 0].cpu().permute(1, 2, 0)
                ax.imshow(rgb_in.clamp(0, 1))
                ax.set(xticks=[], yticks=[])
                set_spines(ax, visible=False)
                fig.savefig(
                    os.path.join(out_path, f"{idx:03d}_rgb_in.png"),
                    bbox_inches='tight', 
                    pad_inches=0,
                    dpi=100,
                )

                fig, ax = plt.subplots(1, 1)
                d_reren = color_tensor(depths_reren.clamp(0, 1), cmap_magma, norm=False)
                ax.imshow(d_reren)
                ax.set(xticks=[], yticks=[])
                set_spines(ax, visible=False)
                fig.savefig(
                    os.path.join(out_path, f"{idx:03d}_d_reren.png"),
                    bbox_inches='tight', 
                    pad_inches=0,
                    dpi=100,
                )
                plt.close(fig)

                fig, ax = plt.subplots(1, 1)
                rgb_reren = data.IMGS_REREN[i, 0].cpu().permute(1, 2, 0)
                ax.imshow(rgb_reren.clamp(0, 1))
                ax.set(xticks=[], yticks=[])
                set_spines(ax, visible=False)
                fig.savefig(
                    os.path.join(out_path, f"{idx:03d}_rgb_reren.png"),
                    bbox_inches='tight', 
                    pad_inches=0,
                    dpi=100,
                )

                fig, ax = plt.subplots(1, 1)
                mask = masks[i, j, 0].cpu()
                ax.imshow((mask == 1).float(), interpolation="none", cmap=cmap_custom)
                ax.set(xticks=[], yticks=[])
                set_spines(ax, visible=False)
                fig.savefig(
                    os.path.join(out_path, f"{idx:03d}_mask.png"),
                    bbox_inches='tight', 
                    pad_inches=0,
                    dpi=100,
                )
                plt.close(fig)

                fig, ax = plt.subplots(1, 1)
                ax.imshow(cond.cpu())
                ax.set(xticks=[], yticks=[])
                set_spines(ax, visible=False)
                fig.savefig(
                    os.path.join(out_path, f"{idx:03d}_cond.png"),
                    bbox_inches='tight', 
                    pad_inches=0,
                    dpi=100,
                )
                plt.close(fig)

                fig, ax = plt.subplots(1, 1)
                imgs_refined, depths_out, imgs_conditioning, (h_out_new, w_out_new) = NVSModel.gt_synthesizer.refine_and_predict_depth(data)
                ax.imshow(imgs_refined[0, 0].permute(1, 2, 0).cpu().clamp(0, 1))
                ax.set(xticks=[], yticks=[])
                set_spines(ax, visible=False)
                fig.savefig(
                    os.path.join(out_path, f"{idx:03d}_rgb_synth.png"),
                    bbox_inches='tight', 
                    pad_inches=0,
                    dpi=100,
                )
                plt.close(fig)


    print("Completed.")


if __name__ == '__main__':
    main()
