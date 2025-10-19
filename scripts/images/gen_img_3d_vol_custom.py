"""
Script to produce 3D voxel grid plots.
"""

import sys
sys.path.append(".")
from tqdm.auto import tqdm
import argparse

import torch
import torch.nn.functional as F
from torchvision.utils import save_image
import torch

from scripts.inference_setup import *
from utils.array_ops import to_tensor_unsqueeze, to
from bts.gt_synthesis.gt_synthesis import Outputs, Data
from utils.plotting import cmap_spectral, get_pts, PAGE_WIDTH_INCHES
from utils.plotting_3d import draw_camera_with_frustum, make_3d_fig, draw_ply_mesh, occupancy_grid_to_ply, dpi2scale
from bts.ignite_evaluation.evaluator import initialize, InferenceWrapper
from PIL.Image import Image

cmap = cmap_spectral


def main(img_path: str, model: str):

    DRY_RUN = False
    PLOT_IMAGES = False
    PLOT_NOVEL_VIEWS = False
    PLOT_IN_CAM = True
    THRESHOLD_OCCUPANCY = 0.5

    config_name = {
        "Waymo": "waymo",
        "KITTI-360": "exp_recon_full",
    }[model]

    config = load_and_setup_config(
        config_name=config_name, 
    )
    dataset, out_path = setup_task(config, "images/img_3d_vol_custom", return_dataset="test")

    up = [0, -1, 0]
    scene_camera_eye = [1.1, -1.1, 0]
    frustum_depth = 1.0
    APPLY_INCL_ADJUST = True
    config.BTS.DATA.VIS_VOLUME.X_RANGE = [-9, 9]
    config.BTS.DATA.VIS_VOLUME.Z_RANGE = [4., 25.]
    config.BTS.DATA.VIS_VOLUME.Y_RANGE = [-0.25, 4]
    config.BTS.DATA.VIS_VOLUME.PPM = 7
    
    x_range, y_range, z_range = [
        config.BTS.DATA.VIS_VOLUME.X_RANGE,
        config.BTS.DATA.VIS_VOLUME.Y_RANGE,
        config.BTS.DATA.VIS_VOLUME.Z_RANGE,
    ]
    x_res, y_res, z_res = [
        config.BTS.DATA.VIS_VOLUME.X_RES_PPM,
        config.BTS.DATA.VIS_VOLUME.Y_RES_PPM,
        config.BTS.DATA.VIS_VOLUME.Z_RES_PPM,
    ]
    cam_incl_adjust = config.BTS.DATA.CAM_INCL_ADJUST.to(device)
    
    config.BTS.RESUME_FROM = "/home/stud/wph/storage/user/BTS/out/recon/EXP334_backend-gloo-4_run_0/best_model.pt"
    model_name = "full"

    img = Image.open(img_path)
    img = torch.Tensor(img).permute(1, 2, 0)
    img = ((img / 255) - 0.5) * 2

    # config.EVAL_OCCUPANCY.MODE = "depth_pred"
    config.SYNTHETIC_GT.DEPTH_PREDICTOR_NAME = "Metric3D"
    config.EVAL_OCCUPANCY.MODE = "recon"
    IS_CONTROLNET = False    
    match config.EVAL_OCCUPANCY.MODE:
        case "depth_pred":
            IS_CONTROLNET = True
            wrapper: InferenceWrapper = initialize(config, refine_output=False)
            net = wrapper.gt_synthesizer.renderer.net
            forward_fn = lambda d: wrapper(d, forward_type="controlnet_novel_view")
            suffix = config.SYNTHETIC_GT.DEPTH_PREDICTOR_NAME
        case "cascade":
            IS_CONTROLNET = True
            wrapper: InferenceWrapper = initialize(config, refine_output=True)
            net = wrapper.gt_synthesizer.renderer.net
            forward_fn = lambda d: wrapper(d, forward_type="controlnet_cascade")
            suffix = "cascade_" + config.NAME
        case "recon":
            wrapper: InferenceWrapper = initialize(config, refine_output=False)
            net = wrapper.renderer.net
            forward_fn = lambda d: wrapper(d, forward_type="recon")
            suffix = "recon_" + config.NAME + "_" + "_".join(config.BTS.RESUME_FROM.split("/")[-2:]).replace(".pt", "")
            suffix = model_name if model_name else suffix
    
    wrapper.to(device)
    
    progress_bar = tqdm(
        range(len(indices)),
        desc="Generating 3D plots",
    )

    indices = [0]
    
    with torch.no_grad():
        for idx in indices:
            
            progress_bar.set_postfix_str("Forward pass")
            
            data = dataset[idx]
            data["imgs"] = img[None, :, :, :]

            data = to_tensor_unsqueeze(data)
            data = to(data, device)
            data = forward_fn(data)
            
            in_pose = torch.eye(4).to(device)
            in_projs = data["projs"][0][0]
            in_img = data["imgs"][0][0] * 0.5 + 0.5
            if "out_synth" in data:
                out: Outputs = data["out_synth"]
                data_synth: Data = data["data_synth"]
            elif "out_synth_l" in data:
                out: Outputs = data["out_synth_l"][-1]
                data_synth: Data = data["data_synth_l"][-1]
            
            xyz = get_pts(x_range, y_range, z_range, x_res, y_res, z_res).to(device)
            if APPLY_INCL_ADJUST:
                xyz_adj = get_pts(x_range, y_range, z_range, x_res, y_res, z_res, cam_incl_adjust)
                in_pose = cam_incl_adjust.inverse() @ in_pose
            else:
                xyz_adj = xyz
                
            xyz = xyz.permute(2, 0, 1, 3)
            xyz_adj = xyz_adj.permute(2, 0, 1, 3)
            
            rgb, invalid, sigmas, _ = net(xyz_adj.reshape(1, -1, 3), only_density=True)

            
            invalid = invalid.reshape(1, 1, x_res, y_res, z_res)
            alphas = sigmas.reshape(1, 1, x_res, y_res, z_res)
            alphas = alphas * (1 - invalid.float())
            c = rgb.size(-1)
            rgb = rgb.reshape(1, 1, x_res, y_res, z_res, c)[..., :3]
            alphas_mean = F.avg_pool3d(alphas, kernel_size=2, stride=1, padding=0)            
            is_occupied = alphas_mean.squeeze() > THRESHOLD_OCCUPANCY

            width_default_px = 500
            fig = make_3d_fig("", up=up, eye=scene_camera_eye, showgrid=False, transparent_plot_bg=True, width=width_default_px, height=300)

            if is_occupied.any():    
                progress_bar.set_postfix_str("Building mesh")
                ply = occupancy_grid_to_ply(is_occupied.cpu(), xyz.cpu(), up, keep_all=False)
                progress_bar.set_postfix_str("Drawing mesh")
                fig = draw_ply_mesh(fig, ply)
            else:
                print("No occupancies detected. Check your setup.")
            
            # Input camera
            if PLOT_IN_CAM:
                fig = draw_camera_with_frustum(
                    fig, in_pose.cpu().numpy(), in_projs.cpu().numpy(), 
                    image=in_img.permute(1, 2, 0).clamp(0, 1).cpu().numpy() if PLOT_IMAGES else None, 
                    frustum_depth=frustum_depth,
                    frustum_line_width=3.,
                    frame_line_width=5.,
                )
            # Novel view cameras
            if IS_CONTROLNET and PLOT_NOVEL_VIEWS:
                for T, K, img in zip(out.POSES_OUT[0], out.PROJS_OUT[0], data_synth.IMGS_NV[0]):
                    if APPLY_INCL_ADJUST:
                        T = cam_incl_adjust.inverse() @ T
                    fig = draw_camera_with_frustum(
                        fig, T.cpu().numpy(), K.cpu().numpy(), 
                        image=img.permute(1, 2, 0).clamp(0, 1).cpu().numpy() if PLOT_IMAGES else None, 
                        frustum_depth=frustum_depth,
                    )

            if not DRY_RUN:
                filepath = os.path.join(out_path, f"volume_{idx:03d}_{suffix}")
                in_img_path = os.path.join(out_path, f"volume_input_{idx:03d}.png")
                
                print(f"Saving to {in_img_path} and {filepath}.png")
                # ply.write(filepath + ".ply")
                save_image(in_img, in_img_path)
                progress_bar.set_postfix_str("Saving PNG")
                fig.write_image(filepath + ".png", scale=dpi2scale(width_default_px, width_in_inches=PAGE_WIDTH_INCHES/4, dpi=500))
                # fig.write_html(filepath + ".html")
                
            progress_bar.update(1)

    print("Completed.")


if __name__ == '__main__':

    parser = argparse.ArgumentParser("GenImg3dVolCustom")
    parser.add_argument("img", type=str, required=True)
    parser.add_argument("model", type=str, choices=["KITTI-360", "Waymo"], default="KITTI-360")
    args = parser.parse_args()

    main(args.img, args.model)
