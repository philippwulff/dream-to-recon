"""
Script to produce 3D voxel grid plots.
"""

import sys
sys.path.append(".")
from tqdm.auto import tqdm
import time

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

cmap = cmap_spectral


def main():

    DRY_RUN = False
    PLOT_IMAGES = False
    PLOT_NOVEL_VIEWS = False
    PLOT_IN_CAM = True
    THRESHOLD_OCCUPANCY = 0.5

    indices = [
        # Kitti-360
        # 0, 28, 
        # 266, 
        # 28, 84, 119, 287, 374, 385,
        # 100, 300,
        # 42, 54, 112, 374, 398,
        # Waymo
        1815, 2580, 2010, 3435,  # work well with BTS
        2775, 4470, 2830,
        0, 2775, 
        1080, 1740, 2830, 4470,
        350, 450, 2340
    ]

    config = load_and_setup_config(
        # config_name="exp_recon_full", 
        config_name="waymo", 
        # config_name="eval_bts_lidar_occ", 
        # config_name="eval_bts_lidar_occ_waymo", 
        # config_name="eval_controlnet_cascade", 
        # data_config_name="co3d",
    )
    config.CONTROLNET.DATA.USE_VAL_FOR_TEST = False
    config.BTS.DATA.USE_VAL_FOR_TEST = False
    config.CONTROLNET.DATA.return_45 = True
    config.BTS.DATA.return_45 = True
    dataset, out_path = setup_task(config, "figures/3d_volume", return_dataset="test")

    if config.BTS.DATA.type == "CO3D":
        up = [-0.0396, -0.8306, -0.5554]
        frustum_depth = 0.5
        APPLY_INCL_ADJUST = False
    else:
        up = [0, -1, 0]
        scene_camera_eye = [1.1, -1.1, 0]
        frustum_depth = 1.0
        APPLY_INCL_ADJUST = True
        config.BTS.DATA.VIS_VOLUME.X_RANGE = [-9, 9]
        # config.BTS.DATA.VIS_VOLUME.Z_RANGE = [3., 25.]
        config.BTS.DATA.VIS_VOLUME.Z_RANGE = [4., 25.]
        # config.BTS.DATA.VIS_VOLUME.Y_RANGE = [-1.5, 4.]

        config.BTS.DATA.VIS_VOLUME.Y_RANGE = [-0.25, 4]
        config.BTS.DATA.VIS_VOLUME.PPM = 7
        # config.BTS.DATA.VIS_VOLUME.PPM = 12
    
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

    # config.BTS.RESUME_FROM = "out/pretrained/kitti_360/training-checkpoint.pt"
    # model_name = "BTS"

    # config.BTS.RESUME_FROM = "out/recon/bts_depth_kitti_360/training_checkpoint_14000.pt"
    # model_name = "BTS_D"

    # config.BTS.RESUME_FROM = "out/recon/waymo_20k_backend-gloo-4_run_0/best_model.pt"
    # config.BTS.RESUME_FROM = "out/recon/waymo_20k_new/best_model.pt"
    # model_name = "full_waymo"

    # config.BTS.RESUME_FROM = "out/recon/waymo_backend-None-1_20250304-191409/training_checkpoint_81500.pt"
    # model_name = "BTS_waymo"
    # THRESHOLD_OCCUPANCY = 0.1

    # config.BTS.RESUME_FROM = "out/recon/waymo_depth_backend-None-1_20250304-192009/training_checkpoint_95000.pt"
    # model_name = "BTS_D_waymo"
    # THRESHOLD_OCCUPANCY = 0.05



    # config.EVAL_OCCUPANCY.MODE = "depth_pred"
    config.SYNTHETIC_GT.DEPTH_PREDICTOR_NAME = "Metric3D"
    config.EVAL_OCCUPANCY.MODE = "recon"
    config.EVAL_OCCUPANCY.MODE = "cascade"
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
    
    # cp_path = "/home/stud/wph/storage/user/BTS/out/recon/EXP99_backend-nccl-4_run_0/training_checkpoint_35500.pt"
    # cp = torch.load(cp_path, map_location=device)
    # wrapper.load_state_dict(cp["model"], strict=False)
    wrapper.to(device)
    
    progress_bar = tqdm(
        range(len(indices)),
        desc="Generating 3D plots",
    )

    inference_time = []
    
    with torch.no_grad():
        for idx in indices:
            
            time_start = time.time()
            progress_bar.set_postfix_str("Forward pass")
            
            data = dataset[idx]
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

            inference_time.append(time.time() - time_start)
            
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
    print(f"Average inference time (ms): {sum(inference_time[1:]) / len(inference_time[1:]) * 1000:.2f}")


if __name__ == '__main__':
    main()
