import sys
sys.path.append(".")
from tqdm.auto import tqdm

import torch
import torch.nn.functional as F
from torchvision.utils import save_image
import torch

from scripts.inference_setup import *
from utils.array_ops import to_tensor_unsqueeze, to
from bts.gt_synthesis.gt_synthesis import Outputs, Data
from utils.plotting import cmap_spectral, get_pts, PAGE_WIDTH_INCHES
from utils.plotting_3d import draw_camera_with_frustum, make_3d_fig, draw_ply_mesh, occupancy_grid_to_ply, dpi2scale, plotly_fig2array
from bts.ignite_evaluation.evaluator import initialize, InferenceWrapper
from moviepy.video.io.ImageSequenceClip import ImageSequenceClip
from configs.structured_configs.synthetic_gt_config import Rig4CameraSamplerConfig

cmap = cmap_spectral


def main():

    DRY_RUN = False
    PLOT_IMAGES = False
    PLOT_NOVEL_VIEWS = False
    PLOT_IN_CAM = False
    model_name = None
    THRESHOLD_OCCUPANCY = 0.5

    # Kitti
    indices = range(0, 3000, 1)
    # NOTE: And use val set!
    # Waymo
    indices = range(0, 4500, 1)
    # NOTE: And use test set!

    # For the driving video in the teaser use this with Waymo, return_dataset="test" and
    # HIDE_OCCUPANCY_OUTSIDE_FRUSTUM=True. The teaser frame in the paper is at 2580.
    indices = range(2500, 2700, 1)
    HIDE_OCCUPANCY_OUTSIDE_FRUSTUM = True

    config = load_and_setup_config(
        # config_name="eval_bts_lidar_occ", 
        # config_name="eval_bts_lidar_occ_waymo", 
        # config_name="exp_recon_full", 
        config_name="waymo", 
        # config_name="exp_bts_synthetic_rig", 
        # config_name="eval_controlnet_cascade",
        # config_name="eval_full_rig12", 
        # data_config_name="co3d",
        
    )
    # config.BTS.DATA.CATEGORY_NAME = "hydrant"
    # config.BTS.DATA.APPLY_ADJUST_UPRIGHT = True
    # config.BTS.DATA.VIS_VOLUME.PPM = 10
    # config.SYNTHETIC_GT.TOP_K_NOVEL_VIEWS_TO_KEEP = 12

    config.CONTROLNET.DATA.USE_VAL_FOR_TEST = False
    config.BTS.DATA.USE_VAL_FOR_TEST = False
    config.CONTROLNET.DATA.return_45 = False
    config.BTS.DATA.return_45 = False
    config.BTS.DATA.return_stereo = False

    dataset, out_path = setup_task(
        config, 
        "videos/3d_volume", 
        # return_dataset="val",
        # return_dataset="test",
    )

    if config.BTS.DATA.type == "CO3D":
        up = [-0.0396, -0.8306, -0.5554]
        frustum_depth = 0.5
        APPLY_INCL_ADJUST = False
    else:
        up = [0, -1, 0]
        # scene_camera_eye = [0, -1.0, -1.7]      # behind both stereo-cams
        scene_camera_eye = [1.1, -1.1, 0]       # from the side ("teaser view")

        frustum_depth = 1.0
        APPLY_INCL_ADJUST = True
        config.BTS.DATA.VIS_VOLUME.X_RANGE = [-9, 9]
        config.BTS.DATA.VIS_VOLUME.Z_RANGE = [3, 25.] # todo with 3.5 or 4
        config.BTS.DATA.VIS_VOLUME.Y_RANGE = [-1.5, 3.]
        config.BTS.DATA.VIS_VOLUME.PPM = 5
    
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
    
    # config.SYNTHETIC_GT.DEPTH_PREDICTOR_NAME = "Metric3D"
    # config.SYNTHETIC_GT.CASCADE.CHOSEN_ANCHOR_TYPE = "explore"
    # config.SYNTHETIC_GT.CASCADE.CHOSEN_ANCHOR_TYPE = "rig12_8_explore3x4_ranged"
    # config.SYNTHETIC_GT.CASCADE.CHOSEN_ANCHOR_TYPE = "rig12_8_explore3x4"
    # config.SYNTHETIC_GT.CASCADE.CHOSEN_ANCHOR_TYPE = "none"
    # config.SYNTHETIC_GT.CASCADE.CHOSEN_ANCHOR_TYPE = "rig4from4"
    # config.SYNTHETIC_GT.CASCADE.ANCHOR_SAMPLER_CONFIGS = [Rig4CameraSamplerConfig(NUM_NOVEL_VIEWS=1)]
    # config.SYNTHETIC_GT.CASCADE.ANCHOR_SAMPLER_CONFIGS = []
    # config.SYNTHETIC_GT.INFERENCE_OCCLUSIONS.POSTPROCESSING_OPS_DEPTH_GRADS = [['opening',3],['closing',15]]
    # config.BTS.RESUME_FROM = "/home/stud/wph/storage/user/BTS/out/recon/EXP205_weight_guided_occl_backend-nccl-4_run_0/training_checkpoint_29000.pt"
    # config.BTS.RESUME_FROM = "/home/stud/wph/storage/user/BTS/out/recon/EXP203_l2_backend-nccl-4_run_0/training_checkpoint_29000.pt"
    # config.BTS.RESUME_FROM = "/home/stud/wph/storage/user/BTS/out/recon/EXP207_metric3d_mse_backend-nccl-4_run_0/training_checkpoint_29000.pt"
    # config.BTS.RESUME_FROM = "/home/stud/wph/storage/user/BTS/out/recon/EXP206_metric3d_gnll_backend-nccl-4_run_0/training_checkpoint_29000.pt"


    # config.BTS.RESUME_FROM = "/home/stud/wph/storage/user/BTS/out/recon/EXP334_backend-gloo-4_run_0/best_model.pt"
    # model_name = "full"

    # config.BTS.RESUME_FROM = "out/pretrained/kitti_360/training-checkpoint.pt"
    # model_name = "BTS"

    # config.BTS.RESUME_FROM = "out/recon/bts_depth_kitti_360/training_checkpoint_14000.pt"
    # model_name = "BTS_D"

    # config.BTS.RESUME_FROM = "out/recon/waymo_20k_backend-gloo-4_run_0/best_model.pt"
    config.BTS.RESUME_FROM = "out/recon/waymo_20k_new/best_model.pt"
    model_name = "full_waymo"

    # config.BTS.RESUME_FROM = "out/recon/waymo_backend-None-1_20250304-191409/training_checkpoint_81500.pt"
    # model_name = "BTS_waymo"
    # THRESHOLD_OCCUPANCY = 0.1

    # config.BTS.RESUME_FROM = "out/recon/waymo_depth_backend-None-1_20250304-192009/training_checkpoint_95000.pt"
    # model_name = "BTS_D_waymo"
    # THRESHOLD_OCCUPANCY = 0.05
    
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
            suffix = "recon_" + config.NAME + "_" + config.BTS.RESUME_FROM.split("/")[-2]
            suffix = model_name if model_name else suffix
    
    # cp_path = "/home/stud/wph/storage/user/BTS/out/recon/EXP99_backend-nccl-4_run_0/training_checkpoint_35500.pt"
    # cp = torch.load(cp_path, map_location=device)
    # wrapper.load_state_dict(cp["model"], strict=False)
    wrapper.to(device)
    
    progress_bar = tqdm(
        range(len(indices)),
        desc="Generating 3D plots",
    )
    
    frames = []
    input_img_frames = []
    with torch.no_grad():
        for idx in indices:
            
            progress_bar.set_postfix_str("Forward pass")
            
            data = dataset[idx]
            data = to_tensor_unsqueeze(data)
            data = to(data, device)
            data = forward_fn(data)

            input_img = data["imgs"][0][0] * 0.5 + 0.5
            input_img_frames.append(input_img.permute(1, 2, 0).cpu().numpy() * 255)
            
            # in_pose = torch.eye(4).to(device)
            # in_projs = data["projs"][0][0]
            # in_img = data["imgs"][0][0] * 0.5 + 0.5
            if "out_synth" in data:
                out: Outputs = data["out_synth"]
                data_synth: Data = data["data_synth"]
            elif "out_synth_l" in data:
                out: Outputs = data["out_synth_l"][-1]
                data_synth: Data = data["data_synth_l"][-1]
                # in_pose = out.POSES_IN[0, 0]
                # in_projs = out.PROJS_IN[0, 0]
                # in_img = out.IMGS_IN[0, 0]
            
            xyz = get_pts(x_range, y_range, z_range, x_res, y_res, z_res).to(device)
            if APPLY_INCL_ADJUST:
                xyz_adj = get_pts(x_range, y_range, z_range, x_res, y_res, z_res, cam_incl_adjust)
                # in_pose = cam_incl_adjust.inverse() @ in_pose
            else:
                xyz_adj = xyz
                
            xyz = xyz.permute(2, 0, 1, 3)
            xyz_adj = xyz_adj.permute(2, 0, 1, 3)
            
            # rgb, sigmas, _ = model.gt_synthesizer.renderer.net.sample_colors_and_density(xyz_adj.reshape(1, -1, 3))
            # Extract the densities from the net
            rgb, invalid, sigmas, _ = net(xyz_adj.reshape(1, -1, 3), only_density=True)

            # from utils.plotting import render_profile
            # vis_volume = config.BTS.DATA.VIS_VOLUME
            # profile = render_profile(net, vis_volume, cam_incl_adjust)
            
            invalid = invalid.reshape(1, 1, x_res, y_res, z_res)
            alphas = sigmas.reshape(1, 1, x_res, y_res, z_res)
            if HIDE_OCCUPANCY_OUTSIDE_FRUSTUM:
                alphas = alphas * (1 - invalid.float())
            c = rgb.size(-1)
            rgb = rgb.reshape(1, 1, x_res, y_res, z_res, c)[..., :3]
            alphas_mean = F.avg_pool3d(alphas, kernel_size=2, stride=1, padding=0)            
            is_occupied = alphas_mean.squeeze() > THRESHOLD_OCCUPANCY

            x_len = x_range[1]-x_range[0]
            y_len = y_range[1]-y_range[0]
            z_len = z_range[1]-z_range[0]
            max_len = max(x_len, y_len, z_len)

            fig = make_3d_fig(
                "", up=up, eye=scene_camera_eye, showgrid=False, transparent_plot_bg=True, transparent_paper_bg=False, width=600, height=300, 
                aspectratio=dict(x=2*x_len/max_len, y=2*y_len/max_len, z=2*z_len/max_len)
            )

            if is_occupied.any():    
                progress_bar.set_postfix_str("Building mesh")
                ply = occupancy_grid_to_ply(is_occupied.cpu(), xyz.cpu(), up)
                progress_bar.set_postfix_str("Drawing mesh")
                fig = draw_ply_mesh(fig, ply)
            else:
                print("No occupancies detected. Check your setup.")
            
            # Input camera(s)
            if PLOT_IN_CAM:
                # for T, K, img in zip(out.POSES_IN[0], out.PROJS_IN[0], out.IMGS_IN[0]):
                for T, K, img in zip(data["poses"][0], data["projs"][0], data["imgs"][0] * 0.5 + 0.5):
                    if APPLY_INCL_ADJUST:
                        T = cam_incl_adjust.inverse() @ T
                    fig = draw_camera_with_frustum(
                        fig, T.cpu().numpy(), K.cpu().numpy(), 
                        image=img.permute(1, 2, 0).clamp(0, 1).cpu().numpy() if PLOT_IMAGES else None, 
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
                    
            frame = plotly_fig2array(fig)
            frames.append(frame)
            progress_bar.update(1)

        if not DRY_RUN:
            if len(input_img_frames):
                fps = 3
                file_name = f"{min(indices)}to{max(indices)}_input_{suffix}"
                video = list(input_img_frames)
                video = ImageSequenceClip(video, fps=fps)
                video.write_videofile(str(out_path / file_name) + ".mp4", fps=fps)
                video.write_gif(str(out_path / file_name) + ".gif", fps=1)
                video.close()
            if len(frames):
                fps = 3
                file_name = f"{min(indices)}to{max(indices)}_{suffix}"
                video = list(frames)
                video = ImageSequenceClip(video, fps=fps)
                video.write_videofile(str(out_path / file_name) + ".mp4", fps=fps)
                video.write_gif(str(out_path / file_name) + ".gif", fps=1)
                video.close()
                
    print("Completed.")


if __name__ == '__main__':
    main()
