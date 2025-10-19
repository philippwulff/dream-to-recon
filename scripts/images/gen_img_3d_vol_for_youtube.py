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
from utils.plotting import cmap_spectral, get_pts, PAGE_WIDTH_INCHES, color_tensor, cmap_magma
from utils.plotting_3d import draw_camera_with_frustum, make_3d_fig, draw_ply_mesh, occupancy_grid_to_ply, dpi2scale
from bts.ignite_evaluation.evaluator import initialize, InferenceWrapper
from bts.gt_synthesis.gt_pose_sampler import OrbitCameraSampler
from configs.structured_configs.synthetic_gt_config import OrbitCameraSamplerConfig, RigCameraSamplerConfig
from utils.utils import invert_depth

cmap = cmap_spectral


def main():

    DRY_RUN = False
    PLOT_IMAGES = True
    PLOT_NOVEL_VIEWS = False
    PLOT_IN_CAM = True
    CONTROLNET_TRAINING_PLOT = True
    ENCODING_PLOT = False

    indices = [
        0,
        # 72, 
        # 84, 
        # 0, 44, 363, 374
    ]

    config = load_and_setup_config(
        # config_name="exp_bts_synthetic_rig", 
        # config_name="eval_controlnet_cascade", 
        config_name="exp_recon_full", 
    )
    # config.BTS.DATA.CATEGORY_NAME = "hydrant"
    # config.BTS.DATA.APPLY_ADJUST_UPRIGHT = True
    # config.BTS.DATA.VIS_VOLUME.PPM = 10
    # config.SYNTHETIC_GT.TOP_K_NOVEL_VIEWS_TO_KEEP = 12
    dataset, out_path = setup_task(config, "figures/3d_cams")


    up = [0, -1, 0]
    frustum_depth = 1.0
    APPLY_INCL_ADJUST = True
    if config.BTS.DATA.type == "CO3D":
        up = [-0.0396, -0.8306, -0.5554]
        frustum_depth = 0.5
        APPLY_INCL_ADJUST = False
    elif CONTROLNET_TRAINING_PLOT:
        # from configs.structured_configs.synthetic_gt_config import ShiftCameraSamplerConfig
        
        # config.SYNTHETIC_GT.CASCADE.CHOSEN_ANCHOR_TYPE = "plot_shift"
        config.SYNTHETIC_GT.NV_CAM_SAMPLER = RigCameraSamplerConfig(CAMS_XYZ=[[4,0,0]])
        scene_camera_eye = [0.5, -0.3, -1.0]
        config.BTS.DATA.VIS_VOLUME.X_RANGE = [-9, 9]
        config.BTS.DATA.VIS_VOLUME.Z_RANGE = [3., 15.]
        config.BTS.DATA.VIS_VOLUME.Y_RANGE = [-1.5, 3.]
        config.BTS.DATA.VIS_VOLUME.PPM = 5
    elif ENCODING_PLOT:
        # from configs.structured_configs.synthetic_gt_config import ShiftCameraSamplerConfig
        
        # config.SYNTHETIC_GT.NV_CAM_SAMPLER = RigCameraSamplerConfig(CAMS_XYZ=[[4,0,3]], CAMS_ALPHA_BETA_GAMMA=[[0,-5,0]])
        # config.SYNTHETIC_GT.NV_CAM_SAMPLER.NUM_NOVEL_VIEWS = 1
        # config.SYNTHETIC_GT.CASCADE.CHOSEN_ANCHOR_TYPE = "none"
        # config.SYNTHETIC_GT.CASCADE.CHOSEN_ANCHOR_TYPE = "plot_rig_final"
        # config.SYNTHETIC_GT.CASCADE.TRAJECTORY_SAMPLER_CONFIG = RigCameraSamplerConfig(CAMS_XYZ=[[4,0,3]], CAMS_ALPHA_BETA_GAMMA=[[0,-5,0]])
        # config.SYNTHETIC_GT.CASCADE.CHOSEN_ANCHOR_TYPE = "plot_rig1"
        # scene_camera_eye = [0.5, -0.3, -1.0]
        scene_camera_eye = [1.0, -0.8, -1.1]
        config.BTS.DATA.VIS_VOLUME.X_RANGE = [-9, 9]
        config.BTS.DATA.VIS_VOLUME.Z_RANGE = [3., 15.]
        config.BTS.DATA.VIS_VOLUME.Y_RANGE = [-1.5, 3.]
        config.BTS.DATA.VIS_VOLUME.PPM = 5
    else:

        # scene_camera_eye = [-1, -1, -1]
        # scene_camera_eye = [0.7, -0.5, -0.9]
        scene_camera_eye = [0.8, -0.5, -1.0]
        APPLY_INCL_ADJUST = True
        config.BTS.DATA.VIS_VOLUME.X_RANGE = [-9, 9]
        config.BTS.DATA.VIS_VOLUME.Z_RANGE = [3., 20.]
        # config.BTS.DATA.VIS_VOLUME.Z_RANGE = [3., 18.]
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
    # config.SYNTHETIC_GT.CASCADE.CHOSEN_ANCHOR_TYPE = "rig12_explore3x_ranged"
    # config.SYNTHETIC_GT.INFERENCE_OCCLUSIONS.POSTPROCESSING_OPS_DEPTH_GRADS = [['opening',3],['closing',15]]
    # config.BTS.RESUME_FROM = "/home/stud/wph/storage/user/BTS/out/recon/EXP201_backend-nccl-4_run_0/training_checkpoint_20500.pt"


    # config.SYNTHETIC_GT.CASCADE.CHOSEN_ANCHOR_TYPE = "rig12_8_explore3x4"
    # config.SYNTHETIC_GT.TOP_K_NOVEL_VIEWS_TO_KEEP = 8
    config.SYNTHETIC_GT.CASCADE.SHOW_PROGRESS_BAR = True
    
    # config.SYNTHETIC_GT.PSEUDO_VOLUME.SET_ONLY_OCCLUSIONS_EMPTY = True
    # config.SYNTHETIC_GT.PSEUDO_VOLUME.COLOR_SAMPLING_MODE = "mean_valid"
    
    config.EVAL_OCCUPANCY.MODE = "depth_pred"
    # config.EVAL_OCCUPANCY.MODE = "cascade"
    # config.EVAL_OCCUPANCY.MODE = "recon"
    
    IS_CONTROLNET = False    
    match config.EVAL_OCCUPANCY.MODE:
        case "depth_pred":
            IS_CONTROLNET = True
            wrapper: InferenceWrapper = initialize(config, refine_output=False)
            net = wrapper.gt_synthesizer.renderer.net
            forward_fn = lambda d: wrapper(d, forward_type="controlnet_input_view" if CONTROLNET_TRAINING_PLOT else "controlnet_novel_view")
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
            suffix = "recon_" + config.NAME
    
    # cp_path = "/home/stud/wph/storage/user/BTS/out/recon/EXP99_backend-nccl-4_run_0/training_checkpoint_35500.pt"
    # cp = torch.load(cp_path, map_location=device)
    # wrapper.load_state_dict(cp["model"], strict=False)
    wrapper.to(device)
    
    progress_bar = tqdm(
        range(len(indices)),
        desc="Generating 3D plots",
    )
    
    with torch.no_grad():
        for idx in indices:
            
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
                out: Outputs = data["out_synth_l"][-2]
                data_synth: Data = data["data_synth_l"][-2]
                # in_pose = out.POSES_IN[0, 0]
                # in_projs = out.PROJS_IN[0, 0]
                # in_img = out.IMGS_IN[0, 0]
            
            xyz = get_pts(x_range, y_range, z_range, x_res, y_res, z_res).to(device)
            if APPLY_INCL_ADJUST:
                xyz_adj = get_pts(x_range, y_range, z_range, x_res, y_res, z_res, cam_incl_adjust)
                in_pose = cam_incl_adjust.inverse() @ in_pose
            else:
                xyz_adj = xyz
                
            xyz = xyz.permute(2, 0, 1, 3)
            xyz_adj = xyz_adj.permute(2, 0, 1, 3)
            
            # rgb, sigmas, _ = model.gt_synthesizer.renderer.net.sample_colors_and_density(xyz_adj.reshape(1, -1, 3))
            # rgb, sigmas, _ = net.sample_colors_and_density(xyz_adj.reshape(1, -1, 3))
            rgb, _, sigmas, _ = net(xyz_adj.reshape(1, -1, 3), only_density=True)
            
            
            alphas = sigmas.reshape(1, 1, x_res, y_res, z_res)
            c = rgb.size(-1)
            rgb = rgb.reshape(1, 1, x_res, y_res, z_res, c)[..., :3]
            alphas_mean = F.avg_pool3d(alphas, kernel_size=2, stride=1, padding=0)            
            is_occupied = alphas_mean.squeeze() > .5

            fig = make_3d_fig("", up=up, eye=scene_camera_eye, showgrid=False, transparent_plot_bg=True, width=500, height=300)

            if is_occupied.any():    
                progress_bar.set_postfix_str("Building mesh")
                ply = occupancy_grid_to_ply(is_occupied.cpu(), xyz.cpu(), up)
                progress_bar.set_postfix_str("Drawing mesh")
                fig = draw_ply_mesh(fig, ply)
            else:
                print("No occupancies detected. Check your setup.")
                continue
            
            # Input camera
            if PLOT_IN_CAM:
                fig = draw_camera_with_frustum(
                    fig, in_pose.cpu().numpy(), in_projs.cpu().numpy(), 
                    image=in_img.permute(1, 2, 0).clamp(0, 1).cpu().numpy() if PLOT_IMAGES else None, 
                    frustum_depth=frustum_depth,
                    frustum_line_width=5.,
                    frame_line_width=3.,
                    # frame_color="red",
                )
            # Novel view cameras
            if IS_CONTROLNET and PLOT_NOVEL_VIEWS:
                for T, K, img in zip(out.POSES_OUT[0], out.PROJS_OUT[0], data_synth.IMGS_NV[0]):
                # for T, K, img in zip(data_synth.POSES[0], data_synth.PROJS[0], data_synth.IMGS[0]):
                    if APPLY_INCL_ADJUST:
                        T = cam_incl_adjust.inverse() @ T
                    fig = draw_camera_with_frustum(
                        fig, T.cpu().numpy(), K.cpu().numpy(), 
                        image=img.permute(1, 2, 0).clamp(0, 1).cpu().numpy() if PLOT_IMAGES else None, 
                        frustum_depth=frustum_depth,
                        frustum_line_width=3.,
                        frame_line_width=3.,
                        # frame_color="red",
                    )
            # fig = draw_camera_with_frustum(
            #     fig, 
            #     # data["data_synth_l"][-1].POSES_NV[0, 0].cpu().numpy(), 
            #     np.array([[0.9961947202682495, -8.997695033485797e-09, -0.08715571463108063, 3.999999523162842], [5.587935447692871e-09, 1.0, 7.450580596923828e-09, 3.0517578125e-05], [0.08715566992759705, 6.693198884022422e-09, 0.9961947202682495, 2.999999523162842], [-2.4273220586290556e-11, 4.496160388445247e-13, -5.148576703861707e-11, 0.9999998807907104]]),
            #     np.array([[1.7441738843917847, 0.0, -0.06928843259811401], [0.0, 2.9391183853149414, 0.2700507640838623], [0.0, 0.0, 1.0]]), 
            #     # image=data["data_synth_l"][-1].IMGS_NV[0, 0].permute(1, 2, 0).clamp(0, 1).cpu().numpy() if PLOT_IMAGES else None, 
            #     frustum_depth=frustum_depth,
            #     frustum_line_width=10.,
            #     frame_line_width=3.,
            #     # frame_color="blue",
            # )

            if not DRY_RUN:
                filepath = os.path.join(out_path, f"recon_{idx:03d}_{suffix}")
                
                print(f"Saving to {filepath}.png")
                # progress_bar.set_postfix_str("Saving PLY")
                ply.write(filepath + ".ply")
                save_image(in_img, f"{filepath}_in_img.png")
                progress_bar.set_postfix_str("Saving PNG")
                fig.write_image(filepath + ".png", scale=dpi2scale(500, width_in_inches=PAGE_WIDTH_INCHES/3, dpi=500))
                progress_bar.set_postfix_str("Saving HTML")
                fig.write_html(filepath + ".html")
                
                if "out_synth_l" in data:
                    anchor_out = data["out_synth_l"][0]
                    save_image(anchor_out.IMGS_SYNTH[0, 0, 0], f"{filepath}_synth_rgb.png")
                    d = anchor_out.DEPTHS_SYNTH[0, 0, 0, 0]
                    d = invert_depth(d, 3, 40)
                    d = color_tensor(d, cmap_magma, norm=False).permute(2, 0, 1)
                    save_image(d, f"{filepath}_synth_d.png")
                else:
                    anchor_out = data["out_synth"]
                save_image(anchor_out.IMGS_COND[0, 0, :3], f"{filepath}_cond_rgb.png")
                save_image(anchor_out.IMGS_COND[0, 0, 3:4], f"{filepath}_cond_d.png")
                save_image(anchor_out.IMGS_COND[0, 0, 4:5], f"{filepath}_cond_m.png")
                
            progress_bar.update(1)

    print("Completed.")


if __name__ == '__main__':
    main()
