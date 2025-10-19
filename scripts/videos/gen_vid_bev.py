import numpy as np
from moviepy.video.io.ImageSequenceClip import ImageSequenceClip
from tqdm import tqdm
from dataclasses import dataclass, asdict
from torchvision.utils import save_image

import sys
sys.path.append(".")

from scripts.inference_setup import *

import copy

import hydra
import torch
import torch.nn.functional as F
from torchvision.transforms import ToPILImage, ToTensor
from utils.array_ops import to_tensor_unsqueeze, to

from utils.array_ops import map_fn, unsqueezer
from utils.plotting import color_tensor, color_occlusion_masked_img
from utils.occlusion_ops import comp_occlusion_map, crop_to_patch_center, mask_occlusions_only, mask_occlusions_and_outside_FOV
from utils.huggingface_helpers import InpainterAndImg2Img, ControlNetConfig, ModelCallConfig, ModelConfig, make_depth_condition, make_inpaint_condition
from vcm.utils.data_utils import make_controlnet_dataloaders, make_conditioning_imgs
from vcm.utils.model_utils import make_model_components
from configs.structured_configs.config_utils import register_default_configs, check_and_post_init_config
from datasets.data_util import make_datasets
from bts.gt_synthesis.gt_synthesis import GTSynthesisWrapper
import torchvision.transforms.functional as VF
from utils.plotting import render_profile
from bts.models.bts_direct import BTSDirect
from tqdm import tqdm

from typing import List
import sys
sys.path.append(".")
import shutil

import torch
from scripts.inference_setup import *
from utils.array_ops import to_tensor_unsqueeze, to
from utils.plotting import set_spines, PAGE_WIDTH_INCHES, set_thesis_rcparams
from bts.gt_synthesis.gt_synthesis import Outputs, Data
from bts.ignite_evaluation.evaluator import InferenceWrapper
from configs.structured_configs.synthetic_gt_config import ExplorationCameraSamplerConfig
import matplotlib.pyplot as plt


def main():
    s_img = True
    s_depth = False
    s_profile = True
    dry_run = False

    indices = list(range(0, 5000, 15))

    config = load_and_setup_config(
        config_name="waymo", 
        # config_name="exp_bts_synthetic_cascade_depth_uncert", 
        # config_name="eval_bts_lidar_occ", 
    )
    dataset, out_path = setup_task(config, "videos/driving", return_dataset="test")
    config.BTS.DATA.return_45 = False

    print('Loading checkpoint')
    # cp_path = "/home/stud/wph/storage/user/BTS/out/recon/EXP99_backend-nccl-4_run_0/training_checkpoint_35500.pt"
    # config.BTS.RESUME_FROM = "/home/stud/wph/storage/user/BTS/out/recon/EXP201_backend-nccl-4_run_0//training_checkpoint_29000.pt"
    # config.BTS.RESUME_FROM = "/home/stud/wph/storage/user/BTS/out/recon/EXP205_weight_guided_occl_backend-nccl-4_run_0//training_checkpoint_29000.pt"
    # config.BTS.RESUME_FROM = "/home/stud/wph/storage/user/BTS/out/recon/EXP205_weight_guided_occl_backend-nccl-4_run_0//training_checkpoint_29000.pt"
    # config.BTS.RESUME_FROM = "/home/stud/wph/storage/user/BTS/out/recon/EXP302_backend-gloo-4_run_0/training_checkpoint_5000.pt"
    # config.BTS.RESUME_FROM = "/home/stud/wph/storage/user/BTS/out/recon/EXP305_backend-gloo-4_run_0/training_checkpoint_19000.pt"
    # config.BTS.RESUME_FROM = "/home/stud/wph/storage/user/BTS/out/recon/EXP307_backend-gloo-4_run_0/training_checkpoint_25000.pt"
    # config.BTS.RESUME_FROM = "/home/stud/wph/storage/user/BTS/out/recon/EXP324_backend-gloo-4_run_0/best_model.pt"
    # config.BTS.RESUME_FROM = "/home/stud/wph/storage/user/BTS/out/recon/EXP325_backend-gloo-4_run_0/best_model.pt"
    config.BTS.RESUME_FROM = "out/recon/waymo_20k_new/best_model.pt"
    # cp = torch.load(cp_path, map_location=device)

    # net = BTSDirect(asdict(config.BTS.MODEL_CONF))
    # renderer = NeRFRenderer.from_conf(asdict(config.BTS.RENDERER))
    # renderer = renderer.bind_parallel(net, gpus=None).eval()
    # renderer.renderer.n_coarse = 64
    # renderer.renderer.lindisp = True

    # class _Wrapper(nn.Module):
    #     def __init__(self):
    #         super().__init__()
    #         self.renderer = renderer

    # _wrapper = _Wrapper()

    # _wrapper.load_state_dict(cp["model"], strict=False)
    # renderer.to(device)
    # renderer.eval()

    # z_near = config.BTS.MODEL_CONF.z_near
    # z_far = config.BTS.MODEL_CONF.z_far
    # ray_sampler = ImageRaySampler(z_near, z_far, (192, 640), norm_dir=False)
    
    model = InferenceWrapper.from_conf(config, refine_output=True)
    model.to(device)
    
    with torch.no_grad():
        frames = []
        for idx in tqdm(indices, desc="Model inference"):
            # data = dataset[idx]
            # data_batch = to_tensor_unsqueeze(data)
            # # data_batch = to(data_batch, device)
            # poses = torch.stack(data_batch["poses"], dim=1).to(device)[:, :1]
            # projs = torch.stack(data_batch["projs"], dim=1).to(device)[:, :1]
            # # Move coordinate system to input frame
            # poses = torch.inverse(poses[:, :1, :, :]) @ poses
            
            # net.encode(images, projs, poses, ids_encoder=[0], ids_render=[0])
            # net.set_scale(0)
            
            data_batch = dataset[idx]
            data_batch = to_tensor_unsqueeze(data_batch)
            data_batch = to(data_batch, device)
            
            with torch.no_grad():
                data_batch = model(data_batch, forward_type="recon")
            
            images = torch.stack(data_batch["imgs"], dim=1).to(device)[:, :1]

            frame = None            
            if s_img:
                frame = images[0, 0].permute(1, 2, 0) * .5 + .5
            if s_depth:
                _, depth = render_poses(renderer, ray_sampler, poses, projs)
                depth = 1 / depth
                depth = ((depth - 1 / z_far) / (1 / z_near - 1 / z_far)).clamp(0, 1)
                depth = color_tensor(depth, "magma", norm=False)
                if frame is not None:
                    frame = torch.concat([frame, depth], dim=2)
                else:
                    frame = depth
            if s_profile:
                profile = render_profile(
                    model.renderer.net, 
                    config.BTS.DATA.VIS_VOLUME, 
                    config.BTS.DATA.CAM_INCL_ADJUST.to(device),
                )[0].permute(2, 0, 1)
                profile_w = 1
                if config.BTS.MODEL_CONF.OUTPUT_UNCERTAINTY:
                    profile_uncert = render_profile(
                        model.renderer.net, 
                        config.BTS.DATA.VIS_VOLUME, 
                        config.BTS.DATA.CAM_INCL_ADJUST.to(device),
                        mode="uncert",
                        color_profile="viridis",
                    )[0].permute(2, 0, 1)
                    profile_w = 2
                    profile = torch.cat([profile, profile_uncert], dim=2)
                if frame is not None:
                    profile = VF.resize(profile, [frame.size(0), profile_w*frame.size(0)], antialias=False).permute(1, 2, 0)
                    frame = torch.concat([frame, profile], dim=1)
                else:
                    frame = profile

            # save_image(frame.permute(2, 0, 1), f"{out_path}/{idx:04d}.png")
            frames.append(frame.cpu().numpy())

        frames = [(frame * 255).astype(np.uint8) for frame in frames]

        if not dry_run:
            file_name = f"{min(indices)}to{max(indices)}_{config.NAME}"
            video = list(frames)
            video = ImageSequenceClip(video, fps=5)
            video.write_videofile(str(out_path / file_name) + ".mp4")
            video.write_gif(str(out_path / file_name) + ".gif")
            video.close()

    print("Completed.")


if __name__ == '__main__':
    main()
