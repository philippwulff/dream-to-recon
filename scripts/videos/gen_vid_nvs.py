import numpy as np
from typing import List
import sys
sys.path.append(".")

import torch
import torchvision.transforms.functional as VF
from moviepy.video.io.ImageSequenceClip import ImageSequenceClip

from scripts.inference_setup import *
from utils.array_ops import to_tensor_unsqueeze, to
from utils.plotting import color_tensor, cmap_magma, cmap_jet
from utils.utils import invert_depth
from bts.gt_synthesis.gt_synthesis import Outputs, Data
from bts.ignite_evaluation.evaluator import InferenceWrapper
from configs.structured_configs.synthetic_gt_config import TrajectoryCameraSamplerConfig, RigCameraSamplerConfig, ExplorationCameraSamplerConfig


def main():
    
    s_img = True
    s_depth = True
    s_anchor = False
    s_reproj = False
    s_mask = False
    s_profiles = False
    dry_run = False
    
    # indices = [0, 50, 80]
    indices = range(0, 51)
    
    config = load_and_setup_config(
        config_name="eval_cascade", 
        # config_name="recon_full", 
    )
    config.BTS.DATA.CATEGORY_NAME = "hydrant"
    dataset, out_path = setup_task(config, "videos/cascade_nvs", return_dataset="val")

    config.SYNTHETIC_GT.CASCADE.TRAJECTORY_SAMPLER_CONFIG.POLICY = "center_left_right_center"
    # config.SYNTHETIC_GT.CASCADE.TRAJECTORY_SAMPLER_CONFIG.NUM_STEPS = 100
    config.SYNTHETIC_GT.CASCADE.TRAJECTORY_SAMPLER_CONFIG.NUM_STEPS = 1
    config.SYNTHETIC_GT.CASCADE.SHOW_PROGRESS_BAR = True
    # config.SYNTHETIC_GT.CASCADE.CHOSEN_ANCHOR_TYPE = "rig12_8_explore3x4"
    config.SYNTHETIC_GT.CASCADE.CHOSEN_ANCHOR_TYPE = "empty"
    # config.SYNTHETIC_GT.RENDERER.n_coarse = 64
    config.SYNTHETIC_GT.INFERENCE_OCCLUSIONS.USE_DEPTH_GRADS = True
    config.SYNTHETIC_GT.INFERENCE_OCCLUSIONS.USE_FLOW = False
    # config.SYNTHETIC_GT.INFERENCE_OCCLUSIONS.POSTPROCESSING_OPS_DEPTH_GRADS=[]
    # config.SYNTHETIC_GT.INFERENCE_OCCLUSIONS.POSTPROCESSING_OPS_FLOW=[]

    # config.SYNTHETIC_GT.PSEUDO_VOLUME.SET_ONLY_OCCLUSIONS_EMPTY = True
    # config.SYNTHETIC_GT.PSEUDO_VOLUME.COLOR_SAMPLING_MODE = "mean_valid_surface"
    
    # samplers: List[ExplorationCameraSamplerConfig] = config.SYNTHETIC_GT.CASCADE._generate_default_anchor_configs()
    # for i, sampler in enumerate(samplers):
    #     sampler.VISUALIZE = True
    #     sampler.VISUALIZATION_PATH = os.path.join(out_path, "exploration_sampler", str(i))
    #     sampler.NUM_NOVEL_VIEWS = 2
    # config.SYNTHETIC_GT.CASCADE.ANCHOR_SAMPLER_CONFIGS = samplers
    
    model = InferenceWrapper.from_conf(config, refine_output=True)
    model.to(device)
    
    z_near = model.gt_synthesizer.z_near
    z_far = model.gt_synthesizer.z_far
    
    for idx in indices:
        data_batch = dataset[idx]
        data_batch = to_tensor_unsqueeze(data_batch)
        data_batch = to(data_batch, device)

        with torch.no_grad():
            data_batch = model(data_batch, forward_type="controlnet_cascade", seed=42)
            
        out: List[Outputs] = data_batch["out_synth_l"]
        data: List[Data] = data_batch["data_synth_l"]
        debug = data_batch["data_debug"]
                    
        ver_idx = b_idx = nv_idx = 0
        
        # Inputs
        imgs_in = data[0].IMGS[:, 0].permute(0, 2, 3, 1).cpu()
        depths_in = color_tensor(invert_depth(data[0].DEPTHS[:, 0, 0], z_near, z_far).clamp(0, 1), cmap_magma).cpu()
        # Frames
        imgs_frames = torch.concat([_.IMGS_SYNTH[ver_idx, :, nv_idx].cpu().permute(0, 2, 3, 1) for _ in out])
        depths_frames = torch.concat([_.DEPTHS_SYNTH[ver_idx, :, nv_idx, 0].cpu() for _ in out])
        depths_frames = color_tensor(invert_depth(depths_frames, z_near, z_far).clamp(0, 1), cmap_magma)
        # Mask frames
        masks_frames = flatten_pad_stack([_.MASKS_NV for _ in data])[:, nv_idx, 0].cpu()
        masks_frames = (masks_frames - masks_frames.min()) / (masks_frames.max() - masks_frames.min()).clamp_min(1)
        masks_frames = color_tensor(1 - masks_frames.clamp(0.0, 0.9), cmap_magma)
        # Reproj frames
        cond_frames = flatten_pad_stack([_.IMGS_CONDITIONING for _ in data])[:, b_idx].permute(0, 2, 3, 1).cpu()
        imgs_reproj_frames = flatten_pad_stack([_.IMGS_NV for _ in data])[:, b_idx].permute(0, 2, 3, 1).cpu()
        depths_reproj_frames = flatten_pad_stack([_.DEPTHS_NV for _ in data])[:, b_idx, 0].cpu()
        depths_reproj_frames = color_tensor(invert_depth(depths_reproj_frames, z_near, z_far).clamp(0, 1), cmap_magma)
        # Anchors
        imgs_anchor = flatten_pad_stack([_.IMGS for _ in data])[:, 0].permute(0, 2, 3, 1).cpu()
        depths_anchor = flatten_pad_stack([_.DEPTHS for _ in data])[:, b_idx, 0].cpu()
        depths_anchor = color_tensor(invert_depth(depths_anchor, z_near, z_far).clamp(0, 1), "magma")
        
        if s_img and s_depth:
            frames = torch.concat((imgs_frames, depths_frames), dim=1)
            # frames = torch.concat((imgs_in, depths_in), dim=1)
            _, h, w, _ = frames.shape
            
            if s_reproj:
                # reproj_frames = torch.concat((imgs_reproj_frames, depths_reproj_frames), dim=1)
                # reproj_frames = torch.concat((cond_frames[:, :, :, :3], depths_reproj_frames), dim=1)
                reproj_frames = torch.concat((imgs_reproj_frames, depths_reproj_frames), dim=1)
                frames = torch.concat([frames, reproj_frames], dim=2)
            if s_mask:
                masks_frames = VF.center_crop(masks_frames.permute(0, 3, 1, 2), (h, w)).permute(0, 2, 3, 1)
                frames = torch.concat([frames, masks_frames], dim=2)
                # frames = masks_frames
            if s_anchor and imgs_anchor.numel():
                anchors = torch.concat([imgs_anchor, depths_anchor], dim=1)
                frames = torch.concat([frames, anchors], dim=2)
                # frames = anchors
            if s_profiles:
                profiles = torch.concat(debug["profiles"]).permute(0, 3, 1, 2).cpu()
                profiles = VF.resize(profiles, [min(h, w)]*2)
                profiles = VF.center_crop(profiles, (h, w)).permute(0, 2, 3, 1)
                frames = torch.concat([frames, profiles], dim=2)
        elif s_depth:
            frames = depths_frames
        elif s_img:
            frames = imgs_frames

        frames = [(frame.numpy() * 255).astype(np.uint8) for frame in frames]

        if not dry_run:
            file_name = f"{idx:010d}"
            video = ImageSequenceClip(frames_final, fps=5)
            video.write_videofile(str(out_path / file_name) + ".mp4")
            video.write_gif(str(out_path / file_name) + ".gif")
            video.close()

        print("Completed.")


if __name__ == '__main__':
    main()
