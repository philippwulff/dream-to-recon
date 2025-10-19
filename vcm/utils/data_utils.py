import torch
import time
from datasets.data_util import make_datasets
from configs.structured_configs.main_config import MainConfig

from typing import List, Optional, Tuple
import random
import torch
import time
from torchvision import transforms
from transformers import AutoTokenizer, PretrainedConfig
from omegaconf import DictConfig, OmegaConf
from dataclasses import dataclass, asdict


# Make sure that ~/storage/user/BTS is on the PYTHONPATH:
# export PYTHONPATH="$PWD/BTS"
# from scripts.inference_setup import *
from bts.models.pseudo_volume import PseudoVolume
from bts.common.ray_sampler import ImageRaySampler
from bts.renderer import NeRFRenderer
from utils.transformation_ops import orientate_poses
from utils.occlusion_ops import comp_occlusion_map
from utils.constants import DEFINITIONS
from utils.plotting import color_occlusion_masked_img
from utils.projection_ops import crop_intrinsics, unnormalize_intrinsics, scale_intrinsics
from utils.transforms import Crop
from utils.utils import get_interval_sample


def collate_fn(rows):
    """Processes a batch while dataloading for Controlnet training."""
    
    # We only want to load single-view
    assert len(rows[0]["imgs"]) == len(rows[0]["predicted_depths"]) == 1
    
    imgs = torch.stack([torch.stack(row["imgs"]) for row in rows])                    # [B, 1, 3, H, W]
    depths = torch.stack([torch.stack(row["predicted_depths"]) for row in rows])      # [B, 1, 1, H, W]
    poses = torch.stack([torch.stack([torch.tensor(_) for _ in row["poses"]]) for row in rows])    # [B, 1, 4, 4]
    projs = torch.stack([torch.stack([torch.tensor(_) for _ in row["projs"]]) for row in rows])    # [B, 1, 3, 3]

    return {
        "imgs": imgs.to(memory_format=torch.contiguous_format).float(),
        "depths": depths.to(memory_format=torch.contiguous_format).float(),
        "poses": poses.to(memory_format=torch.contiguous_format).float(),
        "projs": projs.to(memory_format=torch.contiguous_format).float(),
    }


def make_controlnet_dataloaders(cfg: MainConfig):
    """Returns train and test dataloaders according to config."""
    train_dataset, val_dataset, _ = make_datasets(cfg.CONTROLNET.DATA)
    
    # if cfg.CONTROLNET.TRAIN.MAX_TRAIN_SAMPLES:
    #     indices = torch.randperm(len(train_dataset))[:cfg.CONTROLNET.TRAIN.MAX_TRAIN_SAMPLES]
    #     train_dataset = torch.utils.data.Subset(train_dataset, indices)
    
    # if cfg.CONTROLNET.EVAL.MAX_VALIDATION_SAMPLES < len(test_dataset):
    #     # Non-deterministic
    #     indices = torch.linspace(0, len(test_dataset)-1, steps=cfg.CONTROLNET.EVAL.MAX_VALIDATION_SAMPLES, dtype=torch.int)
    #     test_dataset = torch.utils.data.Subset(test_dataset, indices)
    
    train_dataloader = torch.utils.data.DataLoader(
        train_dataset,
        shuffle=True,
        # collate_fn=collate_fn,
        batch_size=cfg.CONTROLNET.TRAIN.TRAIN_BATCH_SIZE,
        num_workers=cfg.CONTROLNET.TRAIN.DATALOADER_NUM_WORKERS,
    )
    
    val_dataloader = torch.utils.data.DataLoader(
        val_dataset,
        shuffle=False,
        # collate_fn=collate_fn,
        batch_size=1,#cfg.CONTROLNET.EVAL.BATCH_SIZE,
        num_workers=cfg.CONTROLNET.TRAIN.DATALOADER_NUM_WORKERS,
    )
    
    return train_dataloader, val_dataloader


def encode_and_render(
    ray_sampler: ImageRaySampler, 
    renderer: NeRFRenderer, 
    imgs: torch.Tensor, 
    depths: torch.Tensor, 
    projs_enc: torch.Tensor, 
    poses_enc: torch.Tensor, 
    projs_rend: torch.Tensor, 
    poses_rend: torch.Tensor, 
    black_invalid: bool = False
    ):
    """Helper function."""
    renderer.net.encode(imgs, projs_enc, poses_enc, depths)
    renderer.net.set_scale(0)
    
    all_rays, _ = ray_sampler.sample(None, poses_rend, projs_rend)      # [n*nv, n_pts, 8]
    render_dict = renderer(all_rays, want_weights=True, want_alphas=True)

    render_dict["fine"] = dict(render_dict["coarse"])
    render_dict = ray_sampler.reconstruct(render_dict)

    depths = render_dict["coarse"]["depth"].unsqueeze(2)                                                                    # [B, 1, 1, H, W]
    frames = render_dict["coarse"]["rgb"].squeeze(4).permute(0, 1, 4, 2, 3)                                                 # [B, 1, 3, H, W]
    invalid = (render_dict["coarse"]["invalid"].squeeze(-1) * render_dict["coarse"]["weights"]).sum(-1).unsqueeze(2) > .8   # [B, 1, 1, H, W]
    
    if black_invalid:
        depths[invalid] = depths.max()
        frames[invalid.repeat(1, 1, 3, 1, 1)] = 0
    
    return frames, depths, invalid

# @torch.cuda.amp.custom_fwd(cast_inputs=torch.float32)
def comp_reprojected_img_extents(projs_enc, projs_new, depths_enc, c2w_enc, c2w_new, w_new, h_new):
    """Helper function that returns the min and max image coordinates of a re-projected image."""
    
    # assert c2w_enc.shape[1] == projs_enc.shape[1] == 1, "Multiple encoding NVs not implemented"
    
    # c2w_new_float32 = c2w_new.to(dtype=torch.float32)
    # c2w_new_inv = c2w_new_float32.inverse()
    
    B = projs_enc.shape[0]
    corners = torch.tensor([
        [-1., -1., 1.],     # top-left
        [1., -1., 1.],      # top-right
        [1., 1., 1.],       # bottom-right
        [-1., 1., 1.],      # bottom-left
    ], device=projs_enc.device, dtype=projs_enc.dtype).unsqueeze(0).unsqueeze(0).repeat(B, 1, 1, 1)
    corners_depth = depths_enc[..., [0, 0, -1, -1], [0, -1, -1, 0]]
    
    projs_enc_inv = torch.inverse(projs_enc.float()).to(projs_enc.dtype)
    corners = (projs_enc_inv @ corners.transpose(3, 2)).transpose(3, 2)
    # Norm the vectors, scale by depths and make homogeneous coordinates.
    corners = corners #/ corners.norm(p=2, dim=-1, keepdim=True)         # TODO check why this norm is not needed
    corners = corners * corners_depth[:, :, 0, :, None]
    corners = torch.concatenate([corners, torch.ones_like(corners[..., :1])], dim=-1)
    # This automatically works with c2w_new [B, nv, 4, 4] and c2w_enc [B, 1, 4, 4]
    w2c_new = torch.inverse(c2w_new.float()).to(c2w_new.dtype)
    corners_nv = (projs_new @ (w2c_new @ c2w_enc)[..., :3, :] @ corners.transpose(3, 2)).transpose(3, 2)
    corners_nv = (corners_nv / (corners_nv[..., 2:3] + 1e-9))[..., :2]
    # Un-normalize pixel coordinates
    corners_nv = corners_nv * 0.5 + 0.5
    corners_nv[..., 0] *= w_new
    corners_nv[..., 1] *= h_new
    
    min_left_and_top = torch.min(corners_nv, dim=-2)[0]
    max_right_and_bottom = torch.max(corners_nv, dim=-2)[0]
    # l = min_left_and_top[:, :, 0].clamp(0, w_new) 
    # t = min_left_and_top[:, :, 1].clamp(0, h_new)
    # r = max_right_and_bottom[:, :, 0].clamp(0, w_new)
    # b = max_right_and_bottom[:, :, 1].clamp(0, h_new)
    l = min_left_and_top[:, :, 0]
    t = min_left_and_top[:, :, 1]
    r = max_right_and_bottom[:, :, 0]
    b = max_right_and_bottom[:, :, 1]
    return l, t, r, b       # each is [B, NV] in pixels


def sample_rotated_poses_and_projs(cfg: MainConfig, poses_c2w: torch.Tensor, projs: torch.Tensor, depths: torch.Tensor, hw: Tuple[int], hw_rot: Tuple[int], nv: int = 1, generator=None):
    """
    Args:
        cfg (MainConfig): _description_
        poses_c2w (torch.Tensor): [B, 1, 4, 4]
        projs (torch.Tensor): [B, 1, 3, 3]
        depths (torch.Tensor): [B, 1, 1, H, W]
        nv (int, optional): Number of novel views. Defaults to 1.
    """
    B, _, _, _ = poses_c2w.shape
    H, W = hw
    H_ROT, W_ROT = hw_rot
    device = poses_c2w.device
    dtype = poses_c2w.dtype
    
    x_deg_min, x_deg_max = cfg.CONTROLNET.TRAIN.NOVEL_VIEW_X_ROTATION_LIMITS
    y_deg_min, y_deg_max = cfg.CONTROLNET.TRAIN.NOVEL_VIEW_Y_ROTATION_LIMITS
    
    x_degs = torch.rand(B*nv, generator=generator, device=device, dtype=dtype) * (x_deg_max-x_deg_min) + x_deg_min
    y_degs = torch.rand(B*nv, generator=generator, device=device, dtype=dtype) * (y_deg_max-y_deg_min) + y_deg_min
    z_dists = torch.full((B*nv,), cfg.CONTROLNET.TRAIN.NOVEL_VIEW_DIST_ROTATION, device=device, dtype=dtype)  # TODO hardcode
    
    poses_rot = []
    for i in range(B):
        poses_nv = orientate_poses(poses_c2w[i:i+1].repeat(1, nv, 1, 1), x_degs[i*nv:(i+1)*nv], y_degs[i*nv:(i+1)*nv], z_dists[i*nv:(i+1)*nv])
        poses_rot.append(poses_nv)
    
    poses_rot = torch.concat(poses_rot)
    
    # Determine the min and max visible image pixels when viewed from poses_rot (using same intrinsics).
    l, t, r, b = comp_reprojected_img_extents(projs, projs, depths, poses_c2w, poses_rot, W, H)         # FIXME these should maybe be W_RECT and H_RECT???
    
    # Sample crop from visible pixels.
    # l_sample = get_interval_sample(l.clamp(0, W), r.clamp(0, W)-W_ROT-l.clamp(0, W))
    w_interval = (r-W_ROT-l).clamp(0, torch.inf)
    # w_interval = r-W_ROT-l
    
    edge_dist = cfg.CONTROLNET.TRAIN.EDGE_DIST_FOR_PROJ_SAMPLE * w_interval
    # l_sample = get_interval_sample(l + l_edge_dist, torch.where(w_interval>0., w_interval, 0.), generator=generator)
    # l_sample = get_interval_sample(l, torch.where(w_interval>0., w_interval, 0.), generator=generator)
    l_sample = get_interval_sample(l+edge_dist, w_interval-2*edge_dist, generator=generator)
    # l_sample = get_interval_sample(r-W_ROT, 0)
    # l_sample = get_interval_sample(l, 0)
    l_sample = l_sample.round().long()
    # l_sample = l_sample.clamp(0, W-W_ROT).round().long()
    # t_sample = get_interval_sample(t.clamp(0, H), b.clamp(0, H)-H_ROT-t.clamp(0, H))
    h_interval = (b-H_ROT-t).clamp(0, torch.inf)
    edge_dist = cfg.CONTROLNET.TRAIN.EDGE_DIST_FOR_PROJ_SAMPLE * h_interval
    t_sample = get_interval_sample(t+edge_dist, h_interval-2*edge_dist, generator=generator)
    t_sample = t_sample.round().long()
    # FIXME this enforces staying within the projs frustrum... maybe good maybe not?
    t_sample = torch.zeros_like(t_sample)
    # t_sample = t_sample.clamp(0, H-H_ROT).round().long()
    
    projs_rot = crop_intrinsics(projs.repeat(1, nv, 1, 1), l_sample, t_sample, unnormalized_W=W, unnormalized_H=H, new_unnormalized_W=W_ROT, new_unnormalized_H=H_ROT)
    
    return poses_rot, projs_rot, (l, t, r, b, l_sample, t_sample)


# TODO move this into an nn.Module
def make_conditioning_imgs(
    cfg: MainConfig, 
    imgs: torch.Tensor, 
    poses_c2w: torch.Tensor, 
    projs: torch.Tensor, 
    depths: torch.Tensor,
    renderer: NeRFRenderer, 
    tokenizer: Optional[AutoTokenizer] = None,
    poses_rot: Optional[torch.Tensor] = None,
    degs_around_X: Optional[List[float]] = None,
    degs_around_Y: Optional[List[float]] = None,
    lefts_input_crop: Optional[List[float]] = None,
    tops_input_crop: Optional[List[float]] = None,
    lefts_nv_crop: Optional[List[float]] = None,
    tops_nv_crop: Optional[List[float]] = None,
    eval_seed: Optional[int] = None,
    debug: bool = False,
    mode: str = "training",
    ):
    """Preprocesses a batch on the GPU.
    
    Pipeline: 
      (1) input --> novel-view (output during inference) 
      (2) novel-view --> re-rendered input
      (3) re-rendered input --> controlnet conditioning image (output during training)

    Args:
        cfg (ControlnetConfig)
        imgs (torch.Tensor): [B, 1, 3, H, W]
        poses (torch.Tensor): [B, 1, 4, 4]
        projs (torch.Tensor): [B, 1, 3, 3]
        depths (Callable): [B, 1, 1, H, W]
        poses_rot (torch.Tensor): [B, NV, 4, 4]
        density_field (PseudoVolume)
        renderer (NeRFRenderer)
        ray_sampler (ImageRaySampler)
    """
    
    generator = torch.Generator(device=device).manual_seed(eval_seed) if eval_seed else None
    
    debug_dict = {}
    data = {}
    
    # Sample rotation
    batch_size = len(imgs)
    
    # Convert from [-1, 1] to [0, 1]
    imgs = imgs * .5 + .5

    # Move coordinate system to input frame
    poses = poses_c2w #torch.inverse(poses_c2w) @ poses_c2w     # TODO these should be close to eye.... maybe check this?
    
    H, W = imgs.shape[-2:]
    # The resolution of the input-encoding image
    H_INP, W_INP = (H, W) if not cfg.CONTROLNET.TRAIN.ENCODING_RESOLUTION else cfg.CONTROLNET.TRAIN.ENCODING_RESOLUTION
    # The novel view resolution
    H_NV, W_NV = (H, W) if not cfg.CONTROLNET.TRAIN.RERENDERING_RESOLUTION else cfg.CONTROLNET.TRAIN.RERENDERING_RESOLUTION
    # The re-rendered input-pose-resolution is a multiple of the controlnet-training-resolution.
    H_CTRL, W_CTRL = cfg.CONTROLNET.TRAIN.RESOLUTION
    scale_factor = min(H_INP/H_CTRL, W_INP/W_CTRL)
    H_REREN = int(H_CTRL * scale_factor)
    W_REREN = int(W_CTRL * scale_factor)
    W_RECT = 1408
    H_RECT = 376
    if mode == "inference":
        H_NV, W_NV = H_REREN, W_REREN
        
    data["hw_reren"] = (H_REREN, W_REREN)
        
    # --- CROP THE INPUT IF NEEDED ---
    
    # (1) Given input image resolution -> Input encoding image resolution
    if lefts_input_crop is None:
        lefts_input_crop = torch.tensor([0] * batch_size, device=device)
    lefts_input_crop = lefts_input_crop.clamp(0, W-W_INP).round().long()
    if tops_input_crop is None:
        tops_input_crop = torch.tensor([0] * batch_size, device=device)
    tops_input_crop = tops_input_crop.clamp(0, H-H_INP).round().long()
        
    crop_to_INP = Crop(H_INP, W_INP, lefts=lefts_input_crop, tops=tops_input_crop)
    projs_INP = crop_intrinsics(projs, lefts_input_crop, tops_input_crop, unnormalized_W=W, unnormalized_H=H, new_unnormalized_W=W_INP, new_unnormalized_H=H_INP) 
    
    imgs = crop_to_INP(imgs)
    depths = crop_to_INP(depths)
        
    # min_left_nv, min_top_nv, max_right_nv, max_bottom_nv = comp_reprojected_img_extents(projs_INP, projs_INP, depths, poses, poses_rot, W_INP, H_INP)    
    
    # (2) Create the novel view intrinsics
    # if mode == "inference":
    #     # This is a CenterCrop TODO: change this
    #     lefts_nv_crop = torch.tensor([(W_INP-W_NV)//2]*batch_size, device=device)
    #     tops_nv_crop = torch.tensor([(H_INP-H_NV)//2]*batch_size, device=device)
        
    # if lefts_nv_crop is None:
    #     lefts_nv_crop = get_interval_sample(min_left_nv, max_right_nv-W_NV-min_left_nv)
    # lefts_nv_crop = lefts_nv_crop.clamp(0, W_INP-W_NV).round().long()
    # # lefts_nv_crop = lefts_nv_crop.clamp(0, max_right_nv-W_NV).round().long()
    # if tops_nv_crop is None:
    #     tops_nv_crop = get_interval_sample(min_top_nv, max_bottom_nv-H_NV-min_top_nv)
    # tops_nv_crop = tops_nv_crop.clamp(0, H_INP-H_NV).round().long()
    # tops_nv_crop = tops_nv_crop.clamp(0, max_bottom_nv-H_NV).round().long()

    # projs_NV = crop_intrinsics(projs_INP, lefts_nv_crop, tops_nv_crop, unnormalized_W=W_INP, unnormalized_H=H_INP, new_unnormalized_W=W_NV, new_unnormalized_H=H_NV)
    
    nv = cfg.NUM_NOVEL_VIEWS if mode == "inference" else 1
    
    if poses_rot is None:
        # # Build rotated pose matrices
        # if degs_around_X is None or degs_around_Y is None:
        #     # Sample random rotations
        #     x_deg_min, x_deg_max = cfg.CONTROLNET.TRAIN.NOVEL_VIEW_X_ROTATION_LIMITS
        #     y_deg_min, y_deg_max = cfg.CONTROLNET.TRAIN.NOVEL_VIEW_Y_ROTATION_LIMITS
        #     # Deterministic behavior
        #     degs_around_X = torch.rand(batch_size, generator=generator, device=device) * (x_deg_max-x_deg_min) + x_deg_min
        #     degs_around_Y = torch.rand(batch_size, generator=generator, device=device) * (y_deg_max-y_deg_min) + y_deg_min
        
        # for pose in poses:
        
        # poses_rot = rotate_poses(poses, degs_around_X, degs_around_Y, )
        
        # During training, only nv == 1 is supported.
        
        poses_rot, projs_NV, debug_sample_rotated_poses_and_projs = sample_rotated_poses_and_projs(
            cfg, poses, projs, depths, 
            nv=nv, 
            hw=(H, W), 
            hw_rot=(H_NV, W_NV), 
            generator=generator,
        )
    else:
        # Do a CenterCrop
        l_NV = torch.tensor([(W_INP-W_NV)//2]*batch_size, device=device)
        t_NV = torch.tensor([(H_INP-H_NV)//2]*batch_size, device=device)
        projs_NV = crop_intrinsics(projs_INP, l_NV, t_NV, unnormalized_W=W_INP, unnormalized_H=H_INP, new_unnormalized_W=W_NV, new_unnormalized_H=H_NV)
        # crop_INP_to_NV = Crop(H_NV, W_NV, lefts=l_NV, tops=t_NV)

    # --- RERENDER INPUT TO PRODUCE OCCLUSIONS IN THE CONDITIONING IMAGE ---
    
    with torch.no_grad():
        # Encode with input views and render novel views at the input resolution. 
        ray_sampler = ImageRaySampler(cfg.BTS.model_conf.z_near, cfg.BTS.model_conf.z_far, H_NV, W_NV, norm_dir=False)
        imgs_nv, depths_nv, invalid_nv = encode_and_render(ray_sampler, renderer, imgs, depths, projs_INP, poses, projs_NV, poses_rot, black_invalid=True)
        
        if debug:
            debug_dict["imgs_in"] = imgs.clone()
            debug_dict["depths_in"] = depths.clone()
            debug_dict["imgs_nv"] = imgs_nv.clone()
            debug_dict["depths_nv"] = depths_nv.clone()
            debug_dict["poses_in"] = poses.clone()
            debug_dict["poses_rot"] = poses_rot.clone()
            debug_dict["min_left_nv"] = debug_sample_rotated_poses_and_projs[0].clone()
            debug_dict["min_top_nv"] = debug_sample_rotated_poses_and_projs[1].clone()
            debug_dict["max_right_nv"] = debug_sample_rotated_poses_and_projs[2].clone()
            debug_dict["max_bottom_nv"] = debug_sample_rotated_poses_and_projs[3].clone()
            debug_dict["lefts_nv_crop"] = debug_sample_rotated_poses_and_projs[4].clone()            
            debug_dict["top_nv_crop"] = debug_sample_rotated_poses_and_projs[5].clone()            

        # Determine the pixel-boundaries of the re-rendered image
        min_left_reren, min_top_reren, max_right_reren, max_bottom_reren = comp_reprojected_img_extents(
            projs_NV,
            projs_INP, 
            depths_nv, 
            poses_rot, 
            poses, 
            W_INP, H_INP
        )    
        # min_left_reren, min_top_reren, max_right_reren, max_bottom_reren = comp_reprojected_img_extents(
        #     projs_NV.view(batch_size*nv, 1, 3, 3),      # This method is only implemented for nv_dim=1
        #     projs_INP, 
        #     depths_nv.view(batch_size*nv, 1, H_NV, W_NV), 
        #     poses_rot.view(batch_size*nv, 1, 4, 4), 
        #     poses, 
        #     W_INP, H_INP
        # )    
        # We need to clamp these to [0, W_INP or H_INP] because we do not care about input image pixels
        # outside of this range.
        min_left_reren = min_left_reren.clamp(0, W_INP)
        max_right_reren = max_right_reren.clamp(0, W_INP)
        min_top_reren = min_top_reren.clamp(0, H_INP)
        max_bottom_reren = max_bottom_reren.clamp(0, H_INP)
        # Then sample the left and top cropping values.
        w_interval = (max_right_reren-W_REREN-min_left_reren).clamp(0, torch.inf)
        h_interval = (max_bottom_reren-H_REREN-min_top_reren).clamp(0, torch.inf)
        if mode == "training":
            lefts_reren = get_interval_sample(min_left_reren, w_interval, generator=generator)  # [B, nv]
            tops_reren = get_interval_sample(min_top_reren, h_interval, generator=generator)  # [B, nv]
        elif mode == "inference":
            lefts_reren = min_left_reren + w_interval / 2
            tops_reren = min_top_reren + h_interval / 2
        lefts_reren = lefts_reren.clamp(0, W_INP-W_REREN).round().long().flatten()  # [B*NV]
        tops_reren = tops_reren.clamp(0, H_INP-H_REREN).round().long().flatten()    # [B*NV]
        crop_INP_to_REREN = Crop(H_REREN, W_REREN, lefts=lefts_reren, tops=tops_reren)
        
        if debug:
            debug_dict["min_left_reren"] = min_left_reren.clone()
            debug_dict["min_top_reren"] = min_top_reren.clone()
            debug_dict["max_right_reren"] = max_right_reren.clone()
            debug_dict["max_bottom_reren"] = max_bottom_reren.clone()
            debug_dict["lefts_reren"] = lefts_reren.clone()
        
        if mode == "training":
            # (3) Create the re-rendered image cropping function and intrinsics.
            # Apply the new resolution to the intrinsics.
            projs_REREN = crop_intrinsics(projs_INP, lefts_reren, tops_reren, unnormalized_W=W_INP, unnormalized_H=H_INP, new_unnormalized_W=W_REREN, new_unnormalized_H=H_REREN)  
            ray_sampler = ImageRaySampler(cfg.BTS.model_conf.z_near, cfg.BTS.model_conf.z_far, H_REREN, W_REREN, norm_dir=False) 
            # Encode with novel views and re-render input views at controlnet-aspect-ratio.
            img_rerendered, depth_rerendered, invalid_rerendered = encode_and_render(ray_sampler, renderer, imgs_nv, depths_nv, projs_NV, poses_rot, projs_REREN, poses, black_invalid=True)
            
            if debug:
                debug_dict["imgs_rerendered"] = img_rerendered.clone()
                debug_dict["depth_rerendered"] = depth_rerendered.clone()
                
            occlusion_masks_dict = comp_occlusion_map(
                cfg.CONTROLNET.TRAIN.OCCLUSIONS,
                pose1=poses.squeeze(1),
                pose2=poses.squeeze(1),
                # We crop the input depth so it should have the same intrinsics as the rerendering camera.
                proj1=projs_REREN.squeeze(1),
                proj2=projs_REREN.squeeze(1),
                depth1=depth_rerendered.squeeze(1),
                depth2=crop_INP_to_REREN(depths.squeeze(1)),
            )   # [B, 1, H, W]
            
            imgs_conditioning = img_rerendered.squeeze(1)
            depths_conditioning = depth_rerendered.squeeze(1)
            invalid_pixels = invalid_rerendered.squeeze(1)
            masks_conditioning = occlusion_masks_dict["occlusions_full"]       # Visible is 1.0, occluded is 0.0
            data["poses_out"] = poses.squeeze(1)                # [B, 4, 4]
            data["projs_out"] = projs_REREN.squeeze(1)          # [B, 3, 3]
            # Original width and height come from: 
            #   W_REREN/W_INP * W_INP/W * W_RECT = W_CTRL/W * W_RECT
            #   and
            #   H_REREN/H_INP * H_INP/H * H_RECT = H_CTRL/H * H_RECT
            projs_out_unnormalized = unnormalize_intrinsics(data["projs_out"], width=W_REREN/W * W_RECT, height=H_REREN/H * H_RECT)          # [B, 3, 3]
            data["projs_out_unnormalized"] = scale_intrinsics(projs_out_unnormalized, scale_width=W_CTRL/W_REREN, scale_height=H_CTRL/H_REREN)          # [B, 3, 3]
            
            
            # Scale according to resized output.
            # projs_out = scale_proj(projs_REREN.squeeze(1), scale_width=W_REREN/W_CTRL, scale_height=H_REREN/H_CTRL)          # [B, 3, 3]

        elif mode == "inference":
            
            # comp_reprojected_img_extents(projs_rot, projs, depths, poses_rot, poses_c2w, W_ROT, H_ROT) 
    
            # crop_INP_to_NV = Crop(H_ROT, W_ROT, lefts=l_sample, tops=t_sample)
            input_crops_for_optflow = crop_INP_to_REREN(imgs.squeeze(1).repeat_interleave(nv, dim=0))
            imgs_conditioning = imgs_nv.view(batch_size*nv, 3, H_REREN, W_REREN)
            occlusion_masks_dict = comp_occlusion_map(
                cfg.CONTROLNET.INFERENCE_OCCLUSIONS,
                img1=input_crops_for_optflow,
                img2=imgs_conditioning,
            ) # dict with [B*nv, 1, H, W]
            
            depths_conditioning = depths_nv.view(batch_size*nv, 1, H_REREN, W_REREN)
            invalid_pixels = invalid_nv.view(batch_size*nv, 1, H_REREN, W_REREN)
            # Visible is 1.0, occluded is 0.0
            masks_conditioning = occlusion_masks_dict["occlusions_full"].view(batch_size*nv, 1, H_REREN, W_REREN)
            
            data["poses_out"] = poses_rot.view(batch_size*nv, 4, 4)#.view(batch_size, nv, 4, 4)               # [B, 4, 4]
            projs_out_unnormalized = unnormalize_intrinsics(projs_NV, width=W_NV/W * W_RECT, height=H_NV/H * H_RECT)          # [B, 3, 3]
            # projs_out_unnormalized = scale_intrinsics(projs_out_unnormalized, scale_width=W_CTRL/W_NV, scale_height=H_CTRL/H_NV)
            data["projs_out"] = projs_NV.view(batch_size*nv, 3, 3)#.view(batch_size, nv, 3, 3)
            data["projs_out_unnormalized"] = projs_out_unnormalized.view(batch_size*nv, 3, 3)#.view(batch_size, nv, 3, 3)          # [B, 3, 3]
            
            # data["projs_out"] = scale_proj(projs_NV.squeeze(1), scale_width=W_CTRL/W_NV, scale_height=H_CTRL/H_NV)        # [B, 3, 3]
            # projs_out = scale_proj(projs_REREN.squeeze(1), scale_width=W_NV/W_CTRL, scale_height=H_NV/H_CTRL)          # [B, 3, 3]
            
            # imgs_conditioning = crop_NV_to_REREN(imgs_nv.squeeze(1))
            # depths_conditioning = crop_NV_to_REREN(depths_nv.squeeze(1))
            # invalid_pixels = crop_NV_to_REREN(invalid_nv.squeeze(1))
            # masks_conditioning = crop_NV_to_REREN(occlusion_masks_dict["occlusions_full"])       # Visible is 1.0, occluded is 0.0
            # imgs_conditioning = crop_INP_to_REREN(imgs_nv.squeeze(1))
            # depths_conditioning = crop_INP_to_REREN(depths_nv.squeeze(1))
            # invalid_pixels = crop_INP_to_REREN(invalid_nv.squeeze(1))
            # masks_conditioning = crop_INP_to_REREN(occlusion_masks_dict["occlusions_full"])       # Visible is 1.0, occluded is 0.0
            if debug:
                debug_dict["input_crops_for_optflow"] = input_crops_for_optflow.view(batch_size, nv, 3, *input_crops_for_optflow.shape[-2:])
            
        if debug:
            debug_dict["occlusion_masks_dict"] = occlusion_masks_dict
        
        depths_conditioning = (depths_conditioning - cfg.BTS.model_conf.z_near) / (cfg.BTS.model_conf.z_far - cfg.BTS.model_conf.z_near)
        masks_conditioning[invalid_pixels] = DEFINITIONS.IS_INVALID
        
        cond_input_type, cond_input_channels = cfg.CONTROLNET.MODEL.CONDITIONING_INPUT_TYPE_AND_CHANNELS
        if cond_input_type == "rgb":
            pass
        elif cond_input_type == "rgbd":
            imgs_conditioning = torch.concat([imgs_conditioning, depths_conditioning], dim=1)
        elif cond_input_type == "rgbm":
            imgs_conditioning = torch.concat([imgs_conditioning, masks_conditioning], dim=1)
        elif cond_input_type == "rgbdm":
            imgs_conditioning = torch.concat([imgs_conditioning, depths_conditioning, masks_conditioning], dim=1)
        else:
            raise NotImplementedError(f"Conditioning input type {cfg.CONTROLNET.MODEL.CONDITIONING_INPUT_TYPE} does not exist.")

        if cfg.CONTROLNET.MODEL.MASK_CONDITIONING_IMAGE:
            # Set masked pixels in every channel to 0 except in the mask channel if present.
            num_non_mask_channels = cond_input_channels - 1 if "m" in cond_input_type else cond_input_channels
            imgs_conditioning[:, :num_non_mask_channels, :, :] = color_occlusion_masked_img(
                imgs_conditioning[:, :num_non_mask_channels, :, :], 
                masks_conditioning, 
                c_occl=torch.zeros([num_non_mask_channels], device=imgs_conditioning.device),   # black
                c_inv=torch.zeros([num_non_mask_channels], device=imgs_conditioning.device),    # black 
            )
            
    # --- APPLY FINAL TRANSFORMS ---
    # Resize such that the smaller image edge matches the desired resolution
    resize_to_CTRL = transforms.Resize((H_CTRL, W_CTRL), interpolation=transforms.InterpolationMode.BILINEAR, antialias=None)
    resize_to_CTRL_nearest = transforms.Resize((H_CTRL, W_CTRL), interpolation=transforms.InterpolationMode.NEAREST, antialias=None)
    
    imgs_conditioning = resize_to_CTRL(imgs_conditioning)             # [B*NV, C, H, W]
    masks_conditioning = resize_to_CTRL_nearest(masks_conditioning)                        # [B*NV, 1, H, W]

    # data["projs_out_unnormalized"] = unnormalize_intrinsics(projs_out, width=W_CTRL/W * W_RECT, height=H_CTRL/H * H_RECT)          # [B, 3, 3]
    # projs_out = scale_proj(projs_REREN.squeeze(1), scale_width=W_NV/W_CTRL, scale_height=H_NV/H_CTRL)          # [B, 3, 3]


    if mode == "training":
        
        gt_transforms = transforms.Compose([
            crop_INP_to_REREN,
            resize_to_CTRL,
        ])
        
        data["depths"] = gt_transforms(depths.squeeze(1))                     # [B, 3, H, W]
        imgs = gt_transforms(imgs.squeeze(1))                                 # [B, 3, H, W]
        # Normalize to [-1, 1]
        data["imgs"] = transforms.Normalize([0.5], [0.5])(imgs)
        data["conditioning_imgs"] = imgs_conditioning
        data["masks"] = masks_conditioning
    elif mode == "inference":
        data["conditioning_imgs"] = imgs_conditioning#.view(batch_size, nv, 3, *imgs_conditioning.shape[-2:])
        data["masks"] = masks_conditioning#.view(batch_size, nv, 1, *masks_conditioning.shape[-2:])
    
    # --- TOKENIZE CAPTION ---
    
    # from transformers import pipeline

    # captioner = pipeline("image-to-text",model="Salesforce/blip-image-captioning-large", device=0)

    # def caption_image_data(example):
    #     image = example["image"]
    #     image_caption = captioner(image)[0]['generated_text']
    #     example['image_caption'] = image_caption
    #     return example
    
    get_str_fn = lambda x: "" if random.random() < cfg.CONTROLNET.TRAIN.PROPORTION_EMPTY_PROMPTS else x
    caption_strs = [get_str_fn(cfg.CONTROLNET.TRAIN.PROMPT_TEXT) for _ in range(batch_size*nv)]
    
    captions_ids = None
    if tokenizer:
        inputs = tokenizer(
            caption_strs, max_length=tokenizer.model_max_length, padding="max_length", truncation=True, return_tensors="pt"
        )
        
        captions_ids = inputs.input_ids.to(imgs.device)
        
    data["caption_strs"] = caption_strs         # [B]
    data["captions_tokenized"] = captions_ids   # None or [B, tokenizer.model_max_length]
    
    return data, debug_dict



# DEBUGGING
if __name__ == "__main__":
    from configs.structured_configs.config_utils import register_default_configs, check_and_post_init_config
    from omegaconf import OmegaConf
    from hydra import initialize, compose


    register_default_configs()
    config_path="default_controlnet.yaml"
    with initialize(version_base=None, config_path="../../configs"):
        cfg = compose(config_name=config_path)
        
    cfg = check_and_post_init_config(cfg)
    
    cfg.CONTROLNET.TRAIN.TRAIN_BATCH_SIZE = 1
    cfg.NUM_NOVEL_VIEWS = 3
    
    with torch.no_grad():
        
        density_field = PseudoVolume()
        renderer = NeRFRenderer.from_conf(asdict(cfg.BTS)["renderer"])
        renderer = renderer.bind_parallel(density_field, gpus=None).eval().to(device)

        train_dl, test_dl = make_controlnet_dataloaders(cfg)

        batch = next(iter(train_dl))
        
        imgs = batch["imgs"].to(device)
        poses = batch["poses"].to(device)
        projs = batch["projs"].to(device)
        depths = batch["depths"].to(device)
        
        imgs_cond, _ = make_conditioning_imgs(
            cfg,
            imgs,
            poses,
            projs,
            depths,
            renderer=renderer,
            mode="inference",
            debug=True,
        )
