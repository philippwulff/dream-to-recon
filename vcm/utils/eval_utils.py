from typing import Callable, Literal, Optional, List, Dict, Callable
from tqdm.auto import tqdm
from PIL import Image
import numpy as np
import torch
import torch.utils.checkpoint
# from diffusers import StableDiffusionControlNetPipeline, UniPCMultistepScheduler
# from utils.depth_predictors import Metric3DWrapper
from vcm.utils.model_utils import make_model_components
from bts.gt_synthesis.gt_synthesis import GTSynthesisWrapper, Data, Outputs
from utils.projection_ops import comp_align_scale_shift, align_depth, align_inv_depth
from bts.losses.utils import compute_errors_l1ssim, compute_depth_metrics, compute_rgb_metrics, perceptual_metrics
from utils.occlusion_ops import DEFINITIONS
from configs.structured_configs.main_config import MainConfig
import time
from utils.depth_predictors import make_depth_predictor
from torchvision.transforms.functional import resize
from torchvision.transforms import InterpolationMode
from collections import defaultdict
from bts.common.ray_sampler import ImageRaySampler
from utils.utils import invert_depth
from utils.plotting import render_profile


def image_grid(imgs, rows, cols):
    assert len(imgs) == rows * cols

    w, h = imgs[0].size
    grid = Image.new("RGB", size=(cols * w, rows * h))

    for i, img in enumerate(imgs):
        grid.paste(img, box=(i % cols * w, i // cols * h))
    return grid


def format_conditioning_imgs(conditioning_imgs: torch.Tensor) -> torch.Tensor:
    expand_shape = [1] * len(conditioning_imgs.shape)
    expand_shape[-3] = 3
    conditioning_image_parts = [conditioning_imgs[..., :3, :, :]]
    for c in range(3, conditioning_imgs.shape[-3]):
        conditioning_image_parts.append(
            conditioning_imgs[..., c:c+1, :, :].repeat(expand_shape)   # Single channel to RGB
        )
    formatted_imgs = torch.concat(conditioning_image_parts, dim=-1)
    return formatted_imgs


def run_validation_input_view(
    cfg: MainConfig, 
    model: GTSynthesisWrapper,
    device, 
    dataloader: torch.utils.data.DataLoader, 
    callback: Optional[Callable] = None,
    # eval_depth_mode: Literal["depth", "inverse"] = "inverse", 
    # scale_only: bool = False, 
    # max_depth_to_align: Optional[float] = None
):
    
    progress_bar = tqdm(
        range(0, len(dataloader) * cfg.CONTROLNET.EVAL.NUM_VALIDATION_IMAGE_VERSIONS),
        desc="Validation Steps",
        # Only show the progress bar once on each machine.
    )
    
    # mse_fn = lambda x, y: torch.mean((x - y)**2)
    # psnr_fn = lambda mse, peak=1.0: 10 * torch.log10(peak**2/mse)
    
    image_logs = []

    for b, batch in enumerate(dataloader):
        
        callback_inputs = defaultdict(list)
        
        imgs = torch.stack(batch["imgs"], dim=1).to(device)
        poses = torch.stack(batch["poses"], dim=1).to(device)
        poses = poses.inverse() @ poses
        projs = torch.stack(batch["projs"], dim=1).to(device)
        
        time_start = time.time()
        with torch.no_grad(), torch.autocast("cuda"):
            # with :            # TODO why is this needed... possibly fix it?
            # We set the seed such that generated poses and projs are the same 
            # for a given test scene during every validation run.
            out, data, _ = model(imgs, poses, projs, output_in_nv=False, refine_output=False, seed=42 + b)
            data: Data = data

            imgs_gt = out.IMGS_GT
            depths_gt = out.DEPTHS_GT

            imgs_versions = []
            depths_versions = []
            for _ in range(cfg.CONTROLNET.EVAL.NUM_VALIDATION_IMAGE_VERSIONS):
                imgs_refined, depths_out, imgs_conditioning_out, hw_out_new = model.refine_and_predict_depth(data)
                imgs_refined = imgs_refined.squeeze(1)
                depths_out = depths_out.squeeze(1)
                
                imgs_conditioning_out = imgs_conditioning_out.squeeze(1)
                imgs_versions.append(imgs_refined.cpu().float())
                depths_versions.append(depths_out.cpu().float())
                progress_bar.update(1)
                    
            imgs_gt = resize(imgs_gt, hw_out_new, interpolation=InterpolationMode.BILINEAR, antialias=None).cpu().float()
            depths_gt = resize(depths_gt, hw_out_new, interpolation=InterpolationMode.BILINEAR, antialias=None).cpu().float()
        
        torch.cuda.synchronize()
        per_sample_time = (time.time() - time_start) / (dataloader.batch_size * cfg.CONTROLNET.EVAL.NUM_VALIDATION_IMAGE_VERSIONS)

        for i in range(dataloader.batch_size):
            
            formatted_images = format_conditioning_imgs(imgs_conditioning_out[i]).permute(1, 2, 0).cpu()
            img_versions = [_[i] for _ in imgs_versions]
            formatted_images = torch.concatenate([formatted_images, *[_.permute(1, 2, 0) for _ in img_versions]], dim=1)
            
            # GT img, depth, mask
            prompt = data.CAPTION_STRS[i]
            # gt_img = imgs_gt[i].permute(1, 2, 0) * 0.5 + 0.5
            gt_img = imgs_gt[i] * 0.5 + 0.5
            assert gt_img.min() >= 0.0 and gt_img.min() <= 1.0, "NORMALIZING ERROR"
            gt_depth_pt = depths_gt[i].unsqueeze(0)       # [1, 1, H, W]
            depth_versions_pt = [_[i].unsqueeze(0) for _ in depths_versions]   # list of [1, 1, H, W]
            mask = data.MASKS_REREN[i].cpu().squeeze().unsqueeze(0).unsqueeze(0) == DEFINITIONS.IS_VISIBLE
            # rmse, rmse_masked, psnr, psnr_masked, scale, shift = 0, 0, 0, 0, 0, 0
            scale, shift = 0, 0
            rgb_metrics_l, rgb_metrics_masked_l, metrics_l, metrics_masked_l = [], [], [], []
            img_versions_depths = [gt_depth_pt.squeeze().cpu()]
            img_versions_depths_inv = [1 / (gt_depth_pt.squeeze().cpu() + 1e-9)]
            img_versions_depth_diffs = []
            for img, depth in zip(img_versions, depth_versions_pt):
                # mse = mse_fn(gt_img, img)
                # mse_masked = mse_fn(img[mask == 1], gt_img[mask == 1])
                # rmse += np.sqrt(mse)
                # psnr += psnr_fn(mse)
                # rmse_masked += np.sqrt(mse_masked)
                # psnr_masked += psnr_fn(mse_masked)
                
                # Compute RGB statistics
                rgb_metrics_l.append(compute_rgb_metrics(pred=img.unsqueeze(0), gt=gt_img.unsqueeze(0)))
                rgb_metrics_masked_l.append(compute_rgb_metrics(pred=img.unsqueeze(0), gt=gt_img.unsqueeze(0), valid_mask=mask.repeat(1, 3, 1, 1)))
                
                # Predict depth on denoised image
                # gt_depth_align = gt_depth_pt.clone()
                # if eval_depth_mode == "inverse":
                #     depth = 1 / (depth + 1e-9)
                #     gt_depth_align = 1 / (gt_depth_align + 1e-9)
                # # Align depth scale to GT
                # scale, shift = comp_align_scale_shift(depth, gt_depth_align, scale_only=scale_only, max_depth_to_align=max_depth_to_align)
                # depth = depth * scale + shift
                # if eval_depth_mode == "inverse":
                #     depth = 1 / (depth + 1e-9)
                depth, scale, shift = model._align_depth(depth.unsqueeze(1), gt_depth_pt.unsqueeze(1))
                depth = depth.squeeze(1)
                # if eval_depth_mode == "inverse":
                #     depth, scale, shift = align_inv_depth(depth, gt_depth_pt, scale_only=scale_only, max_depth_to_align=max_depth_to_align)
                # else:
                #     depth, scale, shift = align_depth(depth, gt_depth_pt, scale_only=scale_only, max_depth_to_align=max_depth_to_align)
                
                # Compute depth statistics
                metrics_l.append(compute_depth_metrics(pred=depth, gt=gt_depth_pt))
                metrics_masked_l.append(compute_depth_metrics(pred=depth, gt=gt_depth_pt, valid_mask=mask))
                
                img_versions_depths.append(depth.squeeze())
                img_versions_depths_inv.append(1 / (depth.squeeze() + 1e-9))
                img_versions_depth_diffs.append(gt_depth_pt.squeeze() - depth.squeeze())
                scale += float(scale)
                shift += float(shift)
                
            depth_pixel_var = torch.stack(depth_versions_pt).var(dim=0).squeeze()   # [H, W]
            depth_mean = torch.stack(depth_versions_pt).mean()
            depth_var = torch.stack(depth_versions_pt).var()

            results = {
                "formatted_images": formatted_images, 
                "prompt": prompt,
                # "gt_img": gt_img,
                # "mask": mask,
                # "avg_rmse": float(rmse / len(img_versions)),
                # "avg_rmse_masked": float(rmse_masked / len(img_versions)),
                # "avg_psnr": float(psnr / len(img_versions)),
                # "avg_psnr_masked": float(psnr_masked / len(img_versions)),
                "avg_denoising_latency": float(per_sample_time),
                # "formatted_depths": np.concatenate(img_versions_depths, axis=1),
                # "formatted_depths_inv": np.concatenate(img_versions_depths_inv, axis=1),
                # "formatted_depth_diffs": np.concatenate(img_versions_depth_diffs, axis=1),
                "avg_scale": float(scale / len(img_versions)),
                "avg_shift": float(shift / len(img_versions)),
                "depth_mean": float(depth_mean),
                "depth_var": float(depth_var),
                # "depth_pixel_var": depth_pixel_var.numpy(),
            }
            for prefix, metrics_, metrics_masked_ in [("avg_depth", metrics_l, metrics_masked_l), ("avg_rgb", rgb_metrics_l, rgb_metrics_masked_l)]:
                for k in metrics_[0].keys():
                    non_nan = [_[k] for _ in metrics_ if not _[k].isnan()]
                    non_nan_masked = [_[k] for _ in metrics_masked_ if not _[k].isnan()]
                    results[f"{prefix}_{k}"] = float(sum(non_nan) / len(non_nan)) if non_nan else 0.0
                    results[f"{prefix}_{k}_masked"] = float(sum(non_nan_masked) / len(non_nan_masked)) if non_nan_masked else 0.0
                
            image_logs.append(results)
            
            callback_inputs["results"].append(results)
            callback_inputs["formatted_images"].append(formatted_images)
            callback_inputs["prompt"].append(prompt)
            callback_inputs["gt_img"].append(gt_img)
            callback_inputs["mask"].append(mask)
            callback_inputs["formatted_depths"].append(torch.concat(img_versions_depths, dim=1))
            callback_inputs["formatted_depths_inv"].append(torch.concat(img_versions_depths_inv, dim=1))
            callback_inputs["formatted_depth_diffs"].append(torch.concat(img_versions_depth_diffs, dim=1))
            callback_inputs["depth_pixel_var"].append(depth_pixel_var)
            
        if callback is not None:
            # Callback receives current batch as input
            callback(
                batch_idx=b,
                data=data,
                **callback_inputs
            )

    return image_logs  


def run_validation_novel_view(
    cfg: MainConfig, 
    model: GTSynthesisWrapper,
    device, 
    dataloader: torch.utils.data.DataLoader, 
    callback: Optional[Callable] = None,
    ):
    
    progress_bar = tqdm(
        range(len(dataloader) * cfg.CONTROLNET.EVAL.NUM_VALIDATION_IMAGE_VERSIONS),
        desc="Validation Steps",
        # Only show the progress bar once on each machine.
    )
    
    image_logs = []

    for b, batch in enumerate(dataloader):
        callback_inputs = defaultdict(list)
        
        imgs = torch.stack(batch["imgs"], dim=1).to(device)
        poses = torch.stack(batch["poses"], dim=1).to(device)
        poses = poses.inverse() @ poses
        projs = torch.stack(batch["projs"], dim=1).to(device)
        
        with torch.no_grad(), torch.autocast("cuda"):
            seed = 42 + b       # Per-batch deterministic poses and projs in every validation run.
            out, data, debug_data = model(imgs, poses, projs, output_in_nv=True, refine_output=False, seed=seed, debug=True)
            out: Outputs = out
            data: Data = data
            profiles = render_profile(model.renderer.net, cfg.CONTROLNET.DATA.CAM_INCL_ADJUST.to(device)).cpu().float()

            imgs_versions, depths_versions = [], []
            imgs_versions_reren, depths_versions_reren, invalids_versions_reren = [], [], []
            scales, shifts = [], []
            for _ in range(cfg.CONTROLNET.EVAL.NUM_VALIDATION_IMAGE_VERSIONS):
                imgs_refined, depths_out, imgs_conditioning_out, hw_out_new = model.refine_and_predict_depth(data)
                depths_out, scale, shift = model._align_depth(depths_out, data.DEPTHS_NV)#, mask)       # TODO add mask
                scales.append(scale), shifts.append(shift)
                
                # Rerender the input from the refined novel view.
                imgs_gt, depths_gt, invalid_gt = model.encode_and_render(
                    ImageRaySampler(model.z_near, model.z_far, *data.HW_DATA, norm_dir=True),
                    imgs_refined,
                    depths_out,
                    projs_enc=out.PROJS_OUT,
                    poses_enc=out.POSES_OUT,
                    projs_rend=out.PROJS_IN,
                    poses_rend=out.POSES_IN,
                )
                # TODO maybe render the nv-conditioned profile here?
                
                imgs_versions.append(imgs_refined.cpu().float())        # list of [B, NV, 3, H, W]
                depths_versions.append(depths_out.cpu().float())
                imgs_versions_reren.append(imgs_gt.squeeze(1).cpu().float())    # list of [B, 3, H, W]
                depths_versions_reren.append(depths_gt.squeeze(1).cpu().float())
                invalids_versions_reren.append(invalid_gt.squeeze(1).cpu())
                progress_bar.update(1)
                
                
        imgs_in = out.IMGS_IN.cpu().float()
        depths_in = out.DEPTHS_IN.cpu().float()
        imgs_conditioning_out = imgs_conditioning_out.cpu().float()
                    
        for i in range(dataloader.batch_size):
            
            imgs_synth = torch.stack([_[i] for _ in imgs_versions])
            depths_synth = torch.stack([_[i] for _ in depths_versions])
            imgs_reren = torch.stack([_[i] for _ in imgs_versions_reren])
            depths_reren = torch.stack([_[i] for _ in depths_versions_reren])
            invalids_reren = torch.stack([_[i] for _ in invalids_versions_reren])
            imgs_cond = format_conditioning_imgs(imgs_conditioning_out[i]).permute(0, 2, 3, 1)
            # TODO this is hardcoding nv0 for now
            formatted_images = torch.concatenate([imgs_cond[0], *[_[0].permute(1, 2, 0) for _ in imgs_synth]], dim=1)

            # Metrics in the novel view
            perceptual_metrics_dict = perceptual_metrics(pred=imgs_synth.view(-1, *imgs_synth.shape[-3:]), gt=imgs_in.view(-1, *imgs_in.shape[-3:]))
            # TODO compute TSED between input img and refined novel view img
            
            # Metrics in the input view
            rgb_metrics_dict = compute_rgb_metrics(pred=imgs_reren, gt=imgs_in.squeeze(1), valid_mask=invalids_reren.repeat(1, 3, 1, 1))
            depth_metrics_dict = compute_depth_metrics(pred=depths_reren, gt=depths_in.squeeze(1), valid_mask=invalids_reren)
            
            results = {
                "formatted_images": formatted_images, 
                "prompt": data.CAPTION_STRS[i],
                "avg_scale": float(torch.stack([_[i] for _ in scales]).mean().item()),
                "avg_shift": float(torch.stack([_[i] for _ in shifts]).mean().item()),
                "depth_mean": float(depths_synth.mean()),
                "depth_var": float(depths_synth.var()),
            }
            results.update({f"avg_rgb_{k}_masked": float(v) for k, v in rgb_metrics_dict.items()})
            results.update({f"avg_depth_{k}_masked": float(v) for k, v in depth_metrics_dict.items()})
            results.update({f"synth_view_{k}": float(v) for k, v in perceptual_metrics_dict.items()})
                
            image_logs.append(results)
            
            callback_inputs["results"].append(results)
            callback_inputs["mask"].append(invalids_reren)
            callback_inputs["depth_pixel_var"].append(depths_synth.var(dim=0))
            callback_inputs["imgs_synth"].append(imgs_synth)
            callback_inputs["depths_synth"].append(depths_synth)
            callback_inputs["depths_synth_inv"].append(invert_depth(depths_synth, model.z_near, model.z_far))
            callback_inputs["imgs_reren"].append(imgs_reren)
            callback_inputs["depths_reren"].append(depths_reren)
            callback_inputs["depths_reren_inv"].append(invert_depth(depths_reren, model.z_near, model.z_far))
            callback_inputs["invalids_reren"].append(invalids_reren)
            callback_inputs["imgs_cond"].append(imgs_cond)
            
        if callback is not None:
            callback(
                batch_idx=b,
                data=data,
                debug_data=debug_data,
                profiles=profiles,
                imgs_in=imgs_in,
                **callback_inputs
            )

    return image_logs  


def log_validation(cfg: MainConfig, pipeline, controlnet, accelerator, weight_dtype, step, logger, dataloader: torch.utils.data.DataLoader, gt_synthesizer: GTSynthesisWrapper):
    logger.info(f"Running validation on {len(dataloader.dataset)} images ... ")

    # _, _, _, _, _, _, pipeline = make_model_components(cfg, tokenizer, text_encoder, vae, unet, accelerator.unwrap_model(controlnet), build_pipe=True)
    
    pipeline.controlnet = accelerator.unwrap_model(controlnet)
    # pipeline.scheduler = UniPCMultistepScheduler.from_config(pipeline.scheduler.config)     # TODO add util fn
    pipeline = pipeline.to(accelerator.device)
    gt_synthesizer.refine_pipeline = pipeline
    
    image_logs = run_validation_input_view(
        cfg,
        gt_synthesizer,
        accelerator.device, 
        dataloader, 
        # eval_depth_mode="depth", 
        # scale_only=True, 
    )
    
    gt_synthesizer.refine_pipeline = None
    
    tag_scalar_dict = comp_avg_stats(image_logs)

    for tracker in accelerator.trackers:
        if tracker.name == "tensorboard":
            for i, log in enumerate(image_logs[:cfg.CONTROLNET.EVAL.MAX_VALIDATION_IMAGES_TB]):
                tracker.writer.add_images(f"Val/Img{i} Prompt: '{log['prompt']}'", log["formatted_images"].numpy(), step, dataformats="HWC")
            for k, v in tag_scalar_dict.items():
                tracker.writer.add_scalar(f"Val/{k}", v, step)
        else:
            logger.warn(f"image logging not implemented for {tracker.name}")
            
            
def comp_avg_stats(stats_in: List[dict]) -> Dict[str, float]:
    """Averages float values in a list of dicts."""
    # Creates statistics with 0 initially
    stats_out = defaultdict(float)
    for i, stat in enumerate(stats_in):
        for key in stat.keys():
            if isinstance(stat[key], float):
                stats_out[key] += stat.get(key, 0.0)
                
    # Average statistics
    for key in stats_out.keys():
        stats_out[key] /= len(stats_in)
        
    return stats_out