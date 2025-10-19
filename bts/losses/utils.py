import math
import torch
import lpips
import numpy as np
from scipy.spatial import cKDTree as KDTree
from bts.models.layers.layers import ssim as ssim_fn
from ignite.metrics import FID, InceptionScore
from typing import Dict, Optional
import torch.nn.functional as F
import torchvision.transforms.functional as TF
import skimage


def compute_errors_l1ssim(img0, img1, mask=None):
    n, pc, h, w, nv, c = img0.shape
    img1 = img1.expand(img0.shape)
    img0 = img0.permute(0, 1, 4, 5, 2, 3).reshape(-1, c, h, w)
    img1 = img1.permute(0, 1, 4, 5, 2, 3).reshape(-1, c, h, w)
    errors = .85 * torch.mean(ssim_fn(img0, img1, pad_reflection=False, gaussian_average=True, comp_mode=True), dim=1) + .15 * torch.mean(torch.abs(img0 - img1), dim=1)
    errors = errors.view(n, pc, nv, h, w).permute(0, 1, 3, 4, 2).unsqueeze(-1)
    if mask is not None: return errors, mask
    else: return errors


def compute_depth_metrics(pred: torch.Tensor, gt: torch.Tensor, valid_mask: Optional[torch.BoolTensor] = None) -> Dict[str, torch.Tensor]:
    """Returns depth error metrics. Inputs should be [B, 1, H, W]."""
    # TODO: Maybe implement median scaling
    
    if pred.shape != gt.shape:
        pred = F.interpolate(pred, gt.shape[-2:])
        
    if valid_mask is None:
        valid_mask = gt != 0
    
    # Structural Similarity Index
    # https://en.wikipedia.org/wiki/Structural_similarity_index_measure
    gt_ = gt.masked_fill(~valid_mask, 0.)
    pred_ = pred.masked_fill(~valid_mask, 0.)
    ssim = torch.mean(ssim_fn(gt_, pred_, pad_reflection=False, gaussian_average=True, comp_mode=True))
    
    gt = gt[valid_mask]
    pred = pred[valid_mask]
    
    # Absolute and squared absolute relative error
    abs_rel = torch.mean(torch.abs(gt - pred) / gt.clamp_min(1))
    sq_rel = torch.mean(((gt - pred) ** 2) / gt.clamp_min(1))
    
    # RMSE and Log RMSE
    rmse = (gt - pred) ** 2
    rmse = rmse.mean() ** .5
    rmse_log = (torch.log(gt) - torch.log(pred)) ** 2
    rmse_log = rmse_log.mean() ** .5
    
    # Alpha-1/2/3 accuracy
    thresh = torch.maximum((gt / pred), (pred / gt))
    a1 = (thresh < 1.25).to(torch.float).mean()
    a2 = (thresh < 1.25 ** 2).to(torch.float).mean()
    a3 = (thresh < 1.25 ** 3).to(torch.float).mean()
    
    return {
        "abs_rel": abs_rel.view(1),
        "sq_rel": sq_rel.view(1),
        "rmse": rmse.view(1),
        "rmse_log": rmse_log.view(1),
        "a1": a1.view(1),
        "a2": a2.view(1),
        "a3": a3.view(1),
        "ssim": ssim.view(1),
    }


# def rgb_metrics(pred: torch.Tensor, gt: torch.Tensor, mask: Optional[torch.Tensor] = None) -> Dict[str, torch.Tensor]:
#     """Returns RGB error metrics. Inputs should be [B, 1, H, W]."""
#     if mask is None:
#         mask = torch.ones_like(pred).bool()
#         ssim = torch.mean(ssim_fn(gt, pred, pad_reflection=False, gaussian_average=True, comp_mode=True))
#     else:
#         ssim = torch.tensor(torch.nan, device=pred.device)
    
#     mse = torch.mean((gt[mask] - pred[mask]) ** 2)
#     rmse = torch.sqrt(mse) 
#     peak = 1.0      # Maximum signal in RGB channels should be '1'.
#     psnr = 10 * torch.log10(peak**2/mse)
    
#     # TODO
#     fid = FID()
#     is = InceptionScore()
    
#     return {
#         "mse": mse.view(1),
#         "rmse": rmse.view(1),
#         "psnr": psnr.view(1),
#         "ssim": ssim.view(1),
#         "fid": None,
#         "is": None,
#     }
    

def compute_rgb_metrics(pred: torch.Tensor, gt: torch.Tensor, valid_mask: Optional[torch.Tensor] = None, lpips_net: Optional[lpips.LPIPS] = None, crop_5_percent_on_all_sides: bool = False) -> Dict[str, torch.Tensor]:
    """Returns RGB error metrics. Inputs should be [B, 3, H, W]."""
    
    if valid_mask is None:
        valid_mask = torch.ones_like(pred[:, :1, :, :]).bool()
    
    gt = gt.masked_fill(~valid_mask, 0.)
    pred = pred.masked_fill(~valid_mask, 0.)

    if crop_5_percent_on_all_sides:
        # Following tucker et al. and others, we crop 5% on all sides
        _, _, h, w = pred.shape
        y0 = int(math.ceil(0.05 * h))
        y1 = int(math.floor(0.95 * h))
        x0 = int(math.ceil(0.05 * w))
        x1 = int(math.floor(0.95 * w))
        gt = gt[:, :, y0:y1, x0:x1]
        pred = pred[:, :, y0:y1, x0:x1]
    
    # Compute SSIM, MSE, RMSE, PSNR
    ssim = []
    for pred_, gt_ in zip(pred.cpu().numpy(), gt.cpu().numpy()):
        ssim.append(
            skimage.metrics.structural_similarity(pred_, gt_, multichannel=True, channel_axis=0, data_range=1, gaussian_weights=True, sigma=1.5, use_sample_covariance=False)
        )
    ssim = torch.tensor(ssim, dtype=gt.dtype, device=gt.device).mean()
    # ssim = torch.mean(ssim_fn(gt, pred, pad_reflection=False, gaussian_average=True, comp_mode=True))
    
    mse = torch.mean((gt - pred) ** 2)
    rmse = torch.sqrt(mse) 
    peak = 1.0
    psnr = 10 * torch.log10(peak**2/mse)
    
    if lpips_net is not None:
        # https://pypi.org/project/lpips/
        # Expecting an RGB image in [-1, 1]
        lpips_score = lpips_net(pred, gt, normalize=True).mean()
    else:
        lpips_score = torch.tensor(torch.nan, device=mse.device)

    return {
        "rmse": rmse.view(1),
        "psnr": psnr.view(1),
        "ssim": ssim.view(1),
        "lpips": lpips_score.view(1),
    }
    

def perceptual_metrics(pred: torch.Tensor, gt: Optional[torch.Tensor] = None) -> Dict[str, torch.Tensor]:
    """https://pytorch-ignite.ai/blog/gan-evaluation-with-fid-and-is/#evaluation-metrics"""

    # When using less than at least 2048 samples in one of the datasets I get
    # 'ValueError: Imaginary component 5.029033962366753e+75'
    # See: https://github.com/mseitzer/pytorch-fid/issues/13
    # TODO is this a fix? https://github.com/ahmadki/mlperf_sd_inference/issues/4#issuecomment-1806149386
    if len(pred) >= 2048 and gt is not None:
        fid = FID(device=pred.device)
        fid.update((
            TF.resize(pred, (299, 299), antialias=False), 
            TF.resize(gt, (299, 299), antialias=False), 
        ))

        # Compute FID and Inception Score
        fid_result = fid.compute()
    else:
        fid_result = torch.tensor(torch.nan)
    
    inception_score = InceptionScore(output_transform=lambda x: x, device=pred.device)
    inception_score.update(
        TF.resize(pred, (299, 299), antialias=False)
    )
    is_result = inception_score.compute()

    return {
        "fid": fid_result.view(1),
        "is": torch.tensor(is_result),
    }


def edge_aware_smoothness(gt_img, depth, mask=None):
    n, pc, h, w = depth.shape
    gt_img = gt_img.permute(0, 1, 4, 5, 2, 3).reshape(-1, 3, h, w)
    depth = 1 / depth.reshape(-1, 1, h, w).clamp(1e-3, 80)
    depth = depth / torch.mean(depth, dim=[2, 3], keepdim=True)

    gt_img = F.interpolate(gt_img, (h, w))

    d_dx = torch.abs(depth[:, :, :, :-1] - depth[:, :, :, 1:])
    d_dy = torch.abs(depth[:, :, :-1, :] - depth[:, :, 1:, :])

    i_dx = torch.mean(torch.abs(gt_img[:, :, :, :-1] - gt_img[:, :, :, 1:]), 1, keepdim=True)
    i_dy = torch.mean(torch.abs(gt_img[:, :, :-1, :] - gt_img[:, :, 1:, :]), 1, keepdim=True)

    d_dx *= torch.exp(-i_dx)
    d_dy *= torch.exp(-i_dy)

    errors = F.pad(d_dx, pad=(0, 1), mode='constant', value=0) + F.pad(d_dy, pad=(0, 0, 0, 1), mode='constant', value=0)
    errors = errors.view(n, pc, h, w)
    return errors


def compute_chamfer(gen_points_sampled: np.ndarray, gt_points_sampled: np.ndarray) -> float:
    """This function computes a symmetric chamfer distance, i.e. the sum of both chamfers.
    
    From:
    https://github.com/philippwulff/Deep3DComp/blob/main/deep_sdf/metrics/chamfer.py

    gen_points_sampled: np.array of points sampled from the generated mesh surface.
    gt_points_sampled: np.array of points sampled from the GT mesh surface.
    """
    # one direction
    gen_points_kd_tree = KDTree(gen_points_sampled)
    one_distances, one_vertex_ids = gen_points_kd_tree.query(gt_points_sampled)
    gt_to_gen_chamfer = np.mean(np.square(one_distances))

    # other direction
    gt_points_kd_tree = KDTree(gt_points_sampled)
    two_distances, two_vertex_ids = gt_points_kd_tree.query(gen_points_sampled)
    gen_to_gt_chamfer = np.mean(np.square(two_distances))

    return float(gt_to_gen_chamfer + gen_to_gt_chamfer), np.concatenate((one_distances, two_distances), axis=0)