from typing import Tuple, List, Optional
import cv2
import numpy as np
import torch
import torch.nn.functional as F
from utils.constants import DEFINITIONS
from configs.structured_configs.occlusions_config import OcclusionDetectionConfig

import torch
from torchvision.models.optical_flow import Raft_Large_Weights, raft_large, RAFT
from torchvision.utils import flow_to_image
import torchvision.transforms.functional as F_tv
from torchvision.transforms._presets import OpticalFlow
from utils.constants import DEFINITIONS
from utils.projection_ops import project_world_points, unproject_to_world_points
    
    
def comp_occlusions_from_flow(flow0, flow1):
    """Takes forward and reverse optical flow and returns occlusion masks."""
    n, _, h, w = flow0.shape
    device = flow0.device
    
    # Span pixel coordinates.
    x = torch.linspace(0, w-1, w, device=device).view(1, 1, w).expand(1, h, w)
    y = torch.linspace(0, h-1, h, device=device).view(1, h, 1).expand(1, h, w)
    xy = torch.cat((x, y), dim=0).view(1, 2, h, w).expand(n, 2, h, w)

    # Compute new coodinates.
    xy_0 = (xy + flow0).round().long()
    xy_1 = (xy + flow1).round().long()
    
    xy_0[:, 0, :, :] = xy_0[:, 0, :, :].clamp(0, w-1)
    xy_0[:, 1, :, :] = xy_0[:, 1, :, :].clamp(0, h-1)
    xy_1[:, 0, :, :] = xy_1[:, 0, :, :].clamp(0, w-1)
    xy_1[:, 1, :, :] = xy_1[:, 1, :, :].clamp(0, h-1)

    batch_indices = torch.repeat_interleave(torch.arange(n, device=device), h*w)
    xy_0 = xy_0.permute(0, 2, 3, 1).reshape(-1, 2)
    xy_1 = xy_1.permute(0, 2, 3, 1).reshape(-1, 2)

    # Create the mask. 0 is visible and 1 is occupied.
    mask0 = torch.zeros((n, 1, h, w), device=device)
    mask1 = torch.zeros((n, 1, h, w), device=device)

    mask0[batch_indices, :, xy_0[:, 1], xy_0[:, 0]] = 1
    mask1[batch_indices, :, xy_1[:, 1], xy_1[:, 0]] = 1

    return mask0, mask1


def comp_occlusions_from_raft(img1_batch: torch.Tensor, img2_batch: torch.Tensor, raft_model: RAFT, raft_transforms: OpticalFlow) -> torch.Tensor:
    """
    Returns the occlusion map in img2. Pixels that are not seen in img1 are 0, all others are 1.
    :param img1_batch: [B, C, H, W]. Height and width should be divisible by 8.
    :param img2_batch: [B, C, H, W]. Height and width should be divisible by 8.
    :param raft_model: The RAFT model
    :param raft_transforms: The RAFT transform functions
    """
    assert len(img1_batch.shape) == len(img1_batch.shape) == 4, "INCORRECT INPUT SHAPE"
    
    img1_batch, img2_batch = raft_transforms(img1_batch, img2_batch)
    raft_model.eval()
    
    with torch.no_grad():
        flow_img1_to_img2 = raft_model(img1_batch, img2_batch)[-1]
        flow_img2_to_img1 = raft_model(img2_batch, img1_batch)[-1]    
        mask1_to_2, _ = comp_occlusions_from_flow(flow_img1_to_img2, flow_img2_to_img1)
    
    is_visible = mask1_to_2 == 1
    mask1_to_2[is_visible] = DEFINITIONS.IS_VISIBLE
    mask1_to_2[~is_visible] = DEFINITIONS.IS_OCCLUDED
    
    return mask1_to_2    # [B, 1, H, W]


def comp_occlusions_from_depth(
    pose_1, 
    pose_2, 
    intrinsics_1, 
    intrinsics_2, 
    depth_1, 
    depth_2, 
    thresh: float = 0.0, 
    out_of_bound_thresh_h: float = 0.0,
    out_of_bound_thresh_w: float = 0.0,
    max_occlusion_detection_depth: float = 0.0,
    ) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
    """
    Checks if 3D points seen by camera 2 are occluded in camera 1 by reprojecting depth from camera 2 into camera 1 
    and comparing to interpolated depth values from depth map 1.
    :param pose_1: [B, 4, 4]
    :param pose_2: [B, 4, 4]
    :param intrinsics_1: [B, 3, 3]
    :param intrinsics_2: [B, 3, 3]
    :param depth_1: [B, 1, H, W]
    :param depth_2: [B, 1, H, W]
    :param thresh: float 
    """
    
    world_points_3d, _ = unproject_to_world_points(depth_2[:, 0, :, :], intrinsics_2, pose_2)               # [B, N, 3]
    image_points_2_reproj, depth_2_reproj = project_world_points(world_points_3d, intrinsics_1, pose_1)     # [B, N, 2] and [B, N]
    image_points_2_reproj = image_points_2_reproj.to(pose_1.dtype)
    depth_2_reproj = depth_2_reproj.to(pose_1.dtype)
    
    # To visualize
    # plt.scatter(image_points_2_reproj[:, 0], image_points_2_reproj[:, 1], c=depth_2_reproj)
    # plt.gca().invert_yaxis()
    
    out_of_bound = torch.logical_or(
        abs(image_points_2_reproj[:, :, 0]) > 1 - out_of_bound_thresh_h, abs(image_points_2_reproj[:, :, 1]) > 1 - out_of_bound_thresh_w
    )[:, None, :]

    interp_locations = image_points_2_reproj.unsqueeze(1)     # [B, 1, N, 2] mostly within range [-1, 1]

    interp_depths = F.grid_sample(
        depth_1,                # [B, 1, H, W] 
        interp_locations, 
        mode='bilinear', 
        padding_mode='zeros',   # use 0 for out-of-bound grid locations (outside of [-1, 1])
        align_corners=False,
    ).squeeze(1)                # [B, 1, N]

    is_occluded_and_in_range = torch.logical_and(
        depth_2_reproj[:, None, :] > (interp_depths + thresh), interp_depths < max_occlusion_detection_depth
    )
    
    is_behind = torch.ones_like(interp_depths) * DEFINITIONS.IS_VISIBLE
    is_behind[is_occluded_and_in_range] = DEFINITIONS.IS_OCCLUDED
    is_behind[interp_depths == 0] = DEFINITIONS.IS_INVALID
    is_behind[out_of_bound] = DEFINITIONS.IS_INVALID
    is_behind_map = is_behind.reshape(depth_2.shape)        # [B, 1, H, W]
    
    return is_behind_map, world_points_3d, is_behind


def sobel_filter(depth_1, thresh=0.1):
    device = depth_1.device
    dtype = depth_1.dtype
    sobel_x = torch.tensor([[-1, 0, 1], [-2, 0, 2], [-1, 0, 1]], device=device, dtype=dtype).view(1, 1, 3, 3)
    sobel_y = torch.tensor([[-1, -2, -1], [0, 0, 0], [1, 2, 1]], device=device, dtype=dtype).view(1, 1, 3, 3)
    
    inv_depth_1 = (1 / depth_1 - 1 / depth_1.min()) / (1 / depth_1.max() - 1 / depth_1.min())
    # Compute the gradient (https://en.wikipedia.org/wiki/Sobel_operator)
    edge_x = F.conv2d(inv_depth_1, sobel_x, padding=1)
    edge_y = F.conv2d(inv_depth_1, sobel_y, padding=1)
    magnitude = torch.sqrt(edge_x ** 2 + edge_y ** 2)
    # Apply the threshold to the relative depth gradient. 
    magnitude = (magnitude > thresh).to(depth_1.dtype)    # [B, 1, H, W] 
    return magnitude


def compute_occlusions_from_depth_grads(
    pose_1, 
    pose_2, 
    intrinsics_1, 
    intrinsics_2, 
    depth_1, 
    depth_2, 
    thresh: float = 0.1, 
    out_of_bound_thresh_h: float = 0.05,
    out_of_bound_thresh_w: float = 0.05
    ) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
    """
    Checks if 3D points seen by camera 2 are occluded in camera 1 by reprojecting depth from camera 2 into camera 1 
    and comparing to interpolated depth values from depth map 1.
    :param pose_1: [B, 4, 4]
    :param pose_2: [B, 4, 4]
    :param intrinsics_1: [B, 3, 3]
    :param intrinsics_2: [B, 3, 3]
    :param depth_1: [B, 1, H, W]
    :param depth_2: [B, 1, H, W]
    :param thresh: float 
    """
    dtype = pose_1.dtype
    
    world_points_3d, _ = unproject_to_world_points(depth_2[:, 0, :, :], intrinsics_2, pose_2)               # [B, N, 3]
    image_points_2_reproj, _ = project_world_points(world_points_3d, intrinsics_1, pose_1)     # [B, N, 2] and [B, N]
    image_points_2_reproj = image_points_2_reproj.to(dtype)
    
    out_of_bound = torch.logical_or(
        abs(image_points_2_reproj[:, :, 0]) > 1 - out_of_bound_thresh_h, abs(image_points_2_reproj[:, :, 1]) > 1 - out_of_bound_thresh_w
    )[:, None, :]

    interp_locations = image_points_2_reproj.unsqueeze(1)     # [B, 1, N, 2] mostly within range [-1, 1]
    
    magnitude = sobel_filter(depth_1, thresh=thresh) + 1.0

    interp_edges = F.grid_sample(
        magnitude,                # [B, 1, H, W] 
        interp_locations, 
        mode='bilinear', 
        padding_mode='zeros',   # use 0 for out-of-bound grid locations (outside of [-1, 1])
        align_corners=False,
    ).squeeze(1)                # [B, 1, N]

    is_behind = torch.ones_like(interp_edges) * DEFINITIONS.IS_VISIBLE
    is_behind[interp_edges == 2] = DEFINITIONS.IS_OCCLUDED
    is_behind[interp_edges == 0] = DEFINITIONS.IS_INVALID
    is_behind[out_of_bound] = DEFINITIONS.IS_INVALID
    is_behind_map = is_behind.reshape(depth_2.shape)        # [B, 1, H, W]
    
    return is_behind_map, world_points_3d, is_behind


def morphological_op(img: torch.Tensor, method: str, kernel_size: int = 3):
    """Performs a kernel-based morphological operation using OpenCV.
    
    Args:
        img (torch.Tensor): [B, 1, H, W]
    """
    # Dilation will expand the True regions. This step ensures that neighboring True pixels join together to form larger patches.
    dilation_fn = lambda x: F.max_pool2d(x.float(), kernel_size, stride=1, padding=kernel_size // 2)
    # Eliminate the small patches of True pixels and leave behind only the larger patches
    erosion_fn = lambda x: 1 - F.max_pool2d(1 - x.float(), kernel_size, stride=1, padding=kernel_size // 2)

    if method == "opening":
        # Opening: Erosion followed by dilation --> removes noise outside of patches.
        # https://opencv24-python-tutorials.readthedocs.io/en/latest/py_tutorials/py_imgproc/py_morphological_ops/py_morphological_ops.html
        return dilation_fn(erosion_fn(img))
    elif method == "closing":
        # Closing: Dilation followed by erosion --> removes noise inside of patches/closes gap within them.
        return erosion_fn(dilation_fn(img))
    elif method == "erosion":
        return erosion_fn(img)
    elif method == "dilation":
        return dilation_fn(img)
    else:
        raise NotImplementedError


def post_process_occlusion_mask(is_behind: torch.Tensor, morph_ops: List[Tuple[str, int]] = [], area_thresh: Optional[int] = None) -> np.ndarray:
    """
    Post-processes the is_behind map to keep only larger patches of True pixels.
    
    Args:
        is_behind: Boolean tensor representing the is_behind map. [B, 1, H, W]
        kernel_size: Size of the morphological operation kernel.
        area_thres: Minimum area to keep a patch.
        
    Returns:
        - Processed boolean map. [B, 1, H, W]
    """
        
    # is_behind_out = is_behind.copy().squeeze()
    is_behind_out = is_behind.clone()
    is_invalid_mask = is_behind_out == DEFINITIONS.IS_INVALID
        
    # is_behind_uint8 = np.zeros(shape=(is_behind_out.shape), dtype=np.uint8)
    is_behind_filtered = torch.zeros_like(is_behind_out)
    is_behind_filtered[is_behind_out == DEFINITIONS.IS_OCCLUDED] = 1

    if len(morph_ops) > 0 and len(morph_ops[0]):
        for method, kernel_size in morph_ops:
            is_behind_filtered = morphological_op(is_behind_filtered, method=method, kernel_size=kernel_size)
    
    # Area filtering.
    # https://pyimagesearch.com/2021/02/22/opencv-connected-component-labeling-and-analysis/
    
    is_behind_filtered = is_behind_filtered.cpu().numpy().astype(np.uint8)      # [B, 1, H, W]
    
    if area_thresh is not None:
        for i in range(is_behind_filtered.shape[0]):
            num_labels, labels, stats, _ = cv2.connectedComponentsWithStats(is_behind_filtered[i, 0, :, :], connectivity=8)
            for l in range(1, num_labels):
                if stats[l, cv2.CC_STAT_AREA] < area_thresh:
                    is_behind_filtered[i, 0, :, :][labels == l] = 0
                
    is_behind_filtered = torch.tensor(is_behind_filtered, device=is_behind.device)

    # Convert values back to definitions
    is_behind_out[is_behind_filtered == 0] = DEFINITIONS.IS_VISIBLE
    is_behind_out[is_behind_filtered == 1] = DEFINITIONS.IS_OCCLUDED
    is_behind_out[is_invalid_mask] = DEFINITIONS.IS_INVALID

    return is_behind_out
    
    
def keep_overlapping_patches(map1: np.ndarray, map2: np.ndarray, threshold: float = 50, method: str = "keep_overlap") -> np.ndarray:
    """
    Fuses two binary masks. Keeps the union of patches that overlap by more than the threshold. 
    Treats 0 as background and 1 as foreground patches.
    :param map1: Binary map [H, W]
    :param map2: Binary map [H, W]
    :param threshold: Minimum overlap value for patches to keep
    """
    
    in_dtype = map1.dtype
    
    map1 = map1.astype(np.uint8)
    map2 = map2.astype(np.uint8)
    # Find connected components in each map
    num_labels1, labels1, stats1, _ = cv2.connectedComponentsWithStats(map1)
    num_labels2, labels2, stats2, _ = cv2.connectedComponentsWithStats(map2)

    final_map = np.zeros_like(map1, dtype=np.uint8)

    # Iterate through each component in map1, ignoring the background
    for i in range(1, num_labels1):  
        component1 = (labels1 == i)
        # Iterate through each component in map2, ignoring the background
        for j in range(1, num_labels2):  
            component2 = (labels2 == j)
            overlap = np.logical_and(component1, component2)
            # If overlap meets the threshold, add to the final map
            if np.sum(overlap) >= threshold:
                match method:
                    case "keep_overlap":
                        final_map = np.logical_or(final_map, np.logical_or(component1, component2))
                    case "keep_1":
                        final_map = np.logical_or(final_map, component1)
                    case "keep_2":
                        final_map = np.logical_or(final_map, component2)
                    case _:
                        raise ValueError(f"Method {method} not available.")

    return final_map.astype(in_dtype)  


def fuse_occlusion_masks(
    primary_mask: torch.Tensor, 
    secondary_mask: torch.Tensor, 
    method: str = "overlapping_union", 
    min_overlap_area: float = 50
) -> torch.Tensor:
    """Combines two occlusions masks"""
    masks_ = []
    is_invalid_ = []
    
    device = primary_mask.device
    dtype = primary_mask.dtype
    
    for mask in [primary_mask, secondary_mask]:
        mask_ = mask.clone()
        is_invalid = mask_ == DEFINITIONS.IS_INVALID
        is_visible = mask_ == DEFINITIONS.IS_VISIBLE
        is_occluded = mask_ == DEFINITIONS.IS_OCCLUDED
        mask_[is_occluded] = 1
        mask_[is_visible] = 0
        mask_[is_invalid] = 0
        masks_.append(mask_)
        is_invalid_.append(is_invalid)
    
    # TODO why this?
    masks_[1][primary_mask == DEFINITIONS.IS_INVALID] = 0
    
    if method in ["overlapping_union", "overlappting_keep_secondary"]:
        fused_mask = []
        for pm, sm in zip(masks_[0], masks_[1]):
            fm = keep_overlapping_patches(
                pm.squeeze().cpu().numpy(), sm.squeeze().cpu().numpy(), 
                threshold=min_overlap_area, 
                method="keep_2" if "overlappting_keep_secondary" else "keep_overlap"
                )
            fused_mask.append(torch.tensor(fm[None, None, :, :], device=device, dtype=dtype))
        fused_mask = torch.concat(fused_mask)
    elif method == "union":
        fused_mask = torch.logical_or(*masks_)
    else:
        raise NotImplementedError(f"Fusion method {method} does not exist.")
    
    is_occluded = fused_mask == 1
    is_visible = fused_mask == 0
    fused_mask[is_occluded] = DEFINITIONS.IS_OCCLUDED
    fused_mask[is_visible] = DEFINITIONS.IS_VISIBLE
    fused_mask[torch.logical_or(*is_invalid_)] = DEFINITIONS.IS_INVALID
        
    return fused_mask


def crop_to_patch_center(
    mask: torch.Tensor, 
    others_to_crop: torch.Tensor, 
    crop_height: int = 512, 
    crop_width: int = 512, 
    min_patch_area: int = 0,
    ) -> List[List[torch.Tensor]]:
    """Finds patches in the given mask and creates a list of crops centered on the patches"""
    
    mask_tensor = mask.clone()
    _, _, image_height, image_width = mask.shape
    
    # Add 0-padding if input height or width to not match the cropping height or width.
    padding_ltrb = [0, 0, 0, 0]
    if crop_width > image_width or crop_height > image_height:
        padding_ltrb = [
            (crop_width - image_width) // 2 if crop_width > image_width else 0,
            (crop_height - image_height) // 2 if crop_height > image_height else 0,
            (crop_width - image_width + 1) // 2 if crop_width > image_width else 0,
            (crop_height - image_height + 1) // 2 if crop_height > image_height else 0,
        ]
    
    zero_pad = lambda x: F_tv.pad(x, padding_ltrb, fill=0)
    
    mask_array = zero_pad(mask_tensor).squeeze().numpy().astype(np.uint8)
    padded_height, padded_width = mask_array.shape
    
    _, labels, stats, _ = cv2.connectedComponentsWithStats(mask_array)
    # stats is ordered by area in descending order
    crops = []
    for label, stat in enumerate(stats):
        if label == 0:
            # Ignore the background
            continue
        area = stat[cv2.CC_STAT_AREA]
        if area < min_patch_area:
            continue
        ctr_x = int(stat[cv2.CC_STAT_LEFT] + stat[cv2.CC_STAT_WIDTH] / 2)
        ctr_y = int(stat[cv2.CC_STAT_TOP] + stat[cv2.CC_STAT_HEIGHT] / 2)
        upper = max(0, min(padded_height - crop_height, ctr_y - crop_height // 2))
        left = max(0, min(padded_width - crop_width, ctr_x - crop_width // 2))
        
        mask_this_patch_padded = torch.tensor((labels == label).astype(np.uint8))[None, None, :, :]
        mask_this_patch_cropped = mask_this_patch_padded[:, :, upper:upper+crop_height, left:left+crop_width]
        
        mask_this_patch_og_shape = mask_this_patch_padded.clone()
        mask_this_patch_og_shape[:, :, 0:upper, :] = 0
        mask_this_patch_og_shape[:, :, upper+crop_height:, :] = 0
        mask_this_patch_og_shape[:, :, :, 0:left] = 0
        mask_this_patch_og_shape[:, :, ::, left+crop_width:] = 0
        mask_this_patch_og_shape = mask_this_patch_og_shape[:, :, padding_ltrb[1]:padded_height-padding_ltrb[3], padding_ltrb[0]:padded_width-padding_ltrb[2]]
        
        new_crops = [mask_this_patch_cropped, mask_this_patch_og_shape]
        for other in others_to_crop:
            # other = centercrop_transform(other.clone())
            other = zero_pad(other.clone())
            new_crops.append(other[:, :, upper:upper+crop_height, left:left+crop_width])
        crops.append(new_crops)
    
    return crops


def resize_to(img, method="smaller", size=512, mode="bilinear"):
    """
    Resize the image such that the smaller/larger side has the given `size`.
    :param img: [B, C, H, W]
    """
    
    assert isinstance(img, torch.Tensor), "INPUT MUST BE TENSOR"
        
    _, _, w, h = img.shape
    
    if method == "smaller":
        scale_factor = size / min(w, h)
    elif method == "larger":
        scale_factor = size / max(w, h)
    else:
        raise NotImplementedError()
        
    return F.interpolate(img, scale_factor=scale_factor, mode=mode)


def comp_occlusion_map(
    cfg: OcclusionDetectionConfig,
    pose1: Optional[torch.Tensor] = None,
    pose2: Optional[torch.Tensor] = None,
    proj1: Optional[torch.Tensor] = None,
    proj2: Optional[torch.Tensor] = None,
    img1: Optional[torch.Tensor] = None,
    img2: Optional[torch.Tensor] = None,
    depth1: Optional[torch.Tensor] = None,
    depth2: Optional[torch.Tensor] = None,
    exclusion_map: Optional[torch.BoolTensor] = None,
    occlusions_depth_grads: Optional[torch.BoolTensor] = None,
):
    """Computes occlusion masks based on depth and optical flow.
    Occlusions are regions that are occluded in view 1 and visible in view 2. 
    The output maps are in view 2.
    
    Args:
        pose1: [B, 4, 4]
        pose2: [B, 4, 4]
        proj1: [B, 3, 3]
        proj2: [B, 3, 3]
        img1: [B, 3, H, W]
        img2: [B, 3, H, W]
        depth1: [B, 1, H, W]
        depth2: [B, 1, H, W]
        exclusion_map: If given, `True` pixels will not be marked as occluded.
        occlusions_depth_grads: 
            If given, this will be used in place of the depth gradient occlusion map.
            Useful when pre-computing and rendering occlusions via the pseudo volume.
        cfg: OcclusionConfig
    """
    assert any([cfg.USE_FLOW, cfg.USE_DEPTH, cfg.USE_DEPTH_GRADS])
    
    # --- OCCLUSIONS FROM DEPTH REPROJECTION ---
    is_behind, is_behind_post = None, None
    if cfg.USE_DEPTH:
        is_behind, _, _ = comp_occlusions_from_depth(
            pose1,
            pose2,
            proj1,
            proj2,
            depth1,
            depth2,
            thresh=cfg.DEPTH_THRESH,
            out_of_bound_thresh_h=cfg.OUT_OF_BOUND_THRESH_H,
            out_of_bound_thresh_w=cfg.OUT_OF_BOUND_THRESH_W,
            max_occlusion_detection_depth=cfg.MAX_OCCLUSION_DETECTION_DEPTH,
        )
        # FIXME
        # is_behind = torch.ones_like(depth1) * DEFINITIONS.IS_VISIBLE
        # is_behind[depth2 > (depth1 + cfg.DEPTH_THRESH)] = DEFINITIONS.IS_OCCLUDED
        # is_behind[depth2 == 0] = DEFINITIONS.IS_INVALID
        is_behind_post = post_process_occlusion_mask(
            is_behind, 
            morph_ops=cfg.POSTPROCESSING_OPS_DEPTH,
            area_thresh=cfg.AREA_THRESH
        )
    
    # --- OCCLUSIONS FORM OPTICAL FLOW ---
    is_behind_flow, is_behind_flow_post = None, None
    if cfg.USE_FLOW:
        # TODO setting this up takes 0.13s. Do it only once.
        weights = Raft_Large_Weights.DEFAULT
        raft_model = raft_large(weights, progress=False).to(img1.device)
        raft_model.eval()
        
        is_behind_flow = comp_occlusions_from_raft(
            img1, 
            img2.to(img1), 
            raft_model, 
            weights.transforms(),
        )
        is_behind_flow_post = post_process_occlusion_mask(
            is_behind_flow, 
            morph_ops=cfg.POSTPROCESSING_OPS_FLOW,
            area_thresh=cfg.AREA_THRESH
        )
        
    # --- OCCLUSIONS FROM DEPTH GRADIENTS ---
    is_behind_depth_grads, is_behind_depth_grads_post = None, None
    if cfg.USE_DEPTH_GRADS:
        if occlusions_depth_grads is None:
            is_behind_depth_grads, _, _ = compute_occlusions_from_depth_grads(
                pose1,
                pose2,
                proj1,
                proj2,
                depth1,
                depth2,
                thresh=cfg.DEPTH_GRADS_THRESH,
            )
        else: 
            is_behind_depth_grads = torch.ones_like(occlusions_depth_grads) * DEFINITIONS.IS_VISIBLE
            is_behind_depth_grads[occlusions_depth_grads] = DEFINITIONS.IS_OCCLUDED
            is_behind_depth_grads = is_behind_depth_grads.float()
            
        is_behind_depth_grads_post = post_process_occlusion_mask(
            is_behind_depth_grads,
            morph_ops=cfg.POSTPROCESSING_OPS_DEPTH_GRADS,
            area_thresh=cfg.AREA_THRESH,
        )
    
    # If the final exclusion mask is a combination of sources
    if cfg.USE_FLOW and cfg.USE_DEPTH:
        occlusion_mask = fuse_occlusion_masks(is_behind_post, is_behind_flow_post)
    elif cfg.USE_FLOW and cfg.USE_DEPTH_GRADS:
        occlusion_mask = fuse_occlusion_masks(is_behind_depth_grads_post, is_behind_flow_post, method="overlappting_keep_secondary")
    # If the final exclusion mask comes from a single source
    elif cfg.USE_FLOW:
        occlusion_mask = is_behind_flow_post
    elif cfg.USE_DEPTH:
        occlusion_mask = is_behind_post
    elif cfg.USE_DEPTH_GRADS:
        occlusion_mask = is_behind_depth_grads_post
    
    occlusion_mask_non_excluded = occlusion_mask.clone()
    if exclusion_map is not None:
        occlusion_mask[exclusion_map] = DEFINITIONS.IS_VISIBLE
        occlusion_mask = post_process_occlusion_mask(occlusion_mask, morph_ops=[(DEFINITIONS.OPENING, 5)], area_thresh=500)
    
    return {
        "occlusions_full": occlusion_mask,
        "occlusions_full_non_exluded_keep": occlusion_mask_non_excluded, 
        "occlusions_depth": is_behind,
        "occlusions_depth_post": is_behind_post,
        "occlusions_flow": is_behind_flow,
        "occlusions_flow_post": is_behind_flow_post,
        "occlusions_depth_grads": is_behind_depth_grads,
        "occlusions_depth_grads_post": is_behind_depth_grads_post,
    }
    

def mask_occlusions_only(mask):
    is_occluded = mask == DEFINITIONS.IS_OCCLUDED
    new_mask = torch.ones_like(mask)
    new_mask[is_occluded] = 0.0
    return new_mask


def mask_occlusions_and_outside_FOV(mask):
    is_not_visible = ~(mask == DEFINITIONS.IS_VISIBLE)
    new_mask = torch.ones_like(mask) * DEFINITIONS.IS_VISIBLE
    new_mask[is_not_visible] = DEFINITIONS.IS_OCCLUDED
    return new_mask
    