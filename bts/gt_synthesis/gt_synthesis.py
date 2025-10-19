import math
import torch
import time
import random
from copy import copy, deepcopy
from dataclasses import dataclass, asdict, fields, field
from typing import List, Optional, Tuple, Dict, Any, Literal, Callable

import torch.nn as nn
from torchvision.transforms.functional import resize, normalize
from torchvision.transforms import InterpolationMode, transforms

from bts.renderer import NeRFRenderer
from bts.models.pseudo_volume import PseudoVolume
from bts.common.ray_sampler import ImageRaySampler
from bts.gt_synthesis.gt_pose_sampler import CameraSampler, comp_reprojected_img_extents
from bts.gt_synthesis.make_camera_sampler import make_camera_sampler
from bts.gt_synthesis.common import (
    compute_occlusion_center_error,
    compute_occlusion_edges_error,
)
from configs.structured_configs.main_config import MainConfig
from vcm.utils.model_utils import make_model_components, make_tokenizer
from utils.utils import (
    asdict_lowercase_keys_override,
    invert_depth,
    get_interval_sample,
)
from utils.occlusion_ops import comp_occlusion_map, sobel_filter, morphological_op
from utils.constants import DEFINITIONS
from utils.plotting import color_occlusion_masked_img
from utils.projection_ops import (
    crop_intrinsics,
    unnormalize_intrinsics,
    scale_intrinsics,
    align_depth,
    align_inv_depth,
    distance_to_z,
)
from utils.transforms import Crop
from utils.depth_predictors import (
    make_depth_predictor,
    Metric3DWrapper,
    UniDepthWrapper,
)


@dataclass
class Outputs:
    """
    Outputs from GTSynthesisWrapper's 'forward' method containing synthetic GTs.
    """

    IMGS_IN: torch.Tensor
    DEPTHS_IN: torch.Tensor
    PROJS_IN: torch.Tensor
    POSES_IN: torch.Tensor
    # Outputs
    POSES_OUT: torch.Tensor
    PROJS_OUT: torch.Tensor
    # Synthetic imgs
    IMGS_SYNTH: Optional[torch.Tensor] = None
    DEPTHS_SYNTH: Optional[torch.Tensor] = None
    SCALES: Optional[torch.Tensor] = None
    SHIFTS: Optional[torch.Tensor] = None
    INVALID_SYNTH: Optional[torch.Tensor] = None
    AVG_DENOISING_TIME: Optional[float] = None
    # For controlnet training
    IMGS_COND: Optional[torch.Tensor] = None
    IMGS_GT: Optional[torch.Tensor] = None
    DEPTHS_GT: Optional[torch.Tensor] = None
    CAPTIONS_IDS: Optional[torch.Tensor] = None
    IMGS_PREPROCESSED: Optional[torch.Tensor] = None


@dataclass
class Data:
    """
    Intermediary values used in GTSynthesisWrapper.
    """

    # Values extracted from the inputs in GTSynthesis.forward
    GENERATOR: torch.Generator
    IMGS: torch.Tensor
    DEPTHS: torch.Tensor
    POSES: torch.Tensor
    PROJS: torch.Tensor
    MASKS: torch.Tensor | None
    IMGS_IN_FOR_OCCLUSION_DET: torch.Tensor
    DEPTHS_IN_FOR_OCCLUSION_DET: torch.Tensor
    PROJS_IN_FOR_OCCLUSION_DET: torch.Tensor
    IMGS_IN_FOR_OCCLUSION_DET_HW: tuple[int, int]
    HW_DATA: tuple[int, int]
    HW: tuple[int, int]
    HW_NV: tuple[int, int] | None
    HW_REREN: tuple[int, int]
    HW_CTRL: tuple[int, int]
    HW_RECT: tuple[int, int]
    NUM_NV: int
    NUM_TOP_K: int
    N: int
    B: int
    CAPTION_STRS: List[str]
    CAPTIONS_IDS: torch.Tensor | None  # TODO not sure if I need this?
    IMGS_PREPROCESSED: Optional[torch.Tensor] = None
    IMGS_ORIGINAL_SCALE: Optional[torch.Tensor] = None
    RESCALE_SCALE: float = 1.0
    CAM_SAMPLER_KWARGS: Dict[str, Any] = field(default_factory=lambda: {})
    NUM_INPUT_ANCHORS: int = 1

    # Values populated during rendering novel views from the input views
    IMGS_NV: Optional[torch.Tensor] = None
    DEPTHS_NV: Optional[torch.Tensor] = None
    MASKS_NV: Optional[torch.Tensor] = None
    INVALID_NV: Optional[torch.Tensor] = None
    POSES_NV: Optional[torch.Tensor] = None
    PROJS_NV: Optional[torch.Tensor] = None
    PROJS_NV_UNNORMALIZED: Optional[torch.Tensor] = None
    # Has format (L, R, T, B)
    LRTB_REREN: tuple[int] | Tuple[torch.Tensor] | None = None

    # Values populated during re-rending the input from a novel view
    IMGS_REREN: Optional[torch.Tensor] = None
    DEPTHS_REREN: Optional[torch.Tensor] = None
    INVALID_REREN: Optional[torch.Tensor] = None
    MASKS_REREN: Optional[torch.Tensor] = None
    PROJS_REREN: Optional[torch.Tensor] = None
    PROJS_REREN_UNNORMALIZED: Optional[torch.Tensor] = None
    IMGS_CROPPED_TO_REREN: Optional[torch.Tensor] = None
    DEPTHS_CROPPED_TO_REREN: Optional[torch.Tensor] = None

    # Final outputs
    IMGS_CONDITIONING: Optional[torch.Tensor] = None


EPS = 1e-6


class GTSynthesisWrapper(nn.Module):

    def __init__(
        self,
        cfg: MainConfig,  # TODO remove this
        renderer_cfg: Dict,
        camera_sampler: CameraSampler,
        depth_predictor: Optional[Callable] = None,
        align_depth_policy: Literal["none", "direct", "inverse"] = "direct",
        align_depth_mode: Literal["mean", "median", "lstsq"] = "median",
        align_depth_max_rel_error: float = 0.0,
        align_depth_scale_only: bool = False,
        align_depth_min_valid_fraction: float = 0.2,
        resample_invalid_outputs: bool = False,  # TODO implement
        only_occlusions_valid: bool = False,  # TODO implement
        min_mean_valid_pixels: float = 0.5,
        min_mean_occluded_pixels: float = 0.0,
        max_mean_occluded_pixels: float = 0.3,
        z_near: float = 3.0,
        z_far: float = 80.0,
        z_near_gt: Optional[float] = None,
        z_far_gt: Optional[float] = None,
        out_invalid_lrtb_edge_fraction: Tuple[float, float, float, float] = (
            0.0,
            0.0,
            0.0,
            0.0,
        ),
        tokenizer=None,  # Add type
        feature_extractor=None,
        refine_pipeline=None,
        black_invalid: bool = False,  # TODO add to cfg and enable during controlnet training
        n_retries_to_sample_valid: int = 0,
        input_crop_policy: str = "center",
        top_k_novel_views_to_keep: Optional[int] = None,
        set_occlusions_to_random_noise: bool = False,
        depths_conditioning_inverse: bool = False,
        close_conditioning_invalid: bool = False,
        close_conditioning_invalid_kernel_size: int = 5,
        num_synthetic_versions: int = 1,
        depth_grads_thresh=0.5,
        **kwargs,
    ) -> None:

        super().__init__()
        self.cfg = cfg

        # Rendering components.
        self.camera_sampler = camera_sampler
        pseudo_volume = PseudoVolume(
            **asdict_lowercase_keys_override(cfg.SYNTHETIC_GT.PSEUDO_VOLUME)
        ).eval()
        # Assign the renderer as a regular attribute to prevent it from
        # being included in the state_dict. Define persistent=False for the same reason.
        self.renderer: NeRFRenderer
        object.__setattr__(
            self, "renderer", NeRFRenderer.from_conf(renderer_cfg, persistent=False)
        )
        self.renderer = self.renderer.bind_parallel(pseudo_volume, gpus=None).eval()
        self.renderer.net.set_scale(0)

        # Controlnet and depth predictor modules
        self.depth_predictor: Metric3DWrapper | UniDepthWrapper | None
        object.__setattr__(self, "depth_predictor", depth_predictor)
        self.tokenizer = tokenizer
        self.feature_extractor = feature_extractor
        self.refine_pipeline = refine_pipeline

        self.z_near = z_near
        self.z_far = z_far
        self.input_crop_policy = input_crop_policy

        self.depth_grads_thresh = depth_grads_thresh

        # Conditioning image args.
        self.black_invalid = black_invalid
        self.set_occlusions_to_random_noise = set_occlusions_to_random_noise
        self.depths_conditioning_inverse = depths_conditioning_inverse
        self.close_conditioning_invalid = close_conditioning_invalid
        self.close_conditioning_invalid_kernel_size = (
            close_conditioning_invalid_kernel_size
        )

        # Predicted depth alignment args.
        self.align_depth = False
        self.align_inv_depth = False
        match align_depth_policy:
            case "none":
                pass
            case "direct":
                self.align_depth = True
            case "inverse":
                self.align_depth = True
                self.align_inv_depth = True
            case _:
                raise ValueError(
                    f"Depth alignment policy {align_depth_policy} is unavailable."
                )
        self.max_rel_depth_error_to_align = align_depth_max_rel_error
        self.align_depth_mode = align_depth_mode
        self.align_depth_scale_only = align_depth_scale_only
        self.align_depth_min_valid_fraction = align_depth_min_valid_fraction

        # Arguments for when to set the synthetic GTs to invalid
        self.z_near_gt = z_near_gt if z_near_gt else z_near
        self.z_far_gt = z_far_gt if z_far_gt else z_far
        self.resample_invalid_outputs = resample_invalid_outputs
        self.min_mean_valid_pixels = min_mean_valid_pixels
        self.min_mean_occluded_pixels = min_mean_occluded_pixels
        self.max_mean_occluded_pixels = max_mean_occluded_pixels
        # self.max_mean_occluded_pixels_for_scoring = max_mean_occluded_pixels * 0.75   # Penalize this harder
        assert 0 <= self.min_mean_valid_pixels <= 1, "INVALID RANGE"
        self.out_invalid_lrtb_edge_fraction = out_invalid_lrtb_edge_fraction
        assert all(
            [0.0 <= _ < 0.5 for _ in self.out_invalid_lrtb_edge_fraction]
        ), "INVALID RANGE"
        self.only_occlusions_valid = only_occlusions_valid

        # Args for more/less expensive GT sampling.
        self.n_retries_to_sample_valid = n_retries_to_sample_valid
        self.top_k_novel_views_to_keep = top_k_novel_views_to_keep
        self.num_synthetic_versions = num_synthetic_versions
        assert self.num_synthetic_versions > 0, "INVALID NUMBER OF OUTPUT VERSIONS"

    @classmethod
    def from_conf(
        cls,
        config: MainConfig,
        cam_incl_adjust: Optional[torch.Tensor] = None,
        refiner: bool = False,
        depth_pred: bool = False,
        **kwargs,
    ):

        camera_sampler = make_camera_sampler(
            config.SYNTHETIC_GT.NV_CAM_SAMPLER, cam_incl_adjust=cam_incl_adjust
        )
        depth_predictor = (
            make_depth_predictor(
                config.SYNTHETIC_GT, compile=config.SYNTHETIC_GT.COMPILE_DEPTH_PREDICTOR
            ).eval()
            if refiner or depth_pred
            else None
        )
        if refiner:
            components, refine_pipeline = make_model_components(
                config,
                build_pipe=True,
                compile_pipeline=config.SYNTHETIC_GT.COMPILE_REFINER,
            )
            # TODO maybe there is a better place for this
            refine_pipeline = refine_pipeline.to(torch.float16)
            kwargs["tokenizer"] = components["tokenizer"]
            kwargs["feature_extractor"] = components["feature_extractor"]
        elif "tokenizer" not in kwargs:
            kwargs["tokenizer"] = make_tokenizer(config)
            refine_pipeline = None
        else:
            refine_pipeline = None

        return cls(
            camera_sampler=camera_sampler,
            cfg=config,
            refine_pipeline=refine_pipeline,
            depth_predictor=depth_predictor,
            renderer_cfg=asdict(config.SYNTHETIC_GT.RENDERER),
            **asdict_lowercase_keys_override(config.SYNTHETIC_GT, **kwargs),
        )

    def to(self, *args, **kwargs):
        """Moves all submodules."""
        super().to(*args, **kwargs)
        if self.refine_pipeline is not None:
            self.refine_pipeline.to(*args, **kwargs)
        if self.depth_predictor is not None:
            self.depth_predictor.to(*args, **kwargs)
        return self

    def set_camera_sampler_kwargs(self, **kwargs):
        """..."""
        self._camera_sampler_kwargs = kwargs

    def forward(
        self,
        images: List[torch.Tensor] | torch.Tensor,
        poses: List[torch.Tensor] | torch.Tensor,
        projs: List[torch.Tensor] | torch.Tensor,
        depths: Optional[List[torch.Tensor] | torch.Tensor] = None,
        images_non_occlusion_masks: Optional[List[torch.Tensor]] = None,
        hw_unnorm: Optional[tuple[int, int]] = None,
        output_in_nv: bool = True,
        refine_output: bool = False,
        seed: Optional[int] = None,
        debug: bool = False,
        camera_sampler_kwargs: Optional[Dict[str, Any]] = None,
    ) -> tuple[Outputs, Data, Dict]:
        """Synthesizes GTs for controlnet or reconstructor training.

        Notation:
            B = batch size
            NV = number of output novel views
            N = length of the input lists = number of input novel views

        Args:
            images (List[torch.Tensor]): list of [B, 1, 3, H, W]
            poses (List[torch.Tensor]): list of [B, 1, 4, 4]
            projs (List[torch.Tensor]): list of [B, 1, 3, 3]
            depths (Optional[List[torch.Tensor]], optional): list of [B, 1, 1, H, W]. Defaults to None.
            output_in_nv (bool, optional): If False, the novel views are projected back to the input pose. Used for training the controlnet. Defaults to True.
            hw_unnorm (Tuple[int, int], optional): Image height and width used to produce unnormalized camera intrinsics.
                If `None`, the input image shape is used in-place.
            refine_output (bool, optional): Whether to predict refined novel views with depth. Defaults to False.
            seed (int, optional): Seed for deterministic behavior. Defaults to None.
            debug (bool, optional): Enables debugging outputs. Defaults to False.
            camera_sampler_kwargs (dict, optional): Keywords argument inputs to the camera pose sampler. Defaults to None.
        """
        if isinstance(images, torch.Tensor):
            images = [images]
        if isinstance(poses, torch.Tensor):
            poses = [poses]
        if isinstance(projs, torch.Tensor):
            projs = [projs]
        if depths is not None and isinstance(depths, torch.Tensor):
            depths = [depths]
        if images_non_occlusion_masks is not None and isinstance(
            images_non_occlusion_masks, torch.Tensor
        ):
            images_non_occlusion_masks = [images_non_occlusion_masks]

        if hw_unnorm is None:
            hw_unnorm = images[0].shape[-2:]

        if not output_in_nv:
            assert (
                len(images) == 1
            ), "OUTPUT IN INPUT VIEW IS NOT IMPLEMENTED FOR MULTIPLE ENCODING VIEWS."
            assert (
                images[0].shape[1] == 1
            ), "OUTPUT IN INPUT VIEW IS NOT IMPLEMENTED FOR MULTIPLE INPUT NOVEL VIEWS DIM > 1."

        debug_data = {"None": None} if debug else None

        with torch.no_grad():
            data = self._preprocess_inputs(
                images,
                depths,
                poses,
                projs,
                images_non_occlusion_masks,
                hw_unnorm,
                seed,
                output_in_nv,
                camera_sampler_kwargs,
            )

            if output_in_nv:
                out, data, debug_data = self.forward_reconstructor_training_data(
                    data, debug_data, refine_output
                )
            else:
                out, data, debug_data = self.forward_controlnet_training_data(
                    data, debug_data
                )

        return out, data, debug_data

    def _preprocess_inputs(
        self,
        imgs: List[torch.Tensor],
        depths: Optional[List[torch.Tensor]],
        poses: List[torch.Tensor],
        projs: List[torch.Tensor],
        images_non_occlusion_masks: List[torch.Tensor],
        hw_unnorm: tuple[int, int],
        eval_seed: int | None,
        output_in_nv: bool,
        camera_sampler_kwargs: Optional[Dict[str, Any]],
    ) -> Data:
        """
        Extract depth if needed, crop the input images, and tokenize the captions. Extract all helper variables.
        """
        H_RECT, W_RECT = hw_unnorm
        batch_size, n_in_imgs, _, _, _ = imgs[0].shape

        dtype = imgs[0].dtype
        device = imgs[0].device

        NV = self.camera_sampler.nv if output_in_nv else 1

        imgs_in_original_scale = None
        rescale_scale = 1.0
        if len(imgs) == 1:
            # INPUT IMAGE RESCALING IS ONLY IMPLEMENTED FOR (ONE) SAME-RESOLUTION INPUTS.
            H, W = imgs[0].shape[-2:]
            H_NV, W_NV = (
                (H, W)
                if not self.cfg.SYNTHETIC_GT.RERENDERING_RESOLUTION
                else self.cfg.SYNTHETIC_GT.RERENDERING_RESOLUTION
            )
            rescale_scale = max(H_NV / H, W_NV / W)

            if rescale_scale < 1:
                # Use a larger resolution input img for cropping the controlnet GT,
                # but use a smaller res during rendering to save time.
                imgs_in_original_scale = deepcopy(imgs[0] * 0.5 + 0.5)
                h_new = int(imgs[0].size(-2) * rescale_scale)
                w_new = int(imgs[0].size(-1) * rescale_scale)
                imgs[0] = resize(
                    imgs[0].view(batch_size * n_in_imgs, *imgs[0].shape[-3:]),
                    [h_new, w_new],
                ).view(batch_size, n_in_imgs, 3, h_new, w_new)
                projs[0] = scale_intrinsics(
                    projs[0], rescale_scale, rescale_scale, normalized=True
                )
                if depths is not None and len(depths):
                    depths[0] = resize(
                        depths[0].view(batch_size * n_in_imgs, *depths[0].shape[-3:]),
                        [h_new, w_new],
                    ).view(batch_size, n_in_imgs, 1, h_new, w_new)
                # Not implemented for images_non_occlusion_masks.
        else:
            pass
            # raise NotImplementedError()
            # TODO maybe disable use of self.cfg.SYNTHETIC_GT.RERENDERING_RESOLUTION

        # --- Predict depth if needed ---

        if depths is None or not len(depths):
            assert (
                self.depth_predictor is not None
            ), "Missing depth predictor, but wanting to predict depth."
            depths = []
            for i in range(len(imgs)):
                b, n, _, h, w = imgs[i].shape
                d = self.depth_predictor(
                    imgs[i].view(b * n, 3, h, w),
                    unnormalize_intrinsics(projs[i].view(b * n, 3, 3), W_RECT, H_RECT),
                ).view(b, n, 1, h, w)
                d = d.to(imgs[i])  # The depth predictors may up-cast to fp32.

                if n > 1:
                    # If we predict depths for multiple input images, we need to align the depths at
                    # idxs > 0 to the depth at idx = 0.

                    # We only have one "input image" here.
                    prev_num_input_imgs = self.renderer.net._num_input_imgs
                    self.renderer.net._num_input_imgs = 1
                    # Reproject and invalid mask at the depth map borders.
                    depths_to_ignore = torch.zeros_like(d[:, :1])
                    border = 2
                    depths_to_ignore[:, :, :, :, :border] = 1.0
                    depths_to_ignore[:, :, :, :, -border:] = 1.0
                    depths_to_ignore[:, :, :, :border, :] = 1.0
                    depths_to_ignore[:, :, :, -border:, :] = 1.0
                    depths_to_ignore[d[:, :1] < self.z_near_gt] = 1.0
                    depths_for_sobel = (
                        d[:, :1].clamp_min(self.z_near).clamp_max(self.z_far)
                    )
                    high_depth_grads = sobel_filter(
                        depths_for_sobel.view(-1, 1, h, w),
                        thresh=self.cfg.SYNTHETIC_GT.INFERENCE_OCCLUSIONS.DEPTH_GRADS_THRESH,
                    ).view(d[:, :1].shape)
                    depths_to_ignore[high_depth_grads > 0.5] = 1.0

                    ray_sampler = ImageRaySampler(
                        self.z_near, self.z_far, h, w, channels=2, norm_dir=True
                    )
                    img_rend, d_rend, invalid_rend = self.encode_and_render(
                        ray_sampler=ray_sampler,
                        imgs=imgs[i][:, :1].mean(dim=2, keepdim=True),
                        depths=d[:, :1],
                        projs_enc=projs[i][:, :1],
                        poses_enc=poses[i][:, :1],
                        projs_rend=projs[i][:, 1:],
                        poses_rend=poses[i][:, 1:],
                        imgs_features=depths_to_ignore,
                    )
                    self.renderer.net._num_input_imgs = prev_num_input_imgs

                    # Do not use the border pixels of the image at idx = 0 for alignment.
                    # Also filter out out-of-range depths and close the resulting mask
                    borders_invalid_rend = img_rend[:, :, 1:2]
                    invalid_align = torch.logical_or(
                        invalid_rend, borders_invalid_rend > 0.5
                    )
                    invalid_align = torch.logical_or(
                        invalid_align, d_rend < self.z_near_gt
                    )
                    invalid_align = (
                        morphological_op(
                            invalid_align.reshape(b * (n - 1), 1, h, w).float(),
                            "closing",
                            self.close_conditioning_invalid_kernel_size,
                        )
                        .bool()
                        .reshape(b, n - 1, 1, h, w)
                    )
                    # We want at least 5% content from the image at idx = 0 in the renders of the images at idxs > 0.
                    d_aligned, _, _ = self._align_depth(
                        d[:, 1:], d_rend, ~invalid_align, min_valid_fraction=0.2
                    )

                    # # =======================================================================================
                    # # Visualize the alignment of depth at idxs > 0 to the depth at idx = 0.
                    # import matplotlib.pyplot as plt
                    # plt.close()
                    # d_inv = invert_depth(d[0].permute(2, 0, 3, 1).reshape(h, w*n).clamp(self.z_near, self.z_far), self.z_near, self.z_far)
                    # d_aligned_reproj = invert_depth(d_aligned[0].permute(2, 0, 3, 1).reshape(h, w*(n-1)).clamp(self.z_near, self.z_far), self.z_near, self.z_far)
                    # d_aligned_reproj = torch.cat([torch.zeros_like(d[0, 0, 0]), d_aligned_reproj], dim=1)
                    # img_rend_plot = img_rend[0, :, 0:1].permute(2, 0, 3, 1).reshape(h, w*(n-1), 1).repeat(1, 1, 3) * 0.5 + 0.5
                    # img_rend_plot = torch.cat([torch.zeros_like(imgs[i][0, 0].permute(1, 2, 0)), img_rend_plot], dim=1)
                    # d_align_target_plot = invert_depth(d_rend[0].permute(2, 0, 3, 1).reshape(h, w*(n-1)).clamp(self.z_near, self.z_far), self.z_near, self.z_far)
                    # d_align_target_plot = torch.cat([torch.zeros_like(d[0, 0, 0]), d_align_target_plot], dim=1)
                    # plt.subplot(6, 1, 1)
                    # plt.imshow(imgs[i][0].permute(2, 0, 3, 1).reshape(h, w*n, 3).float().cpu() * 0.5 + 0.5)
                    # plt.subplot(6, 1, 2)
                    # plt.imshow(d_inv.float().cpu())
                    # plt.subplot(6, 1, 3)
                    # plt.imshow(img_rend_plot.float().cpu())
                    # plt.subplot(6, 1, 4)
                    # plt.imshow(d_align_target_plot.float().cpu())
                    # plt.subplot(6, 1, 5)
                    # plt.imshow(torch.cat([torch.zeros_like(d[0, 0, 0]), invalid_align[0].permute(2, 0, 3, 1).reshape(h, w*(n-1))], dim=1).float().cpu())
                    # plt.subplot(6, 1, 6)
                    # plt.imshow(d_aligned_reproj.float().cpu())
                    # plt.savefig("temp.png", dpi=300)
                    # # =======================================================================================

                    d[:, 1:] = d_aligned

                depths.append(d)

        # Avoid overflows.
        depths = [_.clamp(0, self.z_far) for _ in depths]
        imgs = [_ * 0.5 + 0.5 for _ in imgs]  # From [-1, 1] to [0, 1]

        # --- CROP THE INPUT IF NEEDED ---
        # Recomp H and W
        H = min([_.shape[-2] for _ in imgs])
        W = min([_.shape[-1] for _ in imgs])

        num_input_anchors = 1
        imgs_in, depths_in, projs_in, poses_in, masks_in = [], [], [], [], []
        for i, (img, depth, proj, pose) in enumerate(zip(imgs, depths, projs, poses)):
            # Anchors cropped from the input
            _, img_n, _, img_h, img_w = img.shape

            mask_cropped = None
            if img_w != W or img_h != H:
                # If the current image does not match the target shape, we need to crop it accordingly.
                match self.input_crop_policy:
                    case "center":  # TODO rename this to "center_crop"
                        l = torch.full(
                            (batch_size, n_in_imgs),
                            fill_value=(img_w - W) / 2,
                            device=device,
                            dtype=dtype,
                        )
                        t = torch.full(
                            (batch_size, n_in_imgs),
                            fill_value=(img_h - H) / 2,
                            device=device,
                            dtype=dtype,
                        )
                    case "center+left+right":
                        # TODO rename this to "overlapping_crops" since it can be > 3
                        num_input_anchors = int(math.ceil(img_w / W))
                        l = (
                            torch.linspace(
                                0,
                                img_w - W,
                                num_input_anchors,
                                device=device,
                                dtype=dtype,
                            )
                            .unsqueeze(1)
                            .repeat(batch_size, n_in_imgs)
                        )
                        t = (
                            torch.linspace(
                                0,
                                img_h - H,
                                num_input_anchors,
                                device=device,
                                dtype=dtype,
                            )
                            .unsqueeze(1)
                            .repeat(batch_size, n_in_imgs)
                        )
                    case _:
                        ValueError(
                            f"Input crop policy {self.input_crop_policy} is not available."
                        )

                # TODO set this in a better place
                self.renderer.net._num_input_imgs = num_input_anchors * n_in_imgs

                crop = Crop(H, W, l, t)
                img_cropped = crop(img.repeat(1, num_input_anchors, 1, 1, 1))
                depth_cropped = crop(depth.repeat(1, num_input_anchors, 1, 1, 1))
                proj_cropped = crop_intrinsics(
                    proj.repeat(1, num_input_anchors, 1, 1), l, t, img_w, img_h, W, H
                )
                pose_cropped = pose.repeat(1, num_input_anchors, 1, 1)
                if images_non_occlusion_masks is not None and len(
                    images_non_occlusion_masks
                ):
                    mask_cropped = crop(
                        images_non_occlusion_masks[i].repeat(
                            1, num_input_anchors, 1, 1, 1
                        )
                    )
            else:
                img_cropped = img
                depth_cropped = depth
                proj_cropped = proj
                pose_cropped = pose
                if images_non_occlusion_masks is not None and len(
                    images_non_occlusion_masks
                ):
                    mask_cropped = images_non_occlusion_masks[i]

            imgs_in.append(img_cropped)
            depths_in.append(depth_cropped)
            projs_in.append(proj_cropped)
            poses_in.append(pose_cropped)
            if images_non_occlusion_masks is not None and len(
                images_non_occlusion_masks
            ):
                masks_in.append(mask_cropped)

        # TODO check if deepcopy needed
        imgs_in = torch.concat(imgs_in, dim=1)
        depths_in = torch.concat(depths_in, dim=1)
        projs_in = torch.concat(projs_in, dim=1)
        poses_in = torch.concat(poses_in, dim=1)
        if images_non_occlusion_masks is not None and len(images_non_occlusion_masks):
            # Convert bool to float
            masks_in = torch.concat(masks_in, dim=1).to(dtype=dtype)

            # Sometimes edge pixels are invalid
            # TODO think of a better solution for this hardcode
        else:
            masks_in = torch.ones_like(depths_in)
        masks_in[:, :, :, 0, :] = 0.0
        masks_in[:, :, :, -1, :] = 0.0
        masks_in[:, :, :, :, 0] = 0.0
        masks_in[:, :, :, :, -1] = 0.0

        N = imgs_in.size(1)
        # The resolution of the input-encoding image
        H_INP, W_INP = (
            H,
            W,
        )  # if not self.cfg.SYNTHETIC_GT.ENCODING_RESOLUTION else self.cfg.SYNTHETIC_GT.ENCODING_RESOLUTION
        # The novel view resolution
        # The re-rendered input-pose-resolution is a multiple of the controlnet-training-resolution.
        H_CTRL, W_CTRL = self.cfg.CONTROLNET.TRAIN.RESOLUTION
        scale_factor = min(H_INP / H_CTRL, W_INP / W_CTRL)
        H_REREN = int(H_CTRL * scale_factor)
        W_REREN = int(W_CTRL * scale_factor)

        generator = (
            torch.Generator(device=device).manual_seed(eval_seed) if eval_seed else None
        )

        # Tokenize captions
        get_str_fn = lambda x: (
            ""
            if random.random() < self.cfg.CONTROLNET.TRAIN.PROPORTION_EMPTY_PROMPTS
            else x
        )
        NUM_TOP_K = (
            NV
            if not self.top_k_novel_views_to_keep
            else min(NV, self.top_k_novel_views_to_keep)
        )
        caption_strs = [
            get_str_fn(self.cfg.CONTROLNET.TRAIN.PROMPT_TEXT)
            for _ in range(batch_size * NUM_TOP_K)
        ]

        captions_ids: torch.Tensor | None = None
        if self.tokenizer:
            inputs = self.tokenizer(
                caption_strs,
                max_length=self.tokenizer.model_max_length,
                padding="max_length",
                truncation=True,
                return_tensors="pt",
            )
            captions_ids = inputs.input_ids.to(device)

        images_clip_preprocessed = None
        if self.feature_extractor:
            # CLIPImageProcessor expects inputs in range [0, 255] if `do_rescale=True`.
            # https://huggingface.co/docs/transformers/en/model_doc/clip#transformers.CLIPImageProcessor.preprocess.images
            # images_clip_preprocessed *= 255.
            self.feature_extractor.do_rescale = False
            images_clip_preprocessed = self.feature_extractor(
                imgs_in[:, :1].view(-1, *imgs_in.shape[-3:]).clamp(0, 1),
                return_tensors="pt",
            ).pixel_values
            images_clip_preprocessed = images_clip_preprocessed.repeat_interleave(
                NUM_TOP_K, dim=0
            )
            images_clip_preprocessed = images_clip_preprocessed.to(device)

        data = Data(
            IMGS=imgs_in,
            DEPTHS=depths_in,
            POSES=poses_in,
            PROJS=projs_in,
            MASKS=masks_in if len(masks_in) else None,
            IMGS_IN_FOR_OCCLUSION_DET=imgs[0],
            IMGS_IN_FOR_OCCLUSION_DET_HW=imgs[0].shape[-2:],
            DEPTHS_IN_FOR_OCCLUSION_DET=depths[0],
            PROJS_IN_FOR_OCCLUSION_DET=projs[0],
            HW_DATA=(H, W),
            HW=(H_INP, W_INP),
            # HW_NV=(H_REREN, W_REREN) if output_in_nv else None,
            HW_NV=(
                (H_REREN, W_REREN)
                if output_in_nv
                else self.cfg.SYNTHETIC_GT.RERENDERING_RESOLUTION
            ),
            HW_REREN=(H_REREN, W_REREN),
            HW_CTRL=(H_CTRL, W_CTRL),
            HW_RECT=(H_RECT, W_RECT),
            NUM_NV=NV,
            NUM_TOP_K=NUM_TOP_K,
            N=N,
            B=batch_size,
            CAPTION_STRS=caption_strs,
            CAPTIONS_IDS=captions_ids,
            IMGS_PREPROCESSED=images_clip_preprocessed,
            GENERATOR=generator,
            IMGS_ORIGINAL_SCALE=imgs_in_original_scale,
            RESCALE_SCALE=rescale_scale,
            CAM_SAMPLER_KWARGS=(
                camera_sampler_kwargs if camera_sampler_kwargs is not None else {}
            ),
            NUM_INPUT_ANCHORS=num_input_anchors * n_in_imgs,
        )
        return data

    def forward_controlnet_training_data(
        self, data: Data, debug_data: Dict
    ) -> tuple[Outputs, Data, Dict]:
        """Project to novel view -> project to input view."""

        data, debug_data = self._render_novel_view_from_input(
            data, debug_dict=debug_data, make_masks=False
        )
        data, debug_data = self._rerender_input_from_novel_view(
            data, debug_dict=debug_data
        )
        data.IMGS_CONDITIONING, data.MASKS_REREN, valid_idxs = (
            self._build_conditioning_input(
                imgs=data.IMGS_REREN,
                depths=data.DEPTHS_REREN,
                masks=data.MASKS_REREN,
                invalid=data.INVALID_REREN,
                hw_out=data.HW_CTRL,
                invalid_to_black=True,
            )
        )
        # Set invalid for samples where the min depth cannot be encoded in the pseudo volume.
        valid_idxs = torch.logical_and(
            valid_idxs,
            (data.DEPTHS_CROPPED_TO_REREN > self.z_near)
            .all(dim=-1)
            .all(dim=-1)
            .all(dim=-1),
        )

        scores = self._compute_novel_view_scores(data.MASKS_REREN, data.INVALID_REREN)
        data = self._keep_top_k(data, scores)

        imgs_gt = resize(
            data.IMGS_CROPPED_TO_REREN.squeeze(1),
            data.HW_CTRL,
            interpolation=InterpolationMode.BILINEAR,
            antialias=None,
        )
        imgs_gt = normalize(imgs_gt, [0.5], [0.5])

        return (
            Outputs(
                DEPTHS_IN=data.DEPTHS_IN_FOR_OCCLUSION_DET,
                IMGS_IN=data.IMGS_IN_FOR_OCCLUSION_DET,
                PROJS_IN=data.PROJS_IN_FOR_OCCLUSION_DET,
                POSES_IN=data.POSES,
                # Controlnet training outputs
                POSES_OUT=data.POSES,
                PROJS_OUT=data.PROJS_REREN,
                IMGS_COND=data.IMGS_CONDITIONING.view(
                    -1, *data.IMGS_CONDITIONING.shape[-3:]
                ),
                IMGS_GT=imgs_gt,
                DEPTHS_GT=data.DEPTHS_CROPPED_TO_REREN.view(
                    -1, *data.DEPTHS_CROPPED_TO_REREN.shape[-3:]
                ),
                CAPTIONS_IDS=data.CAPTIONS_IDS,
                IMGS_PREPROCESSED=data.IMGS_PREPROCESSED,
            ),
            data,
            debug_data,
        )

    def forward_reconstructor_training_data(
        self, data: Data, debug_data: dict | None, refine_output=False
    ) -> tuple[Outputs, Data, Dict]:
        """Project to novel view -> Refine + predict depth."""

        data, debug_data = self._render_novel_view_from_input(
            data, debug_dict=debug_data, make_masks=True
        )
        data.IMGS_CONDITIONING, _, _ = self._build_conditioning_input(
            imgs=data.IMGS_NV,
            depths=data.DEPTHS_NV,
            masks=data.MASKS_NV,
            invalid=data.INVALID_NV,
            hw_out=data.HW_CTRL,
            invalid_to_black=True,
        )

        assert data.MASKS_NV is not None, "MASKS_NV IS NONE"
        assert data.INVALID_NV is not None, "INVALID_NV IS NONE"
        scores = self._compute_novel_view_scores(data.MASKS_NV, data.INVALID_NV)

        # # =======================================================================================
        # # Visualize the scores
        # import matplotlib.pyplot as plt
        # score_indices_sorted = scores[0].sort().indices
        # plot_imgs = data.IMGS_NV[0, score_indices_sorted].permute(2, 0, 3, 1).reshape(192, 288 * data.NUM_NV, 3)
        # plot_mask = data.MASKS_NV[0, score_indices_sorted, 0].permute(1, 0, 2).reshape(192, 288 * data.NUM_NV, 1).repeat(1, 1, 3)
        # plot_imgs = torch.cat([plot_imgs, plot_mask], dim=0).cpu()
        # plt.title(f"Scores: " + ", ".join([f"{_:.3f}" for _ in scores[0].sort().values.cpu()]))
        # plt.imshow(plot_imgs)
        # plt.savefig("scores.png", dpi=300)
        # # =======================================================================================

        # Pick the top-k novel views from the data
        data = self._keep_top_k(data, scores)

        depths_in_valid_range, per_sample_time = None, None
        depths_out, imgs_refined, scales, shifts = [], [], [], []
        if refine_output:

            time_start = time.time()

            # TODO maybe del this
            num_synthetic_versions = 1 if self.training else self.num_synthetic_versions

            for _ in range(num_synthetic_versions):
                imgs_refined_v, depths_out_v, _, hw_out = self.refine_and_predict_depth(
                    data, set_generator_none=num_synthetic_versions > 1
                )
                # Clamp depth to avoid overflows.
                depths_out_v = depths_out_v.clamp(0, self.z_far)

                # Align predicted depth to input depth map.
                if self.align_depth:
                    # Only align:
                    #   - where the reprojected depth is in the valid interval,
                    mask = torch.logical_and(
                        self.z_near_gt < data.DEPTHS_NV, data.DEPTHS_NV < self.z_far_gt
                    )
                    #   - where the pixels are classified as visible,
                    # mask = torch.logical_and(mask, data.MASKS_NV == DEFINITIONS.IS_VISIBLE)
                    mask = torch.logical_and(
                        mask, data.MASKS_NV != DEFINITIONS.IS_OCCLUDED
                    )
                    if self.max_rel_depth_error_to_align > 0.0:
                        #   - and where the relative depth error is smaller than a threshold.
                        #     This does not help if the reprojected and predicted depth scales are too different.
                        mask = torch.logical_and(
                            mask,
                            (depths_out_v - data.DEPTHS_NV).abs()
                            / data.DEPTHS_NV.clamp_min(1.0)
                            < self.max_rel_depth_error_to_align,
                        )

                    if mask.any():
                        depths_out_v, scale_v, shift_v = self._align_depth(
                            depths_out_v, data.DEPTHS_NV, mask
                        )
                        scales.append(scale_v)
                        shifts.append(shift_v)

                depths_out.append(depths_out_v)
                imgs_refined.append(imgs_refined_v)

                # NOTE: Technically we need to store this per version, but we ignore that for now.
                depths_in_valid_range = torch.logical_and(
                    self.z_near_gt < depths_out_v, depths_out_v < self.z_far_gt
                )

            data.INVALID_NV = resize(
                data.INVALID_NV,
                hw_out,
                interpolation=InterpolationMode.NEAREST,
                antialias=None,
            )
            b, nv, c, h, w = data.IMGS_CONDITIONING.shape
            data.IMGS_CONDITIONING = resize(
                data.IMGS_CONDITIONING.view(-1, c, h, w),
                hw_out,
                interpolation=InterpolationMode.BILINEAR,
                antialias=None,
            ).view(b, nv, c, *hw_out)

            if debug_data:
                torch.cuda.synchronize()
                per_sample_time = (time.time() - time_start) / (
                    data.B * num_synthetic_versions
                )

        data.INVALID_NV = data.INVALID_NV.bool()
        m = data.MASKS_NV.clone()
        m[data.INVALID_NV] = DEFINITIONS.IS_INVALID
        has_enough_valid = (m == DEFINITIONS.IS_VISIBLE).to(m.dtype).mean(
            dim=[2, 3, 4]
        ) > self.min_mean_valid_pixels
        data.INVALID_NV[~has_enough_valid] = True
        has_too_many_occluded = (m == DEFINITIONS.IS_OCCLUDED).to(m.dtype).mean(
            dim=[2, 3, 4]
        ) > self.max_mean_occluded_pixels
        data.INVALID_NV[has_too_many_occluded] = True
        if depths_in_valid_range is not None:
            data.INVALID_NV[~depths_in_valid_range] = True
        if self.only_occlusions_valid:
            data.INVALID_NV[data.MASKS_NV != DEFINITIONS.IS_OCCLUDED] = True

        h, w = data.INVALID_NV.shape[-2:]
        l, r = [int(w * _) for _ in self.out_invalid_lrtb_edge_fraction[:2]]
        t, b = [int(h * _) for _ in self.out_invalid_lrtb_edge_fraction[2:]]
        data.INVALID_NV[..., :t, :] = True
        if b > 0:
            data.INVALID_NV[..., -b:, :] = True
        data.INVALID_NV[..., :l] = True
        if r > 0:
            data.INVALID_NV[..., -r:] = True

        return (
            Outputs(
                DEPTHS_IN=data.DEPTHS_IN_FOR_OCCLUSION_DET,
                IMGS_IN=data.IMGS_IN_FOR_OCCLUSION_DET,
                PROJS_IN=data.PROJS_IN_FOR_OCCLUSION_DET,
                POSES_IN=data.POSES,
                # recon training outputs
                POSES_OUT=data.POSES_NV,
                PROJS_OUT=data.PROJS_NV,
                IMGS_SYNTH=torch.stack(imgs_refined) if refine_output else None,
                DEPTHS_SYNTH=torch.stack(depths_out) if refine_output else None,
                SCALES=torch.stack(scales) if scales else None,
                SHIFTS=torch.stack(shifts) if shifts else None,
                AVG_DENOISING_TIME=per_sample_time,
                INVALID_SYNTH=data.INVALID_NV,
                IMGS_COND=data.IMGS_CONDITIONING,
            ),
            data,
            debug_data,
        )

    def encode_and_render(
        self,
        ray_sampler: ImageRaySampler,
        imgs: torch.Tensor,
        depths: torch.Tensor,
        projs_enc: torch.Tensor,
        poses_enc: torch.Tensor,
        projs_rend: torch.Tensor,
        poses_rend: torch.Tensor,
        imgs_features: Optional[torch.Tensor] = None,
    ) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        """Encodes RGBD image, samples novel view camera rays and returns a novel RGBD view."""
        # Encoding depth is expected to be in Z-dim.
        self.renderer.net.encode(
            depths, imgs, projs_enc, poses_enc, features=imgs_features
        )

        all_rays, _ = ray_sampler.sample(
            None, poses_rend, projs_rend
        )  # [n*nv, n_pts, 8]
        render_dict = self.renderer(all_rays, want_weights=True)

        if "fine" not in render_dict:
            # In case no hierachical sampling was used.
            render_dict["fine"] = dict(render_dict["coarse"])

        render_dict = ray_sampler.reconstruct(render_dict)

        depths = render_dict["fine"]["depth"].unsqueeze(2)  # [B, 1, 1, H, W]
        rgb = (
            render_dict["fine"]["rgb"].squeeze(4).permute(0, 1, 4, 2, 3)
        )  # [B, 1, 3, H, W]
        # Do not use a strict invalid policy because this will definately leads to
        # invalid pixels when sampling a novel view outside of the input camera frustum!
        #   e.g.: invalid = render_dict["fine"]["invalid"].squeeze(-1).any(dim=-1).unsqueeze(2)
        # The pseudo-volume does not have density outside any frustum and therefore
        # invalid xyz only have weight when the rays never hits any frustum. We can
        # check if all xyz are invalid:
        # invalid = render_dict["fine"]["invalid"].squeeze(-1).all(dim=-1).unsqueeze(2)
        # If the sigmoid threshold in the pseudo volume is set >.5, this will create
        # content-less border pixels that are not invalid. I found that using a
        # weight-guided policy is better (but does not remove all of those artefacts):
        invalid = (
            render_dict["fine"]["invalid"].squeeze(-1) * render_dict["fine"]["weights"]
        ).sum(-1).unsqueeze(2) > 0.8

        # # =======================================================================================
        # # Visualize depth and rgb
        # import matplotlib.pyplot as plt
        # plt.close()
        # dinv = (1 / depths[0, :, 0].permute(1, 0, 2).reshape(192, 288*8).cpu() - 1 / 100) / (1 / 3 - 1 / 100)
        # rgbs = rgb[0, :, :3].permute(2, 0, 3, 1).reshape(192, 288*8, 3).cpu()
        # occlusions = rgb[0, :, 3].permute(1, 0, 2).reshape(192, 288*8).cpu()
        # invalids = invalid[0, :, 0].permute(1, 0, 2).reshape(192, 288*8).cpu()
        # plt.subplot(4, 1, 1)
        # plt.imshow(dinv.float())
        # plt.subplot(4, 1, 2)
        # plt.imshow(rgbs.float())
        # plt.subplot(4, 1, 3)
        # plt.imshow(invalids)
        # plt.subplot(4, 1, 4)
        # plt.imshow(occlusions)
        # plt.tight_layout()
        # plt.savefig("depths.png", dpi=300)
        # # =======================================================================================

        if self.black_invalid:
            depths[invalid] = depths.max()
            rgb[invalid.repeat(1, 1, 3, 1, 1)] = 0

        # Depths are the distances in Z dir and not the distance along the rays!
        assert (
            ray_sampler.norm_dir
        ), "The ray sampler does not have 'norm_dir=True'. This will give incorrect results!"
        depths = (
            distance_to_z(depths.squeeze(-3), projs_rend).unsqueeze(-3).to(depths.dtype)
        )

        return rgb, depths, invalid

    def _render_novel_view_from_input(
        self,
        data: Data,
        poses_rot: Optional[torch.Tensor] = None,
        debug_dict: Optional[Dict] = None,
        make_masks: bool = True,
    ) -> tuple[Data, Optional[Dict]]:
        """Given data from the inputs views, return the novel views."""
        B = data.B
        N = data.N
        NV = data.NUM_NV
        H, W = data.HW
        H_NV, W_NV = data.HW_NV
        H_RECT, W_RECT = data.HW_RECT

        # Overloading the camera sampler with all arguments
        data.CAM_SAMPLER_KWARGS.update(
            {
                "depths_enc": data.DEPTHS,
                "poses_enc": data.POSES,
                "projs_enc": data.PROJS,
                "num_input_anchors": data.NUM_INPUT_ANCHORS,
                "masks_enc": data.MASKS,
            }
        )
        poses_rot, projs_NV, debug_sample_rotated_poses_and_projs = self.camera_sampler(
            # We use the 0th camera in the input to find the novel view cameras
            poses=data.POSES[:, :1],
            projs=data.PROJS_IN_FOR_OCCLUSION_DET[:, :1],
            depths=data.DEPTHS_IN_FOR_OCCLUSION_DET[:, :1],
            nv=NV,
            hw=data.IMGS_IN_FOR_OCCLUSION_DET_HW,
            hw_rot=(H_NV, W_NV),
            generator=data.GENERATOR,
            hw_unnorm=(H_RECT, W_RECT),
            **data.CAM_SAMPLER_KWARGS,
        )

        # --- RERENDER INPUT TO PRODUCE OCCLUSIONS IN THE CONDITIONING IMAGE ---

        channels = 3
        feature_map = None
        if self.cfg.SYNTHETIC_GT.INFERENCE_OCCLUSIONS.USE_DEPTH_GRADS:
            channels += 1
            depth_grads_thresh = (
                self.cfg.SYNTHETIC_GT.INFERENCE_OCCLUSIONS.DEPTH_GRADS_THRESH
            )
            depths_for_sobel = data.DEPTHS.clamp_min(self.z_near).clamp_max(self.z_far)
            feature_map = sobel_filter(
                depths_for_sobel.view(-1, 1, H, W), thresh=depth_grads_thresh
            ).view(data.DEPTHS.shape)
            if data.NUM_INPUT_ANCHORS == 1:
                border = 3
                # feature_map[:, 0, :, :border, :] = 1.
                # feature_map[:, 0, :, -border:, :] = 1.
                # Only left and right borders because the camera will not move up or down on our data.
                feature_map[:, 0, :, :, :border] = 1.0
                feature_map[:, 0, :, :, -border:] = 1.0
        if data.MASKS is not None:
            channels += 1
            if feature_map is not None:
                feature_map = torch.concat([feature_map, data.MASKS], dim=-3)
            else:
                feature_map = data.MASKS

        # Encode with input views and render novel views at the input resolution.
        ray_sampler = ImageRaySampler(
            self.z_near, self.z_far, H_NV, W_NV, channels=channels, norm_dir=True
        )
        imgs_nv, depths_nv, invalid_nv = self.encode_and_render(
            ray_sampler,
            data.IMGS,
            data.DEPTHS,
            data.PROJS,
            data.POSES,
            projs_NV,
            poses_rot,
            feature_map,
        )
        rgb_nv = imgs_nv[:, :, :3]

        # COMPUTE REGIONS IN THE NOVEL VIEWS THAT ARE OCCLUDED IN THE INPUTS
        # Determine the pixel-boundaries of the re-rendered image in the input image.
        min_left_reren, min_top_reren, max_right_reren, max_bottom_reren = (
            comp_reprojected_img_extents(
                projs_NV,
                data.PROJS_IN_FOR_OCCLUSION_DET[
                    :, :1
                ],  # TODO possibly account for all input images here
                depths_nv,
                poses_rot,
                data.POSES[:, :1],
                data.IMGS_IN_FOR_OCCLUSION_DET_HW[1],
                data.IMGS_IN_FOR_OCCLUSION_DET_HW[0],
                ~invalid_nv,
                # W, H
            )
        )
        # We need to clamp these to [0, W_INP or H_INP] because we do not care about input image pixels
        # outside of this range.
        min_left_reren = min_left_reren.clamp(0, data.IMGS_IN_FOR_OCCLUSION_DET_HW[1])
        max_right_reren = max_right_reren.clamp(0, data.IMGS_IN_FOR_OCCLUSION_DET_HW[1])
        min_top_reren = min_top_reren.clamp(0, data.IMGS_IN_FOR_OCCLUSION_DET_HW[0])
        max_bottom_reren = max_bottom_reren.clamp(
            0, data.IMGS_IN_FOR_OCCLUSION_DET_HW[0]
        )
        # Then sample the left and top cropping values.
        lrtb_reren = (min_left_reren, max_right_reren, min_top_reren, max_bottom_reren)

        # Center crop
        w_interval = (max_right_reren - W_NV - min_left_reren).clamp(0, torch.inf)
        h_interval = (max_bottom_reren - H_NV - min_top_reren).clamp(0, torch.inf)
        lefts_reren = min_left_reren + w_interval / 2
        tops_reren = min_top_reren + h_interval / 2
        lefts_reren = (
            lefts_reren.clamp(0, max(0, data.IMGS_IN_FOR_OCCLUSION_DET_HW[1] - W_NV))
            .round()
            .long()
        )  # [B*NV]
        tops_reren = (
            tops_reren.clamp(0, max(0, data.IMGS_IN_FOR_OCCLUSION_DET_HW[0] - H_NV))
            .round()
            .long()
        )  # [B*NV]
        crop_to_NV = Crop(H_NV, W_NV, lefts=lefts_reren, tops=tops_reren)

        # (3) Create the re-rendered image cropping function and intrinsics.
        # Apply the new resolution to the intrinsics.
        # TODO possibly account for all input images here
        projs_NV_REREN = crop_intrinsics(
            data.PROJS_IN_FOR_OCCLUSION_DET[:, :1].repeat(1, NV, 1, 1),
            lefts_reren,
            tops_reren,
            unnormalized_W=data.IMGS_IN_FOR_OCCLUSION_DET_HW[1],
            unnormalized_H=data.IMGS_IN_FOR_OCCLUSION_DET_HW[0],
            new_unnormalized_W=W_NV,
            new_unnormalized_H=H_NV,
        )

        # TODO maybe stitch :3 together is three input crops
        input_crops_for_optflow = crop_to_NV(
            data.IMGS_IN_FOR_OCCLUSION_DET[:, :1].repeat(1, NV, 1, 1, 1)
        )
        input_depth_crops = crop_to_NV(
            data.DEPTHS_IN_FOR_OCCLUSION_DET[:, :1].repeat(1, NV, 1, 1, 1)
        )

        # TODO add param
        occlusion_mask = None
        if self.cfg.SYNTHETIC_GT.INFERENCE_OCCLUSIONS.USE_DEPTH_GRADS:
            occlusion_mask = (
                imgs_nv[:, :, 3:4].view(B * NV, 1, H_NV, W_NV) > self.depth_grads_thresh
            )
        # data_input_masks = None
        # if data.MASKS is not None:
        #     data_input_masks = imgs_nv[:, :, 4:5].view(B*NV, 1, H_NV, W_NV).bool()

        occlusion_masks_dict, masks_nv = None, None
        if make_masks:
            occlusion_masks_dict = comp_occlusion_map(
                self.cfg.SYNTHETIC_GT.INFERENCE_OCCLUSIONS,
                img1=input_crops_for_optflow.view(B * NV, 3, H_NV, W_NV),
                img2=rgb_nv.view(B * NV, 3, H_NV, W_NV),
                pose1=data.POSES[:, 0].repeat_interleave(NV, dim=0),
                pose2=poses_rot.view(B * NV, 4, 4),
                # We crop the input depth so it should have the same intrinsics as the rerendering camera.
                proj1=projs_NV_REREN.view(B * NV, 3, 3),
                proj2=projs_NV.view(B * NV, 3, 3),
                depth1=input_depth_crops.view(B * NV, 1, H_NV, W_NV),
                depth2=depths_nv.view(B * NV, 1, H_NV, W_NV),
                occlusions_depth_grads=occlusion_mask,
                # exclusion_map=data_input_masks,
            )  # dict with [B*nv, 1, H, W]

            masks_nv = (
                occlusion_masks_dict["occlusions_full"]
                .view(B, NV, 1, H_NV, W_NV)
                .to(data.IMGS)
            )
            # masks_nv = occlusion_masks_dict["occlusions_full_non_excluded"].view(B, NV, 1, H_NV, W_NV)

        # These are at the input data resolution!
        projs_out_unnormalized = unnormalize_intrinsics(
            projs_NV,
            width=W_NV / data.IMGS_IN_FOR_OCCLUSION_DET_HW[1] * W_RECT,
            height=H_NV / data.IMGS_IN_FOR_OCCLUSION_DET_HW[0] * H_RECT,
        )  # [B, 3, 3]

        if debug_dict:
            debug_dict.update(
                {
                    "occlusion_masks_dict": occlusion_masks_dict,
                    "input_crops_for_optflow": input_crops_for_optflow.clone()
                    .detach()
                    .cpu(),
                    "min_left_nv": copy(
                        debug_sample_rotated_poses_and_projs.get("l", None)
                    ),
                    "min_top_nv": copy(
                        debug_sample_rotated_poses_and_projs.get("t", None)
                    ),
                    "max_right_nv": copy(
                        debug_sample_rotated_poses_and_projs.get("r", None)
                    ),
                    "max_bottom_nv": copy(
                        debug_sample_rotated_poses_and_projs.get("b", None)
                    ),
                    "lefts_nv_crop": copy(
                        debug_sample_rotated_poses_and_projs.get("l_sample", None)
                    ),
                    "top_nv_crop": copy(
                        debug_sample_rotated_poses_and_projs.get("t_sample", None)
                    ),
                    "min_left_reren": min_left_reren.clone().detach().cpu(),
                    "min_top_reren": min_top_reren.clone().detach().cpu(),
                    "max_right_reren": max_right_reren.clone().detach().cpu(),
                    "max_bottom_reren": max_bottom_reren.clone().detach().cpu(),
                    "lefts_reren": lefts_reren.clone().detach().cpu(),
                }
            )

        data.IMGS_NV = rgb_nv
        data.DEPTHS_NV = depths_nv
        data.MASKS_NV = masks_nv
        data.INVALID_NV = invalid_nv
        data.POSES_NV = poses_rot
        data.PROJS_NV = projs_NV
        data.PROJS_NV_UNNORMALIZED = projs_out_unnormalized
        data.LRTB_REREN = lrtb_reren

        # # =======================================================================================
        # # Visualize the novel views
        # import matplotlib.pyplot as plt
        # plt.close()
        # imgs = imgs_nv[:, :, :3].view(B*NV, 3, H_NV, W_NV)
        # occlusion_mask = imgs_nv[:, :, 3:4].view(B*NV, 1, H_NV, W_NV) > 0.5
        # d = data.DEPTHS[0, 0, 0].clamp_min(self.z_near)
        # d_inv = invert_depth(d, self.z_near, self.z_far)
        # plt.subplot(5, 1, 1)
        # plt.imshow(d_inv.cpu())
        # plt.subplot(5, 1, 2)
        # plt.imshow(feature_map[0, 0, 0].cpu())
        # plt.subplot(5, 1, 3)
        # plt.imshow(sobel_filter(data.DEPTHS.view(-1, 1, H, W).clamp_min(self.z_near), thresh=0.1)[0, 0].cpu())
        # plt.subplot(5, 1, 4)
        # plt.imshow(imgs.permute(2, 0, 3, 1).reshape(192, -1, 3).cpu().float())
        # plt.subplot(5, 1, 5)
        # plt.imshow(occlusion_mask[:, 0].permute(1, 0, 2).reshape(192, -1).cpu())
        # plt.savefig("temp.png", dpi=300)
        # # =======================================================================================

        return data, debug_dict

    def _rerender_input_from_novel_view(
        self, data: Data, debug_dict: Optional[Dict] = None
    ):

        B = data.B
        NV = data.NUM_NV
        H, W = data.HW
        H_NV, W_NV = data.HW_NV
        H_REREN, W_REREN = data.HW_REREN
        H_RECT, W_RECT = data.HW_RECT
        H_CTRL, W_CTRL = data.HW_CTRL

        min_left_reren, max_right_reren, min_top_reren, max_bottom_reren = (
            data.LRTB_REREN
        )

        # # Then sample the left and top cropping values.
        w_interval = (max_right_reren - W_REREN - min_left_reren).clamp(0, torch.inf)
        h_interval = (max_bottom_reren - H_REREN - min_top_reren).clamp(0, torch.inf)
        lefts_reren = get_interval_sample(
            min_left_reren, w_interval, generator=data.GENERATOR
        )  # [B, nv]
        tops_reren = get_interval_sample(
            min_top_reren, h_interval, generator=data.GENERATOR
        )  # [B, nv]
        lefts_reren = (
            lefts_reren.clamp(0, W - W_REREN).round().long().flatten()
        )  # [B*NV]
        tops_reren = tops_reren.clamp(0, H - H_REREN).round().long().flatten()  # [B*NV]
        crop_INP_to_REREN = Crop(H_REREN, W_REREN, lefts=lefts_reren, tops=tops_reren)
        # (3) Create the re-rendered image cropping function and intrinsics.
        # Apply the new resolution to the intrinsics.
        projs_REREN = crop_intrinsics(
            data.PROJS,
            lefts_reren,
            tops_reren,
            unnormalized_W=W,
            unnormalized_H=H,
            new_unnormalized_W=W_REREN,
            new_unnormalized_H=H_REREN,
        )
        ray_sampler = ImageRaySampler(
            self.z_near, self.z_far, H_REREN, W_REREN, norm_dir=True
        )
        # Encode with novel views and re-render input views at controlnet-aspect-ratio.
        # TODO convert depths to depths along Z
        img_rerendered, depth_rerendered, invalid_rerendered = self.encode_and_render(
            ray_sampler,
            data.IMGS_NV,
            data.DEPTHS_NV,
            data.PROJS_NV,
            data.POSES_NV,
            projs_REREN,
            data.POSES,
        )

        depths_cropped_to_reren = crop_INP_to_REREN(data.DEPTHS.squeeze(1))
        # NOTE: We should align the re-rendered depth to the input depth, but if the
        # pseudo volume reprojects depth sharply, this is not needed.
        # depth_rerendered, _, _ = self._align_depth(depth_rerendered, depths_cropped_to_reren, ~invalid_rerendered)
        occlusion_masks_dict = comp_occlusion_map(
            self.cfg.SYNTHETIC_GT.OCCLUSIONS,
            pose1=data.POSES.squeeze(1),
            pose2=data.POSES.squeeze(1),
            # We crop the input depth so it should have the same intrinsics as the rerendering camera.
            proj1=projs_REREN.squeeze(1),
            proj2=projs_REREN.squeeze(1),
            # depth1=depth_rerendered.squeeze(1),
            depth1=depth_rerendered.squeeze(1),
            depth2=depths_cropped_to_reren,
        )  # [B, 1, H, W]

        if data.IMGS_ORIGINAL_SCALE is not None:
            crop_INP_ORIGINAL_SCALE_to_REREN = Crop(
                H_REREN * 1 / data.RESCALE_SCALE,
                W_REREN * 1 / data.RESCALE_SCALE,
                lefts=lefts_reren * 1 / data.RESCALE_SCALE,
                tops=tops_reren * 1 / data.RESCALE_SCALE,
            )
            data.IMGS_CROPPED_TO_REREN = crop_INP_ORIGINAL_SCALE_to_REREN(
                data.IMGS_ORIGINAL_SCALE.squeeze(1)
            ).unsqueeze(1)
        else:
            data.IMGS_CROPPED_TO_REREN = crop_INP_to_REREN(
                data.IMGS.squeeze(1)
            ).unsqueeze(1)
        data.DEPTHS_CROPPED_TO_REREN = depths_cropped_to_reren.view(
            B, NV, *depths_cropped_to_reren.shape[-3:]
        )

        projs_out_unnormalized = unnormalize_intrinsics(
            projs_REREN.squeeze(1),
            width=W_REREN / W * W_RECT,
            height=H_REREN / H * H_RECT,
        )  # [B, 3, 3]
        projs_out_unnormalized = scale_intrinsics(
            projs_out_unnormalized,
            scale_width=W_CTRL / W_REREN,
            scale_height=H_CTRL / H_REREN,
        )  # [B, 3, 3]

        if debug_dict:
            debug_dict.update(
                {
                    "occlusion_masks_dict": occlusion_masks_dict,
                }
            )

        data.IMGS_REREN = img_rerendered
        # data.DEPTHS_REREN = depth_rerendered_aligned
        data.DEPTHS_REREN = depth_rerendered
        data.INVALID_REREN = invalid_rerendered
        data.MASKS_REREN = occlusion_masks_dict["occlusions_full"].view(
            B, NV, 1, H_REREN, W_REREN
        )
        data.PROJS_REREN = projs_REREN
        data.PROJS_REREN_UNNORMALIZED = projs_out_unnormalized

        return data, debug_dict

    def _build_conditioning_input(
        self, imgs, depths, masks, invalid, hw_out, invalid_to_black: bool = False
    ):
        """
        Returns the conditioning input image for the refinement models by joining the required channels
        from the given images, depths and masks and painting occluded and invalid areas.
        """
        b, nv, _, h, w = imgs.shape
        imgs_conditioning = imgs.clone().view(b * nv, 3, h, w).clamp(0.0, 1.0)
        if self.depths_conditioning_inverse:
            depths_conditioning = (
                invert_depth(depths, self.z_near, self.z_far)
                .view(b * nv, 1, h, w)
                .to(imgs)
            )
        else:
            depths_conditioning = (
                (depths - self.z_near) / (self.z_far - self.z_near)
            ).view(b * nv, 1, h, w)
        # TODO conditioning on inverse depth
        # print("unique", masks.unique())
        masks_conditioning = masks.clone().view(b * nv, 1, h, w)

        is_invalid = invalid.view(b * nv, 1, h, w)
        if self.close_conditioning_invalid:
            is_invalid = morphological_op(
                is_invalid.float(),
                "closing",
                self.close_conditioning_invalid_kernel_size,
            ).bool()
        masks_conditioning[is_invalid] = DEFINITIONS.IS_INVALID

        masks_conditioning_min = min(
            DEFINITIONS.IS_INVALID, DEFINITIONS.IS_OCCLUDED, DEFINITIONS.IS_VISIBLE
        )
        masks_conditioning_max = max(
            DEFINITIONS.IS_INVALID, DEFINITIONS.IS_OCCLUDED, DEFINITIONS.IS_VISIBLE
        )
        masks_conditioning_norm = (masks_conditioning - masks_conditioning_min) / (
            masks_conditioning_max - masks_conditioning_min
        )

        cond_input_type, cond_input_channels = (
            self.cfg.CONTROLNET.MODEL.CONDITIONING_INPUT_TYPE_AND_CHANNELS
        )
        if cond_input_type == "rgb":
            pass
        elif cond_input_type == "rgbd":
            imgs_conditioning = torch.concat(
                [imgs_conditioning, depths_conditioning], dim=1
            )
        elif cond_input_type == "rgbm":
            imgs_conditioning = torch.concat(
                [imgs_conditioning, masks_conditioning_norm], dim=1
            )
        elif cond_input_type == "rgbdm":
            imgs_conditioning = torch.concat(
                [imgs_conditioning, depths_conditioning, masks_conditioning_norm], dim=1
            )
        else:
            raise NotImplementedError(
                f"Conditioning input type {cond_input_type} does not exist."
            )

        if self.cfg.CONTROLNET.MODEL.MASK_CONDITIONING_IMAGE:
            # Set masked pixels in every channel to 0 except in the mask channel if present.
            num_non_mask_channels = (
                cond_input_channels - 1
                if "m" in cond_input_type
                else cond_input_channels
            )
            imgs_conditioning[:, :num_non_mask_channels, :, :] = (
                color_occlusion_masked_img(
                    imgs_conditioning[:, :num_non_mask_channels, :, :],
                    masks_conditioning,
                    c_occl=(
                        "random" if self.set_occlusions_to_random_noise else "zeros"
                    ),  # black
                    c_inv=(
                        "random"
                        if self.set_occlusions_to_random_noise
                        else "zeros" if invalid_to_black else None
                    ),
                )
            )

        # --- APPLY FINAL TRANSFORMS ---
        # Resize such that the smaller image edge matches the desired resolution
        imgs_conditioning = resize(
            imgs_conditioning,
            hw_out,
            interpolation=transforms.InterpolationMode.BILINEAR,
            antialias=None,
        )
        imgs_conditioning = imgs_conditioning.view(
            b, nv, imgs_conditioning.shape[1], *hw_out
        )  # [B, NV, C, H, W]

        valid_idxs = torch.logical_and(
            (masks_conditioning == DEFINITIONS.IS_VISIBLE)
            .to(masks_conditioning.dtype)
            .mean(dim=[1, 2, 3])
            .view(b, nv)
            > self.min_mean_valid_pixels,
            (masks_conditioning == DEFINITIONS.IS_OCCLUDED)
            .to(masks_conditioning.dtype)
            .mean(dim=[1, 2, 3])
            .view(b, nv)
            > self.min_mean_occluded_pixels,
        )

        return imgs_conditioning, masks_conditioning.view(b, nv, 1, h, w), valid_idxs

    def refine_and_predict_depth(self, data: Data, set_generator_none=False):
        assert self.refine_pipeline is not None, "Refine pipeline is None."
        B = data.B
        NV = data.NUM_TOP_K
        h_data, w_data = data.HW_DATA

        device = data.IMGS_CONDITIONING.device
        dtype = data.IMGS_CONDITIONING.dtype

        imgs_conditioning = data.IMGS_CONDITIONING.view(
            -1, *data.IMGS_CONDITIONING.shape[-3:]
        )  # [B*NV, C, H, W]

        shared_kwargs = {
            "prompt": data.CAPTION_STRS,
            "num_inference_steps": self.cfg.CONTROLNET.EVAL.NUM_INFERENCE_STEPS,
            "generator": data.GENERATOR if not set_generator_none else None,
            "output_type": "pt",
            "guidance_scale": self.cfg.CONTROLNET.EVAL.GUIDANCE_SCALE,
        }

        pipe_dtype = self.refine_pipeline.dtype
        if self.cfg.CONTROLNET.MODEL.IS_UNCLIP:

            imgs_refined = self.refine_pipeline(
                control_image=imgs_conditioning.to(pipe_dtype),
                image=data.IMGS_PREPROCESSED.to(
                    pipe_dtype
                ),  # FIXME when using the cascade sampler there is an issue with wrong formatting here
                controlnet_image_embeds_type=self.cfg.CONTROLNET.MODEL.CONTROLNET_IMAGE_EMBEDS_TYPE,
                # noise_level=self.cfg.CONTROLNET.MODEL.NOISE_LEVEL,        # TODO eval the effect of noise during inference
                height=imgs_conditioning.size(-2),
                width=imgs_conditioning.size(-1),
                **shared_kwargs,
            ).images
            # TODO the pipe only outputs np.array for now
            imgs_refined = torch.tensor(imgs_refined).permute(0, 3, 1, 2)
        else:
            imgs_refined = self.refine_pipeline(
                image=imgs_conditioning.to(pipe_dtype), **shared_kwargs
            ).images

        imgs_refined = imgs_refined.to(device, dtype)

        # Resize to the input image resolution
        h_out, w_out = imgs_refined.shape[-2:]
        resize_factor = min(h_data / h_out, w_data / w_out)  # TODO check this
        h_out_new = int(h_out * resize_factor)
        w_out_new = int(w_out * resize_factor)
        imgs_refined = resize(
            imgs_refined,
            (h_out_new, w_out_new),
            interpolation=InterpolationMode.BILINEAR,
            antialias=None,
        )
        imgs_conditioning = resize(
            imgs_conditioning,
            (h_out_new, w_out_new),
            interpolation=InterpolationMode.BILINEAR,
            antialias=None,
        )
        imgs_conditioning = imgs_conditioning.view(B, NV, *imgs_conditioning.shape[-3:])

        # Output depth on refined img
        depths_out = self.depth_predictor(
            imgs_refined, data.PROJS_NV_UNNORMALIZED.view(B * NV, 3, 3)
        )  # n, 1, h, w
        depths_out = depths_out.view(B, NV, 1, h_out_new, w_out_new)
        imgs_refined = imgs_refined.view(B, NV, 3, h_out_new, w_out_new)

        return imgs_refined, depths_out, imgs_conditioning, (h_out_new, w_out_new)

    def _align_depth(
        self,
        depths_to_align: torch.Tensor,
        depths_target: torch.Tensor,
        valid_mask: Optional[torch.Tensor] = None,
        **kwargs,
    ) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        """Depth alignment while ignoring the ."""
        B, NV, _, H, W = depths_to_align.shape

        kwargs_with_attr = {
            "scale_only": self.align_depth_scale_only,
            "mode": self.align_depth_mode,
            "min_valid_fraction": self.align_depth_min_valid_fraction,
        }
        kwargs_with_attr.update(kwargs)

        if self.align_inv_depth:
            depths_to_align, scale, shift = align_inv_depth(
                depths_to_align.reshape(-1, 1, H, W),
                depths_target.reshape(-1, 1, H, W),
                valid_mask.reshape(-1, 1, H, W) if valid_mask is not None else None,
                **kwargs_with_attr,
            )
        else:
            depths_to_align, scale, shift = align_depth(
                depths_to_align.reshape(-1, 1, H, W),
                depths_target.reshape(-1, 1, H, W),
                valid_mask.reshape(-1, 1, H, W) if valid_mask is not None else None,
                **kwargs_with_attr,
            )

        return depths_to_align.reshape(B, NV, 1, H, W), scale, shift

    def _compute_novel_view_scores(
        self, masks: torch.Tensor, invalid: torch.Tensor
    ) -> torch.Tensor:
        """
        Score novel view masks based on whether they have
            1. few invalid pixels
            2. many occluded pixels in the center and few at the edges
        """
        # b, nv, _, _, _ = masks.shape

        # m = masks.clone()
        # m[invalid] = DEFINITIONS.IS_INVALID

        # is_visible = (m == DEFINITIONS.IS_VISIBLE).to(m.dtype)
        # is_occluded = (m == DEFINITIONS.IS_OCCLUDED).to(m.dtype)

        # scores = torch.zeros((b, nv), device=masks.device, dtype=masks.dtype)

        # enough_valid = is_visible.mean(dim=[2, 3, 4]) > self.min_mean_valid_pixels
        # scores[enough_valid] += 1.
        # fraction_px_occluded = is_occluded.mean(dim=[2, 3, 4])

        # edges_error = compute_occlusion_edges_error(is_occluded, torch.zeros_like(is_occluded), invalid)
        # center_error = compute_occlusion_center_error(is_occluded, torch.ones_like(is_occluded), invalid)

        # scores = scores + fraction_px_occluded * 0.5 - edges_error - center_error

        # return scores
        b, nv, _, _, _ = masks.shape

        m = masks.clone()
        m[invalid] = DEFINITIONS.IS_INVALID

        is_visible = (m == DEFINITIONS.IS_VISIBLE).to(m.dtype)
        is_occluded = (m == DEFINITIONS.IS_OCCLUDED).to(m.dtype)

        scores = torch.zeros((b, nv), device=masks.device, dtype=masks.dtype)

        enough_valid = is_visible.mean(dim=[2, 3, 4]) > self.min_mean_valid_pixels
        scores[enough_valid] += 1.0
        enough_occluded = (
            is_occluded.mean(dim=[2, 3, 4]) > self.min_mean_occluded_pixels
        )
        scores[enough_occluded] += 1.0
        # What fraction of pixels are occluded and above the threshold
        fraction_px_occluded = is_occluded.mean(dim=[2, 3, 4])
        overocclusion_error = (
            (fraction_px_occluded > self.max_mean_occluded_pixels)
            * (self.max_mean_occluded_pixels - fraction_px_occluded)
            * 10
        )
        # Low error if few occluded pixels are at the edges
        edges_error = compute_occlusion_edges_error(
            is_occluded, torch.zeros_like(is_occluded), invalid
        )
        # Low error if many occluded pixels are in the center
        center_error = compute_occlusion_center_error(
            is_occluded, torch.ones_like(is_occluded), invalid
        )

        scores = (
            scores
            + fraction_px_occluded
            + overocclusion_error
            - edges_error
            - 0.5 * center_error
        )

        return scores

    def _keep_top_k(self, data: Data, scores: torch.Tensor) -> Data:
        N, NV = scores.shape

        if not self.top_k_novel_views_to_keep or self.top_k_novel_views_to_keep >= NV:
            return data

        k = self.top_k_novel_views_to_keep

        # Creating a new data object
        top_k_data = {}
        for field in fields(data):
            value = getattr(data, field.name)
            if isinstance(value, torch.Tensor) and value.shape[:2] == (N, NV):
                # If tensor matches the batch and novel view dimensions
                top_k_values = torch.empty_like(
                    value[:, :k]
                )  # Prepare tensor to store top k entries

                for n in range(N):
                    _, indices = torch.topk(
                        scores[n], k
                    )  # Get indices of the top k scores
                    top_k_values[n] = value[n, indices]  # Retrieve top k values

                top_k_data[field.name] = top_k_values
            elif (
                isinstance(value, (tuple, list))
                and len(value)
                and isinstance(value[0], torch.Tensor)
                and value[0].shape[:2] == (N, NV)
            ):
                # If it is a non-empty list of tensors with the required shape
                new_values_list = []
                for i, sub_tensor in enumerate(value):
                    top_k_sub_tensor = torch.empty_like(
                        sub_tensor[:, :k]
                    )  # Prepare tensor to store top k entries
                    for n in range(N):
                        _, indices = torch.topk(
                            scores[n], k
                        )  # Get indices of the top k scores
                        top_k_sub_tensor[n] = sub_tensor[
                            n, indices
                        ]  # Retrieve top k values
                    new_values_list.append(top_k_sub_tensor)
                top_k_data[field.name] = new_values_list
            else:
                # Copy other fields as they are
                top_k_data[field.name] = value

        # Return new Data instance with only top k entries kept
        return Data(**top_k_data)

    @staticmethod
    def merge_dicts_based_on_valid_idxs(
        datas: List[Data], valid_idxs: List[torch.Tensor]
    ) -> Data:
        N, NV = valid_idxs[0].shape

        # Clone initial tensors to preserve structure
        out = {}
        for field in fields(datas[0]):
            value = getattr(datas[0], field.name)
            if isinstance(value, torch.Tensor):
                out[field.name] = value.clone()
            elif isinstance(value, (tuple, list)):
                # If it is a list of tensors
                if len(value) and isinstance(value[0], torch.Tensor):
                    out[field.name] = [v.clone() for v in value]
                else:
                    out[field.name] = value
            else:
                out[field.name] = value
        out = Data(**out)

        out_valid = valid_idxs[0].clone()
        valid_idxs_merged = torch.cat(valid_idxs[1:], dim=1)  # [N, NV * num_dicts]
        invalid_idxs_0 = ~valid_idxs[0]

        def merge_concat_tensor(selected, concat_tensor):
            for n in range(N):
                # Collect indices of valid values up to NV
                valid_indices_n = valid_idxs_merged[n].nonzero(as_tuple=True)[0]
                num_valid_n = valid_indices_n.size(0)

                invalid_idxs_0_n = invalid_idxs_0[n].nonzero(as_tuple=True)[0]
                num_invalid_idxs_0_n = invalid_idxs_0_n.size(0)

                if num_valid_n > 0:
                    # Only replace (1) a max of NV values; (2) the invalid values in the 0th dict
                    valid_indices_n = valid_indices_n[
                        : min(NV, num_invalid_idxs_0_n, num_valid_n)
                    ]
                    invalid_idxs_0_n = invalid_idxs_0_n[: len(valid_indices_n)]
                    # Fill in the valid entries for this batch element
                    selected[n, invalid_idxs_0_n] = concat_tensor[n, valid_indices_n]
                    out_valid[n, invalid_idxs_0_n] = True
            return selected

        for field in fields(datas[0]):
            value = getattr(datas[0], field.name)
            if isinstance(value, torch.Tensor) and value.shape[:2] == (N, NV):
                # Concatenate all tensors across the dictionaries for each key
                ds = torch.cat(
                    [getattr(d, field.name) for d in datas[1:]], dim=1
                )  # [N, NV * num_dicts, ...]

                # Initialize with original data to retain in case of insufficient valid values
                # selected = value.clone()

                # for n in range(N):
                #     # Collect indices of valid values up to NV
                #     valid_indices_n = valid_idxs_merged[n].nonzero(as_tuple=True)[0]
                #     num_valid_n = valid_indices_n.size(0)

                #     invalid_idxs_0_n = invalid_idxs_0[n].nonzero(as_tuple=True)[0]
                #     num_invalid_idxs_0_n = invalid_idxs_0_n.size(0)

                #     if num_valid_n > 0:
                #         # Only replace (1) a max of NV values; (2) the invalid values in the 0th dict
                #         valid_indices_n = valid_indices_n[:min(NV, num_invalid_idxs_0_n, num_valid_n)]
                #         invalid_idxs_0_n = invalid_idxs_0_n[:len(valid_indices_n)]
                #         # Fill in the valid entries for this batch element
                #         selected[n, invalid_idxs_0_n] = ds[n, valid_indices_n]
                #         out_valid[n, invalid_idxs_0_n] = True

                selected = merge_concat_tensor(value.clone(), ds)
                # Assign the processed tensor back to the output dataclass
                setattr(out, field.name, selected)
            elif (
                isinstance(value, (tuple, list))
                and len(value)
                and isinstance(value[0], torch.Tensor)
                and value[0].shape[:2] == (N, NV)
            ):
                # If it is a non-empty list of tensors with shape [N, NV, ...]
                new_value = []
                for i in range(len(value)):
                    ds = torch.cat(
                        [getattr(d, field.name)[i] for d in datas[1:]], dim=1
                    )  # [N, NV * num_dicts, ...]
                    selected = merge_concat_tensor(value[i].clone(), ds)
                    new_value.append(selected)
                setattr(out, field.name, selected)

        return out, out_valid
