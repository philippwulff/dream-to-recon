from typing import Optional, List, Dict
import torch
import torch.nn as nn
from bts.gt_synthesis.gt_synthesis import GTSynthesisWrapper, Outputs, Data, Tuple
from configs.structured_configs.main_config import MainConfig
from configs.structured_configs.synthetic_gt_config import TrajectoryCameraSamplerConfig, CameraSamplerConfig
from bts.gt_synthesis.gt_pose_sampler import TrajectoryCameraSampler
from bts.gt_synthesis.make_camera_sampler import make_camera_sampler
from tqdm.auto import tqdm
from utils.occlusion_ops import DEFINITIONS
from utils.plotting import render_profile
from utils.utils import asdict_lowercase_keys_override


class CascadeWrapper(nn.Module):
    def __init__(
        self, 
        config: MainConfig,
        gt_synthesizer: GTSynthesisWrapper,
        trajectory_sampler_config: TrajectoryCameraSamplerConfig,
        anchor_sampler_configs: Optional[List[CameraSamplerConfig]] = None,
        encoding_policy: str = "input",
        show_progress_bar: bool = False,
        cam_incl_adjust: Optional[torch.Tensor] = None,
        non_occlusion_masks: bool = False,
        anchor_seed: Optional[int] = None,
        cascade_seed: Optional[int] = None,
        *args, 
        **kwargs,
        ) -> None:
        super().__init__()
        # super().__init__(*args, **kwargs)
                
        self.config = config
        self.gt_synthesizer = gt_synthesizer
        
        if anchor_sampler_configs is not None:
            self.anchor_camera_samplers = [make_camera_sampler(cfg, cam_incl_adjust=cam_incl_adjust) for cfg in anchor_sampler_configs]
        else:
            self.anchor_camera_samplers = [gt_synthesizer.camera_sampler]       # Do not copy here, since this messes up the cascade vis
            
        self.cascade_camera_sampler = make_camera_sampler(trajectory_sampler_config, cam_incl_adjust=cam_incl_adjust)
        assert isinstance(self.cascade_camera_sampler, TrajectoryCameraSampler), "CASCADE CAMERA SAMPLER SHOULD BE A TrajectoryCameraSampler type."

        self.non_occlusion_masks = non_occlusion_masks
        self.encoding_policy = encoding_policy
        self.show_progress_bar = show_progress_bar
        self.cam_incl_adjust = cam_incl_adjust
        self.anchor_seed = anchor_seed
        self.cascade_seed = cascade_seed

    @classmethod
    def from_conf(cls, config: MainConfig, cam_incl_adjust: Optional[torch.Tensor] = None, gt_synthesizer: Optional[GTSynthesisWrapper] = None, **kwargs):
        if gt_synthesizer is None:
            gt_synthesizer = GTSynthesisWrapper.from_conf(config, cam_incl_adjust, refiner=True)
        
        return cls(
            config=config,
            cam_incl_adjust=cam_incl_adjust,
            gt_synthesizer=gt_synthesizer,
            anchor_sampler_configs=config.SYNTHETIC_GT.CASCADE.ANCHOR_SAMPLER_CONFIGS,      # Property is not kept when calling `asdict(config)`
            **asdict_lowercase_keys_override(config.SYNTHETIC_GT.CASCADE, **kwargs),
        )

    def to(self, *args, **kwargs):
        super().to(*args, **kwargs)
        self.gt_synthesizer.to(*args, **kwargs)
        return self


    def forward(
        self, 
        images: torch.Tensor, 
        poses: torch.Tensor, 
        projs: torch.Tensor, 
        anchor_seed: Optional[int] = None, 
        cascade_seed: Optional[int] = None, 
        return_profile: bool = False,
        **kwargs,
        ) -> Tuple[List[Outputs], List[Data], Dict[str, torch.Tensor]]:
        """Calls SyntheticGTConfig.forward in cascaded manner.

        Args:
            images (torch.Tensor): [B, 1, 3, H, W]
            poses (torch.Tensor): [B, 1, 4, 4]
            projs (torch.Tensor): [B, 1, 3, 3]
            depths (Optional[torch.Tensor], optional): [B, N, 1, H, W]. Defaults to None.
            seed (int, optional): Seed for deterministic behavior. Defaults to None.
        """
        if anchor_seed is None and not self.anchor_seed is None:
            anchor_seed = self.anchor_seed
        if cascade_seed is None and not self.cascade_seed is None:
            cascade_seed = self.cascade_seed

        out_prev: List[Outputs] = []
        data_prev: List[Data] = []
        debug_dicts = []
        profiles = []
        
        num_inputs_from_data = images.size(1)
        num_inputs = 1
        input_idxs = list(range(num_inputs))
        input_idxs = [0]
        num_nv_anchors = len(self.anchor_camera_samplers)
        num_steps = num_nv_anchors + self.cascade_camera_sampler._num_steps
        self.cascade_camera_sampler.reset()
        
        for i in tqdm(range(num_steps), desc="Cascade Steps", disable=not self.show_progress_bar):
            
            match self.encoding_policy:
                case "input":
                    # Encode with the input images
                    idxs = input_idxs    # [0]
                case "anchor":
                    # Encode with the anchor images
                    # idxs = torch.arange(num_input_anchors, num_anchors)
                    raise NotImplementedError("GTSynthesisWrapper needs the 0th camera to be the input camera.")
                case "prev":
                    # Encode only with the images from the previous iteration of the cascade.
                    # Use the input anchors on the first iteration.
                    raise NotImplementedError("GTSynthesisWrapper needs the 0th camera to be the input camera.")
                case "input+anchor":
                    # Encode with the input + anchor images
                    anchor_idxs = [j for j in range(num_inputs, num_inputs+num_nv_anchors) if j<=i]
                    idxs = input_idxs + anchor_idxs
                case "input+prev":
                    # Encode with the input images + the images from the previous iteration of the cascade
                    prev_idxs = [] if i == 0 else [-1]
                    idxs = input_idxs + prev_idxs
                case "input+anchor+prev":
                    # Encode with input and anchor images as well as the last frames in the trajectory.
                    anchor_idxs = [j for j in range(num_inputs, num_inputs+num_nv_anchors) if j<=i]
                    prev_idxs = [] if i <= num_nv_anchors else [-1]
                    idxs = input_idxs + anchor_idxs + prev_idxs
                case _:
                    raise ValueError(f"Encoding policy {self.encoding_policy} is not available.")
            
            imgs_enc, poses_enc, projs_enc, depths_enc, masks_enc = [], [], [], [], []
            for j in idxs:
                if j == 0:
                    # Add the inputs to the encoding lists
                    imgs_enc.append(images)
                    poses_enc.append(poses)
                    projs_enc.append(projs)
                    masks_enc.append(torch.ones_like(images[:, :, :1]))
                    # if self.non_occlusion_masks:
                    #     masks_enc.append(torch.zeros_like(images)[:, :, :1].bool())
                    if len(out_prev) > 0:
                        depths_enc.append(out_prev[0].DEPTHS_IN)
                    continue
                if i == 0:
                    # There are no previous outputs on the first cascade step.
                    continue
                # Add previous outputs to the encoding lists
                # self.gt_synthesizer.renderer.net._num_input_imgs = 3       # TODO is disabling this correct?
                out_jm1 = out_prev[j-1]
                data_jm1 = data_prev[j-1]
                version_idx = 0
                imgs_enc.append(out_jm1.IMGS_SYNTH[version_idx] * 2. - 1.)
                poses_enc.append(out_jm1.POSES_OUT)
                projs_enc.append(out_jm1.PROJS_OUT)
                depths_enc.append(out_jm1.DEPTHS_SYNTH[version_idx])
                masks_enc.append((data_jm1.MASKS_NV == DEFINITIONS.IS_OCCLUDED).float())
                # if self.non_occlusion_masks:
                #     masks_enc.append(make_non_occlusion_masks(out_jm1.DEPTHS_SYNTH[version_idx], data_jm1.DEPTHS_NV, data_jm1.MASKS_NV))
            
            if i == 0:
                self.gt_synthesizer.renderer.net._num_input_imgs = num_inputs_from_data
            
            if i < num_nv_anchors:
                self.gt_synthesizer.camera_sampler = self.anchor_camera_samplers[i]
                seed = anchor_seed
            else:
                self.gt_synthesizer.camera_sampler = self.cascade_camera_sampler
                seed = cascade_seed

            out, data, debug_dict = self.gt_synthesizer(imgs_enc, poses_enc, projs_enc, depths_enc, output_in_nv=True, refine_output=True, seed=seed, images_non_occlusion_masks=masks_enc, **kwargs)

            if return_profile:
                profiles.append(render_profile(self.gt_synthesizer.renderer.net, self.config.BTS.DATA.VIS_VOLUME, cam_incl_adjust=self.cam_incl_adjust.to(images)))

            out_prev.append(out)
            data_prev.append(data)
            debug_dicts.append(debug_dict)
                
        debug = {
            "profiles": profiles,
            "num_nv_anchors": num_nv_anchors,
            "debug_dicts": debug_dicts,
        }
        return out_prev, data_prev, debug
