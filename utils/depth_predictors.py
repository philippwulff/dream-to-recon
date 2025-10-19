import torch
import torch.nn as nn
import torch.nn.functional as F
from typing import Literal

from configs.structured_configs.synthetic_gt_config import SyntheticGTConfig


class Metric3DWrapper(nn.Module):
    """
    Wrapper for inference with Metric3d from torch hub.

    Following:
        https://github.com/YvanYin/Metric3D/blob/16aeb7fe176cd0ecaf6f8bc0af288c62e978354c/hubconf.py#L145
    """
    def __init__(self, *args, **kwargs) -> None:
        super().__init__()
        # https://github.com/YvanYin/Metric3D/tree/main?tab=readme-ov-file#news-pytorch-hub-is-supported
        self.arch: Literal[
            "metric3d_convnext_tiny", 
            "metric3d_convnext_large", 
            "metric3d_vit_small", 
            "metric3d_vit_large", 
            "metric3d_vit_giant2"
        ] = "metric3d_vit_large"
        if "vit" in self.arch:
            self.input_size = (616, 1064)
        elif "convnext" in self.arch:
            self.input_size = (544, 1216)
        else:
            raise ValueError(f"Unknown architecture {self.arch}")
        self.input_size
        self.model = torch.hub.load('yvanyin/metric3d', self.arch, pretrain=True, trust_repo=True, force_reload=False)

    def clear_buffers(self):
        """
        Clears some buffered variables in the Metric3D model that depend on batch size.
        This needs to be run between inference steps with variable batch sizes.
        """
        name = "depth_expectation_anchor"
        if name in self.model.depth_model.decoder._buffers:
            del self.model.depth_model.decoder._buffers[name]

    def prepare_inputs(self, rgb_origin: torch.Tensor, intrinsics_origin: torch.Tensor):
        # intrinsic = [707.0493, 707.0493, 604.0814, 180.5066]
        gt_depth_scale = 256.0
        # rgb_origin = cv2.imread(rgb_file)[:, :, ::-1]

        #### ajust input size to fit pretrained model
        # keep ratio resize
        h_origin, w_origin = rgb_origin.shape[-2:]
        scale = min(self.input_size[0] / h_origin, self.input_size[1] / w_origin)
        # rgb = cv2.resize(rgb_origin, (int(w * scale), int(h * scale)), interpolation=cv2.INTER_LINEAR)

        rgb = F.interpolate(
            rgb_origin, size=(int(h_origin * scale), int(w_origin * scale)), mode='bilinear', align_corners=False
        )
        h, w = rgb.shape[-2:]
        
        mean = torch.tensor([123.675, 116.28, 103.53])[None, :, None, None].to(rgb)
        std = torch.tensor([58.395, 57.12, 57.375])[None, :, None, None].to(rgb)
        # rgb = torch.from_numpy(rgb.transpose((2, 0, 1))).float()
        rgb = torch.div((rgb - mean), std)

        # remember to scale intrinsic, hold depth
        # intrinsic = [intrinsic[0] * scale, intrinsic[1] * scale, intrinsic[2] * scale, intrinsic[3] * scale]
        intrinsics = intrinsics_origin * scale
        # Pad to input_size with the mean value
        # padding = [123.675, 116.28, 103.53]
        pad_h = self.input_size[0] - h
        pad_w = self.input_size[1] - w
        pad_h_half = pad_h // 2
        pad_w_half = pad_w // 2
        pad_info = [pad_h_half, pad_h - pad_h_half, pad_w_half, pad_w - pad_w_half]
        # rgb = cv2.copyMakeBorder(rgb, pad_h_half, pad_h - pad_h_half, pad_w_half, pad_w - pad_w_half, cv2.BORDER_CONSTANT, value=padding)
        rgb = F.pad(rgb, (pad_w_half, pad_w - pad_w_half, pad_h_half, pad_h - pad_h_half), value=0)

        return rgb, intrinsics, pad_info, gt_depth_scale
        #### normalize

    # Running with float16 returns NaNs
    # @torch.cuda.amp.custom_fwd(cast_inputs=torch.float32)
    def forward(self, imgs: torch.Tensor, projs: torch.Tensor) -> torch.Tensor:
        """Returns depth from Metric3d.

        Args:
            imgs (torch.Tensor): [B, 3, H, W]
            projs (torch.Tensor): [B, 3, 3]
        """
        self.clear_buffers()
        B, _, ori_h, ori_w = imgs.shape
        
        # Move from [-1, 1] to [0, 255]
        imgs = (imgs * 0.5 + 0.5) * 255
        # rgb_inputs, cam_models_stacks, pad_infos, scale_infos = self.transform_data(imgs, projs)
        rgb_inputs, intrinsics, pad_info, gt_depth_scale = self.prepare_inputs(imgs, projs)

        with torch.no_grad():
            pred_depths, confidence, output_dict = self.model.inference({'input': rgb_inputs})

        # un pad
        pred_depths = pred_depths[:, :, pad_info[0] : pred_depths.shape[-2] - pad_info[1], pad_info[2] : pred_depths.shape[-1] - pad_info[3]]
        # upsample to original size
        pred_depths = torch.nn.functional.interpolate(pred_depths, imgs.shape[-2:], mode='bilinear')

        # de-canonical transform
        canonical_to_real_scale = (intrinsics[:, 0, 0] + intrinsics[:, 1, 1]) * .5 / 1000.0 # 1000.0 is the focal length of canonical camera
        pred_depths = pred_depths * canonical_to_real_scale[:, None, None, None] # now the depth is metric
        pred_depths = torch.clamp(pred_depths, 0, 300)
        
        return pred_depths.to(imgs)


class UniDepthWrapper(nn.Module):
    """
    Providing intrinsics to UniDepth will make it use those instead of the estimated ones:
    https://github.com/lpiccinelli-eth/UniDepth/issues/50#issuecomment-2143361276
    """
    def __init__(self, compile=False, *args, **kwargs) -> None:
        super().__init__(*args, **kwargs)
        self.version="v1"
        self.backbone="ViTL14"
        # Setting `force_reload=True` results in many different OSError's.
        self.model = torch.hub.load("lpiccinelli-eth/UniDepth", "UniDepth", version=self.version, backbone=self.backbone, pretrained=True, trust_repo=True, force_reload=False)

        if compile:
            pass
            # self.model = torch.compile(self.model, mode="max-autotune", fullgraph=False)

    def forward(self, rgbs: torch.Tensor, intrinsics: torch.Tensor) -> torch.Tensor:
        """
        Args:
            rgbs (torch.Tensor): [B, 3, H, W]
            intrinsics ([torch.Tensor]): [B, 3, 3]
        """
        # # Changing the image resolution
        # # https://github.com/lpiccinelli-eth/UniDepth/issues/30#issuecomment-2067989034
        # H, W = rgbs.shape[-2:]
        # new_H = (H // 14) * 14 if H % 14 != 0 else H
        # new_W = (W // 14) * 14 if W % 14 != 0 else W
        # if new_H != H or new_W != W:
        #     rgbs = F.interpolate(rgbs, size=(new_H, new_W), mode='bilinear', align_corners=False)
        # self.model.image_shape = (new_H, new_W)

        # Outputs are FP32 even if the inputs are FP16
        out = self.model.infer(rgbs, intrinsics)
        depth = out["depth"].to(rgbs.dtype)

        # if new_H != H or new_W != W:
        #     depth = F.interpolate(depth, size=(H, W), mode='bilinear', align_corners=False)

        return depth


def make_depth_predictor(cfg: SyntheticGTConfig, **kwargs):
    cls_name = cfg.DEPTH_PREDICTOR_NAME + "Wrapper"
    try:
        cls = globals()[cls_name]
    except KeyError:
        raise ValueError(f"Depth predictor {cls_name} is not available.")
    return cls(**kwargs)


# TODO
# https://github.com/isl-org/ZoeDepth
