from dataclasses import is_dataclass
from configs.structured_configs.synthetic_gt_config import CameraSamplerConfig
from bts.gt_synthesis.gt_pose_sampler import *


def make_camera_sampler(cfg: CameraSamplerConfig, **kwargs):
    """Helper"""
    try:
        key = cfg.TYPE if is_dataclass(cfg) else cfg["TYPE"]
        pose_sampler_cls: CameraSampler = globals()[key]
    except KeyError:
        raise ValueError(f"Camera sampler type {key} is invalid.")
    
    pose_sampler = pose_sampler_cls.from_conf(cfg, **kwargs)
    
    return pose_sampler