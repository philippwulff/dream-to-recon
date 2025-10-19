from hydra.core.config_store import ConfigStore
from omegaconf import OmegaConf, DictConfig

from configs.structured_configs.main_config import MainConfig
from configs.structured_configs.data_config import DataConfigKitti360, DataConfigCO3D, DataConfigConcatCO3D, DataConfigDAVIS, DataConfigWaymo
from configs.structured_configs.bts_config import SchedulerConfig
from configs.structured_configs.synthetic_gt_config import ALL_CAMERA_SAMPLER_CONFIGS
from configs.structured_configs.loss_config import ALL_LOSS_CONFIGS


def register_default_configs():
    """Registers dataclass configs to Hydra's config store."""
    
    cs = ConfigStore.instance()
    cs.store(name="base_main_config", node=MainConfig)
    for g in ["CONTROLNET/DATA", "BTS/DATA"]:
        cs.store(group=g, name="kitti_360", node=DataConfigKitti360)
        cs.store(group=g, name="co3d", node=DataConfigCO3D)
        cs.store(group=g, name="co3d_concat", node=DataConfigConcatCO3D)
        cs.store(group=g, name="davis", node=DataConfigDAVIS)
        cs.store(group=g, name="waymo", node=DataConfigWaymo)
    
    for cfg in ALL_CAMERA_SAMPLER_CONFIGS:
        if hasattr(cfg, "TYPE"):
            cs.store(group="SYNTHETIC_GT/NV_CAM_SAMPLER", name=cfg.UNIQUE_NAME if cfg.UNIQUE_NAME else cfg.TYPE, node=cfg)

    for cfg in ALL_LOSS_CONFIGS:
        cs.store(group=f"BTS/LOSSES/{cfg.TYPE}", name=cfg.TYPE, node=cfg)
        
    # TODO add schedulers
        

def check_and_post_init_config(cfg: DictConfig) -> MainConfig:
    """Makes sure that the selected config is valid."""
    
    cfg = OmegaConf.to_object(cfg)

    if cfg.CONTROLNET.TRAIN.RESOLUTION[0] % 8 != 0 and cfg.CONTROLNET.TRAIN.RESOLUTION[1] % 8 != 0:
        raise ValueError(
            "`RESOLUTION` must be divisible by 8 for consistently sized encoded images between the VAE and the controlnet encoder."
        ) 
    
    return cfg
    