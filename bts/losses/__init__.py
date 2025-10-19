from typing import List
from .base_loss import BaseLoss
from .depth_loss import DepthReconstructionLoss
from .weight_regul import WeightRegularizationLoss
from .density_grid_regul import DensityGridRegularizationLoss
from .density_grid_loss import DensityGridLoss
# from .rgb_loss import 
from configs.structured_configs.loss_config import LossConfig
from omegaconf import DictConfig



def make_loss(cfg: LossConfig, **kwargs) -> BaseLoss:
    try:
        if not hasattr(cfg, "TYPE"):
            raise AttributeError("Loss config must have a TYPE field")
        loss_cls = globals()[cfg.TYPE]
    except KeyError:
        raise ValueError(f"Loss type {cfg.TYPE} is not available.")
    
    return loss_cls.from_conf(cfg, **kwargs)


def make_losses(cfgs: DictConfig, **kwargs) -> List[BaseLoss]:
    losses = []
    for k, v in cfgs.items():
        try:
            losses.append(make_loss(v, **kwargs))
        except AttributeError as e:
            # TODO if a child config removes a loss via override, but the parent sets fields, this exists. 
            # Should clean up the config after laoding.
            pass           
    return losses




