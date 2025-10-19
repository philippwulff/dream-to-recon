from abc import ABC, abstractmethod
from utils.utils import asdict_lowercase_keys_override
from configs.structured_configs.loss_config import LossConfig

import torch


class BaseLoss(ABC):
    def __init__(self, **kwargs) -> None:
        super().__init__()

    @classmethod 
    def from_conf(cls, cfg: LossConfig, **kwargs):
        return cls(**asdict_lowercase_keys_override(cfg, **kwargs))

    @abstractmethod
    def get_loss_metric_names(self) -> list[str]:
        """
        Returns a list with the keys in the returned dict from the __call__ method.
        This is used to report losses during validation.
        """
        raise NotImplementedError

    @abstractmethod
    def __call__(self, data, **kwargs) -> tuple[torch.Tensor, dict[str, torch.Tensor]]:
        raise NotImplementedError
