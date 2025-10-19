import torch
import torch.nn as nn
from typing import List
from bts.losses.base_loss import BaseLoss
from torch import profiler


class DensityGridRegularizationLoss(BaseLoss):
    def __init__(
        self, 
        threshold: float = 0.0,
        lambda_reg: float = 0.0,
        return_stats: bool = False,
        **kwargs,
        ) -> None:
        super().__init__()
        self.lambda_reg = lambda_reg
        self.threshold = threshold
        self.return_stats = return_stats
        
    def get_loss_metric_names(self):
        return ["density_grid_reg"]

    def __call__(self, data: dict, **kwargs):
        loss_dict = {}
        with profiler.record_function("loss_computation"):
            density_grid = data["density_grid"]
                
            if self.return_stats:
                loss_dict["density_grid_abs_max"] = density_grid.abs().max().item()
                loss_dict["density_grid_abs_mean"] = density_grid.abs().mean().item()
                loss_dict["density_grid_abs_median"] = density_grid.abs().median().item()
            
            density_grid = (density_grid.abs() - self.threshold).clamp_min(0)

            # Attempt to make it more numerically stable
            max_v = density_grid.max().clamp_min(1)
            max_v = max_v.detach()

            loss = (((density_grid / max_v)).mean() * max_v)
            loss = torch.nan_to_num(loss, 0, 0, 0)

            # Black magic to prevent error messages from anomaly detection when using AMP
            if torch.all(loss == 0):
                loss = loss.detach()
            
            loss *= self.lambda_reg
            
            loss_dict["loss_density_grid"] = loss.item()

        return loss, loss_dict
