import torch
import torch.nn as nn
from typing import List
from bts.losses.base_loss import BaseLoss
from torch import profiler


class WeightRegularizationLoss(BaseLoss):
    def __init__(
        self, 
        criterion: str = "l2",
        lambda_weight_reg: float = 0.0,
        **kwargs,
        ) -> None:
        super().__init__()
        self.lambda_weight_reg = lambda_weight_reg
        match criterion:
            case "l2":
                self.crit = lambda x: torch.linalg.vector_norm(x, ord=2)
            case "l1":
                self.crit = lambda x: torch.linalg.vector_norm(x, ord=1)
            case _:
                raise ValueError(f"Criterion {criterion} not available.")

    @staticmethod
    def get_loss_metric_names():
        return ["weight_reg"]

    def __call__(self, data: dict, parameters: List[nn.Parameter], **kwargs):
        loss_dict = {}
        with profiler.record_function("loss_computation"):
            
            loss = torch.tensor(0.0, device=list(parameters)[0].device)
            for p in parameters:
                if p is not None and p.requires_grad:
                    loss += self.crit(p)
            
            loss *= self.lambda_weight_reg
            
            loss_dict["weight_reg"] = loss.item()

        return loss, loss_dict
