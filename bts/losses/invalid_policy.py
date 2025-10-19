import torch
import torch.nn as nn
from typing import Any


class InvalidPolicy(nn.Module):
    def __init__(self, policy: str = "strict", *args, **kwargs) -> None:
        super().__init__(*args, **kwargs)
        self.policies = {
            "none": self.no_policy,
            "strict": self.strict_policy,
            "weight_guided": self.weight_guided_policy,
            "weight_guided_diverse": self.weight_guided_diverse_policy,
        }
        try:
            self.invalid_policy = self.policies[policy]
        except KeyError:
            raise ValueError(f"Unknown invalid policy: {policy}")

    def no_policy(invalids: torch.Tensor, **kwargs) -> torch.Tensor:
        invalid = torch.zeros_like(
            torch.all(torch.any(invalids > 0.5, dim=-2), dim=-1).unsqueeze(-1),
            dtype=torch.bool,
        )
        return invalid
    
    @staticmethod
    def strict_policy(invalids: torch.Tensor, **kwargs: Any) -> torch.Tensor:
        """Consider all rays invalid where there is at least one invalidly sampled color."""
        invalid = torch.all(torch.any(invalids > 0.5, dim=-2), dim=-1).unsqueeze(-1)
        return invalid

    @staticmethod
    def weight_guided_policy(invalids: torch.Tensor, **kwargs) -> torch.Tensor:
        """Integrate invalid indicator function over the weights. It is invalid if > 90% of the mass is invalid. (Arbitrary threshold)"""
        weights = kwargs["weights"]
        invalid = torch.all(
            (invalids.to(torch.float32) * weights.unsqueeze(-1)).sum(-2) > 0.9,
            dim=-1,
            keepdim=True,
        )
        return invalid

    @staticmethod
    def weight_guided_diverse_policy(invalids: torch.Tensor, **kwargs) -> torch.Tensor:
        """We now also consider, whether there is enough variance in the ray colors to give a meaningful supervision signal."""
        rgb_samps = kwargs["rgb_samps"]
        ray_std = torch.std(rgb_samps, dim=-3).mean(-1)
        weights = kwargs["weights"]
        # Integrate invalid indicator function over the weights. 
        # It is invalid if > 90% of the mass is invalid. (Arbitrary threshold)
        invalid = torch.all(
            ((invalids.to(torch.float32) * weights.unsqueeze(-1)).sum(-2) > 0.9)
            | (ray_std < 0.01),
            dim=-1,
            keepdim=True,
        )
        return invalid     
    
    def forward(self, invalids: torch.Tensor, **kwargs) -> torch.Tensor: 
        """Applies invalid policy."""
        return self.invalid_policy(invalids, **kwargs)
