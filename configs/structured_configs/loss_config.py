import os
from typing import Any, Dict, Optional, Literal, Tuple, List
from dataclasses import dataclass, field
from omegaconf import MISSING


@dataclass
class LossConfig:
    # Class name used for loss construction
    TYPE: str
    
    
@dataclass
class DepthLossConfig(LossConfig):
    TYPE: str = "DepthReconstructionLoss"
    criterion: str = "gnll"
    invalid_policy: str = "none"
    lambda_entropy: float = 0.
    lambda_depth_reg: float = 0.
    lambda_alpha_reg: float = 0.
    lambda_surfaceness_reg: float = 0.
    lambda_edge_aware_smoothness: float = 0.
    lambda_depth_smoothness: float = 0.
    lambda_coarse: float = 1.0
    lambda_fine: float = 1.0
    lambda_var: float = 0.0
    median_thresholding: bool = False
    alpha_reg_reduction: str = "ray"
    alpha_reg_fraction: float = 1/8
    # Trades off input image loss (lambda_in) vs. novel view losses (1-lambda_in).
    LAMBDA_IN: float = 0.5
    

@dataclass
class WeightRegularizationLossConfig(LossConfig):
    TYPE: str = "WeightRegularizationLoss"
    criterion: str = "l2"
    LAMBDA_WEIGHT_REG: float = 0.01
    

@dataclass
class DensityGridRegularizationLossConfig(LossConfig):
    TYPE: str = "DensityGridRegularizationLoss"
    LAMBDA_REG: float = 1.0
    THRESHOLD: float = 16.0


@dataclass
class DensityGridLossConfig(LossConfig):
    TYPE: str = "DensityGridLoss"
    CRITERION: str = "l2"
    WEIGHT: float = 1.0
    WEIGHT_OCCL_AND_EMPTY: float = 1.0
    EPS: float = 1.0
    ALIGN: bool = False
    ALIGN_SHIFT: bool = False


@dataclass
class RGBLossConfig(LossConfig):
    pass


@dataclass
class MainLossConfig():
    DEPTH_LOSS: DepthLossConfig = DepthLossConfig()
    WEIGHT_REG: WeightRegularizationLossConfig = WeightRegularizationLossConfig()
    
    def get_enabled_losses(self):
        return [_ for _ in vars(self) if issubclass(_, LossConfig) and _.ENABLED]
    
    
ALL_LOSS_CONFIGS = [
    DepthLossConfig,
    WeightRegularizationLossConfig,
    DensityGridRegularizationLossConfig,
    DensityGridLossConfig
]