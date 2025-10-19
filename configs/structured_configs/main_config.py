import os
from typing import Any, Optional
from datetime import datetime
from dataclasses import dataclass
from omegaconf import MISSING

from configs.structured_configs.controlnet_config import ControlnetConfig
from configs.structured_configs.bts_config import BTSConfig
from configs.structured_configs.profiler_config import ProfilerConfig
from configs.structured_configs.synthetic_gt_config import SyntheticGTConfig
from configs.structured_configs.eval_config import OccupancyEvalConfig

@dataclass
class AMPConfig:
    """Torch automatic mixed precision"""
    ENABLED: bool = False
    # https://github.com/pytorch/pytorch/issues/40497#issuecomment-1262373602
    GROWTH_INTERVAL: int = 2000
    # https://pytorch.org/docs/stable/amp.html#gradient-scaling
    MIN_SCALE: Optional[int] = None
    # AMP training tricks: 
    # https://discuss.pytorch.org/t/training-tricks-to-improve-stability-of-mixed-precision/167310
    

@dataclass
class MainConfig:
    
    JOB_TYPE: str = "default"
    # Experiment name. Used for naming the output folder.
    NAME: str = MISSING
    # The output directory for all controlnet experiments. Model predictions and checkpoints will be written here.
    OUTPUT_DIR: str = "out"
    # Identifier for this training run. Can be any literal that can be converted to `str`.
    # Combined with RESUME_FROM="latest" this allows resuming preempted jobs.
    UNIQUE_ID: Any = datetime.now().strftime("%Y%m%d-%H%M%S")
    UNIQUE_EVAL_ID: Any = datetime.now().strftime("%Y%m%d-%H%M%S")
    
    # Distributed job arguments
    BACKEND: Optional[str] = None           # Depends on GPU model, e.g. "nccl"
    NPROC_PER_NODE: Optional[int] = None    # No of GPUs
    MASTER_PORT: Optional[int] = 25642
    
    SEED: Optional[int] = None
    # Subclass of DataConfig
    CONTROLNET: ControlnetConfig = ControlnetConfig()
    BTS: BTSConfig = BTSConfig()
    SYNTHETIC_GT: SyntheticGTConfig = SyntheticGTConfig()
    
    PROFILER: ProfilerConfig = ProfilerConfig()
    AMP: AMPConfig = AMPConfig()
    WITH_CLEARML: bool = False
    
    LOG_GRADIENTS: bool = False
    DETECT_ANOMALY: bool = False
    DETECT_ANOMALY_CHECK_NAN: bool = False
    STOP_ON_NAN: bool = True
    # Setting these to `0` disables gradient clipping.
    # Clamp values after backprop
    OVERALL_GRAD_CLIP_NORM: float = 0.0
    # Clamp values during backprop
    BACKPROP_GRAD_CLIP_VAL: float = 0.0
    
    EVAL_OCCUPANCY: OccupancyEvalConfig = OccupancyEvalConfig()
    
    @property
    def CONTROLNET_EXP_DIR(self) -> str:
        assert self.NAME != MISSING
        return os.path.join(self.OUTPUT_DIR, "controlnet", self.NAME)
    @property
    def CONTROLNET_EVAL_DIR(self) -> str:
        # We want a different output folder depending on the evaluation task.
        return os.path.join(self.CONTROLNET_EXP_DIR, self.JOB_TYPE, self.UNIQUE_EVAL_ID)
    @property
    def RECON_EXP_DIR(self) -> str:
        assert self.NAME != MISSING
        return os.path.join(self.OUTPUT_DIR, "recon", f"{self.NAME}_backend-{self.BACKEND}-{self.NPROC_PER_NODE}_{self.UNIQUE_ID}")
    @property
    def RECON_EVAL_DIR(self) -> str:
        return os.path.join(self.RECON_EXP_DIR, self.JOB_TYPE, self.UNIQUE_EVAL_ID)
    @property
    def RECON_CACHE_DIR(self) -> str:
        return os.path.join(self.RECON_EXP_DIR, "synth_gt_cache")
    # You can use this arg to set a custom cache dir for the synthetic ground truth.
    CUSTOM_RECON_CACHE_DIR: str | None = None
