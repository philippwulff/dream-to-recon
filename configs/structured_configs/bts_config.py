import os
from typing import Any, Dict, Optional, Literal, Tuple, List
from dataclasses import dataclass, field
from omegaconf import MISSING, DictConfig
from configs.structured_configs.loss_config import LossConfig, DepthLossConfig, MainLossConfig


FINAL_PREDICTION_HEAD_DEFAULT = "normal_head"


@dataclass
class CodeConfig:
    num_freqs: int = 6
    freq_factor: float = 1.5
    include_input: bool = True
    

@dataclass
class EncoderConfig:
    type: str = "monodepth2"
    freeze: bool = False
    pretrained: bool = True
    resnet_layers: int = 50
    num_ch_dec: List[int] = field(default_factory=lambda: [32, 32, 64, 128, 256])
    d_out: int = 64
    decoder_norm: Optional[str] = None
    

class Monodepth2EncoderConfig(EncoderConfig):
    type: str = "monodepth2"
    freeze: bool = False
    pretrained: bool = True
    resnet_layers: int = 50
    num_ch_dec: List[int] = field(default_factory=lambda: [32, 32, 64, 128, 256])
    d_out: int = 64
    decoder_norm: Optional[str] = None

    
@dataclass
class HeadConfig:
    name: str = FINAL_PREDICTION_HEAD_DEFAULT
    freeze: bool = False
    type: str = "mlp"
    ARGS: Dict[str, Any] = field(default_factory=lambda: {})
    

@dataclass
class ModelConfig:
    # "BTSNet" or "BTSDirect"
    ARCH: str = "BTSNet"
    use_code: bool = True
    prediction_mode: str = "default"
    code: CodeConfig = CodeConfig()
    encoder: EncoderConfig = EncoderConfig()
    
    mlp_coarse: dict = field(default_factory=lambda: dict(
        type="resnet",
        n_blocks=0,
        d_hidden=64
    ))
    mlp_fine: dict = field(default_factory=lambda: dict(
        type="empty",
        n_blocks=1,
        d_hidden=128
    ))
    DECODER_HEADS: List[Any] = field(default_factory=lambda: [
        HeadConfig(type="resnet", ARGS={"n_blocks": 0, "d_hidden": 64}),
        HeadConfig(type="empty", ARGS={"n_blocks": 1, "d_hidden": 128}),
        ])
    FINAL_PREDICTION_HEAD: str = FINAL_PREDICTION_HEAD_DEFAULT
    
    z_near: float = 3.0
    z_far: float = 80.0
    inv_z: bool = True
    n_frames_encoder: int = 1
    n_frames_render: int = 2
    frame_sample_mode: str = "kitti360-mono"
    sample_mode: str = "patch"
    patch_size: int = 8
    ray_batch_size: int = 4096
    flip_augmentation: bool = True
    learn_empty: bool = False
    code_mode: str = "z"
    USE_AUTOMASKING: bool = False
    SAMPLE_COLOR: bool = True
    
    SAMPLED_DENSITY_LAMBDA: float = 1.0
    OUTPUT_UNCERTAINTY: bool = False
    OUTPUT_RAW: bool = False

@dataclass
class SchedulerConfig:
    type: str = "fix"
    step_size: int = 120000
    gamma: float = 0.1
    

@dataclass
class SaveBestConfig:
    METRIC: str = "abs_rel"
    SIGN: float = 1.0


@dataclass
class RendererConfig:
    # NeRF renderer arguments.
    
    # Stratified samples.
    n_coarse: int = 64
    # Fine samples placed using statistics pre-computed using the coarse samples. 
    #   - Either samples drawn from from the weight distribution along the ray,
    #   - or depth-guided samples drawn from a Gaussian specified by the pre-computed depth and `depth_std`.
    n_fine: int = 0             # Total number of dine samples
    n_fine_depth: int = 0       # How many of them are using depth-guidance
    depth_std: float = 1.0
    
    sched: List = field(default_factory=lambda: [])
    white_bkgd: bool = False
    lindisp: bool = True
    hard_alpha_cap: bool = True
    eval_batch_size: int = 100_000
    
    
@dataclass
class VisualizeConfig:
    POLICY: str = "eqspaced"
    NUM_SCENES: int = 8
    
    
@dataclass
class BTSConfig:
    # NAME: str = "Default"
    SEED: int = 42
    CHECKPOINT: Optional[str] = None
    
    MODE: str = "depth"
    SUPERVISION_FROM_CASCADE: bool = False
    
    BATCH_SIZE: int = 16
    NUM_WORKERS: int = 4

    # Whether eval and vis steps are in units of steps or epochs.
    VAL_USE_ITERS: bool = True
    VIS_USE_ITERS: bool = True
    # Every how often to run eval and vis.
    VALIDATE_EVERY: int = 2000
    VISUALIZE_EVERY: int = 500
    
    LOG_EVERY_ITERS: int = 1
    log_tb_train_every_iters: int = -1
    log_tb_val_every_iters: int = -1
    log_tb_vis_every_iters: int = 1

    CHECKPOINT_EVERY: int = 500
    RESUME_FROM: Optional[str] = None

    LOSS_DURING_VALIDATION: bool = False#True
    NUM_EPOCHS: int = 150
    STOP_ITERATION: Optional[int] = None

    LR: float = 1e-4
    warmup_steps: int = 10000
    decay_rate: float = 0.5
    decay_steps: int = 100000
    num_steps: int = 100000
    
    WEIGHT_DECAY: float = 0.0
    
    SAVE_BEST: SaveBestConfig = SaveBestConfig()
    
    DATA: Any = MISSING
    # Optionally downscale the input data by this factor.
    DATA_DOWNSCALE_FACTOR: int = 1
    CACHE_SYNTHETIC_GT: bool = False
    BATCH_SIZE_MULTIPLE_AFTER_CLEANUP: int | None = None
    
    MODEL_CONF: ModelConfig = ModelConfig()
    
    LOSSES: DictConfig = DictConfig({})
    SCHEDULER: SchedulerConfig = SchedulerConfig()
    RENDERER: RendererConfig = RendererConfig()
    
    VISUALIZE: VisualizeConfig = VisualizeConfig()
    