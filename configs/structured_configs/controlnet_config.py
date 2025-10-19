import os
from typing import Any, Dict, Optional, Literal, Tuple, List
from dataclasses import dataclass, field
from omegaconf import MISSING
from configs.structured_configs.occlusions_config import OcclusionDetectionConfig
from configs.constants.constants import DEFINITIONS


@dataclass
class ModelConfig:
    # Path to pretrained model or model identifier from huggingface.co/models.
    PRETRAINED_MODEL_NAME_OR_PATH: str = MISSING
    IS_UNCLIP: bool = False
    NOISE_LEVEL: int = 0
    # Options: ["image", "empty"]
    CONTROLNET_IMAGE_EMBEDS_TYPE: str = "empty"
    # Path to pretrained controlnet model or model identifier from huggingface.co/models.
    # If not specified controlnet weights are initialized from unet.
    CONTROLNET_MODEL_NAME_OR_PATH: Optional[str] = None
    # Revision of pretrained model identifier from huggingface.co/models.
    REVISION: Optional[str] = None
    # Variant of the model files of the pretrained model identifier from huggingface.co/models, e.g. 'fp16'.
    VARIANT: Optional[str] = None
    # Pretrained tokenizer name or path if not the same as model_name.
    TOKENIZER_NAME: Optional[str] = None
    # Define how to condition the controlnet. Available options are:
    #   ("rgb": 3)      = RGB
    #   ("rgbm": 4)     = RGB + Mask
    #   ("rgbd": 4)     = RGB + Depth
    #   ("rgbdm": 5)    = RGB + Depth + Mask
    CONDITIONING_INPUT_TYPE_AND_CHANNELS: Tuple[Any] = ("rgb", 3)   # Note: Hydra casts the int to a str if without `Any`
    # Whether to zero-out the masked areas in the conditioning image. 
    MASK_CONDITIONING_IMAGE: bool = True
    

@dataclass
class TrainingConfig:
    NUM_TRAIN_EPOCHS: int = 1
    # Total number of training steps to perform.  If provided, overrides NUM_TRAIN_EPOCHS.
    MAX_TRAIN_STEPS: Optional[int] = None
    # Whether training should be resumed from a previous checkpoint. Use a path saved by
    # `CHECKPOINTING_STEPS`, or `"latest"` to automatically select the last available checkpoint.
    RESUME_FROM_CHECKPOINT: Optional[str] = None
    # Save a checkpoint of the training state every X updates. Checkpoints can be used for resuming training via `--resume_from_checkpoint`. 
    # In the case that the checkpoint is better than the final trained model, the checkpoint can also be used for inference.
    # Using a checkpoint for inference requires separate loading of the original pipeline and the individual checkpointed model components.
    # See https://huggingface.co/docs/diffusers/main/en/training/dreambooth#performing-inference-using-a-saved-checkpoint for step by step
    # instructions.
    CHECKPOINTING_STEPS: int = 10_000
    # Max number of checkpoints to store.
    CHECKPOINTS_TOTAL_LIMIT: Optional[int] = None
    # A seed for reproducible training.
    SEED: Optional[int] = None
    # Number of updates steps to accumulate before performing a backward/update pass.
    GRADIENT_ACCUMULATION_STEPS: int = 1
    # Whether or not to use gradient checkpointing to save memory at the expense of slower backward pass.
    GRADIENT_CHECKPOINTING: bool = False
    # Save more memory by using setting grads to None instead of zero. Be aware, that this changes certain
    # behaviors, so disable this argument if it causes any problems. More info:
    # https://pytorch.org/docs/stable/generated/torch.optim.Optimizer.zero_grad.html
    SET_GRADS_TO_NONE: bool = False
    
    # --- DATALOADER ---
    # Batch size (per device) for the training dataloader.
    TRAIN_BATCH_SIZE: int = 4
    # Number of subprocesses to use for data loading. 0 means that the data will be loaded in the main process.
    DATALOADER_NUM_WORKERS: int = 0
    # For debugging purposes or quicker training, truncate the number of training examples to this
    # value if set.
    MAX_TRAIN_SAMPLES: Optional[int] = None
    # Proportion of image prompts to be replaced with empty strings. Defaults to 0 (no prompt replacement).
    PROPORTION_EMPTY_PROMPTS: float = 0.0
    PROMPT_TEXT: str = ""
    
    # --- TRANSFORMS ---
    # The resolution of the input image used for encoding the pseudo volume.
    ENCODING_RESOLUTION: Optional[Tuple[int, int]] = None
    # The resolution of the rendered novel view.
    RERENDERING_RESOLUTION: Optional[Tuple[int, int]] = None
    # Rotation in degrees about the X and Y axes for producing the novel view pose from the input pose.
    NOVEL_VIEW_X_ROTATION_LIMITS: Tuple[int, int] = (-5, 5)
    NOVEL_VIEW_Y_ROTATION_LIMITS: Tuple[int, int] = (-30, 30)
    # The distance of point in camera-Z-dir around which the rotations happen
    NOVEL_VIEW_DIST_ROTATION: float = 5.0
    EDGE_DIST_FOR_PROJ_SAMPLE: float = 0.0
    # Config for occlusion detection.
    OCCLUSIONS: OcclusionDetectionConfig = OcclusionDetectionConfig()
    # The resolution for input images, all the images in the train/validation dataset will be resized to this resolution.
    RESOLUTION: Tuple[int, int] = (512, 512)
    
    # --- LEARNING RATE ---
    # Initial learning rate (after the potential warmup period) to use.
    LR: float = 5e-6 
    # Scale the learning rate by the number of GPUs, gradient accumulation steps, and batch size.
    SCALE_LR: bool = False
    # The scheduler type to use.
    # Choose from: ["linear", "cosine", "cosine_with_restarts", "polynomial", "constant", "constant_with_warmup"]
    LR_SCHEDULER: str = "constant"
    # Number of steps for the warmup in the lr scheduler.
    LR_WARMUP_STEPS: int = 5_000
    # Number of hard resets of the lr in cosine_with_restarts scheduler.
    LR_NUM_CYCLES: int = 1
    # Power factor of the polynomial scheduler.
    LR_POWER: float = 1.0
    
    # --- OPTIMIZER ---
    # Whether or not to use 8-bit Adam from bitsandbytes.
    USE_8BIT_ADAM: bool = False
    # The beta1 parameter for the Adam optimizer.
    ADAM_BETA1: float = 0.9
    # The beta2 parameter for the Adam optimizer.
    ADAM_BETA2: float = 0.999
    # Weight decay to use.
    ADAM_WEIGHT_DECAY: float = 1e-2
    # Epsilon value for the Adam optimizer.
    ADAM_EPSILON: float = 1e-8
    # Max gradient norm.
    MAX_GRAD_NORM: float = 1.0
    

@dataclass
class ValidationConfig:
    # Batch size during eval.
    BATCH_SIZE: int = 4
    # Maximum.
    # MAX_VALIDATION_SAMPLES: int = 10
    # Number of images to be generated for each `--validation_image`, `--validation_prompt` pair
    NUM_VALIDATION_IMAGE_VERSIONS: int = 4
    # Run validation every X steps. Validation consists of running the latest checkpoint and logging the images.
    VALIDATION_STEPS: int = 5_000
    # Number of denoising steps during evaluation.
    NUM_INFERENCE_STEPS: int = 5
    # Name of the denoising scheduler during evaluation.
    SCHEDULER_NAME: str = "UniPCMultistepScheduler"
    # Max number of validation images from the validation dataset to log to TensorBoard.
    MAX_VALIDATION_IMAGES_TB: int = 20
    # Setting this to 1.0 disables classifier-free guidance.
    GUIDANCE_SCALE: float = 1.0
    
  
@dataclass
class ControlnetConfig:
    
    DATA: Any = MISSING
    MODEL: ModelConfig = ModelConfig()
    TRAIN: TrainingConfig = TrainingConfig()
    EVAL: ValidationConfig = ValidationConfig()
    
    INFERENCE_OCCLUSIONS: OcclusionDetectionConfig = OcclusionDetectionConfig(
        USE_DEPTH=False,
        AREA_THRESH=0.0,
        POSTPROCESSING_OPS_FLOW=[
            (DEFINITIONS.OPENING, 3),
            (DEFINITIONS.CLOSING, 15),
            (DEFINITIONS.DILATION, 15),
        ]
    )
            
    # --- OUTPUT ---
    # [TensorBoard](https://www.tensorflow.org/tensorboard) log directory. Will default to
    # *output_dir/runs/**CURRENT_DATETIME_HOSTNAME***.
    LOGGING_DIR: str = "logs"
    # The directory where the downloaded models and datasets will be stored.
    CACHE_DIR: Optional[str] = None    
    # The integration to report the results and logs to. Supported platforms are `"tensorboard"`
    # (default), `"wandb"` and `"comet_ml"`. Use `"all"` to report to all integrations.
    REPORT_TO: str = "tensorboard" 
    # The `project_name` argument passed to Accelerator.init_trackers for more information see 
    # https://huggingface.co/docs/accelerate/v0.17.0/en/package_reference/accelerator#accelerate.Accelerator
    TRACKER_PROJECT_NAME: str = "train_controlnet"
    
    # --- MEMORY SETTINGS ---
    # Whether or not to allow TF32 on Ampere GPUs. Can be used to speed up training. For more information, see
    # https://pytorch.org/docs/stable/notes/cuda.html#tensorfloat-32-tf32-on-ampere-devices
    ALLOW_TF32: bool = False
    # Whether to use mixed precision. Choose between fp16 and bf16 (bfloat16). Bf16 requires PyTorch >= 1.10. 
    # and an Nvidia Ampere GPU. Default to the value of accelerate config of the current system or the
    # flag passed with the `accelerate.launch` command. Use this argument to override the accelerate config.
    # Options: [None, "fp16", "bf16"]
    MIXED_PRECISION: Optional[str] = None
    # Whether or not to use xformers.
    ENABLE_XFORMERS_MEMORY_EFFICIENT_ATTENTION: bool = False
    
    # --- 🤗 HUB ---
    # Whether or not to push the model to the Hub.
    PUSH_TO_HUB: bool = False
    # The token to use to push to the Model Hub.
    HUB_TOKEN: Optional[str] = None
    # The name of the repository to keep in sync with the local `output_dir`.
    HUB_MODEL_ID: Optional[str] = None


    # parser.add_argument(
    #     "--dataset_name",
    #     type=str,
    #     default=None,
    #     help=(
    #         "The name of the Dataset (from the HuggingFace hub) to train on (could be your own, possibly private,"
    #         " dataset). It can also be a path pointing to a local copy of a dataset in your filesystem,"
    #         " or to a folder containing files that 🤗 Datasets can understand."
    #     ),
    # )
    # parser.add_argument(
    #     "--dataset_config_name",
    #     type=str,
    #     default=None,
    #     help="The config of the Dataset, leave as None if there's only one config.",
    # )
    # parser.add_argument(
    #     "--train_data_dir",
    #     type=str,
    #     default=None,
    #     help=(
    #         "A folder containing the training data. Folder contents must follow the structure described in"
    #         " https://huggingface.co/docs/datasets/image_dataset#imagefolder. In particular, a `metadata.jsonl` file"
    #         " must exist to provide the captions for the images. Ignored if `dataset_name` is specified."
    #     ),
    # )
    # parser.add_argument(
    #     "--stream_dataset", action="store_true", help="Whether to initialize an IterableDataset for streaming."
    # )
    # parser.add_argument(
    #     "--streaming_shuffle_buffer_size",
    #     type=int,
    #     default=None,
    #     help="The size of the shuffle buffer of the IterableDataset. Setting it >0 enables shuffling.",
    # )
    # parser.add_argument(
    #     "--image_column", type=str, default="image", help="The column of the dataset containing the target image."
    # )
    # parser.add_argument(
    #     "--conditioning_image_column",
    #     type=str,
    #     default="conditioning_image",
    #     help="The column of the dataset containing the controlnet conditioning image.",
    # )
    # parser.add_argument(
    #     "--caption_column",
    #     type=str,
    #     default="text",
    #     help="The column of the dataset containing a caption or a list of captions.",
    # )


    # parser.add_argument(
    #     "--validation_prompt",
    #     type=str,
    #     default=None,
    #     nargs="+",
    #     help=(
    #         "A set of prompts evaluated every `--validation_steps` and logged to `--report_to`."
    #         " Provide either a matching number of `--validation_image`s, a single `--validation_image`"
    #         " to be used with all prompts, or a single prompt that will be used with all `--validation_image`s."
    #     ),
    # )
    # parser.add_argument(
    #     "--validation_image",
    #     type=str,
    #     default=None,
    #     nargs="+",
    #     help=(
    #         "A set of paths to the controlnet conditioning image be evaluated every `--validation_steps`"
    #         " and logged to `--report_to`. Provide either a matching number of `--validation_prompt`s, a"
    #         " a single `--validation_prompt` to be used with all `--validation_image`s, or a single"
    #         " `--validation_image` that will be used with all `--validation_prompt`s."
    #     ),
    # )

