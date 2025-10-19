import os
import torch
import diffusers
from diffusers import (
    AutoencoderKL,
    ControlNetModel,
    DDPMScheduler,
    StableDiffusionControlNetPipeline,
    UNet2DConditionModel,
    # UniPCMultistepScheduler,
    # LMSDiscreteScheduler,
    # DPMSolverMultistepScheduler,
    # HeunDiscreteScheduler,
)
from transformers import CLIPImageProcessor, CLIPTextModel, CLIPTokenizer, CLIPVisionModelWithProjection
from diffusers.pipelines.stable_diffusion.stable_unclip_image_normalizer import StableUnCLIPImageNormalizer
from vcm.controlnet_unclip_pipeline import StableDiffusionControlNetUnCLIPPipeline

from diffusers.models.attention_processor import AttnProcessor2_0
from transformers import AutoTokenizer, PretrainedConfig

from configs.structured_configs.main_config import MainConfig



def get_accelerator_checkpoint_path(cfg: MainConfig, use_latest=False):
    if cfg.CONTROLNET.TRAIN.RESUME_FROM_CHECKPOINT != "latest" and not use_latest:
        path = os.path.basename(cfg.CONTROLNET.TRAIN.RESUME_FROM_CHECKPOINT)
    else:
        # Get the most recent checkpoint
        dirs = os.listdir(cfg.CONTROLNET_EXP_DIR)
        dirs = [d for d in dirs if d.startswith("checkpoint")]
        dirs = sorted(dirs, key=lambda x: int(x.split("-")[1]))
        path = dirs[-1] if len(dirs) > 0 else None
        path = os.path.join(cfg.CONTROLNET_EXP_DIR, path)
    return path


def save_model_hook(accelerator):
    
    def save_model_hook_fn(models, weights, output_dir):
        if accelerator.is_main_process:
            i = len(weights) - 1

            while len(weights) > 0:
                weights.pop()
                model = models[i]

                sub_dir = "controlnet"
                model.save_pretrained(os.path.join(output_dir, sub_dir))

                i -= 1
    
    return save_model_hook_fn


def load_model_hook(models, input_dir):
    while len(models) > 0:
        # pop models so that they are not loaded again
        model = models.pop()

        # load diffusers style into model
        load_model = ControlNetModel.from_pretrained(input_dir, subfolder="controlnet")
        model.register_to_config(**load_model.config)

        model.load_state_dict(load_model.state_dict())
        del load_model
        
        
def import_model_class_from_model_name_or_path(pretrained_model_name_or_path: str, revision: str):
    text_encoder_config = PretrainedConfig.from_pretrained(
        pretrained_model_name_or_path,
        subfolder="text_encoder",
        revision=revision,
    )
    model_class = text_encoder_config.architectures[0]

    if model_class == "CLIPTextModel":
        return CLIPTextModel
    elif model_class == "RobertaSeriesModelWithTransformation":
        from diffusers.pipelines.alt_diffusion.modeling_roberta_series import RobertaSeriesModelWithTransformation

        return RobertaSeriesModelWithTransformation
    else:
        raise ValueError(f"{model_class} is not supported.")
    
    
def make_tokenizer(cfg: MainConfig):
    if cfg.CONTROLNET.MODEL.TOKENIZER_NAME:
        tokenizer = AutoTokenizer.from_pretrained(cfg.CONTROLNET.MODEL.TOKENIZER_NAME, revision=cfg.CONTROLNET.MODEL.REVISION, use_fast=False)
    elif cfg.CONTROLNET.MODEL.PRETRAINED_MODEL_NAME_OR_PATH:
        tokenizer = AutoTokenizer.from_pretrained(
            cfg.CONTROLNET.MODEL.PRETRAINED_MODEL_NAME_OR_PATH,
            subfolder="tokenizer",
            revision=cfg.CONTROLNET.MODEL.REVISION,
            use_fast=False,
        )
    else:
        raise ValueError()
    return tokenizer
    

def make_model_components(cfg: MainConfig, feature_extractor=None, image_encoder=None, image_normalizer=None, image_noising_scheduler=None, tokenizer=None, text_encoder=None, vae=None, unet=None, controlnet=None, logger=None, build_pipe=False, compile_pipeline=False):
    
    weight_dtype = torch.float32  
    if cfg.CONTROLNET.MIXED_PRECISION == "fp16":
        weight_dtype = torch.float16
    elif cfg.CONTROLNET.MIXED_PRECISION == "bf16":
        weight_dtype = torch.bfloat16
    
    lprint = lambda m: logger.info(m) if logger is not None else print(m)
    
    if tokenizer is None:
    # Load the tokenizer
        # if cfg.CONTROLNET.MODEL.TOKENIZER_NAME:
        #     tokenizer = AutoTokenizer.from_pretrained(cfg.CONTROLNET.MODEL.TOKENIZER_NAME, revision=cfg.CONTROLNET.MODEL.REVISION, use_fast=False)
        # elif cfg.CONTROLNET.MODEL.PRETRAINED_MODEL_NAME_OR_PATH:
        #     tokenizer = AutoTokenizer.from_pretrained(
        #         cfg.CONTROLNET.MODEL.PRETRAINED_MODEL_NAME_OR_PATH,
        #         subfolder="tokenizer",
        #         revision=cfg.CONTROLNET.MODEL.REVISION,
        #         use_fast=False,
        #     )
        tokenizer = make_tokenizer(cfg)

    if text_encoder is None:
        # import correct text encoder class
        text_encoder_cls = import_model_class_from_model_name_or_path(cfg.CONTROLNET.MODEL.PRETRAINED_MODEL_NAME_OR_PATH, cfg.CONTROLNET.MODEL.REVISION)
        # Load scheduler and models
        text_encoder = text_encoder_cls.from_pretrained(
            cfg.CONTROLNET.MODEL.PRETRAINED_MODEL_NAME_OR_PATH, subfolder="text_encoder", revision=cfg.CONTROLNET.MODEL.REVISION, variant=cfg.CONTROLNET.MODEL.VARIANT
        )
        
    noise_scheduler = DDPMScheduler.from_pretrained(cfg.CONTROLNET.MODEL.PRETRAINED_MODEL_NAME_OR_PATH, subfolder="scheduler")
    
    if vae is None:
        vae = AutoencoderKL.from_pretrained(
            cfg.CONTROLNET.MODEL.PRETRAINED_MODEL_NAME_OR_PATH, subfolder="vae", revision=cfg.CONTROLNET.MODEL.REVISION, variant=cfg.CONTROLNET.MODEL.VARIANT
        )
    if unet is None:
        unet = UNet2DConditionModel.from_pretrained(
            cfg.CONTROLNET.MODEL.PRETRAINED_MODEL_NAME_OR_PATH, subfolder="unet", revision=cfg.CONTROLNET.MODEL.REVISION, variant=cfg.CONTROLNET.MODEL.VARIANT
        )
        
    if cfg.CONTROLNET.MODEL.IS_UNCLIP:
        # image encoding components
        if feature_extractor is None:
            feature_extractor = CLIPImageProcessor.from_pretrained(cfg.CONTROLNET.MODEL.PRETRAINED_MODEL_NAME_OR_PATH, subfolder="feature_extractor")
        if image_encoder is None:
            image_encoder = CLIPVisionModelWithProjection.from_pretrained(cfg.CONTROLNET.MODEL.PRETRAINED_MODEL_NAME_OR_PATH, subfolder="image_encoder")
        # image noising components
        if image_normalizer is None:
            image_normalizer = StableUnCLIPImageNormalizer.from_pretrained(cfg.CONTROLNET.MODEL.PRETRAINED_MODEL_NAME_OR_PATH, subfolder="image_normalizer")
        if image_noising_scheduler is None:
            image_noising_scheduler = DDPMScheduler.from_pretrained(cfg.CONTROLNET.MODEL.PRETRAINED_MODEL_NAME_OR_PATH, subfolder="image_noising_scheduler")

    if controlnet is None:
        if cfg.CONTROLNET.MODEL.CONTROLNET_MODEL_NAME_OR_PATH:
            lprint(f"Loading existing controlnet weights from: {cfg.CONTROLNET.MODEL.CONTROLNET_MODEL_NAME_OR_PATH}")
            controlnet = ControlNetModel.from_pretrained(cfg.CONTROLNET.MODEL.CONTROLNET_MODEL_NAME_OR_PATH)
        else:
            lprint("Initializing controlnet weights from unet")
            controlnet = ControlNetModel.from_unet(unet, conditioning_channels=cfg.CONTROLNET.MODEL.CONDITIONING_INPUT_TYPE_AND_CHANNELS[1])
        
    pipeline = None
    if build_pipe:
        pipe_init_args = {
            "pretrained_model_name_or_path": cfg.CONTROLNET.MODEL.PRETRAINED_MODEL_NAME_OR_PATH,
            "vae": vae,
            "text_encoder": text_encoder,
            "tokenizer": tokenizer,
            "unet": unet,
            # scheduler: is in the validation func
            "controlnet": controlnet,
            "safety_checker": None,
            "revision": cfg.CONTROLNET.MODEL.REVISION,
            "variant": cfg.CONTROLNET.MODEL.VARIANT,
            # Don’t use torch.autocast in any of the pipelines as it can lead to 
            # black images and is always slower than pure float16 precision.
            "torch_dtype": weight_dtype,
        }
        
        if cfg.CONTROLNET.MODEL.IS_UNCLIP:
            pipe_init_args.update({
                "feature_extractor": feature_extractor,
                "image_encoder": image_encoder,
                "image_normalizer": image_normalizer,
                "image_noising_scheduler": image_noising_scheduler,
            })
            pipeline = StableDiffusionControlNetUnCLIPPipeline.from_pretrained(**pipe_init_args)
        else:
            pipeline = StableDiffusionControlNetPipeline.from_pretrained(**pipe_init_args)
            
        # TODO do this only during inference
        # pipeline = pipeline.to(weight_dtype)
        
        # pipeline = StableDiffusionControlNetPipeline.from_pretrained(
        #     cfg.CONTROLNET.MODEL.PRETRAINED_MODEL_NAME_OR_PATH,
        #     vae=vae,
        #     text_encoder=text_encoder,
        #     tokenizer=tokenizer,
        #     unet=unet,
        #     controlnet=controlnet,
        #     # From the docs:
        #     # torch_dtype=weight_dtype,
        #     # TODO what is this:
        #     # use_safetensors=True
        # ).to(weight_dtype)
        # See all schedulers using: pipeline.scheduler.compatibles
        #       [diffusers.schedulers.scheduling_euler_ancestral_discrete.EulerAncestralDiscreteScheduler,
        #       diffusers.schedulers.scheduling_unipc_multistep.UniPCMultistepScheduler,
        #       diffusers.utils.dummy_torch_and_torchsde_objects.DPMSolverSDEScheduler,
        #       diffusers.schedulers.scheduling_ddpm.DDPMScheduler,
        #       diffusers.schedulers.scheduling_euler_discrete.EulerDiscreteScheduler,
        #       diffusers.schedulers.scheduling_heun_discrete.HeunDiscreteScheduler,
        #       diffusers.schedulers.scheduling_k_dpm_2_ancestral_discrete.KDPM2AncestralDiscreteScheduler,
        #       diffusers.schedulers.scheduling_ddim.DDIMScheduler,
        #       diffusers.schedulers.scheduling_lms_discrete.LMSDiscreteScheduler,
        #       diffusers.schedulers.scheduling_dpmsolver_singlestep.DPMSolverSinglestepScheduler,
        #       diffusers.schedulers.scheduling_dpmsolver_multistep.DPMSolverMultistepScheduler,
        #       diffusers.schedulers.scheduling_deis_multistep.DEISMultistepScheduler,
        #       diffusers.schedulers.scheduling_k_dpm_2_discrete.KDPM2DiscreteScheduler,
        #       diffusers.schedulers.scheduling_pndm.PNDMScheduler]
        # scheduler_class = SCHEDULERS.get(cfg.CONTROLNET.EVAL.SCHEDULER_NAME, UniPCMultistepScheduler)
        # scheduler_class = __import__(f"diffusers", fromlist=[cfg.CONTROLNET.EVAL.SCHEDULER_NAME])
        scheduler_class = getattr(diffusers, cfg.CONTROLNET.EVAL.SCHEDULER_NAME)
        pipeline.scheduler = scheduler_class.from_config(pipeline.scheduler.config)
        
        # Accelerate inference of text-to-image diffusion models:
        # https://huggingface.co/docs/diffusers/en/tutorials/fast_diffusion
        
        # When using Pytorch2.0, diffusers automatically selects `AttnProcessor2_0()`.
        
        if compile_pipeline:
            # https://huggingface.co/docs/diffusers/en/optimization/torch2.0#torchcompile
            # Calling the compiled pipeline on a different image size triggers compilation again.
            lprint("Compiling controlnet pipeline...")
            # pipeline.unet = pipeline.unet.to(memory_format=torch.channels_last)
            pipeline.unet = torch.compile(pipeline.unet, mode="max-autotune", fullgraph=False)
            # pipeline.unet = torch.compile(pipeline.unet, mode="max-autotune", fullgraph=False, dynamic=True)
            # pipeline.vae = pipeline.vae.to(memory_format=torch.channels_last)
            # pipeline.vae = torch.compile(pipeline.vae, mode="max-autotune", fullgraph=False)
            # pipeline.vae = torch.compile(pipeline.vae, mode="max-autotune", fullgraph=False, dynamic=True)
            # pass
            # pipeline.vae.decode = torch.compile(pipeline.vae.decode, mode="reduce-overhead", fullgraph=True)

            # pipeline.controlnet = pipeline.controlnet.to(memory_format=torch.channels_last)
            # pipeline.controlnet = torch.compile(pipeline.controlnet, mode="max-autotune", fullgraph=False)
            # pipeline.controlnet = torch.compile(pipeline.controlnet, mode="max-autotune", fullgraph=False, dynamic=True)
            # pipeline = torch.compile(pipeline, mode="max-autotune", fullgraph=True)
        
        # pipeline.scheduler = UniPCMultistepScheduler.from_config(pipeline.scheduler.config)
        # pipeline.scheduler = LMSDiscreteScheduler.from_config(pipeline.scheduler.config)
        # pipeline.scheduler = HeunDiscreteScheduler.from_config(pipeline.scheduler.config)
        # pipeline.scheduler = DPMSolverMultistepScheduler.from_config(pipeline.scheduler.config)

        pipeline.set_progress_bar_config(disable=True)
        
    components = {
        # Only for SD-UnCLIP
        "feature_extractor": feature_extractor, 
        "image_encoder": image_encoder, 
        "image_normalizer": image_normalizer, 
        "image_noising_scheduler": image_noising_scheduler, 
        # Regular SD
        "tokenizer": tokenizer, 
        "text_encoder": text_encoder, 
        "noise_scheduler": noise_scheduler, 
        "vae": vae, 
        "unet": unet, 
        "controlnet": controlnet
    }
        
    return components, pipeline
