#!/usr/bin/env python
# coding=utf-8
# Copyright 2023 The HuggingFace Inc. team. All rights reserved.
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and

# NOTE: Copied from: 
# https://github.com/huggingface/diffusers/blob/main/examples/controlnet/train_controlnet.py
# https://github.com/lllyasviel/ControlNet/blob/main/docs/train.md

# import argparse
import logging
import math
import os
# import random
import shutil
# from pathlib import Path
# from typing import Callable
import time
import sys
from dataclasses import asdict
from omegaconf import DictConfig
import random

import accelerate
# import numpy as np
import torch
import torch.nn.functional as F
import torch.utils.checkpoint
import transformers
from accelerate import Accelerator
from accelerate.logging import get_logger
from accelerate.utils import ProjectConfiguration, set_seed
# from datasets import load_dataset
# from huggingface_hub import create_repo, upload_folder
from packaging import version
# from PIL import Image
# from torchvision import transforms
from tqdm.auto import tqdm
# from transformers import AutoTokenizer, PretrainedConfig
from pandas.io.json._normalize import nested_to_record    


import diffusers
# from diffusers import (
#     AutoencoderKL,
#     ControlNetModel,
#     DDPMScheduler,
#     StableDiffusionControlNetPipeline,
#     UNet2DConditionModel,
#     UniPCMultistepScheduler,
# )
from diffusers.optimization import get_scheduler
from diffusers.utils import check_min_version, is_wandb_available
from diffusers.utils.import_utils import is_xformers_available
from diffusers.utils.torch_utils import is_compiled_module

import hydra
# from hydra.core.config_store import ConfigStore
# from omegaconf import DictConfig, OmegaConf

print(os.getcwd())
print(os.environ["PATH"])
# os.chdir(os.path.join(os.path.dirname(os.path.abspath(__file__)), ".."))
# sys.path.append(os.path.abspath(os.getcwd()))
sys.path.append(os.path.join(os.path.dirname(os.path.abspath(__file__)), ".."))
print(sys.path)

# from configs.defaults.controlnet_config import ControlnetConfig
from configs.structured_configs.main_config import MainConfig
from configs.structured_configs.config_utils import register_default_configs, check_and_post_init_config
from vcm.utils.data_utils import make_controlnet_dataloaders
from vcm.utils.model_utils import save_model_hook, load_model_hook, make_model_components, get_accelerator_checkpoint_path
from vcm.utils.eval_utils import log_validation

# from datasets.data_util import make_datasets

# from scripts.inference_setup import *
# from datasets.kitti_360.kitti_360_dataset import Kitti360Dataset
# from bts.models.pseudo_volume import PseudoVolume
# from bts.renderer.nerf import NeRFRenderer
# from utils.transformation_ops import orbit_poses
# from utils.occlusion_ops import comp_occlusion_map
# from utils.constants import OcclusionConfig, DEFINITIONS
# from utils.plotting import color_occlusion_masked_img
# from bts.training_ignite.trainer import BTSWrapper
from bts.gt_synthesis.gt_synthesis import GTSynthesisWrapper
# from utils.plotting import color_tensor


if is_wandb_available():
    import wandb

# Will error if the minimal version of diffusers is not installed. Remove at your own risks.
check_min_version("0.26.0.dev0")

logger = get_logger(__name__)


# Store the default ControlnetConfig so that it is available to Hydra
register_default_configs()

@hydra.main(version_base=None, config_path="../configs", config_name="base_main_config")
def main(cfg: DictConfig):
    
    cfg: MainConfig = check_and_post_init_config(cfg)
    
    # OmegaConf.set_struct(config, False)

    # cfg.CONTROLNET.OUTPUT_DIR = os.path.join(cfg.CONTROLNET.OUTPUT_DIR, cfg.NAME)
    # logging_dir = os.path.join(cfg.CONTROLNET.OUTPUT_DIR, cfg.CONTROLNET.LOGGING_DIR)
    logging_dir = os.path.join(cfg.CONTROLNET_EXP_DIR, cfg.CONTROLNET.LOGGING_DIR)
    
    # accelerator_project_config = ProjectConfiguration(project_dir=cfg.CONTROLNET.OUTPUT_DIR, logging_dir=logging_dir)
    accelerator_project_config = ProjectConfiguration(project_dir=cfg.CONTROLNET_EXP_DIR, logging_dir=logging_dir)

    accelerator = Accelerator(
        gradient_accumulation_steps=cfg.CONTROLNET.TRAIN.GRADIENT_ACCUMULATION_STEPS,
        mixed_precision=cfg.CONTROLNET.MIXED_PRECISION,
        log_with=cfg.CONTROLNET.REPORT_TO,      # TODO remove wandb
        project_config=accelerator_project_config,
    )

    # Make one log on every process with the configuration for debugging.
    logging.basicConfig(
        format="%(asctime)s - %(levelname)s - %(name)s - %(message)s",
        datefmt="%m/%d/%Y %H:%M:%S",
        level=logging.INFO,
    )
    logger.info(accelerator.state, main_process_only=False)
    if accelerator.is_local_main_process:
        transformers.utils.logging.set_verbosity_warning()
        diffusers.utils.logging.set_verbosity_info()
    else:
        transformers.utils.logging.set_verbosity_error()
        diffusers.utils.logging.set_verbosity_error()

    # If passed along, set the training seed now.
    if cfg.CONTROLNET.TRAIN.SEED is not None:
        set_seed(cfg.CONTROLNET.TRAIN.SEED)

    # Handle the repository creation
    if accelerator.is_main_process:
        if cfg.CONTROLNET_EXP_DIR is not None:
            os.makedirs(cfg.CONTROLNET_EXP_DIR, exist_ok=True)

        # if cfg.CONTROLNET.PUSH_TO_HUB:
        #     repo_id = create_repo(
        #         repo_id=args.hub_model_id or Path(args.output_dir).name, exist_ok=True, token=args.hub_token
        #     ).repo_id

    components, validation_pipeline = make_model_components(cfg, build_pipe=True)
    
    tokenizer = components["tokenizer"] 
    text_encoder = components["text_encoder"] 
    noise_scheduler = components["noise_scheduler"] 
    vae = components["vae"] 
    unet = components["unet"] 
    controlnet = components["controlnet"] 
    # For SD-UnCLIP
    feature_extractor = components["feature_extractor"] 
    image_encoder = components["image_encoder"] 
    image_normalizer = components["image_normalizer"] 
    image_noising_scheduler = components["image_noising_scheduler"] 
    
    # Taken from [Sayak Paul's Diffusers PR #6511](https://github.com/huggingface/diffusers/pull/6511/files)
    def unwrap_model(model):
        model = accelerator.unwrap_model(model)
        model = model._orig_mod if is_compiled_module(model) else model
        return model

    # `accelerate` 0.16.0 will have better support for customized saving
    if version.parse(accelerate.__version__) >= version.parse("0.16.0"):
        # create custom saving & loading hooks so that `accelerator.save_state(...)` serializes in a nice format

        accelerator.register_save_state_pre_hook(save_model_hook(accelerator))
        accelerator.register_load_state_pre_hook(load_model_hook)

    vae.requires_grad_(False)
    unet.requires_grad_(False)
    text_encoder.requires_grad_(False)
    if image_encoder is not None:
        image_encoder.requires_grad_(False)
    controlnet.train()

    if cfg.CONTROLNET.ENABLE_XFORMERS_MEMORY_EFFICIENT_ATTENTION:
        if is_xformers_available():
            import xformers

            xformers_version = version.parse(xformers.__version__)
            if xformers_version == version.parse("0.0.16"):
                logger.warn(
                    "xFormers 0.0.16 cannot be used for training in some GPUs. If you observe problems during training, please update xFormers to at least 0.0.17. See https://huggingface.co/docs/diffusers/main/en/optimization/xformers for more details."
                )
            unet.enable_xformers_memory_efficient_attention()
            controlnet.enable_xformers_memory_efficient_attention()
        else:
            raise ValueError("xformers is not available. Make sure it is installed correctly")

    if cfg.CONTROLNET.TRAIN.GRADIENT_CHECKPOINTING:
        controlnet.enable_gradient_checkpointing()

    # Check that all trainable models are in full precision
    low_precision_error_string = (
        " Please make sure to always have all model weights in full float32 precision when starting training - even if"
        " doing mixed precision training, copy of the weights should still be float32."
    )

    if unwrap_model(controlnet).dtype != torch.float32:
        raise ValueError(
            f"Controlnet loaded as datatype {unwrap_model(controlnet).dtype}. {low_precision_error_string}"
        )

    # Enable TF32 for faster training on Ampere GPUs,
    # cf https://pytorch.org/docs/stable/notes/cuda.html#tensorfloat-32-tf32-on-ampere-devices
    if cfg.CONTROLNET.ALLOW_TF32:
        torch.backends.cuda.matmul.allow_tf32 = True

    if cfg.CONTROLNET.TRAIN.SCALE_LR:
        cfg.CONTROLNET.TRAIN.LR = (
            cfg.CONTROLNET.TRAIN.LR * cfg.CONTROLNET.TRAIN.GRADIENT_ACCUMULATION_STEPS * cfg.CONTROLNET.TRAIN.TRAIN_BATCH_SIZE * accelerator.num_processes
        )

    # Use 8-bit Adam for lower memory usage or to fine-tune the model in 16GB GPUs
    if cfg.CONTROLNET.TRAIN.USE_8BIT_ADAM:
        try:
            import bitsandbytes as bnb
        except ImportError:
            raise ImportError(
                "To use 8-bit Adam, please install the bitsandbytes library: `pip install bitsandbytes`."
            )

        optimizer_class = bnb.optim.AdamW8bit
    else:
        optimizer_class = torch.optim.AdamW

    # Optimizer creation
    params_to_optimize = controlnet.parameters()
    optimizer = optimizer_class(
        params_to_optimize,
        lr=cfg.CONTROLNET.TRAIN.LR,
        betas=(cfg.CONTROLNET.TRAIN.ADAM_BETA1, cfg.CONTROLNET.TRAIN.ADAM_BETA2),
        weight_decay=cfg.CONTROLNET.TRAIN.ADAM_WEIGHT_DECAY,
        eps=cfg.CONTROLNET.TRAIN.ADAM_EPSILON,
    )

    # train_dataset = make_train_dataset(cfg, tokenizer, accelerator)#.with_format("torch")
    # if args.stream_dataset:
    #     assert isinstance(train_dataset, torch.utils.data.IterableDataset)
    # dataset_len = len(train_dataset) if not args.stream_dataset else args.max_train_steps * args.train_batch_size
    # train_dataset, test_dataset = make_datasets(cfg.DATA)
    
    
    # train_dataloader = torch.utils.data.DataLoader(
    #     train_dataset,
    #     collate_fn=collate_fn,
    #     batch_size=cfg.TRAIN.TRAIN_BATCH_SIZE,
    #     num_workers=cfg.TRAIN.DATALOADER_NUM_WORKERS,
    # )
    
    train_dataloader, val_dataloader = make_controlnet_dataloaders(cfg)#, loading_batch_size=cfg.CONTROLNET.TRAIN.TRAIN_BATCH_SIZE//2)
    
    # dataloader_len = len(train_dataloader) if not args.stream_dataset else args.max_train_steps

    # Scheduler and math around the number of training steps.
    overrode_max_train_steps = False
    num_steps_per_epoch = math.ceil(len(train_dataloader.dataset) / cfg.CONTROLNET.TRAIN.GRADIENT_ACCUMULATION_STEPS)
    num_update_steps_per_epoch = math.ceil(len(train_dataloader) / cfg.CONTROLNET.TRAIN.GRADIENT_ACCUMULATION_STEPS)
    # num_update_steps_per_epoch = math.ceil(len(train_dataloader) / cfg.CONTROLNET.TRAIN.GRADIENT_ACCUMULATION_STEPS)
    if cfg.CONTROLNET.TRAIN.MAX_TRAIN_STEPS is None:
        cfg.CONTROLNET.TRAIN.MAX_TRAIN_STEPS = cfg.CONTROLNET.TRAIN.NUM_TRAIN_EPOCHS * num_steps_per_epoch
        overrode_max_train_steps = True

    lr_scheduler = get_scheduler(
        cfg.CONTROLNET.TRAIN.LR_SCHEDULER,
        optimizer=optimizer,
        num_warmup_steps=cfg.CONTROLNET.TRAIN.LR_WARMUP_STEPS * accelerator.num_processes // cfg.CONTROLNET.TRAIN.TRAIN_BATCH_SIZE,
        num_training_steps=cfg.CONTROLNET.TRAIN.MAX_TRAIN_STEPS * accelerator.num_processes // cfg.CONTROLNET.TRAIN.TRAIN_BATCH_SIZE,
        num_cycles=cfg.CONTROLNET.TRAIN.LR_NUM_CYCLES,
        power=cfg.CONTROLNET.TRAIN.LR_POWER,
    )

    # Prepare everything with our `accelerator`.
    controlnet, optimizer, train_dataloader, lr_scheduler = accelerator.prepare(
        controlnet, optimizer, train_dataloader, lr_scheduler
    )

    # For mixed precision training we cast the text_encoder and vae weights to half-precision
    # as these models are only used for inference, keeping weights in full precision is not required.
    weight_dtype = torch.float32
    if accelerator.mixed_precision == "fp16":
        weight_dtype = torch.float16
    elif accelerator.mixed_precision == "bf16":
        weight_dtype = torch.bfloat16

    # Move vae, unet and text_encoder to device and cast to weight_dtype
    vae.to(accelerator.device, dtype=weight_dtype)
    unet.to(accelerator.device, dtype=weight_dtype)
    text_encoder.to(accelerator.device, dtype=weight_dtype)
    if image_encoder is not None:
        image_encoder.to(accelerator.device, dtype=weight_dtype)
        image_normalizer.to(accelerator.device, weight_dtype)

    # We need to recalculate our total training steps as the size of the training dataloader may have changed.
    num_steps_per_epoch = math.ceil(len(train_dataloader.dataset) / cfg.CONTROLNET.TRAIN.GRADIENT_ACCUMULATION_STEPS)
    num_update_steps_per_epoch = math.ceil(len(train_dataloader) / cfg.CONTROLNET.TRAIN.GRADIENT_ACCUMULATION_STEPS)
    # num_update_steps_per_epoch = math.ceil(len(train_dataloader) / cfg.CONTROLNET.TRAIN.GRADIENT_ACCUMULATION_STEPS)
    if overrode_max_train_steps:
        cfg.CONTROLNET.TRAIN.MAX_TRAIN_STEPS = cfg.CONTROLNET.TRAIN.NUM_TRAIN_EPOCHS * num_steps_per_epoch
    # Afterwards we recalculate our number of training epochs
    cfg.CONTROLNET.TRAIN.NUM_TRAIN_EPOCHS = math.ceil(cfg.CONTROLNET.TRAIN.MAX_TRAIN_STEPS / num_steps_per_epoch)

    # We need to initialize the trackers we use, and also store our configuration.
    # The trackers initializes automatically on the main process.
    if accelerator.is_main_process:
        # Cast to dict.
        tracker_config = asdict(cfg)# OmegaConf.to_container(cfg, resolve=True)  
        tracker_config.pop("BTS")
        # Flatten so that TB understands it
        tracker_config = nested_to_record(tracker_config, sep='_')
        valid_tb_type_fn = lambda x: x if isinstance(x, (str, int, float, bool, torch.Tensor)) else str(x)
        tracker_config = {k: valid_tb_type_fn(v) for k, v in tracker_config.items()}
        accelerator.init_trackers(cfg.NAME, config=tracker_config)

    # Train!
    total_batch_size = cfg.CONTROLNET.TRAIN.TRAIN_BATCH_SIZE * accelerator.num_processes * cfg.CONTROLNET.TRAIN.GRADIENT_ACCUMULATION_STEPS

    logger.info("***** Running training *****")
    logger.info(f"  Experiment name = {cfg.NAME}")
    logger.info(f"  Num examples = {len(train_dataloader.dataset)}")
    logger.info(f"  Num batches each epoch = {len(train_dataloader)}")
    logger.info(f"  Num Epochs = {cfg.CONTROLNET.TRAIN.NUM_TRAIN_EPOCHS}")
    logger.info(f"  Instantaneous batch size per device = {cfg.CONTROLNET.TRAIN.TRAIN_BATCH_SIZE}")
    logger.info(f"  Total train batch size (w. parallel, distributed & accumulation) = {total_batch_size}")
    logger.info(f"  Gradient Accumulation steps = {cfg.CONTROLNET.TRAIN.GRADIENT_ACCUMULATION_STEPS}")
    logger.info(f"  Total optimization steps = {cfg.CONTROLNET.TRAIN.MAX_TRAIN_STEPS // cfg.CONTROLNET.TRAIN.TRAIN_BATCH_SIZE}")
    logger.info(f"  Total global steps (# examples * # epochs) = {cfg.CONTROLNET.TRAIN.MAX_TRAIN_STEPS}")
    logger.info(f"  Running Validation every {cfg.CONTROLNET.EVAL.VALIDATION_STEPS} global steps.")
    global_step = 0
    first_epoch = 0

    # Potentially load in the weights and states from a previous save
    if cfg.CONTROLNET.TRAIN.RESUME_FROM_CHECKPOINT:
        path = get_accelerator_checkpoint_path(cfg)

        if path is None:
            accelerator.print(
                f"Checkpoint '{cfg.CONTROLNET.TRAIN.RESUME_FROM_CHECKPOINT}' does not exist. Starting a new training run."
            )
            cfg.CONTROLNET.TRAIN.RESUME_FROM_CHECKPOINT = None
            initial_global_step = 0
        else:
            accelerator.print(f"Resuming from checkpoint {path}")
            accelerator.load_state(path)
            global_step = int(path.split("-")[1])

            initial_global_step = global_step
            first_epoch = global_step // num_steps_per_epoch
    else:
        initial_global_step = 0
        
    last_validation_at_step = initial_global_step
    last_checkpoint_at_step = initial_global_step

    progress_bar = tqdm(
        range(0, cfg.CONTROLNET.TRAIN.MAX_TRAIN_STEPS),
        initial=initial_global_step,
        desc="Global Training Steps",
        # Only show the progress bar once on each machine.
        disable=not accelerator.is_local_main_process,
    )
    
    device = accelerator.device
    
    # model_cfg_dict = OmegaConf.to_container(cfg.BTS, resolve=True)
    
    # density_field = PseudoVolume()
    # renderer = NeRFRenderer.from_conf(asdict(cfg.BTS.renderer))
    # renderer = renderer.bind_parallel(density_field, gpus=None).eval().to(device)
    
    # Helper function that can be reused during evaluation.
    # prep_batch_fn = lambda batch, eval=False: make_conditioning_imgs(
    #     cfg,
    #     batch["imgs"].to(device),
    #     batch["poses"].to(device),
    #     batch["projs"].to(device),
    #     batch["depths"].to(device),
    #     renderer=renderer,
    #     tokenizer=tokenizer,
    #     mode="training",
    #     eval_seed=42 if eval else None,
    #     debug=False,
    # )
    gt_synthesizer = GTSynthesisWrapper.from_conf(cfg, cam_incl_adjust=cfg.CONTROLNET.DATA.CAM_INCL_ADJUST, refiner=False, depth_pred=True, tokenizer=tokenizer, feature_extractor=feature_extractor)
    gt_synthesizer = gt_synthesizer.eval()
    gt_synthesizer.requires_grad_(False)
    gt_synthesizer = gt_synthesizer.to(device, weight_dtype)

    for epoch in range(first_epoch, cfg.CONTROLNET.TRAIN.NUM_TRAIN_EPOCHS):
        
        progress_bar.set_description(f"Global Training Steps (Epoch {epoch})")
        
        for step, batch in enumerate(train_dataloader):
            
            # start_time = time.time()
            # --- PREP DATA ---
        
            # batch, _ = prep_batch_fn(batch)
            images = torch.stack(batch["imgs"], dim=1).to(device, weight_dtype)
            poses = torch.stack(batch["poses"], dim=1)
            poses = (poses.inverse() @ poses).to(device, weight_dtype)
            projs = torch.stack(batch["projs"], dim=1).to(device, weight_dtype)
            with torch.cuda.amp.autocast(dtype=weight_dtype):
                out, _, _ = gt_synthesizer(images, poses, projs, output_in_nv=False, refine_output=False)

            imgs_gt = out.IMGS_GT.to(dtype=weight_dtype)
            controlnet_image = out.IMGS_COND.to(dtype=weight_dtype)
            captions_tokenized = out.CAPTIONS_IDS
            if out.IMGS_PREPROCESSED is not None:
                images_preprocessed = out.IMGS_PREPROCESSED.to(dtype=weight_dtype)
            # print(f"prep_batch_fn time: {time.time() - start_time:.2f}s")
            
            # --- FORWARD PASS, LOSS, OPTIMIZE ---
            
            with accelerator.accumulate(controlnet):
                # Convert images to latent space
                latents = vae.encode(imgs_gt).latent_dist.sample()
                latents = latents * vae.config.scaling_factor

                # Sample noise that we'll add to the latents
                noise = torch.randn_like(latents)
                bsz = latents.shape[0]
                # Sample a random timestep for each image
                timesteps = torch.randint(0, noise_scheduler.config.num_train_timesteps, (bsz,), device=latents.device)
                timesteps = timesteps.long()

                # Add noise to the latents according to the noise magnitude at each timestep
                # (this is the forward diffusion process)
                noisy_latents = noise_scheduler.add_noise(latents, noise, timesteps)

                # Get the text embedding for conditioning
                encoder_hidden_states = text_encoder(captions_tokenized, return_dict=False)[0]     # TODO replace [0] with .last_hidden_state
                
                # Optionally, get CLIP image embeddings to condition
                image_embeds, controlnet_image_embeds = None, None
                if image_encoder is not None:
                    ## get image embeddings
                    # images_preprocessed = imgs_gt       # TODO maybe pass trhough img prepro
                    image_embeds = image_encoder(images_preprocessed).image_embeds
                    ## add noise to image embeddings
                    if cfg.CONTROLNET.MODEL.NOISE_LEVEL >= 1000:
                        train_noise = random.randint(0, 999)
                    else:
                        train_noise = cfg.CONTROLNET.MODEL.NOISE_LEVEL

                    image_embeds = validation_pipeline.noise_image_embeddings(
                                    image_embeds=image_embeds,
                                    noise_level=train_noise,
                                    generator=None,
                                )
                    
                    # controlnet_image_embeds_type = cfg.get("controlnet_image_embeds_type", "empty")
                    if cfg.CONTROLNET.MODEL.CONTROLNET_IMAGE_EMBEDS_TYPE == "image":
                        controlnet_image_embeds = image_embeds
                    else:
                        controlnet_image_embeds = torch.zeros_like(image_embeds)

                down_block_res_samples, mid_block_res_sample = controlnet(
                    noisy_latents,
                    timesteps,
                    # text encoder condition
                    encoder_hidden_states=encoder_hidden_states,
                    # optional image encoder condition for SD-unCLIP
                    class_labels=controlnet_image_embeds,
                    controlnet_cond=controlnet_image,
                    return_dict=False,
                )
                
                # Predict the noise residual
                model_pred = unet(
                    noisy_latents,
                    timesteps,
                    encoder_hidden_states=encoder_hidden_states,
                    class_labels=image_embeds,      # None if not SD-unclip
                    down_block_additional_residuals=[
                        sample.to(dtype=weight_dtype) for sample in down_block_res_samples
                    ],
                    mid_block_additional_residual=mid_block_res_sample.to(dtype=weight_dtype),
                    return_dict=False,
                )[0]

                # Get the target for loss depending on the prediction type
                if noise_scheduler.config.prediction_type == "epsilon":
                    target = noise
                elif noise_scheduler.config.prediction_type == "v_prediction":
                    target = noise_scheduler.get_velocity(latents, noise, timesteps)
                else:
                    raise ValueError(f"Unknown prediction type {noise_scheduler.config.prediction_type}")
                loss = F.mse_loss(model_pred.float(), target.float(), reduction="mean")

                accelerator.backward(loss)
                if accelerator.sync_gradients:
                    params_to_clip = controlnet.parameters()
                    accelerator.clip_grad_norm_(params_to_clip, cfg.CONTROLNET.TRAIN.MAX_GRAD_NORM)
                optimizer.step()
                lr_scheduler.step()
                optimizer.zero_grad(set_to_none=cfg.CONTROLNET.TRAIN.SET_GRADS_TO_NONE)
                
            # --- OPTIMIZATION STEP DONE ---

            # Checks if the accelerator has performed an optimization step behind the scenes
            if accelerator.sync_gradients:
                
                progress_bar.update(cfg.CONTROLNET.TRAIN.TRAIN_BATCH_SIZE)
                global_step += cfg.CONTROLNET.TRAIN.TRAIN_BATCH_SIZE

                if accelerator.is_main_process:
                    
                    # --- CHECKPOINTING ---
                    
                    if (global_step - last_checkpoint_at_step) >= cfg.CONTROLNET.TRAIN.CHECKPOINTING_STEPS:
                        last_checkpoint_at_step = global_step
                        # _before_ saving state, check if this save would set us over the `checkpoints_total_limit`
                        if cfg.CONTROLNET.TRAIN.CHECKPOINTS_TOTAL_LIMIT is not None:
                            checkpoints = os.listdir(cfg.CONTROLNET_EXP_DIR)
                            checkpoints = [d for d in checkpoints if d.startswith("checkpoint")]
                            checkpoints = sorted(checkpoints, key=lambda x: int(x.split("-")[1]))

                            # before we save the new checkpoint, we need to have at _most_ `checkpoints_total_limit - 1` checkpoints
                            if len(checkpoints) >= cfg.CONTROLNET.TRAIN.CHECKPOINTS_TOTAL_LIMIT:
                                num_to_remove = len(checkpoints) - cfg.CONTROLNET.TRAIN.CHECKPOINTS_TOTAL_LIMIT + 1
                                removing_checkpoints = checkpoints[0:num_to_remove]

                                logger.info(
                                    f"{len(checkpoints)} checkpoints already exist, removing {len(removing_checkpoints)} checkpoints"
                                )
                                logger.info(f"removing checkpoints: {', '.join(removing_checkpoints)}")

                                for removing_checkpoint in removing_checkpoints:
                                    removing_checkpoint = os.path.join(cfg.CONTROLNET_EXP_DIR, removing_checkpoint)
                                    shutil.rmtree(removing_checkpoint)

                        save_path = os.path.join(cfg.CONTROLNET_EXP_DIR, f"checkpoint-{global_step}")
                        accelerator.save_state(save_path)
                        logger.info(f"Saved state to {save_path}")

                    # --- VALIDATION ---
                    
                    # if cfg.CONTROLNET.EVAL.MAX_VALIDATION_SAMPLES != 0 and global_step % cfg.CONTROLNET.EVAL.VALIDATION_STEPS == 0:
                    if len(val_dataloader) > 0 and (global_step - last_validation_at_step) >= cfg.CONTROLNET.EVAL.VALIDATION_STEPS:
                        last_validation_at_step = global_step
                        log_validation(
                            # vae,
                            # text_encoder,
                            # tokenizer,
                            # unet,
                            # controlnet,
                            cfg=cfg,
                            pipeline=validation_pipeline,
                            controlnet=controlnet,
                            accelerator=accelerator,
                            weight_dtype=weight_dtype,
                            step=global_step,
                            logger=logger,
                            dataloader=val_dataloader,
                            gt_synthesizer=gt_synthesizer,
                        )

            # --- CLEANUP UPDATE STEP ---

            logs = {"Train/Loss": loss.detach().item(), "Train/LR": lr_scheduler.get_last_lr()[0]}
            progress_bar.set_postfix(**logs)
            accelerator.log(logs, step=global_step)

            if global_step >= cfg.CONTROLNET.TRAIN.MAX_TRAIN_STEPS:
                break

    # --- CLEANUP ---

    # Create the pipeline using using the trained modules and save it.
    accelerator.wait_for_everyone()
    if accelerator.is_main_process:
        controlnet = unwrap_model(controlnet)
        controlnet.save_pretrained(cfg.CONTROLNET_EXP_DIR)
        
    accelerator.end_training()


if __name__ == "__main__":
    main()