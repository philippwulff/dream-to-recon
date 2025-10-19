from typing import Any, List, Optional, Callable
import torch
import numpy as np
from torchvision.transforms import ToPILImage
from transformers import DPTImageProcessor, DPTForDepthEstimation
from diffusers import (
    StableDiffusionPipeline,
    StableDiffusionImg2ImgPipeline,
    StableDiffusionInpaintPipeline,
    StableDiffusionControlNetPipeline,
    StableDiffusionControlNetImg2ImgPipeline,
    StableDiffusionControlNetInpaintPipeline, 
    ControlNetModel
)
from dataclasses import dataclass



def make_inpaint_condition(init_image, mask_image) -> torch.Tensor:
    """Controlnet for inpainting conditioning.
    From: https://huggingface.co/lllyasviel/control_v11p_sd15_inpaint#example
    :param init_image: [B, C, H, W]
    :param mask_image: [B, 1, H, W]
    """
    img = init_image.clone()
    
    assert init_image.shape[2:3] == mask_image.shape[2:3], "image and image_mask must have the same image size"
    
    C = init_image.shape[1]
    img[mask_image.repeat(1, C, 1, 1) > 0.5] = -1.0  # set as masked pixel
    return img


def make_depth_condition(depth_image):
    """Controlnet for depth conditioning.
    :param depth_image: [1, 1, H, W]
    """
    return ToPILImage()(depth_image.repeat(1, 3, 1, 1).squeeze())


@dataclass
class ControlNetConfig:
    NAME: str
    MODEL_PATH: str
    CONTROLNET_CONDITIONING_SCALE: float = 1.0
    CONDITIONING_FUNC: Callable = lambda x: x
    
    
@dataclass
class ModelCallConfig:
    STRENGTH: float = 1.0
    NUM_INFERENCE_STEPS: int = 50
    GUIDANCE_SCALE: float = 7.5
    PROMPT: str = ""
    NEGATIVE_PROMPT: str = ""
    SEED: Optional[str] = None
    

@dataclass
class ModelConfig:
    NAME: str
    MODEL_PATH: str
    CONTROLNET_CONFIGS: Optional[List[ControlNetConfig]] = None
    

class InpainterAndImg2Img:
    """
    Models:
    -------
    CompVis/stable-diffusion-v1-4
    """
    
    def __init__(
        self, 
        model_cfgs_inpaint_img2img: List[ModelConfig], 
        model_call_cfgs_inpaint_img2img: List[ModelConfig], 
        device: str = "cpu"
        ) -> None:
        
        assert len(model_cfgs_inpaint_img2img) <= 2, "CAN ONLY CREATE MAX TWO DIFFERENT MODELS: INPAINT + IMG2IMG"
        assert len(model_call_cfgs_inpaint_img2img) <= 2, "CAN ONLY CREATE MAX TWO DIFFERENT MODELS: INPAINT + IMG2IMG"
        
        self.device = device
        
        model_cfg = None
        model_cfg_inpaint = None
        model_cfg_img2img = None
        self.controlnet_conditioning_scale_inpaint = []
        self.controlnet_conditioning_scale_img2img = []
        self.controlnet_conditioning_funcs_inpaint = []
        self.controlnet_conditioning_funcs_img2img = []
        
        if len(model_cfgs_inpaint_img2img) == 1:
            # We use the same model for inpainting and img2img
            
            model_cfg = model_cfgs_inpaint_img2img[0]
            if model_cfg.CONTROLNET_CONFIGS:
                # If both use a controlnet
                controlnet = [
                    ControlNetModel.from_pretrained(_.MODEL_PATH, torch_dtype=torch.float16).to(device) for _ in model_cfg.CONTROLNET_CONFIGS
                ]
                self.text2img = StableDiffusionControlNetPipeline.from_pretrained(model_cfg.MODEL_PATH, controlnet=controlnet, torch_dtype=torch.float16).to(device)
                components_wo_image_encoder = self.text2img.components
                components_wo_image_encoder.pop("image_encoder")
                self.img2img = StableDiffusionControlNetImg2ImgPipeline(**components_wo_image_encoder).to(device)
                self.inpaint = StableDiffusionControlNetInpaintPipeline(**self.text2img.components).to(device)
                # self.text2img = None
                # self.img2img = StableDiffusionControlNetImg2ImgPipeline.from_pretrained(model_cfg.MODEL_PATH, controlnet=controlnet, torch_dtype=torch.float16).to(device)
                # self.img2img = StableDiffusionImg2ImgPipeline.from_pretrained(model_cfg.MODEL_PATH, torch_dtype=torch.float16).to(device)
                # self.inpaint = StableDiffusionControlNetInpaintPipeline.from_pretrained(model_cfg.MODEL_PATH, controlnet=controlnet, torch_dtype=torch.float16).to(device)
                
                self.controlnet_conditioning_scale_inpaint = [_.CONTROLNET_CONDITIONING_SCALE for _ in model_cfg.CONTROLNET_CONFIGS]
                self.controlnet_conditioning_scale_img2img = self.controlnet_conditioning_scale_inpaint
                self.controlnet_conditioning_funcs_inpaint = [_.CONDITIONING_FUNC for _ in model_cfg.CONTROLNET_CONFIGS]
                self.controlnet_conditioning_funcs_img2img = self.controlnet_conditioning_funcs_inpaint
            else:
                self.text2img = StableDiffusionPipeline.from_pretrained(model_cfg.MODEL_PATH, torch_dtype=torch.float16).to(device)
                self.img2img = StableDiffusionImg2ImgPipeline(**self.text2img.components).to(device)
                self.inpaint = StableDiffusionInpaintPipeline(**self.text2img.components).to(device)
            
        else:
            # There are two different models for inpainting and img2img
            
            model_cfg_inpaint = model_cfgs_inpaint_img2img[0]
            model_cfg_img2img = model_cfgs_inpaint_img2img[1]
            
            if model_cfg_inpaint.CONTROLNET_CONFIGS:
                # Inpainting has controlnet
                controlnet = [
                    ControlNetModel.from_pretrained(_.MODEL_PATH, torch_dtype=torch.float16).to(device) for _ in model_cfg_inpaint.CONTROLNET_CONFIGS
                ]
                self.inpaint = StableDiffusionControlNetInpaintPipeline.from_pretrained(model_cfg_inpaint.MODEL_PATH, controlnet=controlnet, torch_dtype=torch.float16).to(device)
                self.controlnet_conditioning_scale_inpaint = [_.CONTROLNET_CONDITIONING_SCALE for _ in model_cfg_inpaint.CONTROLNET_CONFIGS]
                self.controlnet_conditioning_funcs_inpaint = [_.CONDITIONING_FUNC for _ in model_cfg_inpaint.CONTROLNET_CONFIGS]
                self.inpaint.enable_model_cpu_offload()     # Calling this for the normal SD pipeline causes an OOM error.
            else:
                # Inpainting has no controlnet
                self.inpaint = StableDiffusionInpaintPipeline.from_pretrained(model_cfg_inpaint.MODEL_PATH, torch_dtype=torch.float16).to(device)

            if model_cfg_img2img.CONTROLNET_CONFIGS:
                controlnet = [
                    ControlNetModel.from_pretrained(_.MODEL_PATH, torch_dtype=torch.float16).to(device) for _ in model_cfg_img2img.CONTROLNET_CONFIGS
                ]
                self.img2img = StableDiffusionControlNetImg2ImgPipeline.from_pretrained(model_cfg_img2img.MODEL_PATH, controlnet=controlnet, torch_dtype=torch.float16).to(device)
                self.controlnet_conditioning_scale_img2img = [_.CONTROLNET_CONDITIONING_SCALE for _ in model_cfg_img2img.CONTROLNET_CONFIGS]
                self.controlnet_conditioning_funcs_img2img = [_.CONDITIONING_FUNC for _ in model_cfg_img2img.CONTROLNET_CONFIGS]
                self.img2img.enable_model_cpu_offload()
            else:
                self.img2img = StableDiffusionImg2ImgPipeline.from_pretrained(model_cfg_img2img.MODEL_PATH, torch_dtype=torch.float16).to(device)
            
            
        self.img2img.enable_xformers_memory_efficient_attention()
        self.img2img.set_progress_bar_config(disable=True)
        self.inpaint.enable_xformers_memory_efficient_attention()
        self.inpaint.set_progress_bar_config(disable=True)
        
        if len(model_call_cfgs_inpaint_img2img) == 2:
            self.model_call_cfg_inpaint, self.model_call_cfg_img2img = model_call_cfgs_inpaint_img2img
        else:
            self.model_call_cfg_inpaint = model_call_cfgs_inpaint_img2img[0]
            self.model_call_cfg_img2img = model_call_cfgs_inpaint_img2img[0]
        
    
    def run_img2img(self, init_img_b, control_imgs_b=None, verbose=False):
        """
        :param init_img_b: [B, C, H, W]
        :param control_imgs_b: list of [1, C, H, W]
        """
        B, _, _, _ = init_img_b.shape
        NUM_CONTROL_NETS = len(self.controlnet_conditioning_scale_inpaint)
        
        assert NUM_CONTROL_NETS == 0 or B == 1 #or  == 1
        
        imgs_PIL = [ToPILImage()(init_img_b[i]) for i in range(B)]
        
        kwargs = dict(
            image=imgs_PIL[0],
            prompt=self.model_call_cfg_inpaint.PROMPT, 
            negative_prompt=self.model_call_cfg_inpaint.NEGATIVE_PROMPT,
            guidance_scale=self.model_call_cfg_img2img.GUIDANCE_SCALE,
        )
        if self.model_call_cfg_img2img.STRENGTH:
            kwargs["strength"] = self.model_call_cfg_img2img.STRENGTH
        if self.model_call_cfg_img2img.NUM_INFERENCE_STEPS:
            kwargs["num_inference_steps"] = self.model_call_cfg_img2img.NUM_INFERENCE_STEPS 
        if self.controlnet_conditioning_scale_img2img:
            kwargs["control_image"] = [f(img.to(self.device)) for f, img in zip(self.controlnet_conditioning_funcs_img2img, control_imgs_b)]
            kwargs["controlnet_conditioning_scale"] = self.controlnet_conditioning_scale_img2img
            # For guess_mode=True: "A guidance_scale value between 3.0 and 5.0 is recommended." 
            # https://huggingface.co/docs/diffusers/api/pipelines/controlnet#diffusers.StableDiffusionControlNetImg2ImgPipeline.__call__
            kwargs["guess_mode"] = True 
        if self.model_call_cfg_img2img.SEED:
            kwargs["generator"] = torch.Generator("cuda").manual_seed(self.model_call_cfg_img2img.SEED)
            
        if verbose:
            print("RUNNING IMG2IMG WITH ARGS: ", kwargs)
            self.img2img.set_progress_bar_config(disable=False)
            
        output = self.img2img(**kwargs)
        return output.images


    def run_inpaint(self, init_img_b, mask_img_b, control_imgs_b, verbose=False):
        """
        :param init_img_b: [B, C, H, W]
        :param mask_img_b: [B, 1, H, W]
        :param control_imgs_b: list of [1, C, H, W]
        """
        B, _, _, _ = init_img_b.shape
        NUM_CONTROL_NETS = len(self.controlnet_conditioning_scale_inpaint)
        
        assert NUM_CONTROL_NETS == 0 or B == 1 #or  == 1
        
        imgs_PIL = [ToPILImage()(init_img_b[i]) for i in range(B)]
        masks_PIL = [ToPILImage()(mask_img_b[i].squeeze().numpy().astype(np.uint8)*255) for i in range(B)]
        
        kwargs = dict(
            image=imgs_PIL,
            mask_image=masks_PIL,
            prompt=self.model_call_cfg_inpaint.PROMPT, 
            negative_prompt=self.model_call_cfg_inpaint.NEGATIVE_PROMPT,
            guidance_scale=self.model_call_cfg_inpaint.GUIDANCE_SCALE,
        )
        
        control_image = []
        if self.model_call_cfg_inpaint.STRENGTH:
            kwargs["strength"] = self.model_call_cfg_inpaint.STRENGTH
        if self.model_call_cfg_inpaint.NUM_INFERENCE_STEPS:
            kwargs["num_inference_steps"] = self.model_call_cfg_inpaint.NUM_INFERENCE_STEPS 
        if self.controlnet_conditioning_scale_inpaint:
            # if NUM_CONTROL_NETS == 1:
            #     f = self.controlnet_conditioning_funcs_inpaint[0]
            #     control_image = [f(img.to(self.device)) for img in control_imgs_b]
            # else:
            #     control_image = [f(img.to(self.device)) for f, img in zip(self.controlnet_conditioning_funcs_inpaint, control_imgs_b)]
            # control_image = [[f(img.to(self.device)) for f, img in zip(self.controlnet_conditioning_funcs_inpaint, control_imgs_b)]] * B
            control_image = [f(img.to(self.device)) for f, img in zip(self.controlnet_conditioning_funcs_inpaint, control_imgs_b)]
            kwargs["control_image"] = control_image
            kwargs["controlnet_conditioning_scale"] = self.controlnet_conditioning_scale_inpaint
            # For guess_mode=True: "A guidance_scale value between 3.0 and 5.0 is recommended." 
            # https://huggingface.co/docs/diffusers/api/pipelines/controlnet#diffusers.StableDiffusionControlNetImg2ImgPipeline.__call__
            # kwargs["guess_mode"] = True 
            # kwargs["controlnet_conditioning_scale"] = [self.controlnet_conditioning_scale_inpaint] * B
        if self.model_call_cfg_inpaint.SEED:
            kwargs["generator"] = torch.Generator(self.device).manual_seed(self.model_call_cfg_inpaint.SEED)
        
        if verbose:
            print("RUNNING INPAINT WITH ARGS: ", kwargs)
            self.inpaint.set_progress_bar_config(disable=False)
        
        output = self.inpaint(**kwargs)
        
        return output.images, [imgs_PIL, masks_PIL, control_image]
        
    
    def __call__(self, init_img_b, mask_img_b, control_img_b) -> Any:
        
        output_inpaint = self.run_inpaint(init_img_b, mask_img_b, control_img_b)
        
        output_img2img = self.run_inpaint(output_inpaint, control_img_b)
        
        return output_inpaint, output_img2img
    
    
def get_DPT_large_depth(imgs: torch.Tensor) -> torch.Tensor:
    """Predicts depth using DPT-L.
    :param imgs: [B, C, H, W] of type torch.uint8 in [0, 255].
    """
    assert imgs.dtype == torch.uint8, "DTYPE SHOULD BE UINT8"
    # depth maps on SD images
    processor = DPTImageProcessor.from_pretrained("Intel/dpt-large")
    model = DPTForDepthEstimation.from_pretrained("Intel/dpt-large")
    inputs = processor(images=imgs, return_tensors="pt")
    
    with torch.no_grad():
        outputs = model(**inputs)
        predicted_depth = outputs.predicted_depth 
        if len(predicted_depth.shape) == 3:
            predicted_depth = predicted_depth.unsqueeze(0)
    
    # interpolate to original size
    prediction = torch.nn.functional.interpolate(
        predicted_depth,
        size=imgs.shape[2:],
        mode="bicubic",
        align_corners=False,
    )
    
    output = prediction.squeeze().cpu().numpy()
    formatted = (output * 255 / np.max(output)).astype("uint8")
    #depth_sd = [Image.fromarray(formatted[i, :, :]) for i in range(len(formatted))]
    return torch.tensor(formatted[None, None, :, :])


def get_zoe_depth(imgs):
    """"""
    
    