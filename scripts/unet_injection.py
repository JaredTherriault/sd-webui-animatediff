import os
import gc
import ast
import torch
from torch import Tensor
from torch.nn.functional import group_norm
from einops import rearrange
from typing import List, Tuple

# From AnimateDiff extension
from scripts.logging_animatediff import logger_animatediff
from motion_module import MotionWrapper, VanillaTemporalModule

# From webui
from modules import shared
from modules.devices import torch_gc, device, cpu

# Modules from ldm
from ldm.modules.diffusionmodules.openaimodel import TimestepBlock, TimestepEmbedSequential
from ldm.modules.diffusionmodules.util import GroupNorm32
from ldm.modules.attention import SpatialTransformer

#############################################
#### Code Injection #########################
MM_INJECTED_ATTR = "_mm_injected"

GROUPNORM32_ORIGINAL_FORWARD = GroupNorm32.forward

TIMESTEP_ORIGINAL_FORWARD = TimestepEmbedSequential.forward

# DDIM alpha
ORIGINAL_DDIM_BETAS = dict()

class InjectionParams:
    def __init__(self, video_length: int, unlimited_area_hack: bool) -> None:
        self.video_length = video_length
        self.unlimited_area_hack = unlimited_area_hack

def is_mm_injected_into_model(model):
    return hasattr(model.model.diffusion_model, MM_INJECTED_ATTR)

def get_mm_injected_params(model) -> InjectionParams:
    return getattr(model.model.diffusion_model, MM_INJECTED_ATTR)

def set_mm_injected_params(model, injection_params: InjectionParams):
    setattr(model.model.diffusion_model, MM_INJECTED_ATTR, injection_params)
    
def groupnorm_mm_factory(params: InjectionParams):
    def groupnorm_mm_forward(self, input):
        # axes_factor normalizes batch based on total conds and unconds passed in batch;
        # the conds and unconds per batch can change based on VRAM optimizations that may kick in
        axes_factor = input.size(0)//params.video_length

        input = rearrange(input, "(b f) c h w -> b c f h w", b=axes_factor)
        input = group_norm(input, self.num_groups, self.weight, self.bias, self.eps)
        input = rearrange(input, "b c f h w -> (b f) c h w", b=axes_factor)
        return input
    return groupnorm_mm_forward

def hack_groupnorm(params: InjectionParams):
    logger_animatediff.info(f"Hacking GroupNorm32 forward function.")
    GroupNorm32.forward = groupnorm_mm_factory(params)
    
def restore_original_groupnorm():
    logger_animatediff.info(f"Restoring GroupNorm32 forward function.")
    GroupNorm32.forward = GROUPNORM32_ORIGINAL_FORWARD
    
def hack_timestep():
    def mm_tes_forward(self, x, emb, context=None):
        for layer in self:
            if isinstance(layer, TimestepBlock):
                x = layer(x, emb)
            elif isinstance(layer, (SpatialTransformer, VanillaTemporalModule)):
                x = layer(x, context)
            else:
                x = layer(x)
        return x
    
    logger_animatediff.info(f"Hacking TimestepEmbedSequential.forward function.")
    TimestepEmbedSequential.forward = mm_tes_forward
    
def restore_original_timestep():
    logger_animatediff.info(f"Restoring TimestepEmbedSequential.forward function.")
    TimestepEmbedSequential.forward = TIMESTEP_ORIGINAL_FORWARD
    
def get_unet_input_block_indices():
    return [1, 2, 4, 5, 7, 8, 10, 11]

def get_unet_output_block_index_range():
    return 12

def evaluate_unet_output_block_index(unet_idx):
    return unet_idx % 3 == 2 and unet_idx != 11

def is_unet_middle_block_injected(unet):
    return len(unet.middle_block) > 3 # SD1.5 UNet has 3 expected middle_blocks - more means injected

def inject_motion_module_to_unet(unet, motion_module: MotionWrapper, injection_params: InjectionParams):
    logger_animatediff.info(f"Injecting motion module into UNet input blocks.")
    for mm_idx, unet_idx in enumerate(get_unet_input_block_indices()):
        mm_idx0, mm_idx1 = mm_idx // 2, mm_idx % 2
        unet.input_blocks[unet_idx].append(
            motion_module.down_blocks[mm_idx0].motion_modules[mm_idx1]
        )

    logger_animatediff.info(f"Injecting motion module into UNet output blocks.")
    for unet_idx in range(get_unet_output_block_index_range()):
        mm_idx0, mm_idx1 = unet_idx // 3, unet_idx % 3
        if evaluate_unet_output_block_index(unet_idx):
            unet.output_blocks[unet_idx].insert(
                -1, motion_module.up_blocks[mm_idx0].motion_modules[mm_idx1]
            )
        else:
            unet.output_blocks[unet_idx].append(
                motion_module.up_blocks[mm_idx0].motion_modules[mm_idx1]
            )

    if motion_module.mid_block is not None:
        logger_animatediff.info(f"Injecting motion module into UNet middle blocks.")
        unet.middle_block.insert(-1, motion_module.mid_block.motion_modules[0]) # only 1 VanillaTemporalModule
    setattr(unet, MM_INJECTED_ATTR, injection_params)

def eject_motion_module_from_unet(unet):
    logger_animatediff.info(f"Ejecting motion module from UNet input blocks.")
    for unet_idx in get_unet_input_block_indices():
        unet.input_blocks[unet_idx].pop(-1)

    logger_animatediff.info(f"Ejecting motion module from UNet output blocks.")
    for unet_idx in range(get_unet_output_block_index_range()):
        if evaluate_unet_output_block_index(unet_idx):
            unet.output_blocks[unet_idx].pop(-2)
        else:
            unet.output_blocks[unet_idx].pop(-1)
    
    if is_unet_middle_block_injected(unet):
        logger_animatediff.info(f"Ejecting motion module from UNet middle blocks.")
        unet.middle_block.pop(-2)
    delattr(unet, MM_INJECTED_ATTR)
    
def set_ddim_alpha(p, betas, alphas_cumprod, alphas_cumprod_prev):
    p.sd_model.betas = betas
    p.sd_model.alphas_cumprod = alphas_cumprod
    p.sd_model.alphas_cumprod_prev = alphas_cumprod_prev
    
def set_ddim_alpha_for_animatediff(p):
    global ORIGINAL_DDIM_BETAS
    logger_animatediff.info(f"Setting DDIM alpha for AnimateDiff compatibility.")
    beta_start = 0.00085
    beta_end = 0.012
    betas = torch.linspace(beta_start, beta_end, p.sd_model.num_timesteps, dtype=torch.float32, device=device)
    alphas = 1.0 - betas
    alphas_cumprod = torch.cumprod(alphas, dim=0)
    alphas_cumprod_prev = torch.cat(
        (torch.tensor([1.0], dtype=torch.float32, device=device), alphas_cumprod[:-1]))
    
    # Backup original DDIM data
    ORIGINAL_DDIM_BETAS = {
        "betas": p.sd_model.betas, "alphas_cumprod": p.sd_model.alphas_cumprod, "alphas_cumprod_prev": p.sd_model.alphas_cumprod_prev 
        }
    
    set_ddim_alpha(p, betas, alphas_cumprod, alphas_cumprod_prev)
    
def restore_original_ddim_alpha(p):
    global ORIGINAL_DDIM_BETAS
    assert len(ORIGINAL_DDIM_BETAS.items()) > 0 and "betas" in ORIGINAL_DDIM_BETAS and "alphas_cumprod" in ORIGINAL_DDIM_BETAS and "alphas_cumprod_prev" in ORIGINAL_DDIM_BETAS
    logger_animatediff.info(f"Restoring original DDIM alpha.")
    
    set_ddim_alpha(p, 
        ORIGINAL_DDIM_BETAS["betas"], 
        ORIGINAL_DDIM_BETAS["alphas_cumprod"], 
        ORIGINAL_DDIM_BETAS["alphas_cumprod_prev"])
    
    ORIGINAL_DDIM_BETAS = {}
    