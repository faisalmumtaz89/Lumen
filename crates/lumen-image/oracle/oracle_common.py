"""Shared loading contract for the Qwen-Image-2.1 oracle.

`oracle_dump.py` imports this.

Precision rule: no global dtype override. The checkpoint configures the
autoencoder for fp32 and it must stay there; casting it (either by a global
dtype or by widening an already-rounded bf16 tensor) changes every decoded
pixel. The transformer is requested in bf16.
"""
import os

import torch
from diffusers import QwenImage21Pipeline, QwenImage21Transformer2DModel

def load_pipe(model):
    pipe = QwenImage21Pipeline.from_pretrained(
        model,
        transformer=QwenImage21Transformer2DModel.from_pretrained(
            model, subfolder="transformer", dtype=torch.bfloat16),
    )
    return pipe


def component_dtypes(pipe):
    return {name: sorted({str(p.dtype) for p in getattr(pipe, name).parameters()})
            for name in ("text_encoder", "transformer", "vae")}


def latent_dims(pipe, height, width):
    lat_h = 2 * (height // (pipe.vae_scale_factor * 2))
    lat_w = 2 * (width // (pipe.vae_scale_factor * 2))
    return lat_h, lat_w


def generate_noise(pipe, height, width, seed):
    """The noise the pipeline itself would draw, in the latent dtype."""
    lat_h, lat_w = latent_dims(pipe, height, width)
    g = torch.Generator(device="cpu").manual_seed(seed)
    raw = torch.randn((1, 1, pipe.latent_channels, lat_h, lat_w),
                      generator=g, dtype=torch.bfloat16)
    return raw.view(1, pipe.latent_channels, lat_h * lat_w).transpose(1, 2).contiguous()


def load_frozen_noise(path, pipe, height, width, seed):
    """Load frozen noise, checking the stored values exactly.

    The comparison is on the stored array, not on a re-quantized copy: an
    altered value that happens to round back to the same bf16 would otherwise
    pass, which verifies the consumed values rather than the file.
    """
    import numpy as np
    stored = np.load(path)
    fresh = generate_noise(pipe, height, width, seed).to(torch.float32).cpu().numpy()
    if stored.shape != fresh.shape:
        raise AssertionError(f"frozen noise shape {tuple(stored.shape)} != {tuple(fresh.shape)}")
    if stored.dtype != fresh.dtype:
        raise AssertionError(f"frozen noise dtype {stored.dtype} != {fresh.dtype}")
    if not np.array_equal(stored, fresh):
        raise AssertionError("frozen noise does not match the seeded generation")
    return torch.from_numpy(stored).to(torch.bfloat16)
