#!/usr/bin/env python3
"""Generate the NVFP4 / FP8 E4M3 reference vectors in planar_reference/, which the planar
decode test in crates/lumen-format/src/planar_dequant.rs checks bit for bit.

The codes and scales come from one seeded generator; no checkpoint is read. The expected
values come from NVIDIA ModelOpt's own reference dequantizer
(`NVFP4QTensor.dequantize` / `FP8QTensor.dequantize`), never from a re-implementation of the
decode. The committed files were written with nvidia-modelopt 0.46.0, torch 2.13.0 and
numpy 2.3.5; the script runs on the CPU.

Usage:
    pip install nvidia-modelopt==0.46.0 torch==2.13.0 numpy==2.3.5 requests huggingface_hub
    CUDA_VISIBLE_DEVICES="" python3 crates/lumen-format/tests/fixtures/generate_planar_reference.py

Output (default: planar_reference/ next to this script, or the directory given as the first
argument), per tensor as raw little-endian row-major data under the names the Rust test reads:
  NVFP4: <tag>.packed.u8 (two E2M1 nibbles per byte, low nibble first),
         <tag>.blockscale.u8 (one E4M3 code per 16 weights), <tag>.globalscale.f32,
         <tag>.expected.f32
  FP8:   <tag>.e4m3.u8, <tag>.scale.f32, <tag>.expected.f32
The Rust test holds the directory to 6 tensors and 6144 values per format.
"""
import json
import os
import sys
from pathlib import Path

import numpy as np
import torch
from modelopt.torch.quantization.qtensor import FP8QTensor, NVFP4QTensor
from modelopt.torch.quantization.qtensor.nvfp4_tensor import e2m1_values

SEED = 20260926

# (tag, logical [rows, cols]); each format sums to 6144 values.
NVFP4 = [("nvfp4_a", (1, 16)), ("nvfp4_b", (16, 128)), ("nvfp4_c", (15, 272))]
FP8 = [("fp8_a", (3, 5)), ("fp8_b", (16, 128)), ("fp8_c", (53, 77))]

FINITE_E4M3 = np.array([c for c in range(256) if c not in (0x7F, 0xFF)], dtype=np.uint8)
POSITIVE_E4M3 = np.arange(0x00, 0x7F, dtype=np.uint8)  # +0 .. 448, no NaN
E4M3_SUBNORMALS = [c for c in range(256) if c & 0x78 == 0 and c & 7]  # 14 nonzero codes


def realistic_scale(rng, divisor):
    """A per-tensor scale as an exporter derives it: amax / divisor, amax in [0.05, 2)."""
    return np.float32(rng.uniform(0.05, 2.0) / divisor)


def write(out, name, arr):
    with open(os.path.join(out, name), "wb") as f:
        f.write(arr.tobytes())


def nvfp4_tensors(rng, out):
    """Write the NVFP4 tensors; return what each covers."""
    table = e2m1_values.to(torch.float32)
    covered = []
    for tag, (rows, cols) in NVFP4:
        blocks = rows * cols // 16
        if tag == "nvfp4_a":
            # One block holding each E2M1 code once, under the largest block scale.
            nibbles = rng.permutation(16).astype(np.uint8)
            packed = (nibbles[0::2] | (nibbles[1::2] << 4)).astype(np.uint8)
            block_scales = np.array([0x7E], dtype=np.uint8)
        else:
            packed = rng.integers(0, 256, size=rows * cols // 2, dtype=np.uint8)
            block_scales = rng.choice(POSITIVE_E4M3, size=blocks).astype(np.uint8)
            if tag == "nvfp4_c":
                # A zero block scale and the seven positive E4M3 subnormals as block scales.
                block_scales[rng.permutation(blocks)[:8]] = np.arange(8, dtype=np.uint8)
        global_scale = realistic_scale(rng, 6 * 448)

        qt = NVFP4QTensor(torch.Size([rows, cols]), torch.float32,
                          torch.from_numpy(packed.reshape(rows, cols // 2).copy()))
        scale = torch.from_numpy(block_scales.reshape(rows, cols // 16).copy())
        expected = qt.dequantize(dtype=torch.float32,
                                 scale=scale.view(torch.float8_e4m3fn),
                                 double_scale=torch.tensor(global_scale, dtype=torch.float32),
                                 block_sizes={-1: 16})
        expected = expected.contiguous().numpy().astype("<f4").reshape(-1)

        # How many values the other fold order, (nibble * block scale) * global, would change.
        nibble_pairs = np.stack([packed & 0x0F, packed >> 4], axis=1)
        codes = torch.from_numpy(nibble_pairs.reshape(-1).astype(np.int64))
        bs = torch.from_numpy(block_scales.copy()).view(torch.float8_e4m3fn).to(torch.float32)
        other = (table[codes].view(-1, 16) * bs.unsqueeze(-1)).reshape(-1)
        other = other * torch.tensor(global_scale)
        changed = other.numpy().view(np.uint32) != expected.view(np.uint32)
        covered.append({
            "e2m1_codes": set(codes.tolist()),
            "code_0x8_count": int((codes == 8).sum()),
            "block_scale_subnormals": {int(c) for c in block_scales if c in E4M3_SUBNORMALS},
            "fold_order_sensitive_values": int(changed.sum()),
        })
        write(out, f"{tag}.packed.u8", packed)
        write(out, f"{tag}.blockscale.u8", block_scales)
        write(out, f"{tag}.globalscale.f32", np.array([global_scale], "<f4"))
        write(out, f"{tag}.expected.f32", expected)
    return covered


def fp8_tensors(rng, out):
    """Write the FP8 tensors; return what each covers."""
    covered = []
    for tag, (rows, cols) in FP8:
        n = rows * cols
        if tag == "fp8_a":
            # Both zeros, the smallest and largest subnormals, the smallest normal, the extremes.
            fixed = np.array([0x00, 0x80, 0x01, 0x81, 0x07, 0x87, 0x08, 0x88, 0x7E, 0xFE, 0x38,
                              0xB8], dtype=np.uint8)
            rest = rng.choice(FINITE_E4M3, n - len(fixed))
            weights = rng.permutation(np.concatenate([fixed, rest]))
        elif tag == "fp8_b":
            # Every finite E4M3 code at least once.
            rest = rng.choice(FINITE_E4M3, n - 254)
            weights = rng.permutation(np.concatenate([FINITE_E4M3, rest]))
        else:
            weights = rng.choice(FINITE_E4M3, size=n)
        weights = weights.astype(np.uint8)
        scale = realistic_scale(rng, 448)

        codes = torch.from_numpy(weights.reshape(rows, cols).copy()).view(torch.float8_e4m3fn)
        qt = FP8QTensor(torch.Size([rows, cols]), torch.float32, codes)
        expected = qt.dequantize(dtype=torch.float32,
                                 scale=torch.tensor(scale, dtype=torch.float32))
        expected = expected.contiguous().numpy().astype("<f4").reshape(-1)

        covered.append({
            "codes": set(weights.tolist()),
            "subnormal_codes": {int(c) for c in weights if c in E4M3_SUBNORMALS},
            "code_0x80_count": int((weights == 0x80).sum()),
        })
        write(out, f"{tag}.e4m3.u8", weights)
        write(out, f"{tag}.scale.f32", np.array([scale], "<f4"))
        write(out, f"{tag}.expected.f32", expected)
    return covered


def main():
    out = sys.argv[1] if len(sys.argv) > 1 else Path(__file__).parent / "planar_reference"
    assert sum(r * c for _, (r, c) in NVFP4) == 6144
    assert sum(r * c for _, (r, c) in FP8) == 6144
    os.makedirs(out, exist_ok=True)
    rng = np.random.default_rng(SEED)
    nv = nvfp4_tensors(rng, out)
    fp = fp8_tensors(rng, out)

    union = lambda sets: len(set().union(*sets))
    summary = {
        "e2m1_codes_covered": union(t["e2m1_codes"] for t in nv),
        "e4m3_nonzero_subnormals_in_fp8_weights": union(t["subnormal_codes"] for t in fp),
        "e4m3_nonzero_subnormals_in_block_scales": union(t["block_scale_subnormals"] for t in nv),
        "e4m3_finite_codes_in_fp8_weights": union(t["codes"] for t in fp),
        "code_0x8_nibbles": sum(t["code_0x8_count"] for t in nv),
        "code_0x80_bytes": sum(t["code_0x80_count"] for t in fp),
        "fold_order_sensitive_values": sum(t["fold_order_sensitive_values"] for t in nv),
        "total_bytes": sum(os.path.getsize(os.path.join(out, f)) for f in os.listdir(out)),
        "cuda_initialized": torch.cuda.is_initialized(),
    }
    assert summary["e2m1_codes_covered"] == 16
    assert summary["e4m3_nonzero_subnormals_in_fp8_weights"] == 14
    assert summary["e4m3_nonzero_subnormals_in_block_scales"] == 7
    assert summary["e4m3_finite_codes_in_fp8_weights"] == 254
    assert summary["code_0x8_nibbles"] > 0 and summary["code_0x80_bytes"] > 0
    assert summary["fold_order_sensitive_values"] > 0
    assert summary["total_bytes"] <= 64 * 1024
    assert not summary["cuda_initialized"]
    print(json.dumps(summary))


if __name__ == "__main__":
    main()
