# LBC Binary Format

The Layer-Blob Container (`.lbc`) is Lumen's on-disk model format. It is designed for zero-copy mmap, per-tensor mixed quantization, and one-layer-at-a-time conversion from GGUF.

## Current version

`LBC_VERSION = 4`; a file whose embedding is an as-stored K-quant plane is written as version 5 (`LBC_VERSION_KQUANT_EMBEDDING`), the newest the reader accepts. Source-of-truth: [`crates/lumen-format/`](../crates/lumen-format/).

## Properties

- **128 KiB-aligned blobs** so that the kernel can mmap into HugeTLB / page-cached memory cleanly
- **CRC32 header** for corruption detection at load
- **Zero-copy mmap** at the runtime side — weights are not copied into a Rust `Vec`
- **Per-tensor quantization**: a single file may mix BF16, Q8_0, and Q4_0 tensors (e.g. dense FFN at Q8_0 while output projection stays BF16)
- **Backward compatibility policy**: the LBC reader rejects `version > LBC_VERSION_KQUANT_EMBEDDING` with `UnsupportedVersion`; a Lumen before this release refuses a version-5 file (a `Q4_K_M` / `Q5_K_M` artifact whose embedding is an as-stored K-quant plane) the same way; every other file keeps version 4 and opens as before. Backward-compat for v1 / v2 is in the code path but unverified at runtime (no older LBC files on disk to test against). **Policy: rebuild LBCs after major Lumen upgrades** via `lumen convert` or `lumen pull --quant <scheme>`.

## Layout (high level)

```text
+----------------------+
| Magic + version      |
| CRC32 header         |
| Tensor table         |
+----------------------+
| 128 KiB-aligned blob 0
| 128 KiB-aligned blob 1
| ...
+----------------------+
```

Each tensor entry in the table includes name, dtype, dimensions, byte offset, and byte length. The runtime memo-izes (name → layer index, role) via [`crates/lumen-convert/src/tensor_names.rs`](../crates/lumen-convert/src/tensor_names.rs).

## Quantization descriptors

| Quant | Bytes per element (effective) | Notes |
|---|---|---|
| BF16  | 2 | Reference precision; highest quality |
| Q8_0  | ~1.06 | 32-element groups, F16 scale per group |
| Q4_0  | ~0.56 | 32-element groups, F16 scale per group |
| CtInt4G32 | ~0.58 | Imported compressed-tensors "pack-quantized" INT4: 32-element groups, BF16 scale + 4-bit zero-point per group. Every quantized value is preserved exactly (no dequantization; rows/column-blocks are reindexed where the GGUF tensor conventions require it). CUDA runtime only. |
| Q4_K / Q5_K / Q6_K | 0.5625 / 0.6875 / 0.8203 | GGML superblocks as stored: 256 elements in 144 / 176 / 210 bytes (Q4_K, Q5_K: f16 `d` and `dmin`, eight 6-bit scales and mins packed in 12 bytes, the nibbles, Q5_K's fifth bits; Q6_K: the low nibbles, the high 2-bit pairs, 16 int8 scales, f16 `d`). Carried by the generic target; served by the CUDA K-quant kernels. |

Q4_K / Q5_K / Q6_K FFN planes of a K-quant source (`Q4_K_M`, `Q5_K_M`: a file with K-quant dense FFN projections; its Q4_K / Q5_K / Q6_K planes are the ones served natively, a Q2_K / Q3_K layer plane the generic target carries keeps the Q8_0 upcast on Metal and the host dequant at load on CUDA) are carried verbatim on the generic target and served natively by the CUDA K-quant kernels; so are a K-quant embedding of whole superblocks, a kept `ssm_out` and a preserved Q6_K head, while a Q4_K / Q5_K head is re-quantised (Q8_0 by default). The header carries the source's scheme. The Metal target has no K-quant kernel, so it converts such a source exactly as it did before: every K-quant layer plane upcast to Q8_0, the head re-quantised, a K-quant embedding dequantised to F32 — and a Metal runtime refuses an as-stored K-quant layer plane and names the re-conversion (an as-stored K-quant embedding or a preserved Q6_K head it serves through its F32 dequant copy, the head exactly as it always has). On any other source (a Q4_0 / Q8_0 / BF16 file with an occasional K-quant layer plane) the Metal target upcasts such a plane to Q8_0 and the generic target carries it for CUDA's host dequant at load (with dedicated CUDA kernels for a fidelity-preserved Q5_K `ssm_out` and Q6_K output head). Q2_K / Q3_K have no general kernels on either backend. MXFP4 has no LBC representation: required MXFP4 layer tensors are rejected at conversion; optional MXFP4 tensors are dequantized to F32 (MoE shared-expert gate/up planes to Q4_0).

## Conversion

```bash
# Convert GGUF -> LBC, preserving source quantization
lumen convert --input model.gguf --output model.lbc

# Convert + re-quantize
lumen convert --input model.gguf --output model.lbc --requant q4_0   # dense models only

# Import a Hugging Face compressed-tensors checkpoint (pack-quantized INT4
# group-32, indexed sharded safetensors, dense qwen35-family models only);
# the GGUF supplies tokenizer + hyperparameters only
lumen convert --input donor.gguf --from-hf /path/to/hf-checkpoint --output model.lbc
```

The GGUF converter streams one layer at a time, filters the MTP (Next-N) head, and currently accepts the v1 architecture set (`qwen35` / `qwen35moe`); the HF import path accepts dense `qwen35` models only. Architectures outside these sets are rejected at conversion ([`crates/lumen-convert/src/hyperparams.rs`](../crates/lumen-convert/src/hyperparams.rs)) and additional architecture entries will be added as new model families ship.
