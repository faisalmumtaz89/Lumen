# LBC Binary Format

The Layer-Blob Container (`.lbc`) is Lumen's on-disk model format. It is designed for zero-copy mmap, per-tensor mixed quantization, and one-layer-at-a-time conversion from GGUF.

## Current version

`LBC_VERSION = 3`. Source-of-truth: [`crates/lumen-format/`](../crates/lumen-format/).

## Properties

- **128 KiB-aligned blobs** so that the kernel can mmap into HugeTLB / page-cached memory cleanly
- **CRC32 header** for corruption detection at load
- **Zero-copy mmap** at the runtime side — weights are not copied into a Rust `Vec`
- **Per-tensor quantization**: a single file may mix BF16, Q8_0, and Q4_0 tensors (e.g. dense FFN at Q8_0 while output projection stays BF16)
- **Backward compatibility policy**: the LBC reader rejects `version > LBC_VERSION` with `UnsupportedVersion`. Backward-compat for v1 / v2 is in the code path but unverified at runtime (no older LBC files on disk to test against). **Policy: rebuild LBCs after major Lumen upgrades** via `lumen convert` or `lumen pull --quant <scheme>`.

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

K-quants (Q4_K / Q5_K / Q6_K / Q2_K / Q3_K) and MXFP4 are not runtime-supported; they are dequantized to Q8_0 / BF16 at convert-time.

## Conversion

```bash
# Convert GGUF -> LBC, preserving source quantization
lumen convert --input model.gguf --output model.lbc

# Convert + re-quantize
lumen convert --input model.gguf --output model.lbc --requant q4_0
```

The converter streams one layer at a time, filters the MTP (Next-N) head, and currently accepts the v1 architecture set (`qwen35` / `qwen35moe`); architectures outside the v1 set are rejected at conversion ([`crates/lumen-convert/src/hyperparams.rs`](../crates/lumen-convert/src/hyperparams.rs)) and additional architecture entries will be added as new model families ship.
