# Lumen Phase-1 cross-architecture validation (Modal) + Docker image

This directory holds the Phase-1 CUDA deliverable in two forms:

1. **`Dockerfile`** — the shippable Lumen CUDA inference-server image. One
   image runs on any supported NVIDIA arch because the binary JIT-compiles its
   kernels at runtime via NVRTC.
2. **`validate_arches.py`** — a Modal app that runs that exact binary on five
   NVIDIA compute capabilities and collects a pass/fail validation matrix.

## Why one binary covers every arch

`lumen-server` is a Linux x86_64 binary with **no build-time CUDA SDK link**.
It dlopens `libcuda` (host driver), `libnvrtc`, and `libcublas` at startup and
compiles every GPU kernel from source via NVRTC for whatever compute capability
the device reports. The persistent **PTX disk cache** (`crates/lumen-runtime/
src/cuda/ptx_cache.rs`) keys each compiled module on
`sha256(source) || arch || fast_math || cc || nvrtc_version || driver_version`,
so every arch gets its own cache entries and a cold→warm speedup independently.

Cache dir resolution: `$LUMEN_CUDA_PTX_CACHE_DIR` > `$LUMEN_CACHE_DIR/ptx` >
`$XDG_CACHE_HOME/lumen/ptx` > `~/.cache/lumen/ptx`. Kill switch:
`LUMEN_CUDA_PTX_CACHE=0`.

## Arch matrix (the llama.cpp arch set)

| Modal GPU | Compute cap | Notes                          |
|-----------|-------------|--------------------------------|
| T4        | sm_75       | oldest supported; int8 IMMA opt-in paths are default-OFF |
| A100      | sm_80       | Lumen's build/validation baseline |
| A10G      | sm_86       |                                |
| L4        | sm_89       |                                |
| H100      | sm_90       | newest                         |

## One-time setup (binaries + model into the image / a volume)

The binaries are baked into the image from `./bin`. Populate them from the
build host (the `cuda-ptx-cache @ b623977` release build):

```bash
scp <build-host>:/.../release/lumen-server packaging/modal/bin/lumen-server
scp <build-host>:/.../release/lumen        packaging/modal/bin/lumen
```

Upload the 9B-q8 LBC once into the shared `lumen-models` volume (mounted
read-only on every arch, so the 9 GB is downloaded once, not per-GPU):

```bash
modal volume create lumen-models      # idempotent
modal volume create lumen-ptxcache
modal volume put lumen-models /path/to/qwen3-5-9b-Q8_0.lbc /9bq8.lbc
```

## Run the validation

```bash
modal run packaging/modal/validate_arches.py                  # all five arches (parallel)
modal run packaging/modal/validate_arches.py --arches T4,H100 # a subset
```

Each arch:
1. wipes its own PTX-cache subdir, **cold-launches** the server (NVRTC compiles
   all ~252 modules for this arch), times "listening", records the
   `PTX cache cold` log line and the number of `.ptxc` files written;
2. runs **DET-001** (20× temp-0 greedy on a fixed prompt → must be 1 distinct);
3. runs a **coherence** smoke test (capital of France → Paris, 17×23 → 391, …);
4. measures rough **decode tok/s**;
5. **warm-relaunches** (cache now populated) and times "listening" again —
   warm must be much faster than cold and report `PTX cache warm`.

The container stops as soon as the function returns, so the whole matrix is a
few GPU-minutes. The driver prints a JSON matrix to stdout.

## Cost

Per-arch run is ~5–10 min on a single GPU and the container stops immediately
after. Five arches in parallel = a few GPU-minutes total (a few dollars at
most). Always confirm nothing lingers afterwards:

```bash
modal app list           # lumen-phase1-arch-validation should be 'stopped'
modal app stop lumen-phase1-arch-validation   # if it lingers
```
