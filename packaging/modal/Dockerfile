# Lumen Phase-1 CUDA Docker image (shippable inference server).
#
# This is the Phase-1 Docker deliverable: a single image that runs Lumen's
# CUDA inference server on ANY NVIDIA GPU whose compute capability the runtime
# NVRTC toolchain supports (validated sm_75 / sm_80 / sm_86 / sm_89 / sm_90 --
# T4 / A100 / A10G / L4 / H100).
#
# Why this works across architectures with ONE image:
#   * lumen-server is a Linux x86_64 binary that defers ALL GPU-kernel
#     compilation to runtime via NVRTC (no build-time CUDA SDK link). It
#     dlopens libcuda (the driver, provided by the host via the NVIDIA
#     Container Toolkit), libnvrtc, and libcublas at startup.
#   * On first launch NVRTC compiles every kernel for the *actual* device's
#     compute capability; the persistent PTX disk cache (keyed on
#     arch || cc || nvrtc_version || driver_version) makes subsequent launches
#     on the same arch a sub-second cuModuleLoadData sweep.
#
# Base: CUDA 12.2 *devel* (NOT runtime) -- the binary dlopens the standalone
# libnvrtc.so that ships only in the devel image. libcublas is also present.
# The host driver (libcuda) is injected at run time, so it is NOT in the image.
#
# Build:
#   docker build -t lumen-cuda:phase1 packaging/modal
# Run (requires the NVIDIA Container Toolkit on the host):
#   docker run --gpus all -p 8000:8000 \
#     -v /path/to/models:/models \
#     -v lumen-ptxcache:/cache \
#     -e LUMEN_CACHE_DIR=/cache \
#     lumen-cuda:phase1 \
#     --model /models/qwen3-5-9b-Q8_0.lbc --backend cuda --host 0.0.0.0 --port 8000

FROM nvidia/cuda:12.2.2-devel-ubuntu22.04

RUN apt-get update \
    && apt-get install -y --no-install-recommends ca-certificates \
    && rm -rf /var/lib/apt/lists/*

# The prebuilt shippable binaries. Populate ./bin first (see README.md):
#   scp <build-host>:.../release/lumen-server packaging/modal/bin/
#   scp <build-host>:.../release/lumen        packaging/modal/bin/
COPY bin/lumen-server /opt/lumen/bin/lumen-server
COPY bin/lumen        /opt/lumen/bin/lumen
RUN chmod +x /opt/lumen/bin/lumen-server /opt/lumen/bin/lumen

ENV PATH="/opt/lumen/bin:${PATH}"
# Default PTX cache location (override by mounting a volume + LUMEN_CACHE_DIR).
ENV LUMEN_CACHE_DIR=/cache
# Make the runtime NVRTC / cuBLAS discoverable (devel image installs them here).
ENV LD_LIBRARY_PATH="/usr/local/cuda/lib64:${LD_LIBRARY_PATH}"

EXPOSE 8000
ENTRYPOINT ["/opt/lumen/bin/lumen-server"]
CMD ["--help"]
