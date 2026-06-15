#!/usr/bin/env bash
#
# finalize-release.sh — from the already-built, already-VALIDATED artifacts,
# package the Linux/CUDA tarball + checksums and generate a FILLED Homebrew
# formula. Run by release.yml's publish-release job (after validation is green).
#
# Inputs (set up by the workflow):
#   $TAG          git tag, e.g. v1.0.0 or v1.2.0-rc.1
#   dist/         macOS arm64 tarball + .sha256 (downloaded build-macos artifact)
#   linux-bins/   lumen, lumen-server (downloaded, validated Linux/CUDA binaries)
#
# Outputs (into dist/, which the Release step uploads):
#   lumen-<tag>-linux-x86_64-cuda.tar.gz (+ .sha256)
#   lumen.rb     (Homebrew formula with the real url/version/sha256 filled in)
set -euo pipefail

ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/../.." && pwd)"
cd "$ROOT"
: "${TAG:?TAG env required}"
REPO="faisalmumtaz89/Lumen"
mkdir -p dist

# ── Linux/CUDA raw-binary tarball (for users who don't want Docker) ───────────
stage="$(mktemp -d)/lumen-${TAG}-linux-x86_64-cuda"
mkdir -p "$stage/bin"
cp linux-bins/lumen linux-bins/lumen-server "$stage/bin/"
chmod +x "$stage/bin/"*
for lic in LICENSE-APACHE LICENSE-MIT LICENSE; do [ -f "$lic" ] && cp "$lic" "$stage/"; done
cat > "$stage/README.txt" <<EOF
Lumen — LLM inference for Linux x86_64 / NVIDIA CUDA
Build: ${TAG}

PREREQUISITES
  - NVIDIA GPU + driver (>= 525 recommended).
  - CUDA runtime libs libnvrtc + libcublas present at run time (loaded dynamically;
    no build-time CUDA SDK). On Ubuntu: the cuda-nvrtc-12-2 + libcublas-12-2
    packages — or just run the published Docker image: ghcr.io/${REPO}.
  - Kernels JIT-compile at first run via NVRTC; the PTX disk cache makes subsequent
    launches sub-second (set LUMEN_CACHE_DIR to persist it).

INSTALL   sudo cp bin/lumen bin/lumen-server /usr/local/bin/
RUN       lumen pull qwen3.5-9b:q8_0
          lumen-server --model qwen3.5-9b --quant q8_0 --backend cuda --port 8000
EOF
tar -C "$(dirname "$stage")" -czf "dist/lumen-${TAG}-linux-x86_64-cuda.tar.gz" "$(basename "$stage")"
( cd dist && shasum -a 256 "lumen-${TAG}-linux-x86_64-cuda.tar.gz" > "lumen-${TAG}-linux-x86_64-cuda.tar.gz.sha256" )

# Tag-less Linux alias so `releases/latest/download/lumen-linux-x86_64-cuda.tar.gz`
# resolves (mirrors the macOS alias in build-tarball.sh) — the installer's stable URL.
cp "dist/lumen-${TAG}-linux-x86_64-cuda.tar.gz" "dist/lumen-linux-x86_64-cuda.tar.gz"
( cd dist && shasum -a 256 "lumen-linux-x86_64-cuda.tar.gz" > "lumen-linux-x86_64-cuda.tar.gz.sha256" )

# ── Filled Homebrew formula (from the macOS tarball that will ship) ───────────
mac_tb="lumen-${TAG}-macos-arm64-metal.tar.gz"
[ -f "dist/$mac_tb" ] || { echo "error: dist/$mac_tb missing (build-macos artifact not downloaded?)" >&2; exit 1; }
mac_sha="$(shasum -a 256 "dist/$mac_tb" | awk '{print $1}')"
url="https://github.com/${REPO}/releases/download/${TAG}/${mac_tb}"
ver="${TAG#v}"; ver="${ver#b}"   # strip v / b prefix for the formula version field
sed -e "s|__URL__|${url}|g" -e "s|__VERSION__|${ver}|g" -e "s|__SHA256__|${mac_sha}|g" \
    packaging/homebrew/lumen.rb.in > dist/lumen.rb

echo "[finalize] dist/ contents:"; ls -1 dist
echo "[finalize] homebrew sha256=${mac_sha} version=${ver}"
echo "[finalize] To publish to a tap: copy dist/lumen.rb into faisalmumtaz89/homebrew-lumen (Formula/lumen.rb)."
