#!/usr/bin/env bash
#
# finalize-release.sh — from the already-built, already-VALIDATED artifacts,
# package the Linux/CUDA tarball + checksums and generate a FILLED Homebrew
# formula. Run by release.yml's publish-release job (after validation is green).
#
# Inputs (set up by the workflow):
#   $TAG          git tag, e.g. v1.0.0 or v1.2.0-rc.1
#   dist/         macOS arm64 tarball + .sha256 (downloaded build-macos artifact)
#   linux-bins/   lumen, lumen-server, lbi-convert (downloaded, validated Linux/CUDA binaries)
#
# Outputs (into dist/, which the Release step uploads):
#   lumen-<tag>-linux-x86_64-cuda.tar.gz (+ .sha256)
#   lumen.rb     (Homebrew formula with the real url/version/sha256 filled in)
set -euo pipefail

ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/../.." && pwd)"
cd "$ROOT"
: "${TAG:?TAG env required}"
REPO="faisalmumtaz89/Lumen"
IMAGE="ghcr.io/$(printf '%s' "$REPO" | tr '[:upper:]' '[:lower:]')"  # ghcr names are lowercase
mkdir -p dist

# ── Linux/CUDA raw-binary tarball (for users who don't want Docker) ───────────
stage="$(mktemp -d)/lumen-${TAG}-linux-x86_64-cuda"
mkdir -p "$stage/bin"
cp linux-bins/lumen linux-bins/lumen-server linux-bins/lbi-convert "$stage/bin/"
chmod +x "$stage/bin/"*
# Unguarded cp: a missing legal file must fail the release, not skip silently.
for lic in LICENSE-APACHE LICENSE-MIT THIRD_PARTY_NOTICES.md; do cp "$lic" "$stage/"; done
cat > "$stage/README.txt" <<EOF
Lumen — LLM and image inference for Linux x86_64 / NVIDIA CUDA
Build: ${TAG}

PREREQUISITES
  - NVIDIA GPU with compute capability 8.0 or newer, and its driver: 535 or newer
    with CUDA 12, 580 or newer with CUDA 13.
  - cuBLAS and NVRTC loadable at run time (no build-time CUDA SDK): on the system's
    library paths from CUDA 12 or 13, or in lib/lumen beside bin/, where the
    installer (https://servelumen.com/install.sh) puts NVIDIA's copies when the
    system has none — or just run the published Docker image:
    ${IMAGE}
  - Kernels compile at the first run via NVRTC and are cached, so later launches
    skip the compile.

INSTALL   sudo cp bin/lumen bin/lumen-server bin/lbi-convert /usr/local/bin/
RUN       lumen pull qwen3.5-9b:q8_0
          lumen-server --model qwen3.5-9b --quant q8_0 --backend cuda --port 8000
IMAGES    lbi-convert /path/to/Qwen-Image-2.1 /path/to/lbi
          LUMEN_IMAGE_LBI=/path/to/lbi LUMEN_IMAGE_CKPT=/path/to/Qwen-Image-2.1 lumen-server --port 8001
          (an image-only server, separate from the text server; give it no model)
          https://github.com/${REPO}/blob/main/docs/image-generation.md
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
