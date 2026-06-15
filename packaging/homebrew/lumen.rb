# frozen_string_literal: true

# Homebrew formula for Lumen (Apple Silicon / Metal).
#
# Phase 1 distribution model: an UNSIGNED bottle-style tarball downloaded from
# GitHub Releases. Homebrew installs into its own prefix, so Gatekeeper
# notarization is not required for `brew install` of a CLI (this is the
# llama.cpp model). When a tag is cut, update `version`, `url`, and `sha256`
# (the build-tarball.sh output prints the sha256), then `brew install --build-from-source`
# is NOT needed — this is a precompiled tarball formula.
#
# Tap usage (once published):
#   brew tap OWNER/lumen
#   brew install lumen
#
# This formula is arm64-only by design: Lumen's kernels are M-series-tuned and
# there is no Intel build.
class Lumen < Formula
  desc "GPU-resident LLM inference engine for Apple Silicon (Metal)"
  homepage "https://github.com/OWNER/REPO"
  url "https://github.com/OWNER/REPO/releases/download/b0/lumen-b0-macos-arm64-metal.tar.gz"
  version "0.0.0" # replace with b<N> or vX.Y.Z at release
  sha256 "0000000000000000000000000000000000000000000000000000000000000000"
  license any_of: ["MIT", "Apache-2.0"] # repo is dual-licensed

  # arm64-only: refuse on Intel rather than ship a broken bottle.
  depends_on arch: :arm64
  depends_on macos: :sonoma # macOS 14+ floor (MSL 3.1)

  def install
    bin.install "bin/lumen"
    bin.install "bin/lumen-server"
  end

  def caveats
    <<~EOS
      Lumen runs on Apple Silicon with the Metal backend (no Xcode/Python/CUDA).
      Shaders compile at first launch (~1 s).

      Pull a model and chat:
        lumen pull qwen3.5-9b:q8_0
        lumen "Write a haiku about Rust"

      Or run the OpenAI-compatible server:
        lumen-server --model qwen3.5-9b --quant q8_0 --port 8000

      Models cache under ~/.cache/lumen (override with LUMEN_CACHE_DIR).
    EOS
  end

  test do
    assert_match "lumen", shell_output("#{bin}/lumen --help")
    assert_match "lumen-server", shell_output("#{bin}/lumen-server --help")
  end
end
