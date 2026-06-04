# Contributing to Lumen

Thanks for the interest. Lumen is a single-maintainer, evidence-driven inference engine. The contribution bar is high but the path is clear.

## Scope

Lumen is a general LLM inference engine, but the production-validated surface
area today is intentionally narrow:

- **Models**: v1 ships verified-against-llama.cpp support for the Qwen3.5
  family (dense-9B and MoE-30B-A3B). Additional model families are planned;
  the converter currently rejects architectures outside the v1 set so the
  runtime only dispatches code that has been gated end-to-end. New-family
  PRs are welcome but should start with a scope-change issue describing the
  reference-token / decode-coherence gate plan.
- **Backends**: NVIDIA CUDA (Linux) and Apple Silicon Metal (macOS) as
  production targets; CPU SIMD as correctness reference.
- **Workloads**: single-stream decode (batch=1).

Pull requests that broaden scope (batched serving, speculative decoding) are
unlikely to merge without a written scope-change discussion in an issue first.
New-architecture PRs should plan for the same end-to-end gates the current
v1 family is held to (see `bench/METHODOLOGY.md`).

## Before you start

1. Engineering bar — every change is held to:
   - Every claim is anchored to a `file:line` in this repo.
   - "I think" / "it seems" is not evidence; runs of benchmarks are.
   - Performance claims require paired A/B measurements.
2. Read [`CHANGELOG.md`](CHANGELOG.md) for what has shipped.

## Build & test

```bash
# CUDA path (Linux)
cargo build --release --features cuda
cargo test --workspace --release

# Metal path (macOS)
cargo build --release
cargo test --workspace --release
```

The CPU reference suite runs without a GPU and is the minimum bar for PRs that
do not touch GPU kernels.

For exact published test counts, run `cargo test --workspace --release` locally.

## What makes a PR likely to merge

- Bit-exact output preservation on existing verified models (or explicit
  reference-token gate analysis if behavior changes).
- Performance changes ship with paired A/B benchmarks at the harness in
  `bench/` (CUDA via `modal/`/remote-A100; Metal via `bench/run_bench.sh`).
- Correctness changes ship with a unit test demonstrating the bug and the fix.
- Doc-only PRs are welcome and reviewed quickly.

## What gets rejected

- Performance claims without a paired bench
- Speculative architecture changes ("this should be faster")
- New scope (new model families, new backends, batched serving) without prior issue

## License

By contributing you agree your contribution is dual-licensed under MIT or
Apache-2.0, matching the project license (`LICENSE-MIT`, `LICENSE-APACHE`).
