# Third-party notices

Lumen's GPU kernels include ports of implementation techniques and work
decompositions from the MIT-licensed projects below. The MIT license requires
their copyright and permission notices to accompany copies or substantial
portions of the software; this file carries those notices for the third-party
code ported into this source tree.

Lumen's binaries additionally link Rust crate dependencies under their own
licenses — MIT, Apache-2.0, Unicode-3.0, ISC, BSD-3-Clause,
CDLA-Permissive-2.0, and MPL-2.0 (`option-ext`, whose unmodified source is
available at <https://crates.io/crates/option-ext>). The full set is
enumerable from the lockfile with `cargo metadata`; each crate's license
text ships in its crates.io source archive.

## MLX (Apple)

Portions of the Metal shaders (notably the `qmv_*` decode GEMV family in
`crates/lumen-runtime/src/metal/shaders/matmul_q4.msl`, which ports MLX's
`qmv_fast_impl` work decomposition) are derived from
[MLX](https://github.com/ml-explore/mlx).

```
MIT License

Copyright © 2023 Apple Inc.

Permission is hereby granted, free of charge, to any person obtaining a copy
of this software and associated documentation files (the "Software"), to deal
in the Software without restriction, including without limitation the rights
to use, copy, modify, merge, publish, distribute, sublicense, and/or sell
copies of the Software, and to permit persons to whom the Software is
furnished to do so, subject to the following conditions:

The above copyright notice and this permission notice shall be included in all
copies or substantial portions of the Software.

THE SOFTWARE IS PROVIDED "AS IS", WITHOUT WARRANTY OF ANY KIND, EXPRESS OR
IMPLIED, INCLUDING BUT NOT LIMITED TO THE WARRANTIES OF MERCHANTABILITY,
FITNESS FOR A PARTICULAR PURPOSE AND NONINFRINGEMENT. IN NO EVENT SHALL THE
AUTHORS OR COPYRIGHT HOLDERS BE LIABLE FOR ANY CLAIM, DAMAGES OR OTHER
LIABILITY, WHETHER IN AN ACTION OF CONTRACT, TORT OR OTHERWISE, ARISING FROM,
OUT OF OR IN CONNECTION WITH THE SOFTWARE OR THE USE OR OTHER DEALINGS IN THE
SOFTWARE.
```

## llama.cpp / ggml

Portions of the CUDA and Metal kernels are derived from
[llama.cpp](https://github.com/ggml-org/llama.cpp), notably:

- the `mul_mat_vec_q`-style work decompositions in
  `crates/lumen-runtime/src/cuda/shaders/matvec_*_mmvq.cu`,
  `fused_glu_gemv_q8_split_mmvq.cu`, `mmv_q.cu`, and `mmv_q_moe.cu`;
- the `mul_mat_q`-style quantized GEMMs in
  `crates/lumen-runtime/src/cuda/shaders/mmq_q4_0.cu` and `mmq_q8_0.cu`, and
  the `mul_mat_vec_f`-style float matvec in `mul_mat_vec_f_bf16.cu`;
- the `mul_mat_id`-style expert-grouped MoE dispatch in
  `crates/lumen-runtime/src/cuda/shaders/moe_grouped.cu` and
  `crates/lumen-runtime/src/metal/shaders/moe.msl`;
- the ggml-metal-style tiled GEMM family in
  `crates/lumen-runtime/src/metal/shaders/gemm_*.msl` and `batched_ops.msl`.

Lumen also implements the GGML/GGUF quantization block formats defined by that
project, and `crates/lumen-convert/src/dequant.rs` reproduces reference
dequantization excerpts from `ggml-quants.c`.

```
MIT License

Copyright (c) 2023-2026 The ggml authors

Permission is hereby granted, free of charge, to any person obtaining a copy
of this software and associated documentation files (the "Software"), to deal
in the Software without restriction, including without limitation the rights
to use, copy, modify, merge, publish, distribute, sublicense, and/or sell
copies of the Software, and to permit persons to whom the Software is
furnished to do so, subject to the following conditions:

The above copyright notice and this permission notice shall be included in all
copies or substantial portions of the Software.

THE SOFTWARE IS PROVIDED "AS IS", WITHOUT WARRANTY OF ANY KIND, EXPRESS OR
IMPLIED, INCLUDING BUT NOT LIMITED TO THE WARRANTIES OF MERCHANTABILITY,
FITNESS FOR A PARTICULAR PURPOSE AND NONINFRINGEMENT. IN NO EVENT SHALL THE
AUTHORS OR COPYRIGHT HOLDERS BE LIABLE FOR ANY CLAIM, DAMAGES OR OTHER
LIABILITY, WHETHER IN AN ACTION OF CONTRACT, TORT OR OTHERWISE, ARISING FROM,
OUT OF OR IN CONNECTION WITH THE SOFTWARE OR THE USE OR OTHER DEALINGS IN THE
SOFTWARE.
```

## Qwen chat templates (Alibaba Cloud)

The test fixtures `crates/lumen-runtime/tests/fixtures/qwen35_chat_template.jinja`
and `qwen38_chat_template.jinja` reproduce the chat templates shipped in the
Qwen model repositories' `tokenizer_config.json` (Qwen Team, Alibaba Cloud),
which are distributed under the Apache License 2.0. The license text is
available at <https://www.apache.org/licenses/LICENSE-2.0>.

## Dynamically loaded system libraries

Lumen's binaries do not bundle or redistribute any NVIDIA or Apple libraries.
On CUDA hosts the runtime dynamically loads the system's NVIDIA driver and
CUDA toolkit libraries (libcuda, libnvrtc, libcublas) at run time; on macOS it
links Apple's system-provided Metal and Accelerate frameworks. Those
components remain governed by their own licenses.
