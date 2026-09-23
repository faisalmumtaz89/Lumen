//! Every CUDA op against the CPU reference, on the same inputs.
//!
//! The elementwise kernels compute the same expression in the same
//! accumulation order as the CPU and are held within 1e-6 relative error, the
//! row norms and the f32 attention within 1e-5, so a disagreement there is a
//! bug, not a precision question. The tensor-core
//! paths — the bf16 projections and the fused attention — are held at bf16
//! tolerance (1e-5 and 5e-3) against the same references on bf16-exact inputs,
//! with fixtures chosen so a wrong mask, a dropped epsilon, a softmax without
//! its running maximum or the wrong rounding cannot pass.
//!
//! The ops the DiT forward adds (`dit_ops.cu`) are checked through the same
//! helpers the forward uses, against a CPU rendering of the same bf16
//! arithmetic on bf16 inputs: rounded where the reference rounds, with the
//! AdaLN kernels reading the per-token modulation row. They are held within
//! 1e-3 — above the transcendental and reduction-order ulp differences, below
//! what one missing intermediate rounding produces — and `pack_rows` and
//! `add_gated` exactly.
//!
//! Usage: `cuda-ops-check`

use std::process::ExitCode;

use lumen_image::cuda::dit_gpu::ops;
use lumen_image::cuda::{attention, launch, ImageKernels};
use lumen_image::tensor::{bf16_bits, bf16_round};
use lumen_image::tensor::{gelu_tanh, layer_norm_rows, zero_center_rms_norm_rows, Matrix};
use lumen_runtime::cuda::ffi::CudaDevice;

fn rel_l2(got: &[f32], want: &[f32]) -> f32 {
    let mut num = 0f64;
    let mut den = 0f64;
    for (g, w) in got.iter().zip(want) {
        let g = *g as f64;
        let w = *w as f64;
        num += (g - w) * (g - w);
        den += w * w;
    }
    (num.sqrt() / den.sqrt().max(1e-30)) as f32
}

fn max_abs(got: &[f32], want: &[f32]) -> f32 {
    got.iter()
        .zip(want)
        .map(|(g, w)| (g - w).abs())
        .fold(0.0f32, f32::max)
}

/// Deterministic pseudo-random input, so a failure can be reproduced.
fn pseudo(n: usize, seed: u64) -> Vec<f32> {
    let mut s = seed.wrapping_mul(6364136223846793005).wrapping_add(1);
    (0..n)
        .map(|_| {
            s = s
                .wrapping_mul(6364136223846793005)
                .wrapping_add(1442695040888963407);
            ((s >> 33) as f32 / (1u64 << 31) as f32) - 0.5
        })
        .collect()
}

/// The fused attention from f32 operands, each rounded to bf16 first.
///
/// # Safety
///
/// `q`, `k` and `v` must each hold `seq * heads * 128` f32 elements.
#[allow(clippy::too_many_arguments)]
unsafe fn fused_attention(
    dev: &CudaDevice,
    k: &ImageKernels,
    q: &launch::DevVec,
    kk: &launch::DevVec,
    v: &launch::DevVec,
    text_count: usize,
    seq: usize,
    heads: usize,
) -> Result<lumen_image::cuda::blas::Bf16Activation, String> {
    let q16 = attention::to_bf16(dev, &k.f32_to_bf16_bits, &q.buf).map_err(|e| e.to_string())?;
    let k16 = attention::to_bf16(dev, &k.f32_to_bf16_bits, &kk.buf).map_err(|e| e.to_string())?;
    let v16 = attention::to_bf16(dev, &k.f32_to_bf16_bits, &v.buf).map_err(|e| e.to_string())?;
    attention::fused_block_causal_attention(dev, k, &q16, &k16, &v16, text_count, seq, heads)
        .map_err(|e| e.to_string())
}

/// bf16 bits back on the host as f32.
fn download_bits(
    dev: &CudaDevice,
    bits: &cudarc::driver::CudaSlice<u16>,
) -> Result<Vec<f32>, String> {
    let bits = dev.dtoh_copy(bits).map_err(|e| e.to_string())?;
    dev.synchronize().map_err(|e| e.to_string())?;
    Ok(bits
        .iter()
        .map(|&b| f32::from_bits((b as u32) << 16))
        .collect())
}

struct Report {
    failures: usize,
}

impl Report {
    fn check(&mut self, name: &str, got: &[f32], want: &[f32], bar: f32) {
        let r = rel_l2(got, want);
        let m = max_abs(got, want);
        let ok = r <= bar && got.len() == want.len();
        if !ok {
            self.failures += 1;
        }
        println!(
            "  {name:24} rel-L2={r:.3e}  max_abs={m:.3e}  {}",
            if ok { "ok" } else { "FAIL" }
        );
    }
}

fn main() -> ExitCode {
    match run() {
        Ok(failures) => {
            if failures > 0 {
                eprintln!("{failures} op(s) disagreed with the CPU reference");
                ExitCode::FAILURE
            } else {
                println!("every CUDA op matched the CPU reference");
                ExitCode::SUCCESS
            }
        }
        Err(e) => {
            eprintln!("error: {e}");
            ExitCode::FAILURE
        }
    }
}

fn run() -> Result<usize, String> {
    let dev = CudaDevice::new(0).map_err(|e| format!("no CUDA device: {e}"))?;
    println!(
        "device: {}",
        dev.name().unwrap_or_else(|_| "<unknown>".into())
    );
    let k = ImageKernels::load(&dev).map_err(|e| format!("compiling kernels: {e}"))?;
    let mut rep = Report { failures: 0 };

    // --- LayerNorm with no affine params ---------------------------------
    // At scale 1e-3 the row variance (~1e-7) no longer swamps the epsilon, so
    // a kernel that dropped or misplaced it cannot pass.
    let (rows, dim) = (16usize, 128usize);
    for scale in [1.0f32, 1e-3] {
        let x: Vec<f32> = pseudo(rows * dim, 11).iter().map(|v| v * scale).collect();
        let cpu = layer_norm_rows(&Matrix::new(rows, dim, x.clone()), 1e-6);
        let gx = launch::upload(&dev, &x).map_err(|e| e.to_string())?;
        let out = launch::layernorm_noaffine(&dev, &k, &gx, rows, dim, 1e-6)
            .map_err(|e| e.to_string())?;
        let got = launch::download(&dev, &out).map_err(|e| e.to_string())?;
        rep.check(
            &format!("layernorm_noaffine x{scale}"),
            &got,
            &cpu.data,
            1e-5,
        );
    }
    let x = pseudo(rows * dim, 11);
    let gx = launch::upload(&dev, &x).map_err(|e| e.to_string())?;

    // --- scale * (1 + x) --------------------------------------------------
    let sc = pseudo(dim, 12);
    let mut want = x.clone();
    for r in 0..rows {
        for c in 0..dim {
            want[r * dim + c] *= 1.0 + sc[c];
        }
    }
    let gs = launch::upload(&dev, &sc).map_err(|e| e.to_string())?;
    let out = launch::scale_one_plus(&dev, &k, &gx, &gs, dim).map_err(|e| e.to_string())?;
    let got = launch::download(&dev, &out).map_err(|e| e.to_string())?;
    rep.check("scale_one_plus", &got, &want, 1e-6);

    // --- complex RoPE, against the same arithmetic done on the CPU --------
    let (seq, heads, head_dim) = (5usize, 4usize, 8usize);
    let q = pseudo(seq * heads * head_dim, 21);
    let mut want = q.clone();
    let freqs: Vec<f32> = pseudo(seq * head_dim, 22);
    for s in 0..seq {
        for h in 0..heads {
            for p in 0..head_dim / 2 {
                let base = (s * heads + h) * head_dim + 2 * p;
                let (xr, xi) = (q[base], q[base + 1]);
                let (cr, ci) = (freqs[s * head_dim + 2 * p], freqs[s * head_dim + 2 * p + 1]);
                want[base] = xr * cr - xi * ci;
                want[base + 1] = xr * ci + xi * cr;
            }
        }
    }
    let mut gq = launch::upload(&dev, &q).map_err(|e| e.to_string())?;
    let gf = launch::upload(&dev, &freqs).map_err(|e| e.to_string())?;
    launch::rope_complex(&dev, &k, &mut gq, &gf, seq, heads, head_dim)
        .map_err(|e| e.to_string())?;
    let got = launch::download(&dev, &gq).map_err(|e| e.to_string())?;
    rep.check("rope_complex", &got, &want, 1e-6);

    // --- block-causal attention ------------------------------------------
    // Six keys, two of which are an image block; the text prefix is causal.
    let (seq, heads, head_dim) = (6usize, 2usize, 8usize);
    let q = pseudo(seq * heads * head_dim, 31);
    let kk = pseudo(seq * heads * head_dim, 32);
    let v = pseudo(seq * heads * head_dim, 33);
    let ids: Vec<i32> = vec![-1, -1, -1, 0, 0, 0];
    let kv: Vec<i32> = vec![1; seq];
    let want = attn_cpu(&q, &kk, &v, &ids, &kv, seq, heads, head_dim);
    let gq = launch::upload(&dev, &q).map_err(|e| e.to_string())?;
    let gk = launch::upload(&dev, &kk).map_err(|e| e.to_string())?;
    let gv = launch::upload(&dev, &v).map_err(|e| e.to_string())?;
    let out = launch::attn_block_causal(&dev, &k, &gq, &gk, &gv, &ids, &kv, seq, heads, head_dim)
        .map_err(|e| e.to_string())?;
    let got = launch::download(&dev, &out).map_err(|e| e.to_string())?;
    rep.check("attn_block_causal", &got, &want, 1e-5);

    // --- padding is never attended to ------------------------------------
    let kv2: Vec<i32> = vec![1, 1, 1, 1, 0, 0];
    let want2 = attn_cpu(&q, &kk, &v, &ids, &kv2, seq, heads, head_dim);
    let out = launch::attn_block_causal(&dev, &k, &gq, &gk, &gv, &ids, &kv2, seq, heads, head_dim)
        .map_err(|e| e.to_string())?;
    let got = launch::download(&dev, &out).map_err(|e| e.to_string())?;
    rep.check("attn_block_causal padded", &got, &want2, 1e-5);

    // --- ids from the forward's own slot expansion ---------------------------
    // `image_id` is -1 at text positions, so the kernel's
    // `(qi >= kj) || (image_id[qi] >= 0 && image_id[qi] == image_id[kj])` must
    // reduce to `qi >= kj` for a text query. A kernel that dropped the sign
    // check would let text attend all text and pass the cases above, where the
    // queries are text-plus-image; this one is text-only against an image block.
    //
    // The ids are taken from `dit.rs`'s own `token_metadata` rather than written
    // out by hand, because hand-written ids are what a caller gets wrong: the
    // reference expands the *slot* mask four-fold before calling it, and an ids
    // table that skips that expansion disagrees with the kernel's joint
    // sequence without disagreeing about its length.
    let slots: Vec<bool> = vec![false, false, false, false, true];
    let expanded: Vec<bool> = slots
        .iter()
        .flat_map(|&b| std::iter::repeat(b).take(if b { 4 } else { 1 }))
        .collect();
    let (ids3, _) = lumen_image::dit::policy::token_metadata(&expanded, &[(1, 2, 2)])
        .map_err(|e| e.to_string())?;
    let ids3: Vec<i32> = ids3.iter().map(|&v| v as i32).collect();
    let (seq, heads, head_dim) = (expanded.len(), 2usize, 8usize);
    let q = pseudo(seq * heads * head_dim, 61);
    let kk = pseudo(seq * heads * head_dim, 62);
    let v = pseudo(seq * heads * head_dim, 63);
    let kv: Vec<i32> = vec![1; seq];
    let want3 = attn_cpu(&q, &kk, &v, &ids3, &kv, seq, heads, head_dim);
    let gq = launch::upload(&dev, &q).map_err(|e| e.to_string())?;
    let gk = launch::upload(&dev, &kk).map_err(|e| e.to_string())?;
    let gv = launch::upload(&dev, &v).map_err(|e| e.to_string())?;
    let out = launch::attn_block_causal(&dev, &k, &gq, &gk, &gv, &ids3, &kv, seq, heads, head_dim)
        .map_err(|e| e.to_string())?;
    let got = launch::download(&dev, &out).map_err(|e| e.to_string())?;
    rep.check("attn_block_causal slot-expanded ids", &got, &want3, 1e-5);

    // --- the production attention: fused kernel against the unfused path ---
    // The 5e-3 bar pins the arithmetic and the mask, not the rounding policy
    // of the probabilities and the output: those differ from each other by
    // less than the bar and are settled end to end on the generated image.
    // Both consume the same bf16 operands, so the inputs are rounded
    // to bf16 on the host first and the CPU reference sees exactly what the
    // kernels see. What remains is the bf16 rounding of the probabilities and
    // the accumulation order, which is why the bar is above the f32 ops'.
    {
        let (seq, heads, head_dim, text_count) = (203usize, 2usize, 128usize, 7usize);
        let bf16 = bf16_values;
        let q = bf16(
            pseudo(seq * heads * head_dim, 71)
                .iter()
                .map(|x| x * 4.0)
                .collect(),
        );
        let kk = bf16(
            pseudo(seq * heads * head_dim, 72)
                .iter()
                .map(|x| x * 4.0)
                .collect(),
        );
        let v = bf16(pseudo(seq * heads * head_dim, 73));
        let ids: Vec<i32> = (0..seq)
            .map(|i| if i < text_count { -1 } else { 0 })
            .collect();
        let kv: Vec<i32> = vec![1; seq];
        let want = attn_cpu(&q, &kk, &v, &ids, &kv, seq, heads, head_dim);
        let gq = launch::upload(&dev, &q).map_err(|e| e.to_string())?;
        let gk = launch::upload(&dev, &kk).map_err(|e| e.to_string())?;
        let gv = launch::upload(&dev, &v).map_err(|e| e.to_string())?;
        // Safety: the three buffers are `seq * heads * head_dim` floats.
        let unfused = unsafe {
            attention::block_causal_attention(
                &dev, &k, &gq, &gk, &gv, text_count, seq, heads, head_dim,
            )
        }
        .map_err(|e| e.to_string())?;
        let fused = unsafe { fused_attention(&dev, &k, &gq, &gk, &gv, text_count, seq, heads) }?;
        let got_unfused = launch::download(&dev, &unfused).map_err(|e| e.to_string())?;
        let got_fused = download_bits(&dev, &fused.bits)?;
        rep.check("attention unfused vs cpu", &got_unfused, &want, 5e-3);
        rep.check("attention fused vs cpu", &got_fused, &want, 5e-3);
        rep.check("attention fused vs unfused", &got_fused, &got_unfused, 5e-3);
    }
    // The production shape, fused against unfused only (the CPU reference is
    // too slow there): 1024x1024 is 4096 image tokens after 18 text tokens.
    {
        let (seq, heads, head_dim, text_count) = (4114usize, 32usize, 128usize, 18usize);
        let q: Vec<f32> = pseudo(seq * heads * head_dim, 81)
            .iter()
            .map(|x| x * 4.0)
            .collect();
        let kk: Vec<f32> = pseudo(seq * heads * head_dim, 82)
            .iter()
            .map(|x| x * 4.0)
            .collect();
        let v = pseudo(seq * heads * head_dim, 83);
        let gq = launch::upload(&dev, &q).map_err(|e| e.to_string())?;
        let gk = launch::upload(&dev, &kk).map_err(|e| e.to_string())?;
        let gv = launch::upload(&dev, &v).map_err(|e| e.to_string())?;
        // Safety: the three buffers are `seq * heads * head_dim` floats.
        let unfused = unsafe {
            attention::block_causal_attention(
                &dev, &k, &gq, &gk, &gv, text_count, seq, heads, head_dim,
            )
        }
        .map_err(|e| e.to_string())?;
        let fused = unsafe { fused_attention(&dev, &k, &gq, &gk, &gv, text_count, seq, heads) }?;
        let got_unfused = launch::download(&dev, &unfused).map_err(|e| e.to_string())?;
        let got_fused = download_bits(&dev, &fused.bits)?;
        rep.check(
            "attention fused vs unfused, 1024x1024 shape",
            &got_fused,
            &got_unfused,
            5e-3,
        );
        // Timing, one warm launch each.
        dev.synchronize().map_err(|e| e.to_string())?;
        let started = std::time::Instant::now();
        let out = unsafe {
            attention::block_causal_attention(
                &dev, &k, &gq, &gk, &gv, text_count, seq, heads, head_dim,
            )
        }
        .map_err(|e| e.to_string())?;
        dev.synchronize().map_err(|e| e.to_string())?;
        println!(
            "  attention unfused: {:.2} ms",
            started.elapsed().as_secs_f64() * 1e3
        );
        drop(out);
        let started = std::time::Instant::now();
        let out = unsafe { fused_attention(&dev, &k, &gq, &gk, &gv, text_count, seq, heads) }?;
        dev.synchronize().map_err(|e| e.to_string())?;
        println!(
            "  attention fused: {:.2} ms",
            started.elapsed().as_secs_f64() * 1e3
        );
        drop(out);
    }

    // --- the fused attention's mask at the block boundaries -----------------
    // A text prefix longer than one 64-row block puts text queries in a block
    // that starts inside the prefix; a prefix of exactly one block ends where
    // a block begins; no prefix at all leaves every query unmasked; and a
    // sequence that is a whole number of tiles has no partial last tile.
    // The last case scales the operands so the logits reach ~600 in log2
    // units: a softmax that did not subtract the running maximum overflows to
    // NaN there, while an ordinary fixture never leaves the range where the
    // subtraction is a no-op.
    for (seq, text_count, gain) in [
        (200usize, 70usize, 4.0f32),
        (128, 64, 4.0),
        (128, 0, 4.0),
        (128, 0, 40.0),
    ] {
        let (heads, head_dim) = (1usize, 128usize);
        let bf16 = bf16_values;
        let q = bf16(
            pseudo(seq * heads * head_dim, 91)
                .iter()
                .map(|x| x * gain)
                .collect(),
        );
        let kk = bf16(
            pseudo(seq * heads * head_dim, 92)
                .iter()
                .map(|x| x * gain)
                .collect(),
        );
        let v = bf16(pseudo(seq * heads * head_dim, 93));
        let ids: Vec<i32> = (0..seq)
            .map(|i| if i < text_count { -1 } else { 0 })
            .collect();
        let kv: Vec<i32> = vec![1; seq];
        let want = attn_cpu(&q, &kk, &v, &ids, &kv, seq, heads, head_dim);
        let gq = launch::upload(&dev, &q).map_err(|e| e.to_string())?;
        let gk = launch::upload(&dev, &kk).map_err(|e| e.to_string())?;
        let gv = launch::upload(&dev, &v).map_err(|e| e.to_string())?;
        // Safety: the three buffers are `seq * heads * head_dim` floats.
        let fused = unsafe { fused_attention(&dev, &k, &gq, &gk, &gv, text_count, seq, heads) }?;
        let got = download_bits(&dev, &fused.bits)?;
        rep.check(
            &format!("attention fused, seq {seq} text {text_count} gain {gain}"),
            &got,
            &want,
            5e-3,
        );
    }

    // --- the DiT's own additions, through the forward's helpers ------------
    let ops = ops::load(&dev).map_err(|e| e.to_string())?;

    // The f32 -> bf16 conversion on inputs that are not representable in
    // bf16, so a kernel that rounded the wrong way could not pass.
    {
        let n = 4096usize;
        let mut x = pseudo(n, 68);
        // Exact ties, where nearest-even and round-half-up differ.
        x[0] = f32::from_bits(0x3f80_8000);
        x[1] = f32::from_bits(0x3f81_8000);
        x[2] = f32::from_bits(0xbf80_8000);
        let gx = launch::upload(&dev, &x).map_err(|e| e.to_string())?;
        let rounded =
            attention::to_bf16(&dev, &k.f32_to_bf16_bits, &gx.buf).map_err(|e| e.to_string())?;
        let want: Vec<f32> = x.iter().map(|&v| bf16_round(v)).collect();
        rep.check(
            "f32_to_bf16_bits nearest-even",
            &download_bits(&dev, &rounded)?,
            &want,
            0.0,
        );
    }

    // pack_rows against the reference's joint sequence: text slots by slot
    // index, image tokens in order, encoded as negative sources.
    {
        let (text_rows, image_rows, cols) = (3usize, 8usize, 5usize);
        let txt = bf16_values(pseudo(text_rows * cols, 64));
        let img = bf16_values(pseudo(image_rows * cols, 65));
        // Two text slots, then 8 image tokens, then one more text slot.
        let source: Vec<i32> = [0, 1]
            .into_iter()
            .chain((0..image_rows as i32).map(|i| -i - 1))
            .chain([2])
            .collect();
        let want: Vec<f32> = source
            .iter()
            .flat_map(|&s| {
                if s >= 0 {
                    txt[s as usize * cols..(s as usize + 1) * cols].to_vec()
                } else {
                    let r = (-s - 1) as usize;
                    img[r * cols..(r + 1) * cols].to_vec()
                }
            })
            .collect();
        let gt = upload_bits(&dev, &txt)?;
        let gi = upload_bits(&dev, &img)?;
        let out = ops::pack_rows(&dev, &ops, &gt, &gi, &source, cols).map_err(|e| e.to_string())?;
        rep.check("pack_rows", &download_bits(&dev, &out)?, &want, 0.0);
        // A source past either buffer is refused before any launch.
        for bad in [text_rows as i32, -(image_rows as i32) - 1] {
            let refused = ops::pack_rows(&dev, &ops, &gt, &gi, &[bad], cols).is_err();
            rep.check(
                &format!("pack_rows refuses source {bad}"),
                &[if refused { 1.0 } else { 0.0 }],
                &[1.0],
                0.0,
            );
        }
    }

    // The production projection path: bf16 weight x bf16 activation on the
    // tensor cores, against `Matrix::linear` on the same bf16 values. The
    // operands are exact in bf16, so what remains is f32 accumulation order;
    // the bf16-output form is that result rounded to nearest even.
    {
        let (m, n, kd) = (37usize, 48usize, 96usize);
        let a = bf16_values(pseudo(m * kd, 66));
        let w = bf16_values(pseudo(n * kd, 67));
        let want = Matrix::new(m, kd, a.clone())
            .linear(&Matrix::new(n, kd, w.clone()), None)
            .data;
        let act = lumen_image::cuda::blas::Bf16Activation::from_bits(upload_bits(&dev, &a)?, m, kd)
            .map_err(|e| e.to_string())?;
        let gw = upload_bits(&dev, &w)?;
        // Safety: `gw` holds `n * kd` bf16 elements.
        let out = unsafe { lumen_image::cuda::blas::gemm_bf16(&dev, &gw, &act, n) }
            .map_err(|e| e.to_string())?;
        let got = dev.dtoh_copy(&out).map_err(|e| e.to_string())?;
        dev.synchronize().map_err(|e| e.to_string())?;
        rep.check("gemm_bf16 cuBLAS", &got, &want, 1e-5);
        // Safety: as above.
        let out = unsafe { lumen_image::cuda::blas::gemm_bf16_out(&dev, &gw, &act, n) }
            .map_err(|e| e.to_string())?;
        let rounded: Vec<f32> = want.iter().map(|&v| bf16_round(v)).collect();
        rep.check(
            "gemm_bf16_out cuBLAS",
            &download_bits(&dev, &out.bits)?,
            &rounded,
            1e-3,
        );
    }

    // head_norm_rope against the reference's `RMSNorm` (one `[head_dim]` weight
    // shared by every head, rounded after the normalisation and after the
    // weight) followed by the complex rotation, rounded once. Twice: on
    // ordinary inputs, and on inputs small enough that `eps` dominates the
    // mean square, where a kernel that dropped the epsilon would be off by
    // orders of magnitude rather than an ulp.
    for scale in [1.0f32, 1e-4] {
        let (seq, heads, head_dim) = (5usize, 3usize, 128usize);
        let x = bf16_values(
            pseudo(seq * heads * head_dim, 71)
                .iter()
                .map(|v| v * scale)
                .collect(),
        );
        let w = bf16_values(pseudo(head_dim, 72));
        let freqs: Vec<f32> = pseudo(seq * head_dim, 73);
        let mut normed = vec![0.0f32; x.len()];
        for (row, out) in x.chunks(head_dim).zip(normed.chunks_mut(head_dim)) {
            let ms = row.iter().map(|v| v * v).sum::<f32>() / head_dim as f32;
            let rms = 1.0 / (ms + 1e-6).sqrt();
            for ((o, &v), &wv) in out.iter_mut().zip(row).zip(&w) {
                *o = bf16_round(bf16_round(v * rms) * wv);
            }
        }
        let mut want = normed.clone();
        for s_ in 0..seq {
            for h in 0..heads {
                for p in 0..head_dim / 2 {
                    let base = (s_ * heads + h) * head_dim + 2 * p;
                    let (xr, xi) = (normed[base], normed[base + 1]);
                    let (cr, ci) = (
                        freqs[s_ * head_dim + 2 * p],
                        freqs[s_ * head_dim + 2 * p + 1],
                    );
                    want[base] = bf16_round(xr * cr - xi * ci);
                    want[base + 1] = bf16_round(xr * ci + xi * cr);
                }
            }
        }
        let gx = upload_bits(&dev, &x)?;
        let gw = launch::upload(&dev, &w).map_err(|e| e.to_string())?;
        let gf = launch::upload(&dev, &freqs).map_err(|e| e.to_string())?;
        let out = ops::head_norm_rope(&dev, &ops, &gx, &gw, &gf, seq, heads, 1e-6)
            .map_err(|e| e.to_string())?;
        rep.check(
            &format!("head_norm_rope x{scale}"),
            &download_bits(&dev, &out)?,
            &want,
            1e-3,
        );
    }

    // swiglu against `bf16(bf16(silu(g)) * u)`.
    let n = 1000usize;
    let g = bf16_values(pseudo(n, 74));
    let u = bf16_values(pseudo(n, 75));
    let want: Vec<f32> = g
        .iter()
        .zip(&u)
        .map(|(&gv, &uv)| bf16_round(bf16_round(lumen_image::tensor::silu(gv)) * uv))
        .collect();
    let out = ops::swiglu(&dev, &ops, &upload_bits(&dev, &g)?, &upload_bits(&dev, &u)?)
        .map_err(|e| e.to_string())?;
    rep.check("swiglu", &download_bits(&dev, &out)?, &want, 1e-3);

    // gelu_tanh against the reference's `gelu_tanh`, rounded.
    let n = 512usize;
    let x = bf16_values(pseudo(n, 41));
    let want: Vec<f32> = x.iter().map(|&v| bf16_round(gelu_tanh(v))).collect();
    let out = ops::gelu_tanh(&dev, &ops, &upload_bits(&dev, &x)?).map_err(|e| e.to_string())?;
    rep.check("gelu_tanh", &download_bits(&dev, &out)?, &want, 1e-3);

    // zero_center_rmsnorm against `zero_center_rms_norm_rows`, rounded once.
    // The weights are small and signed, so a kernel that forgot the `+ 1`
    // cannot pass.
    let (rows, dim) = (7usize, 384usize);
    for scale in [1.0f32, 1e-3] {
        let x = bf16_values(pseudo(rows * dim, 42).iter().map(|v| v * scale).collect());
        let w = bf16_values(pseudo(dim, 43));
        let cpu = zero_center_rms_norm_rows(&Matrix::new(rows, dim, x.clone()), &w, 1e-6);
        let want: Vec<f32> = cpu.data.iter().map(|&v| bf16_round(v)).collect();
        let gw = launch::upload(&dev, &w).map_err(|e| e.to_string())?;
        let out =
            ops::zero_center_rmsnorm(&dev, &ops, &upload_bits(&dev, &x)?, &gw, rows, dim, 1e-6)
                .map_err(|e| e.to_string())?;
        rep.check(
            &format!("zero_center_rmsnorm x{scale}"),
            &download_bits(&dev, &out)?,
            &want,
            1e-3,
        );
    }

    // --- the AdaLN gathers, against the reference's per-token expression ----
    // `mod_row` alternates, so a kernel that ignores it and broadcasts row 0
    // differs from the reference on every second token.
    for (case, rows, cols, off, stride) in [
        ("mod 4x", 5usize, 64usize, 0usize, 256usize),
        ("mod mlp chunk", 5, 64, 128, 256),
        ("mod narrow", 5, 48, 0, 48),
    ] {
        // Rows at scale 1e-3 make the epsilon count; a per-row offset of 100
        // makes a one-pass `E[x^2] - mean^2` variance cancel to noise.
        let x = bf16_values(
            pseudo(rows * cols, 44)
                .iter()
                .enumerate()
                .map(|(i, v)| match (i / cols) % 3 {
                    0 => *v,
                    1 => v * 1e-3,
                    _ => v + 100.0,
                })
                .collect(),
        );
        let y = bf16_values(pseudo(rows * cols, 45));
        // A `stride == cols` modulation is the single-row `norm_out` scale,
        // which every token reads from row 0.
        let single_row = stride == cols;
        let factors = bf16_values(pseudo(if single_row { cols } else { 2 * stride }, 46));
        let mod_row: Vec<i32> = (0..rows)
            .map(|r| if single_row { 0 } else { (r % 2) as i32 })
            .collect();
        let g_row = dev.htod_copy(&mod_row).map_err(|e| e.to_string())?;
        let g_factors = upload_bits(&dev, &factors)?;
        let row_factors = |r: usize| &factors[mod_row[r] as usize * stride + off..][..cols];

        // layernorm_scale == `bf16(bf16(layer_norm_rows(x)) * f)`.
        let ln = layer_norm_rows(&Matrix::new(rows, cols, x.clone()), 1e-6);
        let want: Vec<f32> = (0..rows)
            .flat_map(|r| {
                ln.data[r * cols..(r + 1) * cols]
                    .iter()
                    .zip(row_factors(r))
                    .map(|(&xv, &f)| bf16_round(bf16_round(xv) * f))
                    .collect::<Vec<_>>()
            })
            .collect();
        let gx = upload_bits(&dev, &x)?;
        let out =
            ops::layernorm_scale(&dev, &ops, &gx, &g_factors, &g_row, off, cols, stride, 1e-6)
                .map_err(|e| e.to_string())?;
        rep.check(
            &format!("layernorm_scale {case}"),
            &download_bits(&dev, &out)?,
            &want,
            1e-3,
        );

        // add_gated == `x = bf16(x + bf16(t * y))`, in place.
        let want: Vec<f32> = (0..rows)
            .flat_map(|r| {
                x[r * cols..(r + 1) * cols]
                    .iter()
                    .zip(&y[r * cols..(r + 1) * cols])
                    .zip(row_factors(r))
                    .map(|((&xv, &yv), &t)| bf16_round(xv + bf16_round(t * yv)))
                    .collect::<Vec<_>>()
            })
            .collect();
        let mut gx = upload_bits(&dev, &x)?;
        ops::add_gated(
            &dev,
            &ops,
            &mut gx,
            &upload_bits(&dev, &y)?,
            &g_factors,
            &g_row,
            off,
            cols,
            stride,
        )
        .map_err(|e| e.to_string())?;
        rep.check(
            &format!("add_gated {case}"),
            &download_bits(&dev, &gx)?,
            &want,
            0.0,
        );
    }

    Ok(rep.failures)
}

/// `values` rounded to bf16, so a kernel's bf16 input is exactly the value the
/// CPU side computes with.
fn bf16_values(values: Vec<f32>) -> Vec<f32> {
    values.into_iter().map(bf16_round).collect()
}

/// Upload bf16 values as their bits.
fn upload_bits(dev: &CudaDevice, values: &[f32]) -> Result<cudarc::driver::CudaSlice<u16>, String> {
    let bits: Vec<u16> = values.iter().map(|&v| bf16_bits(v)).collect();
    dev.htod_copy(&bits).map_err(|e| e.to_string())
}

/// The same attention the kernel implements, computed on the host.
fn attn_cpu(
    q: &[f32],
    k: &[f32],
    v: &[f32],
    ids: &[i32],
    key_valid: &[i32],
    seq: usize,
    heads: usize,
    head_dim: usize,
) -> Vec<f32> {
    let mut out = vec![0.0f32; q.len()];
    let scale = 1.0 / (head_dim as f32).sqrt();
    for qi in 0..seq {
        for h in 0..heads {
            let qbase = (qi * heads + h) * head_dim;
            let mut scores: Vec<(usize, f32)> = Vec::new();
            for kj in 0..seq {
                if key_valid[kj] == 0 {
                    continue;
                }
                if !(qi >= kj || (ids[qi] >= 0 && ids[qi] == ids[kj])) {
                    continue;
                }
                let kbase = (kj * heads + h) * head_dim;
                let mut dot = 0.0f32;
                for d in 0..head_dim {
                    dot += q[qbase + d] * k[kbase + d];
                }
                scores.push((kj, dot * scale));
            }
            let max = scores.iter().map(|(_, s)| *s).fold(f32::MIN, f32::max);
            let mut denom = 0.0f32;
            let mut acc = vec![0.0f32; head_dim];
            for (kj, s) in &scores {
                let w = (s - max).exp();
                denom += w;
                let vbase = (kj * heads + h) * head_dim;
                for d in 0..head_dim {
                    acc[d] += w * v[vbase + d];
                }
            }
            for d in 0..head_dim {
                out[qbase + d] = if denom > 0.0 { acc[d] / denom } else { 0.0 };
            }
        }
    }
    out
}
