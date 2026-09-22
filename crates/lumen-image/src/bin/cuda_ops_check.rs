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
//! helpers the forward uses: a 16-bit weight must give the result its f32
//! widening would, and the AdaLN kernels must evaluate the reference's
//! per-token modulation expression.
//!
//! Usage: `cuda-ops-check`

use std::process::ExitCode;

use lumen_image::cuda::dit_gpu::ops::{self, Gemm16};
use lumen_image::cuda::{attention, launch, ImageKernels};
use lumen_image::lbi::half_to_f32;
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

/// The fused attention from f32 operands: q and k truncated to bf16 the way
/// the DiT's fused head norm writes them, v handed over as f32.
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
    let q16 = attention::to_bf16(dev, &k.f32_to_bf16_trunc, &q.buf).map_err(|e| e.to_string())?;
    let k16 = attention::to_bf16(dev, &k.f32_to_bf16_trunc, &kk.buf).map_err(|e| e.to_string())?;
    attention::fused_block_causal_attention(dev, k, &q16, &k16, v, text_count, seq, heads)
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

/// f32 -> bf16 -> f32 by truncation, the attention operands' conversion.
fn bf16_trunc(v: f32) -> f32 {
    f32::from_bits(v.to_bits() & 0xffff_0000)
}

/// Round to nearest-even bf16, the kernels' `bf16_rne`, as an f32.
fn bf16_rne(v: f32) -> f32 {
    if v.is_nan() {
        return f32::from_bits(0x7fc0_0000);
    }
    let bits = v.to_bits();
    let lsb = (bits >> 16) & 1;
    let rounded = bits.wrapping_add(0x7fff + lsb);
    f32::from_bits(rounded & 0xffff_0000)
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

    // --- linear with and without bias ------------------------------------
    for (m, n, kd, bias) in [
        (7usize, 5usize, 9usize, false),
        (33, 40, 64, true),
        (64, 128, 96, true),
    ] {
        let a = pseudo(m * kd, 1);
        let w = pseudo(n * kd, 2);
        let b = pseudo(n, 3);
        let cpu = Matrix::new(m, kd, a.clone()).linear(
            &Matrix::new(n, kd, w.clone()),
            if bias { Some(&b) } else { None },
        );
        let ga = launch::upload(&dev, &a).map_err(|e| e.to_string())?;
        let gw = launch::upload(&dev, &w).map_err(|e| e.to_string())?;
        let gb = if bias {
            Some(launch::upload(&dev, &b).map_err(|e| e.to_string())?)
        } else {
            None
        };
        let out =
            launch::linear(&dev, &k, &ga, &gw, gb.as_ref(), m, n, kd).map_err(|e| e.to_string())?;
        let got = launch::download(&dev, &out).map_err(|e| e.to_string())?;
        rep.check(
            &format!("linear {m}x{n}x{kd} bias={bias}"),
            &got,
            &cpu.data,
            1e-6,
        );
    }

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
    // Both consume the same bf16-truncated operands, so the inputs are rounded
    // to bf16 on the host first and the CPU reference sees exactly what the
    // kernels see. What remains is the bf16 rounding of the probabilities and
    // the accumulation order, which is why the bar is above the f32 ops'.
    {
        let (seq, heads, head_dim, text_count) = (203usize, 2usize, 128usize, 7usize);
        let bf16 = |v: Vec<f32>| -> Vec<f32> { v.into_iter().map(bf16_trunc).collect() };
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
        let bf16 = |v: Vec<f32>| -> Vec<f32> { v.into_iter().map(bf16_trunc).collect() };
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

    // The two f32 -> bf16 conversions on inputs that are not representable in
    // bf16, so a kernel that rounded the wrong way could not pass: nearest-even
    // for the projections' inputs, truncation for the attention operands.
    {
        let n = 4096usize;
        let mut x = pseudo(n, 68);
        // Exact ties, where nearest-even and round-half-up differ.
        x[0] = f32::from_bits(0x3f80_8000);
        x[1] = f32::from_bits(0x3f81_8000);
        x[2] = f32::from_bits(0xbf80_8000);
        let gx = launch::upload(&dev, &x).map_err(|e| e.to_string())?;
        // Safety: `gx` holds `n` floats.
        let rounded = unsafe {
            lumen_image::cuda::blas::Bf16Activation::new(&dev, &k.f32_to_bf16_bits, &gx, 1, n)
        }
        .map_err(|e| e.to_string())?;
        let want: Vec<f32> = x.iter().map(|&v| bf16_rne(v)).collect();
        rep.check(
            "f32_to_bf16_bits nearest-even",
            &download_bits(&dev, &rounded.bits)?,
            &want,
            0.0,
        );
        let truncated =
            attention::to_bf16(&dev, &k.f32_to_bf16_trunc, &gx.buf).map_err(|e| e.to_string())?;
        let want: Vec<f32> = x.iter().map(|v| bf16_trunc(*v)).collect();
        rep.check(
            "f32_to_bf16_trunc",
            &download_bits(&dev, &truncated)?,
            &want,
            0.0,
        );
    }

    // pack_rows against the reference's joint sequence: text slots by slot
    // index, image tokens in order, encoded as negative sources.
    {
        let (text_rows, image_rows, cols) = (3usize, 8usize, 5usize);
        let txt = pseudo(text_rows * cols, 64);
        let img = pseudo(image_rows * cols, 65);
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
        let gt = launch::upload(&dev, &txt).map_err(|e| e.to_string())?;
        let gi = launch::upload(&dev, &img).map_err(|e| e.to_string())?;
        let out = ops::pack_rows(&dev, &ops, &gt, &gi, &source, cols).map_err(|e| e.to_string())?;
        let got = launch::download(&dev, &out).map_err(|e| e.to_string())?;
        rep.check("pack_rows", &got, &want, 0.0);
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
    // tensor cores, against `Matrix::linear` on the same bf16-rounded values.
    // The operands are exact in bf16, so what remains is f32 accumulation
    // order.
    {
        let (m, n, kd) = (37usize, 48usize, 96usize);
        let bf16 = |v: Vec<f32>| -> Vec<f32> { v.into_iter().map(bf16_trunc).collect() };
        let a = bf16(pseudo(m * kd, 66));
        let w = bf16(pseudo(n * kd, 67));
        let want = Matrix::new(m, kd, a.clone())
            .linear(&Matrix::new(n, kd, w.clone()), None)
            .data;
        let ga = launch::upload(&dev, &a).map_err(|e| e.to_string())?;
        let w_bits: Vec<u16> = w.iter().map(|x| (x.to_bits() >> 16) as u16).collect();
        let gw = dev.htod_copy(&w_bits).map_err(|e| e.to_string())?;
        // Safety: `ga` holds `m * kd` floats and `gw` `n * kd` bf16 elements.
        let act = unsafe {
            lumen_image::cuda::blas::Bf16Activation::new(&dev, &k.f32_to_bf16_bits, &ga, m, kd)
        }
        .map_err(|e| e.to_string())?;
        let out = unsafe { lumen_image::cuda::blas::gemm_bf16(&dev, &gw, &act, n) }
            .map_err(|e| e.to_string())?;
        let got = dev.dtoh_copy(&out).map_err(|e| e.to_string())?;
        dev.synchronize().map_err(|e| e.to_string())?;
        rep.check("gemm_bf16 cuBLAS", &got, &want, 1e-5);
    }

    // head_norm_rope_bf16 against the reference's per-head `rms_norm_rows`
    // (one `[head_dim]` weight shared by every head) followed by the complex
    // rotation, truncated to bf16 as the attention operands are. The
    // comparison is on the truncated values, so the bar is bf16's.
    // Twice: on ordinary inputs, and on inputs small enough that `eps`
    // dominates the mean square, where a kernel that dropped the epsilon
    // would be off by orders of magnitude rather than an ulp.
    for scale in [1.0f32, 1e-4] {
        let (seq, heads, head_dim) = (5usize, 3usize, 128usize);
        let x: Vec<f32> = pseudo(seq * heads * head_dim, 71)
            .iter()
            .map(|v| v * scale)
            .collect();
        let w = pseudo(head_dim, 72);
        let freqs: Vec<f32> = pseudo(seq * head_dim, 73);
        let normed = lumen_image::tensor::rms_norm_rows(
            &Matrix::new(seq * heads, head_dim, x.clone()),
            &w,
            1e-6,
        );
        let mut want = normed.data.clone();
        for s_ in 0..seq {
            for h in 0..heads {
                for p in 0..head_dim / 2 {
                    let base = (s_ * heads + h) * head_dim + 2 * p;
                    let (xr, xi) = (normed.data[base], normed.data[base + 1]);
                    let (cr, ci) = (
                        freqs[s_ * head_dim + 2 * p],
                        freqs[s_ * head_dim + 2 * p + 1],
                    );
                    want[base] = xr * cr - xi * ci;
                    want[base + 1] = xr * ci + xi * cr;
                }
            }
        }
        let want: Vec<f32> = want.iter().map(|v| bf16_trunc(*v)).collect();
        let gx = launch::upload(&dev, &x).map_err(|e| e.to_string())?;
        let gw = launch::upload(&dev, &w).map_err(|e| e.to_string())?;
        let gf = launch::upload(&dev, &freqs).map_err(|e| e.to_string())?;
        let out = ops::head_norm_rope_bf16(&dev, &ops, &gx, &gw, &gf, seq, heads, 1e-6)
            .map_err(|e| e.to_string())?;
        let got = download_bits(&dev, &out)?;
        rep.check(&format!("head_norm_rope_bf16 x{scale}"), &got, &want, 1e-3);
    }

    // swiglu_bf16 against `silu(g) * u`, rounded to nearest-even bf16.
    let n = 1000usize;
    let g = pseudo(n, 74);
    let u = pseudo(n, 75);
    let want: Vec<f32> = g
        .iter()
        .zip(&u)
        .map(|(&gv, &uv)| bf16_rne(lumen_image::tensor::silu(gv) * uv))
        .collect();
    let gg = launch::upload(&dev, &g).map_err(|e| e.to_string())?;
    let gu = launch::upload(&dev, &u).map_err(|e| e.to_string())?;
    let out = ops::swiglu_bf16(&dev, &ops, &gg, &gu).map_err(|e| e.to_string())?;
    let got = download_bits(&dev, &out)?;
    rep.check("swiglu_bf16", &got, &want, 1e-6);

    // gelu_tanh_inplace against the reference's `gelu_tanh`.
    let n = 512usize;
    let x = pseudo(n, 41);
    let want: Vec<f32> = x.iter().map(|&v| gelu_tanh(v)).collect();
    let mut g = launch::upload(&dev, &x).map_err(|e| e.to_string())?;
    ops::gelu_tanh_inplace(&dev, &ops, &mut g).map_err(|e| e.to_string())?;
    let got = launch::download(&dev, &g).map_err(|e| e.to_string())?;
    rep.check("gelu_tanh_inplace", &got, &want, 1e-6);

    // zero_center_rmsnorm against `zero_center_rms_norm_rows`. The weights are
    // small and signed, so a kernel that forgot the `+ 1` cannot pass.
    let (rows, dim) = (7usize, 384usize);
    for scale in [1.0f32, 1e-3] {
        let x: Vec<f32> = pseudo(rows * dim, 42).iter().map(|v| v * scale).collect();
        let w = pseudo(dim, 43);
        let cpu = zero_center_rms_norm_rows(&Matrix::new(rows, dim, x.clone()), &w, 1e-6);
        let gx = launch::upload(&dev, &x).map_err(|e| e.to_string())?;
        let gw = launch::upload(&dev, &w).map_err(|e| e.to_string())?;
        let out = ops::zero_center_rmsnorm(&dev, &ops, &gx, &gw, rows, dim, 1e-6)
            .map_err(|e| e.to_string())?;
        let got = launch::download(&dev, &out).map_err(|e| e.to_string())?;
        rep.check(
            &format!("zero_center_rmsnorm x{scale}"),
            &got,
            &cpu.data,
            1e-5,
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
        let x: Vec<f32> = pseudo(rows * cols, 44)
            .iter()
            .enumerate()
            .map(|(i, v)| match (i / cols) % 3 {
                0 => *v,
                1 => v * 1e-3,
                _ => v + 100.0,
            })
            .collect();
        let y = pseudo(rows * cols, 45);
        // A `stride == cols` modulation is the single-row `norm_out` scale,
        // which every token reads from row 0.
        let single_row = stride == cols;
        let modu = pseudo(if single_row { cols } else { 2 * stride }, 46);
        let mod_row: Vec<i32> = (0..rows)
            .map(|r| if single_row { 0 } else { (r % 2) as i32 })
            .collect();
        let g_row = dev.htod_copy(&mod_row).map_err(|e| e.to_string())?;

        // layernorm_scale_bf16 == `bf16(layer_norm_rows(x) * (1.0 + s))`.
        let ln = layer_norm_rows(&Matrix::new(rows, cols, x.clone()), 1e-6);
        let want: Vec<f32> = (0..rows)
            .flat_map(|r| {
                let m = &modu[mod_row[r] as usize * stride + off..];
                ln.data[r * cols..(r + 1) * cols]
                    .iter()
                    .zip(&m[..cols])
                    .map(|(&xv, &s)| bf16_rne(xv * (1.0 + s)))
            })
            .collect();
        let gx = launch::upload(&dev, &x).map_err(|e| e.to_string())?;
        let gm = launch::upload(&dev, &modu).map_err(|e| e.to_string())?;
        let out = ops::layernorm_scale_bf16(&dev, &ops, &gx, &gm, &g_row, off, cols, stride, 1e-6)
            .map_err(|e| e.to_string())?;
        let got = download_bits(&dev, &out)?;
        rep.check(&format!("layernorm_scale_bf16 {case}"), &got, &want, 1e-6);

        // add_gated == `*v += g.tanh() * yv`.
        let want: Vec<f32> = (0..rows)
            .flat_map(|r| {
                let m = &modu[mod_row[r] as usize * stride + off..];
                x[r * cols..(r + 1) * cols]
                    .iter()
                    .zip(&y[r * cols..(r + 1) * cols])
                    .zip(&m[..cols])
                    .map(|((&xv, &yv), &g)| xv + g.tanh() * yv)
            })
            .collect();
        let gy = launch::upload(&dev, &y).map_err(|e| e.to_string())?;
        let out = ops::add_gated(&dev, &ops, &gx, &gy, &gm, &g_row, off, cols, stride)
            .map_err(|e| e.to_string())?;
        let got = launch::download(&dev, &out).map_err(|e| e.to_string())?;
        rep.check(&format!("add_gated_gather {case}"), &got, &want, 1e-6);
    }

    // --- the 16-bit GEMM, against the f32 widening of the same weights ------
    // The reference is `Matrix::linear` on the widened weights, so this pins the
    // widening and the accumulation order at once.
    let cases: [(usize, usize, usize, u64); 3] =
        [(7, 5, 9, 51), (33, 40, 64, 52), (64, 128, 96, 53)];
    for (m, n, kd, seed) in cases {
        let a = pseudo(m * kd, seed);
        let w = pseudo(n * kd, seed + 100);
        let ga = launch::upload(&dev, &a).map_err(|e| e.to_string())?;
        for (name, kind) in [("f16", Gemm16::F16), ("bf16", Gemm16::Bf16)] {
            // Pack the weights into the storage dtype, then widen them back the
            // way a reference implementation would. For f16 that is
            // `half_to_f32`, the container's own software decoder, so the two
            // sides of this comparison share no code: the kernel widens with the
            // hardware convert and the reference with bit manipulation.
            let bits = pack_16(&w, kind);
            let widened: Vec<f32> = bits
                .iter()
                .map(|&b| match kind {
                    Gemm16::F16 => half_to_f32(b),
                    Gemm16::Bf16 => f32::from_bits((b as u32) << 16),
                })
                .collect();
            assert!(
                widened.iter().all(|v| v.is_finite()),
                "{name}: the packing produced a non-finite weight"
            );
            let gw = dev.htod_copy(&bits).map_err(|e| e.to_string())?;
            let out =
                ops::gemm_16bit(&dev, &ops, &gw, &ga, m, n, kd, kind).map_err(|e| e.to_string())?;
            let got = launch::download(&dev, &out).map_err(|e| e.to_string())?;
            let want = Matrix::new(m, kd, a.clone()).linear(&Matrix::new(n, kd, widened), None);
            rep.check(
                &format!("gemm_16bit {name} {m}x{n}x{kd}"),
                &got,
                &want.data,
                1e-6,
            );
        }
    }

    Ok(rep.failures)
}

/// Pack f32 values into a 16-bit storage dtype the way a checkpoint would.
///
/// bf16 is the top half of an f32, exactly. f16 is not, so it is encoded here
/// with round-to-nearest-even — the IEEE 754 default the container's decoder
/// assumes — rather than truncation, so the values straddling a rounding
/// boundary are the ones a real checkpoint would produce.
fn pack_16(values: &[f32], kind: Gemm16) -> Vec<u16> {
    values
        .iter()
        .map(|&v| match kind {
            Gemm16::Bf16 => {
                // Round to nearest even on the 16 dropped bits.
                let bits = v.to_bits();
                let rem = bits & 0xffff;
                let mut out = bits >> 16;
                if rem > 0x8000 || (rem == 0x8000 && (out & 1) == 1) {
                    out += 1;
                }
                out as u16
            }
            Gemm16::F16 => f32_to_f16_bits(v),
        })
        .collect()
}

/// f32 to f16 bits, round-to-nearest-even, denormals included.
fn f32_to_f16_bits(v: f32) -> u16 {
    let bits = v.to_bits();
    let sign = ((bits >> 16) & 0x8000) as u16;
    let biased = ((bits >> 23) & 0xff) as i32;
    let mut man = bits & 0x7f_ffff;
    if biased == 0xff {
        // Infinity keeps a zero mantissa; NaN keeps a non-zero one.
        return sign | 0x7c00 | if man != 0 { 0x200 } else { 0 };
    }
    let mut exp = biased - 127 + 15;
    if exp >= 0x1f {
        return sign | 0x7c00;
    }
    if exp <= 0 {
        // Subnormal: shift the implicit bit in and round on the way down.
        if exp < -10 {
            return sign;
        }
        man |= 0x80_0000;
        let shift = (14 - exp) as u32;
        let rem = man & ((1u32 << shift) - 1);
        let mut out = (man >> shift) as u16;
        let half = 1u32 << (shift - 1);
        if rem > half || (rem == half && (out & 1) == 1) {
            out += 1;
        }
        return sign | out;
    }
    let rem = man & 0x1fff;
    man >>= 13;
    if rem > 0x1000 || (rem == 0x1000 && (man & 1) == 1) {
        man += 1;
        if man == 0x400 {
            man = 0;
            exp += 1;
            if exp >= 0x1f {
                return sign | 0x7c00;
            }
        }
    }
    sign | ((exp as u16) << 10) | (man as u16)
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
