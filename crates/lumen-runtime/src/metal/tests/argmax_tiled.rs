//! Differential exactness test: the two-pass tiled argmax
//! (`argmax_tiled_partial` + `argmax_tiled_reduce`) must select a BIT-IDENTICAL
//! token index to the single-TG `argmax` for EVERY input — the hard requirement
//! for it to be a pure speed change over the single-TG kernel.
//!
//! Ground truth is the incumbent `argmax` kernel itself (dispatched on the GPU),
//! so this is true differential testing, not a re-derivation of the expected
//! value. Cases stress the tie topology (both kernels now select the LOWEST
//! GLOBAL index on a value tie — CORR-011, see ffn_elementwise.msl), plus
//! +/-Inf, NaN, all-(-Inf) -> 0, and boundary indices, across several tile counts
//! (the selection must be tile-count-invariant).
//!
//! Run with:
//!   cargo test --release --lib -p lumen-runtime \
//!     metal::tests::argmax_tiled -- --nocapture

use crate::metal::ffi::MTLSize;
use crate::metal::shaders::METAL_SHADER_SOURCE;
use crate::metal::MetalF32Backend;

fn as_bytes_f32(v: &[f32]) -> &[u8] {
    unsafe { std::slice::from_raw_parts(v.as_ptr() as *const u8, std::mem::size_of_val(v)) }
}

struct Harness {
    backend: MetalF32Backend,
    lib: crate::metal::ffi::MetalLibrary,
}

impl Harness {
    fn new() -> Self {
        let backend = MetalF32Backend::new().expect("Metal backend create");
        let lib = backend
            .device
            .new_library_with_source(METAL_SHADER_SOURCE)
            .expect("compile shader library");
        Harness { backend, lib }
    }

    fn pso(&self, name: &str) -> crate::metal::ffi::MetalPipelineState {
        let f = self.lib.get_function(name).expect("get_function");
        self.backend
            .device
            .new_compute_pipeline_state(&f)
            .expect("pipeline state")
    }

    /// Incumbent single-TG `argmax`: 1 TG x 256 threads over n logits.
    fn incumbent(&self, logits: &[f32]) -> u32 {
        let n = logits.len() as u32;
        let logits_buf = self
            .backend
            .device
            .new_buffer_with_bytes(as_bytes_f32(logits))
            .expect("logits buf");
        let result_buf = self.backend.device.new_buffer(4).expect("result buf");
        let pso = self.pso("argmax");
        let cmd = self.backend.queue.new_command_buffer().expect("cmd");
        let enc = cmd.new_compute_encoder().expect("enc");
        enc.set_pipeline_state(&pso);
        enc.set_buffer(&logits_buf, 0, 0);
        enc.set_buffer(&result_buf, 0, 1);
        enc.set_bytes(&n.to_le_bytes(), 2);
        enc.dispatch_threadgroups(MTLSize::new(1, 1, 1), MTLSize::new(256, 1, 1));
        enc.end_encoding();
        cmd.commit_and_wait();
        let mut out = [0u32; 1];
        result_buf.read_u32(&mut out);
        out[0]
    }

    /// Two-pass tiled argmax with `num_tiles` pass-1 threadgroups.
    fn tiled(&self, logits: &[f32], num_tiles: u32) -> u32 {
        let n = logits.len() as u32;
        let tile_size = (n + num_tiles - 1) / num_tiles;
        let actual_tiles = (n + tile_size - 1) / tile_size;
        let logits_buf = self
            .backend
            .device
            .new_buffer_with_bytes(as_bytes_f32(logits))
            .expect("logits buf");
        let part_val = self
            .backend
            .device
            .new_buffer((actual_tiles as usize).max(1) * 4)
            .expect("part_val");
        let part_idx = self
            .backend
            .device
            .new_buffer((actual_tiles as usize).max(1) * 4)
            .expect("part_idx");
        let result_buf = self.backend.device.new_buffer(4).expect("result buf");
        let p1 = self.pso("argmax_tiled_partial");
        let p2 = self.pso("argmax_tiled_reduce");
        // Serial compute encoder: sequential dispatches auto-order (pass-2 reads
        // pass-1's partials), matching the production serial-encoder path.
        let cmd = self.backend.queue.new_command_buffer().expect("cmd");
        let enc = cmd.new_compute_encoder().expect("enc");
        enc.set_pipeline_state(&p1);
        enc.set_buffer(&logits_buf, 0, 0);
        enc.set_buffer(&part_val, 0, 1);
        enc.set_buffer(&part_idx, 0, 2);
        enc.set_bytes(&n.to_le_bytes(), 3);
        enc.set_bytes(&tile_size.to_le_bytes(), 4);
        enc.dispatch_threadgroups(
            MTLSize::new(actual_tiles as u64, 1, 1),
            MTLSize::new(256, 1, 1),
        );
        enc.set_pipeline_state(&p2);
        enc.set_buffer(&part_val, 0, 0);
        enc.set_buffer(&part_idx, 0, 1);
        enc.set_buffer(&result_buf, 0, 2);
        enc.set_bytes(&actual_tiles.to_le_bytes(), 3);
        enc.dispatch_threadgroups(MTLSize::new(1, 1, 1), MTLSize::new(256, 1, 1));
        enc.end_encoding();
        cmd.commit_and_wait();
        let mut out = [0u32; 1];
        result_buf.read_u32(&mut out);
        out[0]
    }

    fn assert_match(&self, label: &str, logits: &[f32]) {
        let want = self.incumbent(logits);
        // Selection must be identical AND tile-count-invariant.
        for &nt in &[1u32, 2, 7, 64, 100, 128, 200, 256] {
            if nt as usize > logits.len() {
                continue;
            }
            let got = self.tiled(logits, nt);
            assert_eq!(
                got, want,
                "{label}: tiled(tiles={nt})={got} != incumbent={want}"
            );
        }
        println!("[argmax-tiled] OK {label}: idx={want} (n={})", logits.len());
    }
}

impl Harness {
    /// One isolated CB containing ONLY the incumbent single-TG argmax; returns
    /// its GPU span (GPUEndTime-GPUStartTime) in microseconds.
    fn incumbent_span_us(&self, logits_buf: &crate::metal::ffi::MetalBuffer, n: u32) -> f64 {
        let result_buf = self.backend.device.new_buffer(4).expect("result");
        let pso = self.pso("argmax");
        let cmd = self.backend.queue.new_command_buffer().expect("cmd");
        let enc = cmd.new_compute_encoder().expect("enc");
        enc.set_pipeline_state(&pso);
        enc.set_buffer(logits_buf, 0, 0);
        enc.set_buffer(&result_buf, 0, 1);
        enc.set_bytes(&n.to_le_bytes(), 2);
        enc.dispatch_threadgroups(MTLSize::new(1, 1, 1), MTLSize::new(256, 1, 1));
        enc.end_encoding();
        cmd.commit_and_wait();
        cmd.gpu_elapsed_secs() * 1e6
    }

    /// One isolated CB containing ONLY the two-pass tiled argmax; returns its GPU
    /// span in microseconds.
    fn tiled_span_us(
        &self,
        logits_buf: &crate::metal::ffi::MetalBuffer,
        n: u32,
        num_tiles: u32,
    ) -> f64 {
        let tile_size = (n + num_tiles - 1) / num_tiles;
        let actual_tiles = (n + tile_size - 1) / tile_size;
        let part_val = self
            .backend
            .device
            .new_buffer((actual_tiles as usize) * 4)
            .expect("pv");
        let part_idx = self
            .backend
            .device
            .new_buffer((actual_tiles as usize) * 4)
            .expect("pi");
        let result_buf = self.backend.device.new_buffer(4).expect("result");
        let p1 = self.pso("argmax_tiled_partial");
        let p2 = self.pso("argmax_tiled_reduce");
        let cmd = self.backend.queue.new_command_buffer().expect("cmd");
        let enc = cmd.new_compute_encoder().expect("enc");
        enc.set_pipeline_state(&p1);
        enc.set_buffer(logits_buf, 0, 0);
        enc.set_buffer(&part_val, 0, 1);
        enc.set_buffer(&part_idx, 0, 2);
        enc.set_bytes(&n.to_le_bytes(), 3);
        enc.set_bytes(&tile_size.to_le_bytes(), 4);
        enc.dispatch_threadgroups(
            MTLSize::new(actual_tiles as u64, 1, 1),
            MTLSize::new(256, 1, 1),
        );
        enc.set_pipeline_state(&p2);
        enc.set_buffer(&part_val, 0, 0);
        enc.set_buffer(&part_idx, 0, 1);
        enc.set_buffer(&result_buf, 0, 2);
        enc.set_bytes(&actual_tiles.to_le_bytes(), 3);
        enc.dispatch_threadgroups(MTLSize::new(1, 1, 1), MTLSize::new(256, 1, 1));
        enc.end_encoding();
        cmd.commit_and_wait();
        cmd.gpu_elapsed_secs() * 1e6
    }
}

fn median(mut xs: Vec<f64>) -> f64 {
    xs.sort_by(|a, b| a.partial_cmp(b).unwrap());
    xs[xs.len() / 2]
}

/// Isolated GPU-span microbenchmark (run explicitly): times argmax-alone in a
/// split command buffer, single-TG vs tiled at several tile counts, on the
/// production vocab (248320).
///   cargo test --release --lib -p lumen-runtime \
///     metal::tests::argmax_tiled::argmax_tiled_span_bench -- --ignored --nocapture
#[test]
#[ignore]
fn argmax_tiled_span_bench() {
    let h = Harness::new();
    let n = 248_320u32;
    let mut rng = lcg(99);
    let v: Vec<f32> = (0..n as usize).map(|_| rng()).collect();
    let logits_buf = h
        .backend
        .device
        .new_buffer_with_bytes(as_bytes_f32(&v))
        .expect("logits");
    let iters = 60usize;
    let warm = 10usize;

    // Incumbent baseline span.
    for _ in 0..warm {
        let _ = h.incumbent_span_us(&logits_buf, n);
    }
    let off: Vec<f64> = (0..iters)
        .map(|_| h.incumbent_span_us(&logits_buf, n))
        .collect();
    let off_med = median(off);
    println!("[argmax-bench] single-TG argmax span: {off_med:.1} us (median of {iters})");

    for &t in &[64u32, 128, 256] {
        for _ in 0..warm {
            let _ = h.tiled_span_us(&logits_buf, n, t);
        }
        let on: Vec<f64> = (0..iters)
            .map(|_| h.tiled_span_us(&logits_buf, n, t))
            .collect();
        let on_med = median(on);
        let saved = off_med - on_med;
        println!("[argmax-bench] tiled(tiles={t}) span: {on_med:.1} us | saved {saved:.1} us");
    }
}

fn lcg(seed: u64) -> impl FnMut() -> f32 {
    let mut s = seed;
    move || {
        s = s
            .wrapping_mul(6364136223846793005)
            .wrapping_add(1442695040888963407);
        let u = (s >> 32) as u32;
        (u as f32) / (u32::MAX as f32) - 0.5
    }
}

#[test]
fn tiled_argmax_matches_incumbent() {
    let h = Harness::new();

    // 1. Random unique-max arrays (production vocab size + smaller), several seeds.
    for &n in &[248_320usize, 4096, 970, 257] {
        for seed in 0..4u64 {
            let mut rng = lcg(seed * 2654435761 + 12345);
            let v: Vec<f32> = (0..n).map(|_| rng()).collect();
            h.assert_match(&format!("random n={n} seed={seed}"), &v);
        }
    }

    // 2. Tie topology: duplicated max at indices 1 and 512. Both kernels now
    //    select the LOWEST GLOBAL index (CORR-011), i.e. 1. assert_match verifies
    //    the tiled kernel reproduces whatever the single-TG incumbent selects.
    {
        let mut v = vec![-1.0f32; 1024];
        v[1] = 9.0;
        v[512] = 9.0;
        h.assert_match("tie {1,512} lowest-index", &v);
    }

    // 3. Many duplicated maxima across residue classes and blocks.
    {
        let mut v = vec![0.0f32; 4096];
        for &i in &[3usize, 259, 515, 256, 512, 2, 770, 1026] {
            v[i] = 5.0;
        }
        h.assert_match("tie multi residue", &v);
    }

    // 4. +Inf duplicated (finite ties among +Inf).
    {
        let mut v = vec![0.0f32; 2048];
        v[100] = f32::INFINITY;
        v[356] = f32::INFINITY; // residue(356)=100, residue(100)=100 -> lower index 100
        v[612] = f32::INFINITY;
        h.assert_match("tie +Inf", &v);
    }

    // 5. All -Inf -> both return 0 (untouched sentinel).
    {
        let v = vec![f32::NEG_INFINITY; 1000];
        h.assert_match("all -Inf", &v);
    }

    // 6. NaN mixed with a finite max: NaN must never win.
    {
        let mut v = vec![-3.0f32; 2000];
        v[7] = f32::NAN;
        v[1234] = f32::NAN;
        v[999] = 4.0; // the true (unique) max
        h.assert_match("NaN + finite max", &v);
    }

    // 7. All NaN -> both return 0 (nothing beats the -Inf sentinel).
    {
        let v = vec![f32::NAN; 800];
        h.assert_match("all NaN", &v);
    }

    // 8. Max at index 0 (boundary).
    {
        let mut v = vec![-1.0f32; 3000];
        v[0] = 10.0;
        h.assert_match("max at 0", &v);
    }

    // 9. Max at last index (boundary).
    {
        let mut v = vec![-1.0f32; 3000];
        let last = v.len() - 1;
        v[last] = 10.0;
        h.assert_match("max at n-1", &v);
    }

    // 10. -Inf with a single finite value below zero (finite must win over -Inf).
    {
        let mut v = vec![f32::NEG_INFINITY; 1500];
        v[777] = -42.0;
        h.assert_match("single finite among -Inf", &v);
    }
}
