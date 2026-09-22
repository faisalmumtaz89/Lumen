//! A minimal dense f32 tensor and the matrix operations the image pipeline uses.
//!
//! This is the CPU reference: clarity over speed, and one fixed accumulation
//! order per operation so a comparison against the reference implementation is
//! reproducible rather than merely close.

/// A row-major 2-D matrix.
#[derive(Debug, Clone, PartialEq)]
pub struct Matrix {
    pub rows: usize,
    pub cols: usize,
    pub data: Vec<f32>,
}

impl Matrix {
    pub fn new(rows: usize, cols: usize, data: Vec<f32>) -> Self {
        assert_eq!(rows * cols, data.len(), "shape does not match the data");
        Self { rows, cols, data }
    }

    pub fn zeros(rows: usize, cols: usize) -> Self {
        Self {
            rows,
            cols,
            data: vec![0.0; rows * cols],
        }
    }

    pub fn row(&self, r: usize) -> &[f32] {
        &self.data[r * self.cols..(r + 1) * self.cols]
    }

    pub fn row_mut(&mut self, r: usize) -> &mut [f32] {
        &mut self.data[r * self.cols..(r + 1) * self.cols]
    }

    /// `self @ other^T` is not what this does: this is the plain product
    /// `self[r, k] * rhs[k, c]`, accumulated in index order.
    pub fn matmul(&self, rhs: &Matrix) -> Matrix {
        assert_eq!(self.cols, rhs.rows, "inner dimensions differ");
        let mut out = Self::zeros(self.rows, rhs.cols);
        for r in 0..self.rows {
            let a = self.row(r);
            let o = out.row_mut(r);
            for (k, &av) in a.iter().enumerate() {
                if av == 0.0 {
                    continue;
                }
                let b = rhs.row(k);
                for c in 0..rhs.cols {
                    o[c] += av * b[c];
                }
            }
        }
        out
    }

    /// Apply a linear layer whose weight is stored `[out, in]`: `x @ W^T`.
    ///
    /// This is the layout every `nn.Linear` in the checkpoint uses.
    pub fn linear(&self, weight: &Matrix, bias: Option<&[f32]>) -> Matrix {
        self.linear_with_threads(weight, bias, default_threads())
    }

    /// As [`linear`](Self::linear), split across `threads` row blocks.
    ///
    /// Rows are independent and each output element accumulates in the same
    /// index order whether or not threads are used, so the result is bit-for-bit
    /// the same as the single-threaded path — parallelising a reference must not
    /// move the number it produces.
    pub fn linear_with_threads(
        &self,
        weight: &Matrix,
        bias: Option<&[f32]>,
        threads: usize,
    ) -> Matrix {
        assert_eq!(self.cols, weight.cols, "linear input width differs");
        let mut out = Self::zeros(self.rows, weight.rows);
        let threads = threads.max(1).min(self.rows.max(1));

        let compute_rows = |rows: &mut [f32], first: usize| {
            for (i, orow) in rows.chunks_exact_mut(weight.rows).enumerate() {
                let x = self.row(first + i);
                for (j, wrow) in weight.data.chunks_exact(weight.cols).enumerate() {
                    let mut acc = 0.0f32;
                    for (k, &xk) in x.iter().enumerate() {
                        acc += xk * wrow[k];
                    }
                    orow[j] = acc;
                }
                if let Some(b) = bias {
                    for (o, &bv) in orow.iter_mut().zip(b) {
                        *o += bv;
                    }
                }
            }
        };

        if threads <= 1 {
            compute_rows(&mut out.data, 0);
            return out;
        }
        let chunk_rows = self.rows.div_ceil(threads);
        let width = weight.rows;
        std::thread::scope(|scope| {
            let mut first = 0usize;
            for block in out.data.chunks_mut(chunk_rows * width) {
                let start = first;
                scope.spawn(move || compute_rows(block, start));
                first += chunk_rows;
            }
        });
        out
    }
}

/// Threads to use when the caller does not say. Bounded so a large machine does
/// not oversubscribe on small matrices.
pub fn default_threads() -> usize {
    std::thread::available_parallelism()
        .map(|n| n.get().min(16))
        .unwrap_or(1)
}

/// Elementwise add.
pub fn add(a: &Matrix, b: &Matrix) -> Matrix {
    assert_eq!((a.rows, a.cols), (b.rows, b.cols), "shape mismatch");
    Matrix::new(
        a.rows,
        a.cols,
        a.data.iter().zip(&b.data).map(|(x, y)| x + y).collect(),
    )
}

/// Multiply a row by a per-column scale, broadcasting over rows.
/// GELU with the tanh approximation, as `nn.GELU(approximate="tanh")`.
pub fn gelu_tanh(x: f32) -> f32 {
    const C: f32 = 0.797_884_6; // sqrt(2/pi)
    0.5 * x * (1.0 + (C * (x + 0.044_715 * x * x * x)).tanh())
}

/// SiLU.
pub fn silu(x: f32) -> f32 {
    x / (1.0 + (-x).exp())
}

/// RMS normalise each row with a learned scale, matching `QwenImage21RMSNorm`
/// and the text encoder's `Qwen3VLTextRMSNorm`: compute in f32, multiply by the
/// weight, then cast back.
pub fn rms_norm_rows(m: &Matrix, weight: &[f32], eps: f32) -> Matrix {
    assert_eq!(m.cols, weight.len(), "norm width differs");
    let mut out = m.clone();
    for r in 0..out.rows {
        let row = out.row_mut(r);
        let mean_sq: f32 = row.iter().map(|v| v * v).sum::<f32>() / row.len() as f32;
        let inv = 1.0 / (mean_sq + eps).sqrt();
        for (v, &w) in row.iter_mut().zip(weight) {
            *v = *v * inv * w;
        }
    }
    out
}

/// Zero-centred RMS normalise, as `QwenImage21ZeroCenterRMSNorm`: the stored
/// weight is `scale - 1`, and the effective scale is computed in f32.
pub fn zero_center_rms_norm_rows(m: &Matrix, weight: &[f32], eps: f32) -> Matrix {
    assert_eq!(m.cols, weight.len(), "norm width differs");
    let mut out = m.clone();
    for r in 0..out.rows {
        let row = out.row_mut(r);
        let mean_sq: f32 = row.iter().map(|v| v * v).sum::<f32>() / row.len() as f32;
        let inv = 1.0 / (mean_sq + eps).sqrt();
        for (v, &w) in row.iter_mut().zip(weight) {
            *v = *v * inv * (w + 1.0);
        }
    }
    out
}

/// LayerNorm with no affine parameters, as the DiT's per-block `img_norm`.
pub fn layer_norm_rows(m: &Matrix, eps: f32) -> Matrix {
    let mut out = m.clone();
    for r in 0..out.rows {
        let row = out.row_mut(r);
        let n = row.len() as f32;
        let mean: f32 = row.iter().sum::<f32>() / n;
        let var: f32 = row.iter().map(|v| (v - mean) * (v - mean)).sum::<f32>() / n;
        let inv = 1.0 / (var + eps).sqrt();
        for v in row.iter_mut() {
            *v = (*v - mean) * inv;
        }
    }
    out
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn linear_matches_a_hand_computation() {
        // W is [out=2, in=3]; x is [1, 3].
        let w = Matrix::new(2, 3, vec![1.0, 2.0, 3.0, 4.0, 5.0, 6.0]);
        let x = Matrix::new(1, 3, vec![1.0, 1.0, 1.0]);
        let y = x.linear(&w, None);
        assert_eq!(y.data, vec![6.0, 15.0]);
    }

    #[test]
    fn rms_norm_scales_a_unit_row_to_the_weight() {
        // A row of ones has rms 1, so the output is the weight.
        let m = Matrix::new(1, 3, vec![1.0, 1.0, 1.0]);
        let out = rms_norm_rows(&m, &[2.0, 3.0, 4.0], 1e-6);
        for (got, want) in out.data.iter().zip(&[2.0, 3.0, 4.0]) {
            assert!((got - want).abs() < 1e-5, "got {got}, want {want}");
        }
    }

    #[test]
    fn zero_center_norm_uses_weight_plus_one() {
        // A stored weight of 0 means an effective scale of 1.
        let m = Matrix::new(1, 2, vec![3.0, 4.0]); // rms 5/sqrt(2)... check via ratio
        let out = zero_center_rms_norm_rows(&m, &[0.0, 0.0], 1e-12);
        let ratio = out.data[1] / out.data[0];
        assert!((ratio - 4.0 / 3.0).abs() < 1e-5, "ratio {ratio}");
    }

    #[test]
    fn layer_norm_centres_and_scales() {
        let m = Matrix::new(1, 4, vec![1.0, 2.0, 3.0, 4.0]);
        let out = layer_norm_rows(&m, 1e-12);
        let mean: f32 = out.data.iter().sum::<f32>() / 4.0;
        assert!(mean.abs() < 1e-5, "mean {mean}");
        let var: f32 = out.data.iter().map(|v| v * v).sum::<f32>() / 4.0;
        assert!((var - 1.0).abs() < 1e-4, "var {var}");
    }

    #[test]
    fn gelu_and_silu_match_reference_values() {
        assert!((gelu_tanh(0.0) - 0.0).abs() < 1e-7);
        // Known value: gelu_tanh(1.0) ~= 0.84119199
        assert!((gelu_tanh(1.0) - 0.841_192).abs() < 1e-5);
        assert!((silu(0.0) - 0.0).abs() < 1e-7);
        assert!((silu(1.0) - 0.731_058_6).abs() < 1e-6);
    }

    #[test]
    fn matmul_is_the_plain_product() {
        let a = Matrix::new(2, 2, vec![1.0, 2.0, 3.0, 4.0]);
        let b = Matrix::new(2, 2, vec![5.0, 6.0, 7.0, 8.0]);
        let c = a.matmul(&b);
        assert_eq!(c.data, vec![19.0, 22.0, 43.0, 50.0]);
    }
}

#[cfg(test)]
mod threading_tests {
    use super::*;

    /// Threading must not change a single output bit: rows are independent and
    /// each output accumulates in the same order either way.
    #[test]
    fn threaded_linear_is_bit_identical_to_serial() {
        let w = Matrix::new(7, 5, (0..35).map(|i| (i as f32) * 0.031 - 0.5).collect());
        let x = Matrix::new(13, 5, (0..65).map(|i| ((i * 7) as f32).sin()).collect());
        let serial = x.linear_with_threads(&w, None, 1);
        for threads in [2, 3, 4, 8, 16] {
            let par = x.linear_with_threads(&w, None, threads);
            assert_eq!(
                serial.data, par.data,
                "threads={threads} produced different values"
            );
        }
    }

    #[test]
    fn threaded_linear_with_bias_matches_serial() {
        let w = Matrix::new(4, 3, (0..12).map(|i| i as f32 * 0.25).collect());
        let x = Matrix::new(9, 3, (0..27).map(|i| (i as f32).cos()).collect());
        let b = vec![0.1f32, -0.2, 0.3, 0.4];
        let serial = x.linear_with_threads(&w, Some(&b), 1);
        let par = x.linear_with_threads(&w, Some(&b), 4);
        assert_eq!(serial.data, par.data);
    }
}
