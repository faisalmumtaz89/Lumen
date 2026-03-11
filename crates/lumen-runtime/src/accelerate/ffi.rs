//! Apple Accelerate framework FFI bindings for `cblas_sgemm`.
//!
//! Links against the Accelerate framework (which includes vecLib/BLAS)
//! using the same `#[link(kind = "framework")]` pattern as `metal_ffi.rs`.
//!
//! Only the single function we need (`cblas_sgemm`) is declared. On Apple
//! Silicon, Accelerate dispatches SGEMM to the AMX coprocessor for large
//! matrix-matrix multiplies, achieving near-peak FLOPS.

/// CBLAS row/column ordering.
#[repr(u32)]
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
#[allow(dead_code)]
pub enum CblasOrder {
    RowMajor = 101,
    ColMajor = 102,
}

/// CBLAS transpose flags.
#[repr(u32)]
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum CblasTranspose {
    NoTrans = 111,
    Trans = 112,
}

#[link(name = "Accelerate", kind = "framework")]
extern "C" {
    /// General single-precision matrix-matrix multiply.
    ///
    /// Computes: C = alpha * op(A) * op(B) + beta * C
    ///
    /// where op(X) is X or X^T depending on the transpose flags.
    ///
    /// For our use case (batched projection):
    ///   - RowMajor layout
    ///   - A = X_batch [M x K], NoTrans
    ///   - B = W^T [K x N], Trans (weights stored row-major as [N x K])
    ///   - C = out [M x N]
    ///   - alpha = 1.0, beta = 0.0
    pub fn cblas_sgemm(
        order: CblasOrder,
        trans_a: CblasTranspose,
        trans_b: CblasTranspose,
        m: i32,       // rows of op(A) and C
        n: i32,       // cols of op(B) and C
        k: i32,       // cols of op(A) / rows of op(B)
        alpha: f32,
        a: *const f32, // [lda x ...] matrix A
        lda: i32,      // leading dimension of A
        b: *const f32, // [ldb x ...] matrix B
        ldb: i32,      // leading dimension of B
        beta: f32,
        c: *mut f32,   // [ldc x ...] matrix C (output)
        ldc: i32,      // leading dimension of C
    );
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_cblas_sgemm_identity() {
        // Smoke test: multiply a 4x4 matrix by identity.
        // A = [[1,2,3,4],[5,6,7,8],[9,10,11,12],[13,14,15,16]]
        // B = I_4x4
        // C should equal A.
        let a: [f32; 16] = [
            1.0, 2.0, 3.0, 4.0,
            5.0, 6.0, 7.0, 8.0,
            9.0, 10.0, 11.0, 12.0,
            13.0, 14.0, 15.0, 16.0,
        ];
        let b: [f32; 16] = [
            1.0, 0.0, 0.0, 0.0,
            0.0, 1.0, 0.0, 0.0,
            0.0, 0.0, 1.0, 0.0,
            0.0, 0.0, 0.0, 1.0,
        ];
        let mut c = [0.0f32; 16];

        unsafe {
            cblas_sgemm(
                CblasOrder::RowMajor,
                CblasTranspose::NoTrans,
                CblasTranspose::NoTrans,
                4, 4, 4,  // M, N, K
                1.0,      // alpha
                a.as_ptr(), 4,
                b.as_ptr(), 4,
                0.0,      // beta
                c.as_mut_ptr(), 4,
            );
        }

        for i in 0..16 {
            assert!(
                (c[i] - a[i]).abs() < 1e-6,
                "c[{i}] = {}, expected {}", c[i], a[i]
            );
        }
    }

    #[test]
    fn test_cblas_sgemm_known_product() {
        // A = [[1, 2], [3, 4]]  (2x2, row-major)
        // B = [[5, 6], [7, 8]]  (2x2, row-major)
        // C = A * B = [[19, 22], [43, 50]]
        let a: [f32; 4] = [1.0, 2.0, 3.0, 4.0];
        let b: [f32; 4] = [5.0, 6.0, 7.0, 8.0];
        let mut c = [0.0f32; 4];

        unsafe {
            cblas_sgemm(
                CblasOrder::RowMajor,
                CblasTranspose::NoTrans,
                CblasTranspose::NoTrans,
                2, 2, 2,
                1.0,
                a.as_ptr(), 2,
                b.as_ptr(), 2,
                0.0,
                c.as_mut_ptr(), 2,
            );
        }

        let expected = [19.0, 22.0, 43.0, 50.0];
        for i in 0..4 {
            assert!(
                (c[i] - expected[i]).abs() < 1e-5,
                "c[{i}] = {}, expected {}", c[i], expected[i]
            );
        }
    }

    #[test]
    fn test_cblas_sgemm_transpose_b() {
        // Test the transpose-B path we use for weight projection:
        // A = [[1, 2, 3], [4, 5, 6]]  (2x3)
        // B = [[1, 4], [2, 5], [3, 6]] stored row-major as (3x2),
        //     but with Trans flag so op(B) = B^T = [[1,2,3],[4,5,6]] (2x3)
        // Wait, Trans on B means op(B) = B^T. B is 3x2, B^T is 2x3.
        // C = A * B^T doesn't work dimension-wise with M=2,N=2,K=3.
        // Actually: let B be stored row-major as [2x3] = [[1,2,3],[4,5,6]].
        // With Trans, op(B) = B^T = [3x2]. Then A[2x3] * B^T[3x2] = C[2x2].
        let a: [f32; 6] = [1.0, 2.0, 3.0, 4.0, 5.0, 6.0]; // 2x3
        let b: [f32; 6] = [1.0, 2.0, 3.0, 4.0, 5.0, 6.0]; // 2x3, transposed to 3x2
        let mut c = [0.0f32; 4];

        // C = A * B^T, M=2, N=2, K=3
        // C[0,0] = 1*1 + 2*2 + 3*3 = 14
        // C[0,1] = 1*4 + 2*5 + 3*6 = 32
        // C[1,0] = 4*1 + 5*2 + 6*3 = 32
        // C[1,1] = 4*4 + 5*5 + 6*6 = 77
        unsafe {
            cblas_sgemm(
                CblasOrder::RowMajor,
                CblasTranspose::NoTrans,
                CblasTranspose::Trans,
                2, 2, 3,  // M, N, K
                1.0,
                a.as_ptr(), 3,  // lda = K = 3
                b.as_ptr(), 3,  // ldb = K = 3 (B is stored as N x K before transpose)
                0.0,
                c.as_mut_ptr(), 2,  // ldc = N = 2
            );
        }

        let expected = [14.0, 32.0, 32.0, 77.0];
        for i in 0..4 {
            assert!(
                (c[i] - expected[i]).abs() < 1e-4,
                "c[{i}] = {}, expected {}", c[i], expected[i]
            );
        }
    }
}
