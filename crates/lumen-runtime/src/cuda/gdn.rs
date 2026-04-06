//! GatedDeltaNet (GDN) scratch allocation and dispatch for the CUDA backend.
//!
//! GDN layers use linear attention with a recurrent state matrix instead of
//! softmax attention with a KV cache. This module provides:
//! - `GdnScratch`: GPU buffer allocations for GDN-specific state and intermediates
//! - `GdnParams`: Dimension parameters derived from the model hyperparameters
//! - `dispatch_gdn_layer`: Orchestrates the full GDN layer forward pass
//!
//! GDN layer pipeline (single-token decode):
//!   1. RMSNorm(x) -> normed
//!   2. Fused QKV matvec: normed @ attn_qkv^T -> qkv_buf [Q(2048)+K(2048)+V(4096)]
//!   3. Conv1D on QKV channels (circular buffer state)
//!   4. SiLU activation on conv output
//!   5. Compute gates: alpha (decay) and beta (mixing) from SSM parameters
//!   6. L2-normalize Q and K per head
//!   7. State update: h = alpha*h + beta*outer(k,v), output = q @ h
//!   8. RMSNorm + learned scale on output
//!   9. Attention gate: silu(attn_gate_weight * normed) * normed_output
//!  10. Output projection: wo * gated_output -> residual add

use crate::error::RuntimeError;
use lumen_format::hyperparams::ModelHyperparams;

/// Dimension parameters for GDN layers, derived from model hyperparameters.
///
/// Qwen3.5 GDN layout:
/// - 16 KV heads (pre-GQA), 32 state heads (post-GQA repeat)
/// - head_dim = 128
/// - Q and K: 16 heads x 128 = 2048 each
/// - V: 32 heads x 128 = 4096
/// - QKV total: 8192
/// - Conv kernel size: 4 (3 history slots in circular buffer)
#[derive(Debug, Clone, Copy)]
pub struct GdnParams {
    /// Number of state/V heads (typically 32 for Qwen3.5).
    pub num_heads: usize,
    /// Number of KV heads pre-GQA repeat (typically 16 for Qwen3.5).
    pub num_kv_heads: usize,
    /// Per-head dimension (typically 128).
    pub head_dim: usize,
    /// Q and K dimension: num_kv_heads * head_dim (typically 2048).
    pub qk_dim: usize,
    /// V / inner_size dimension: num_heads * head_dim (typically 4096).
    pub value_dim: usize,
    /// Total QKV dimension: qk_dim + qk_dim + value_dim (typically 8192).
    pub qkv_dim: usize,
    /// Conv1d kernel size (typically 4).
    pub conv_kernel_size: usize,
    /// Model hidden dimension (typically 2048 for Qwen3.5-9B).
    pub hidden_dim: usize,
    /// RMSNorm epsilon.
    pub eps: f32,
}

impl GdnParams {
    /// Derive GDN parameters from model hyperparameters.
    ///
    /// GDN dimensions come from SSM metadata, NOT from the model's attention head count:
    /// - num_kv_heads = ssm.group_count (16 for Qwen3.5-9B, NOT model's num_kv_heads=4)
    /// - num_heads = 2 * num_kv_heads = 32 (GQA repeat factor 2)
    /// - head_dim = ssm.state_size = 128 (NOT model's head_dim=256)
    /// - inner_size = num_heads * head_dim = 4096
    /// - conv_kernel_size = 4
    ///
    /// The model hyperparams only provide hidden_dim (4096) and eps. The SSM-specific
    /// dimensions are hardcoded here because ModelHyperparams doesn't carry them.
    pub fn from_hyperparams(hp: &ModelHyperparams) -> Self {
        // Qwen3.5-9B GDN: group_count=16, state_size=128, inner_size=4096
        let num_kv_heads = 16;  // ssm.group_count
        let head_dim = 128;     // ssm.state_size
        let num_heads = num_kv_heads * 2; // GQA repeat factor 2 = 32
        Self::with_dims(num_heads, num_kv_heads, head_dim, 4, hp.hidden_dim as usize, hp.norm_eps)
    }

    /// Construct GDN parameters with explicit dimensions.
    /// Used by tests and models with non-standard SSM dimensions.
    pub fn with_dims(
        num_heads: usize,
        num_kv_heads: usize,
        head_dim: usize,
        conv_kernel_size: usize,
        hidden_dim: usize,
        eps: f32,
    ) -> Self {
        let qk_dim = num_kv_heads * head_dim;
        let value_dim = num_heads * head_dim;
        let qkv_dim = qk_dim + qk_dim + value_dim;
        Self {
            num_heads,
            num_kv_heads,
            head_dim,
            qk_dim,
            value_dim,
            qkv_dim,
            conv_kernel_size,
            hidden_dim,
            eps,
        }
    }

    /// Size of the recurrent state matrix per GDN layer in f32 elements.
    /// Layout: [num_heads, head_dim (val), head_dim (key)] transposed for coalesced access.
    pub fn h_state_elements(&self) -> usize {
        self.num_heads * self.head_dim * self.head_dim
    }

    /// Size of the conv1d circular buffer per GDN layer in f32 elements.
    /// Layout: [conv_kernel_size - 1, qkv_dim] -- stores history for all QKV channels.
    pub fn conv_state_elements(&self) -> usize {
        (self.conv_kernel_size - 1) * self.qkv_dim
    }
}

/// GPU-resident scratch buffers for GDN layer computation.
///
/// All buffers are allocated once during `init()` and reused across tokens.
/// Per-layer state (h_states, conv_states) persists across tokens within a
/// sequence. Shared scratch buffers (qkv_buf, gate_buf, etc.) are ephemeral
/// and overwritten each layer.
pub struct GdnScratch {
    /// GDN dimension parameters.
    pub params: GdnParams,

    /// Recurrent hidden state per GDN layer.
    /// Each entry: [num_heads * head_dim * head_dim] f32, transposed layout.
    /// Persists across tokens, reset between sequences.
    pub h_states: Vec<Vec<f32>>,

    /// Conv1d circular buffer state per GDN layer.
    /// Each entry: [(conv_kernel_size - 1) * qkv_dim] f32.
    /// Shifted each token, reset between sequences.
    pub conv_states: Vec<Vec<f32>>,

    /// Current write position in each conv circular buffer [0..kernel_size-2].
    pub conv_positions: Vec<u32>,

    // Shared scratch buffers (ephemeral, rewritten each layer)

    /// QKV matvec output: [qkv_dim] f32.
    pub qkv_buf: Vec<f32>,

    /// Conv1d output: [qkv_dim] f32.
    pub qkv_conv_buf: Vec<f32>,

    /// Alpha (decay) per head: [num_heads] f32.
    pub alpha_buf: Vec<f32>,

    /// Beta (mixing rate) per head: [num_heads] f32.
    pub beta_buf: Vec<f32>,

    /// Raw alpha projection output (pre-gate transform): [num_heads] f32.
    pub alpha_raw_buf: Vec<f32>,

    /// Raw beta projection output (pre-gate transform): [num_heads] f32.
    pub beta_raw_buf: Vec<f32>,

    /// GDN output after state query: [value_dim] f32.
    pub output_buf: Vec<f32>,

    /// RMSNorm + scale output: [value_dim] f32.
    pub normed_out_buf: Vec<f32>,

    /// Attention gate sigmoid output: [value_dim] f32.
    pub gate_sigmoid_buf: Vec<f32>,

    /// SSM output projection result: [hidden_dim] f32.
    pub ssm_proj_buf: Vec<f32>,
}

#[allow(dead_code)]
impl GdnScratch {
    /// Allocate GDN scratch buffers for the given model configuration.
    ///
    /// # Arguments
    /// - `params`: GDN dimension parameters
    /// - `num_gdn_layers`: Number of GDN layers in the model (layer_type=1)
    pub fn allocate(params: GdnParams, num_gdn_layers: usize) -> Self {
        let h_state_size = params.h_state_elements();
        let conv_state_size = params.conv_state_elements();

        Self {
            params,
            h_states: (0..num_gdn_layers).map(|_| vec![0.0f32; h_state_size]).collect(),
            conv_states: (0..num_gdn_layers).map(|_| vec![0.0f32; conv_state_size]).collect(),
            conv_positions: vec![0u32; num_gdn_layers],
            qkv_buf: vec![0.0f32; params.qkv_dim],
            qkv_conv_buf: vec![0.0f32; params.qkv_dim],
            alpha_buf: vec![0.0f32; params.num_heads],
            beta_buf: vec![0.0f32; params.num_heads],
            alpha_raw_buf: vec![0.0f32; params.num_heads],
            beta_raw_buf: vec![0.0f32; params.num_heads],
            output_buf: vec![0.0f32; params.value_dim],
            normed_out_buf: vec![0.0f32; params.value_dim],
            gate_sigmoid_buf: vec![0.0f32; params.value_dim],
            ssm_proj_buf: vec![0.0f32; params.hidden_dim],
        }
    }

    /// Reset all recurrent state to zero for a new sequence.
    /// Clears h_states and conv_states, resets conv positions.
    pub fn reset(&mut self) {
        for h in &mut self.h_states {
            h.fill(0.0);
        }
        for c in &mut self.conv_states {
            c.fill(0.0);
        }
        self.conv_positions.fill(0);
    }

    /// Number of GDN layers this scratch was allocated for.
    pub fn num_layers(&self) -> usize {
        self.h_states.len()
    }
}

// ============================================================================
// GDN CPU reference kernels (placeholder for CUDA kernel dispatch)
// ============================================================================

/// Apply 1D convolution on QKV channels using circular buffer state.
///
/// For each channel i in [0..dim):
///   1. Write current input into conv_state at current position
///   2. Compute dot product of conv_state (circularly) with conv_weight
///   3. Write result to output
///
/// Returns the next conv position (circular, wraps at kernel_size - 1).
#[allow(dead_code)]
pub fn conv1d_decode(
    input: &[f32],
    conv_state: &mut [f32],
    conv_weight: &[f32],
    output: &mut [f32],
    dim: usize,
    kernel_size: usize,
    conv_pos: u32,
) -> u32 {
    let buf_slots = (kernel_size - 1) as u32;

    // Write current input into conv state at current position
    let pos = conv_pos as usize;
    for i in 0..dim {
        conv_state[pos * dim + i] = input[i];
    }

    // Convolve: for each channel, dot product of state history with weight
    // Weight layout: [kernel_size, dim] where kernel_size includes current input
    for i in 0..dim {
        let mut sum = input[i] * conv_weight[(kernel_size - 1) * dim + i];
        for k in 0..buf_slots as usize {
            let state_pos = ((pos + buf_slots as usize - k) % buf_slots as usize) * dim + i;
            sum += conv_state[state_pos] * conv_weight[(kernel_size - 2 - k) * dim + i];
        }
        output[i] = sum;
    }

    (conv_pos + 1) % buf_slots
}

/// SiLU activation in-place: x = x * sigmoid(x).
#[allow(dead_code)]
pub fn silu_inplace(x: &mut [f32]) {
    for v in x.iter_mut() {
        *v = *v / (1.0 + (-*v).exp());
    }
}

/// Softplus: y = ln(1 + exp(x)).
#[allow(dead_code)]
fn softplus(x: f32) -> f32 {
    if x > 20.0 {
        x // Avoid overflow: softplus(x) -> x for large x
    } else {
        (1.0 + x.exp()).ln()
    }
}

/// Compute GDN gate values from SSM parameters.
///
/// For each head h:
///   dt = softplus(dt_bias[h])
///   alpha[h] = exp(ssm_a[h] * dt) * sigmoid(alpha_raw[h])
///   beta[h] = sigmoid(beta_raw[h])
///
/// Note: ssm_a stores -exp(A_log), so ssm_a * dt is already negative,
/// making exp(ssm_a * dt) a decay in (0, 1).
#[allow(dead_code)]
pub fn compute_gates(
    dt_bias: &[f32],
    ssm_a: &[f32],
    alpha_raw: &[f32],
    beta_raw: &[f32],
    alpha_out: &mut [f32],
    beta_out: &mut [f32],
    num_heads: usize,
) {
    for h in 0..num_heads {
        let dt = softplus(dt_bias[h]);
        let decay = (ssm_a[h] * dt).exp();
        let alpha_gate = 1.0 / (1.0 + (-alpha_raw[h]).exp()); // sigmoid
        alpha_out[h] = decay * alpha_gate;
        let beta_gate = 1.0 / (1.0 + (-beta_raw[h]).exp()); // sigmoid
        beta_out[h] = beta_gate;
    }
}

/// L2-normalize vectors per head, in-place.
///
/// For each head, compute L2 norm and divide each element by it.
/// eps prevents division by zero.
#[allow(dead_code)]
pub fn l2_normalize_heads(data: &mut [f32], num_heads: usize, head_dim: usize, eps: f32) {
    for h in 0..num_heads {
        let start = h * head_dim;
        let end = start + head_dim;
        let head = &mut data[start..end];

        let norm_sq: f32 = head.iter().map(|x| x * x).sum();
        let norm = (norm_sq + eps).sqrt();
        let inv_norm = 1.0 / norm;
        for v in head.iter_mut() {
            *v *= inv_norm;
        }
    }
}

/// GDN recurrent state update and output computation.
///
/// For each head h:
///   h_state[h] = alpha[h] * h_state[h] + beta[h] * outer(k[h], v[h])
///   output[h] = h_state[h]^T @ q[h]
///
/// Uses GQA: Q and K have num_kv_heads, mapped to num_heads via repeat.
#[allow(dead_code)]
pub fn state_update_and_output(
    h_state: &mut [f32],
    q_norm: &[f32],
    k_norm: &[f32],
    v: &[f32],
    alpha: &[f32],
    beta: &[f32],
    output: &mut [f32],
    num_heads: usize,
    num_kv_heads: usize,
    head_dim: usize,
) {
    let gqa_ratio = num_heads / num_kv_heads;

    for h in 0..num_heads {
        let kv_h = h / gqa_ratio;
        let k_start = kv_h * head_dim;
        let v_start = h * head_dim;
        let q_start = kv_h * head_dim;

        // State: h_state[h] is [head_dim(val) x head_dim(key)] in transposed layout
        let state_start = h * head_dim * head_dim;
        let a = alpha[h];
        let b = beta[h];

        // Update: h = alpha * h + beta * outer(k, v)
        for vi in 0..head_dim {
            for ki in 0..head_dim {
                let idx = state_start + vi * head_dim + ki;
                h_state[idx] = a * h_state[idx] + b * k_norm[k_start + ki] * v[v_start + vi];
            }
        }

        // Output: o[h, vi] = sum_ki(h_state[h, vi, ki] * q[h, ki])
        let o_start = h * head_dim;
        for vi in 0..head_dim {
            let mut sum = 0.0f32;
            for ki in 0..head_dim {
                sum += h_state[state_start + vi * head_dim + ki] * q_norm[q_start + ki];
            }
            output[o_start + vi] = sum;
        }
    }
}

/// RMSNorm with learned scale, applied per-element.
///
/// out[i] = (x[i] / rms(x)) * scale[i % scale_len]
/// where rms(x) = sqrt(mean(x^2) + eps)
#[allow(dead_code)]
pub fn rmsnorm_scale(x: &[f32], scale: &[f32], out: &mut [f32], eps: f32) {
    let n = x.len();
    let sum_sq: f32 = x.iter().map(|v| v * v).sum();
    let rms = (sum_sq / n as f32 + eps).sqrt();
    let inv_rms = 1.0 / rms;
    for i in 0..n {
        out[i] = x[i] * inv_rms * scale[i % scale.len()];
    }
}

/// Matrix-vector multiply reading weights from raw LE bytes.
///
/// out[row] = dot(weight_row[row], input)
/// Weight layout: [num_rows, in_dim] stored as LE f32 bytes.
#[allow(dead_code)]
fn matvec_bytes(out: &mut [f32], weight_bytes: &[u8], input: &[f32], num_rows: usize, in_dim: usize) {
    let row_bytes = in_dim * 4;
    for row in 0..num_rows {
        let w_start = row * row_bytes;
        let mut sum = 0.0f32;
        for col in 0..in_dim {
            let byte_off = w_start + col * 4;
            let w = f32::from_le_bytes([
                weight_bytes[byte_off],
                weight_bytes[byte_off + 1],
                weight_bytes[byte_off + 2],
                weight_bytes[byte_off + 3],
            ]);
            sum += w * input[col];
        }
        out[row] = sum;
    }
}

/// RMSNorm reading weights from raw LE bytes.
#[allow(dead_code)]
fn rmsnorm_bytes(out: &mut [f32], input: &[f32], norm_bytes: &[u8], eps: f32) {
    let n = input.len();
    let sum_sq: f32 = input.iter().map(|v| v * v).sum();
    let rms = (sum_sq / n as f32 + eps).sqrt();
    let inv_rms = 1.0 / rms;
    for i in 0..n {
        let byte_off = i * 4;
        let w = f32::from_le_bytes([
            norm_bytes[byte_off],
            norm_bytes[byte_off + 1],
            norm_bytes[byte_off + 2],
            norm_bytes[byte_off + 3],
        ]);
        out[i] = input[i] * inv_rms * w;
    }
}

/// Read f32 values from a byte slice at the given offset and length.
#[allow(dead_code)]
fn read_f32_slice(bytes: &[u8], offset: u64, count: usize) -> Vec<f32> {
    let start = offset as usize;
    let mut out = Vec::with_capacity(count);
    for i in 0..count {
        let byte_off = start + i * 4;
        out.push(f32::from_le_bytes([
            bytes[byte_off],
            bytes[byte_off + 1],
            bytes[byte_off + 2],
            bytes[byte_off + 3],
        ]));
    }
    out
}

/// Execute one GDN layer on the CPU (reference implementation for CUDA backend).
///
/// This implements the full GDN decode pipeline using the scratch buffers.
/// On actual CUDA hardware, each step would dispatch a GPU kernel instead.
///
/// Returns Ok(()) on success. The residual is added in-place to `x`.
#[allow(clippy::too_many_arguments)]
#[allow(dead_code)]
pub fn dispatch_gdn_layer(
    x: &mut [f32],
    normed: &mut [f32],
    scratch: &mut GdnScratch,
    gdn_idx: usize,
    layer_bytes: &[u8],
    attn_norm_off: u64,
    wq_off: u64,
    ssm_conv1d_off: u64,
    ssm_dt_off: u64,
    ssm_a_off: u64,
    ssm_alpha_off: u64,
    ssm_beta_off: u64,
    ssm_norm_off: u64,
    ssm_out_off: u64,
    attn_gate_off: u64,
) -> Result<(), RuntimeError> {
    let p = scratch.params;

    // Step 1: RMSNorm(x) -> normed
    let norm_bytes = &layer_bytes[attn_norm_off as usize..(attn_norm_off as usize + p.hidden_dim * 4)];
    rmsnorm_bytes(normed, x, norm_bytes, p.eps);

    // Step 2: QKV matvec: normed @ attn_qkv^T -> qkv_buf
    let wq_bytes = &layer_bytes[wq_off as usize..];
    matvec_bytes(&mut scratch.qkv_buf, wq_bytes, normed, p.qkv_dim, p.hidden_dim);

    // Step 3: Conv1d decode on QKV channels
    let conv_weight = read_f32_slice(layer_bytes, ssm_conv1d_off, p.conv_kernel_size * p.qkv_dim);
    let new_pos = conv1d_decode(
        &scratch.qkv_buf,
        &mut scratch.conv_states[gdn_idx],
        &conv_weight,
        &mut scratch.qkv_conv_buf,
        p.qkv_dim,
        p.conv_kernel_size,
        scratch.conv_positions[gdn_idx],
    );
    scratch.conv_positions[gdn_idx] = new_pos;

    // Step 3b: SiLU activation on conv output
    silu_inplace(&mut scratch.qkv_conv_buf);

    // Step 4a: alpha_raw = matvec(ssm_alpha, normed)
    let alpha_weight_bytes = &layer_bytes[ssm_alpha_off as usize..];
    matvec_bytes(&mut scratch.alpha_raw_buf, alpha_weight_bytes, normed, p.num_heads, p.hidden_dim);

    // Step 4b: beta_raw = matvec(ssm_beta, normed)
    let beta_weight_bytes = &layer_bytes[ssm_beta_off as usize..];
    matvec_bytes(&mut scratch.beta_raw_buf, beta_weight_bytes, normed, p.num_heads, p.hidden_dim);

    // Step 4c: Compute gates (alpha, beta) from SSM parameters
    let dt_bias = read_f32_slice(layer_bytes, ssm_dt_off, p.num_heads);
    let ssm_a = read_f32_slice(layer_bytes, ssm_a_off, p.num_heads);
    compute_gates(
        &dt_bias,
        &ssm_a,
        &scratch.alpha_raw_buf,
        &scratch.beta_raw_buf,
        &mut scratch.alpha_buf,
        &mut scratch.beta_buf,
        p.num_heads,
    );

    // Step 5: L2-normalize Q and K per head
    // Q is at qkv_conv_buf[0..qk_dim], K at [qk_dim..2*qk_dim]
    l2_normalize_heads(&mut scratch.qkv_conv_buf[..p.qk_dim], p.num_kv_heads, p.head_dim, 1e-12);
    l2_normalize_heads(&mut scratch.qkv_conv_buf[p.qk_dim..2 * p.qk_dim], p.num_kv_heads, p.head_dim, 1e-12);

    // Steps 6+7: State update + output
    let q_norm = &scratch.qkv_conv_buf[..p.qk_dim];
    let k_norm = &scratch.qkv_conv_buf[p.qk_dim..2 * p.qk_dim];
    let v = &scratch.qkv_conv_buf[2 * p.qk_dim..];

    // Need to split borrow: clone q, k, v slices to avoid borrow conflict
    let q_norm_copy: Vec<f32> = q_norm.to_vec();
    let k_norm_copy: Vec<f32> = k_norm.to_vec();
    let v_copy: Vec<f32> = v.to_vec();
    let alpha_copy: Vec<f32> = scratch.alpha_buf.clone();
    let beta_copy: Vec<f32> = scratch.beta_buf.clone();

    state_update_and_output(
        &mut scratch.h_states[gdn_idx],
        &q_norm_copy,
        &k_norm_copy,
        &v_copy,
        &alpha_copy,
        &beta_copy,
        &mut scratch.output_buf,
        p.num_heads,
        p.num_kv_heads,
        p.head_dim,
    );

    // Step 8: RMSNorm + scale on output
    let ssm_norm = read_f32_slice(layer_bytes, ssm_norm_off, p.head_dim);
    rmsnorm_scale(&scratch.output_buf, &ssm_norm, &mut scratch.normed_out_buf, p.eps);

    // Step 9: Attention gate: gate = silu(attn_gate_weight * normed)
    let attn_gate_bytes = &layer_bytes[attn_gate_off as usize..];
    matvec_bytes(&mut scratch.gate_sigmoid_buf, attn_gate_bytes, normed, p.value_dim, p.hidden_dim);
    // Apply silu-gated multiply: gate_sigmoid = silu(gate) * normed_out
    for i in 0..p.value_dim {
        let gate = scratch.gate_sigmoid_buf[i];
        let silu_gate = gate / (1.0 + (-gate).exp());
        scratch.gate_sigmoid_buf[i] = silu_gate * scratch.normed_out_buf[i];
    }

    // Step 10: Output projection: ssm_proj = ssm_out_weight * gate_sigmoid
    let ssm_out_bytes = &layer_bytes[ssm_out_off as usize..];
    matvec_bytes(&mut scratch.ssm_proj_buf, ssm_out_bytes, &scratch.gate_sigmoid_buf, p.hidden_dim, p.value_dim);

    // Step 11: Residual: x += ssm_proj
    for i in 0..p.hidden_dim {
        x[i] += scratch.ssm_proj_buf[i];
    }

    Ok(())
}

/// Batched causal conv1d + SiLU for T tokens (CPU reference for CUDA kernel).
///
/// Processes all T tokens in sequence, reading from conv_state for taps that
/// precede the batch and from earlier tokens in the batch for later taps.
/// After processing, updates conv_state with the last (kernel_size-1) tokens.
///
/// Returns the new conv_position after processing all T tokens.
#[allow(dead_code)]
pub fn conv1d_silu_prefill(
    input: &[f32],       // [T * dim]
    conv_state: &mut [f32], // [(kernel_size-1) * dim]
    conv_weight: &[f32],  // [dim * kernel_size]
    output: &mut [f32],   // [T * dim]
    dim: usize,
    kernel_size: usize,
    conv_pos: u32,
    t_count: usize,
) -> u32 {
    let buf_slots = (kernel_size - 1) as u32;

    for t in 0..t_count {
        for d in 0..dim {
            let mut sum = 0.0f32;

            for tap in 0..kernel_size {
                let tokens_back = kernel_size - 1 - tap;
                let source_t = t as i32 - tokens_back as i32;

                let input_val = if source_t >= 0 {
                    input[source_t as usize * dim + d]
                } else {
                    let slot = (conv_pos as usize + buf_slots as usize
                        + (buf_slots as i32 + source_t) as usize) % buf_slots as usize;
                    conv_state[slot * dim + d]
                };

                sum += conv_weight[d * kernel_size + tap] * input_val;
            }

            // SiLU: x / (1 + exp(-x))
            output[t * dim + d] = sum / (1.0 + (-sum).exp());
        }
    }

    // Update conv_state with last buf_slots tokens
    for s in 0..buf_slots as usize {
        let src_t = t_count as i32 - buf_slots as i32 + s as i32;
        if src_t >= 0 {
            let new_slot = (conv_pos as usize + s) % buf_slots as usize;
            for d in 0..dim {
                conv_state[new_slot * dim + d] = input[src_t as usize * dim + d];
            }
        }
    }

    (conv_pos + t_count as u32) % buf_slots
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn gdn_params_from_hyperparams_qwen35() {
        let hp = ModelHyperparams {
            num_layers: 32,
            num_heads: 16,
            num_kv_heads: 4,
            head_dim: 256,
            hidden_dim: 2048,
            intermediate_dim: 12288,
            vocab_size: 248320,
            max_seq_len: 2048,
            rope_params: None,
            num_experts: None,
            num_active_experts: None,
            norm_eps: 1e-6,
            rotary_dim: None, rope_neox: false,
        };
        let p = GdnParams::from_hyperparams(&hp);
        assert_eq!(p.num_kv_heads, 16);
        assert_eq!(p.num_heads, 32);
        assert_eq!(p.head_dim, 128);
        assert_eq!(p.qk_dim, 2048);
        assert_eq!(p.value_dim, 4096);
        assert_eq!(p.qkv_dim, 8192);
        assert_eq!(p.conv_kernel_size, 4);
    }

    #[test]
    fn gdn_scratch_allocation_and_reset() {
        let params = GdnParams {
            num_heads: 4,
            num_kv_heads: 2,
            head_dim: 4,
            qk_dim: 8,
            value_dim: 16,
            qkv_dim: 32,
            conv_kernel_size: 4,
            hidden_dim: 8,
            eps: 1e-6,
        };
        let mut scratch = GdnScratch::allocate(params, 3);

        assert_eq!(scratch.num_layers(), 3);
        assert_eq!(scratch.h_states[0].len(), 4 * 4 * 4);
        assert_eq!(scratch.conv_states[0].len(), 3 * 32);

        // Modify state
        scratch.h_states[0][0] = 42.0;
        scratch.conv_states[1][5] = 7.0;
        scratch.conv_positions[2] = 2;

        // Reset clears everything
        scratch.reset();
        assert_eq!(scratch.h_states[0][0], 0.0);
        assert_eq!(scratch.conv_states[1][5], 0.0);
        assert_eq!(scratch.conv_positions[2], 0);
    }

    #[test]
    fn silu_inplace_basic() {
        let mut x = vec![0.0, 1.0, -1.0, 5.0];
        silu_inplace(&mut x);
        // silu(0) = 0, silu(1) ~= 0.7311, silu(-1) ~= -0.2689, silu(5) ~= 4.9665
        assert!((x[0] - 0.0).abs() < 1e-6);
        assert!((x[1] - 0.7311).abs() < 1e-3);
        assert!((x[2] - (-0.2689)).abs() < 1e-3);
        assert!((x[3] - 4.9665).abs() < 1e-3);
    }

    #[test]
    fn l2_normalize_heads_basic() {
        let mut data = vec![3.0, 4.0, 0.0, 0.0, 0.0, 5.0, 0.0, 0.0];
        l2_normalize_heads(&mut data, 2, 4, 1e-12);
        // Head 0: [3,4,0,0] -> norm=5, [0.6, 0.8, 0, 0]
        assert!((data[0] - 0.6).abs() < 1e-5);
        assert!((data[1] - 0.8).abs() < 1e-5);
        // Head 1: [0,5,0,0] -> norm=5, [0, 1, 0, 0]
        assert!((data[5] - 1.0).abs() < 1e-5);
    }

    #[test]
    fn compute_gates_basic() {
        let dt_bias = vec![0.0f32]; // softplus(0) = ln(2) ~= 0.693
        let ssm_a = vec![-1.0f32]; // ssm_a = -exp(A_log), decay = exp(-1 * 0.693) = exp(-0.693) ~= 0.5
        let alpha_raw = vec![0.0f32]; // sigmoid(0) = 0.5
        let beta_raw = vec![0.0f32]; // sigmoid(0) = 0.5
        let mut alpha = vec![0.0f32];
        let mut beta = vec![0.0f32];
        compute_gates(&dt_bias, &ssm_a, &alpha_raw, &beta_raw, &mut alpha, &mut beta, 1);
        // alpha = decay * sigmoid(alpha_raw) = 0.5 * 0.5 = 0.25
        assert!((alpha[0] - 0.25).abs() < 1e-3);
        assert!((beta[0] - 0.5).abs() < 1e-3);
    }

    /// Verify that batched conv1d+SiLU produces the same output as sequential decode.
    #[test]
    fn conv1d_silu_prefill_matches_sequential() {
        let dim = 8;
        let kernel_size = 4;
        let t_count = 6; // Process 6 tokens
        let buf_slots = kernel_size - 1; // 3

        // Random-ish input and weights
        let input: Vec<f32> = (0..t_count * dim)
            .map(|i| ((i as f32) * 0.1 + 0.3).sin())
            .collect();
        let conv_weight: Vec<f32> = (0..dim * kernel_size)
            .map(|i| ((i as f32) * 0.2 + 0.5).cos())
            .collect();

        // --- Sequential path: conv1d_decode + silu_inplace for each token ---
        let mut seq_state = vec![0.0f32; buf_slots * dim];
        let mut seq_pos = 0u32;
        let mut seq_output = vec![0.0f32; t_count * dim];
        let mut conv_out = vec![0.0f32; dim];

        for t in 0..t_count {
            let token_input = &input[t * dim..(t + 1) * dim];
            seq_pos = conv1d_decode(
                token_input, &mut seq_state, &conv_weight,
                &mut conv_out, dim, kernel_size, seq_pos,
            );
            silu_inplace(&mut conv_out);
            seq_output[t * dim..(t + 1) * dim].copy_from_slice(&conv_out);
        }

        // --- Batched path: conv1d_silu_prefill ---
        let mut batch_state = vec![0.0f32; buf_slots * dim];
        let mut batch_output = vec![0.0f32; t_count * dim];
        let batch_new_pos = conv1d_silu_prefill(
            &input, &mut batch_state, &conv_weight,
            &mut batch_output, dim, kernel_size, 0, t_count,
        );

        // Verify outputs match
        for i in 0..t_count * dim {
            assert!(
                (seq_output[i] - batch_output[i]).abs() < 1e-5,
                "Mismatch at index {i}: seq={} batch={}",
                seq_output[i], batch_output[i]
            );
        }

        // Verify final conv position matches
        assert_eq!(seq_pos, batch_new_pos,
            "Conv positions differ: seq={seq_pos} batch={batch_new_pos}");

        // Verify conv_state matches
        for i in 0..buf_slots * dim {
            assert!(
                (seq_state[i] - batch_state[i]).abs() < 1e-6,
                "Conv state mismatch at index {i}: seq={} batch={}",
                seq_state[i], batch_state[i]
            );
        }
    }
}
