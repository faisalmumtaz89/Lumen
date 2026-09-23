//! The flow-matching Euler scheduler.
//!
//! Mirrors `FlowMatchEulerDiscreteScheduler` with the settings Qwen-Image-2.1
//! ships: dynamic shifting of the exponential kind, a terminal sigma, and a
//! non-stochastic Euler update. The arithmetic follows the reference's order —
//! including the final cast back to the model dtype — because reordering it
//! changes the last bits and this is the reference everything else is checked
//! against.

/// Scheduler settings a caller must supply.
#[derive(Debug, Clone, Copy)]
pub struct SchedulerConfig {
    pub base_image_seq_len: f32,
    pub max_image_seq_len: f32,
    pub base_shift: f32,
    pub max_shift: f32,
    /// Stretch the schedule so it terminates here; `None` disables it.
    pub shift_terminal: Option<f32>,
}

impl SchedulerConfig {
    /// The settings recorded in the model's `scheduler_config.json`.
    pub fn qwen_image_2_1() -> Self {
        Self {
            base_image_seq_len: 256.0,
            max_image_seq_len: 8192.0,
            base_shift: 0.5,
            max_shift: 0.9,
            shift_terminal: Some(0.02),
        }
    }
}

/// The sigma schedule for one generation.
#[derive(Debug, Clone)]
pub struct SigmaSchedule {
    /// One per denoising step, plus the terminal zero.
    pub sigmas: Vec<f32>,
}

/// `mu` for a sequence length, as the pipeline computes it.
///
/// The pipeline passes `base_image_seq_len=256`, `max_image_seq_len=8192`,
/// `base_shift=0.5`, `max_shift=0.9` explicitly from the scheduler config.
pub fn calculate_shift(seq_len: f32, cfg: &SchedulerConfig) -> f32 {
    let m = (cfg.max_shift - cfg.base_shift) / (cfg.max_image_seq_len - cfg.base_image_seq_len);
    let b = cfg.base_shift - m * cfg.base_image_seq_len;
    seq_len * m + b
}

/// The exponential time shift: `exp(mu) / (exp(mu) + (1/t - 1)^sigma)`.
fn time_shift_exponential(mu: f32, sigma: f32, t: f32) -> f32 {
    mu.exp() / (mu.exp() + (1.0 / t - 1.0).powf(sigma))
}

/// Stretch the schedule so the last value equals `shift_terminal`.
fn stretch_shift_to_terminal(values: &[f32], terminal: f32) -> Vec<f32> {
    let one_minus_z_last = 1.0 - values[values.len() - 1];
    let scale = one_minus_z_last / (1.0 - terminal);
    values.iter().map(|v| 1.0 - ((1.0 - v) / scale)).collect()
}

impl SigmaSchedule {
    /// Build the schedule for `num_inference_steps` at a given latent sequence
    /// length, exactly as `set_timesteps` does.
    pub fn new(num_inference_steps: usize, seq_len: usize, cfg: &SchedulerConfig) -> Self {
        let mu = calculate_shift(seq_len as f32, cfg);
        let n = num_inference_steps;
        // numpy's linspace(1.0, 1/n, n): the endpoint is exact, the interior is
        // `start + i * step` in f64 then narrowed, which is what this mirrors.
        // numpy's `linspace(1.0, 1/n, n)` computes in f64 with the endpoint
        // pinned, which is what this reproduces before narrowing to f32.
        let mut sigmas: Vec<f32> = (0..n)
            .map(|i| {
                if n == 1 {
                    return 1.0f32;
                }
                let start = 1.0f64;
                let stop = 1.0f64 / n as f64;
                let step = (stop - start) / (n - 1) as f64;
                let t = if i == n - 1 {
                    stop
                } else {
                    start + step * i as f64
                };
                t as f32
            })
            .collect();
        sigmas = sigmas
            .iter()
            .map(|&t| time_shift_exponential(mu, 1.0, t))
            .collect();
        if let Some(terminal) = cfg.shift_terminal {
            sigmas = stretch_shift_to_terminal(&sigmas, terminal);
        }
        sigmas.push(0.0);
        Self { sigmas }
    }

    pub fn len(&self) -> usize {
        self.sigmas.len()
    }

    pub fn is_empty(&self) -> bool {
        self.sigmas.is_empty()
    }

    /// The timestep the reference hands a bf16 transformer at `step`:
    /// `bf16(bf16(sigma * 1000) / 1000)`, its timestep cast to the latents'
    /// dtype and divided by 1000 in that dtype.
    pub fn model_timestep(&self, step: usize) -> f32 {
        let round = crate::tensor::bf16_round;
        round(round(self.sigmas[step] * 1000.0) / 1000.0)
    }

    /// One Euler step: `sample + (sigma_next - sigma) * model_output`, then cast
    /// to the model output's dtype.
    ///
    /// The values arrive in f32 and `out_bf16` records whether the model output
    /// was bf16. When it was, the reference's type promotion makes the product
    /// bf16 — `dt` is cast to bf16 and the product rounded — before it is
    /// added to the f32 sample, and the sum is rounded to bf16 again; each
    /// rounding is to nearest even.
    pub fn step(
        &self,
        step_index: usize,
        sample: &[f32],
        model_output: &[f32],
        out_bf16: bool,
    ) -> Vec<f32> {
        let sigma = self.sigmas[step_index];
        let sigma_next = self.sigmas[step_index + 1];
        let dt = sigma_next - sigma;
        let dt_bf16 = crate::tensor::bf16_round(dt);
        sample
            .iter()
            .zip(model_output)
            .map(|(&s, &m)| {
                if out_bf16 {
                    crate::tensor::bf16_round(s + crate::tensor::bf16_round(dt_bf16 * m))
                } else {
                    s + dt * m
                }
            })
            .collect()
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    /// `mu` at the two sizes the oracle used.
    #[test]
    fn shift_matches_the_pipeline_formula() {
        let cfg = SchedulerConfig::qwen_image_2_1();
        // 1024x1024 -> 64x64 latent tokens.
        let mu = calculate_shift(64.0 * 64.0, &cfg);
        assert!((mu - 0.6935).abs() < 5e-4, "mu was {mu}");
        let mu2 = calculate_shift(128.0 * 128.0, &cfg);
        assert!((mu2 - 1.3129).abs() < 5e-4, "mu was {mu2}");
    }

    /// The schedule must start at 1.0 and end at the terminal sigma.
    #[test]
    fn schedule_spans_one_to_terminal() {
        let cfg = SchedulerConfig::qwen_image_2_1();
        let s = SigmaSchedule::new(40, 64 * 64, &cfg);
        assert_eq!(s.len(), 41);
        assert!(
            (s.sigmas[0] - 1.0).abs() < 1e-6,
            "starts at {}",
            s.sigmas[0]
        );
        assert!(
            (s.sigmas[39] - 0.02).abs() < 1e-6,
            "ends at {}",
            s.sigmas[39]
        );
        assert_eq!(s.sigmas[40], 0.0);
        // Strictly decreasing.
        for w in s.sigmas.windows(2) {
            assert!(w[1] < w[0], "not monotonic at {:?}", w);
        }
    }

    /// The bf16 step rounds `dt`, the product and the sum, each to nearest
    /// even: a sample on a tie rounds to the even neighbour, and a product that
    /// is not a bf16 value is rounded before it is added.
    #[test]
    fn bf16_step_rounds_like_the_reference() {
        let cfg = SchedulerConfig::qwen_image_2_1();
        let s = SigmaSchedule::new(4, 4096, &cfg);
        let tie_down = f32::from_bits(0x3f80_8000);
        let tie_up = f32::from_bits(0x3f81_8000);
        let out = s.step(0, &[tie_down, tie_up], &[0.0, 0.0], true);
        assert_eq!(out[0].to_bits(), 0x3f80_0000);
        assert_eq!(out[1].to_bits(), 0x3f82_0000);

        // A model output whose product with `dt` is not a bf16 value, and a
        // sample for which rounding that product first changes the sum.
        // `dt` itself is not a bf16 value, so the case is also one where
        // rounding it changes the result.
        let round = crate::tensor::bf16_round;
        let raw = s.sigmas[1] - s.sigmas[0];
        let dt = round(raw);
        let (sample, m) = (0..1000)
            .map(|k| (1.0f32, round(1.0 + k as f32 / 64.0)))
            .find(|&(x, m)| {
                let want = round(x + round(dt * m));
                want != round(x + dt * m) && want != round(x + round(raw * m))
            })
            .expect("a case where both roundings matter");
        assert_eq!(
            s.step(0, &[sample], &[m], true),
            vec![round(sample + round(dt * m))]
        );
    }

    /// A step with zero model output leaves the sample alone.
    #[test]
    fn zero_model_output_is_identity() {
        let cfg = SchedulerConfig::qwen_image_2_1();
        let s = SigmaSchedule::new(4, 4096, &cfg);
        let sample = vec![1.0f32, -2.0, 3.5, 0.0];
        let out = s.step(0, &sample, &[0.0; 4], false);
        assert_eq!(out, sample);
    }

    /// The oracle's own sigma schedule for 8 steps at 32x32 latent tokens, read
    /// from `scheduler_sigmas_smoke.npy` of the f32 reference run. The Rust
    /// schedule must reproduce it to within f32 rounding.
    #[test]
    fn schedule_matches_the_oracle_fixture() {
        let cfg = SchedulerConfig::qwen_image_2_1();
        let s = SigmaSchedule::new(8, 32 * 32, &cfg);
        let expected = [
            1.0f32, 0.90613424, 0.8013588, 0.6836543, 0.55047023, 0.39853805, 0.22359914,
            0.02000004, 0.0,
        ];
        assert_eq!(s.len(), expected.len());
        for (i, (got, want)) in s.sigmas.iter().zip(&expected).enumerate() {
            assert!(
                (got - want).abs() < 1e-6,
                "sigma[{i}] = {got}, oracle says {want}"
            );
        }
    }
}
