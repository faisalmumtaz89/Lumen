//! Static equivalence check: `dit_gpu`'s policy helpers against `dit`'s.
//!
//! The forward's control flow (the rope position table, the block-causal mask,
//! the timestep embedding) is pure integer or trig policy, and `dit_gpu`
//! mirrors it because the CPU reference is the specification. This binary
//! compares the two copies directly rather than by inspection, over a sweep that
//! covers the cases the reference's own tests pin and the neighbours they do
//! not.
//!
//! Three outcomes are compared, not two: a value, a rejected shape, and a panic.
//! The reference has an unreachable-in-practice index panic for a block with
//! more than one frame (see `rope_indices`), and a mirror that merely agreed on
//! the happy path would hide a divergence on the others.
//!
//! It compiles under the `cuda` feature because that is the feature the module
//! it checks lives behind, but it touches no device: the helpers are free
//! functions.
//!
//! Usage: `dit-policy-check`

use std::panic::{catch_unwind, AssertUnwindSafe};
use std::process::ExitCode;

use lumen_image::cuda::dit_gpu::policy as gpu;
use lumen_image::dit as cpu;

/// One shape in a sweep: `(img_shapes, img_mask)`, the mask still in slots.
type Case = (Vec<(u64, u64, u64)>, Vec<bool>);

/// What a call did, with a panic counted as an outcome rather than a crash.
#[derive(Debug, PartialEq)]
enum Outcome<T> {
    Value(T),
    /// Returned an error, e.g. a shape the reference refuses.
    Rejected,
    Panicked,
}

fn capture<T, E, F: FnOnce() -> Result<T, E>>(f: F) -> Outcome<T> {
    match catch_unwind(AssertUnwindSafe(f)) {
        Ok(Ok(v)) => Outcome::Value(v),
        Ok(Err(_)) => Outcome::Rejected,
        Err(_) => Outcome::Panicked,
    }
}

/// `text` text-only slots then one image block of `(f, h, w)` latent tokens.
///
/// `f * h * w` must be a multiple of four: one slot stands for a 2x2 group, and
/// the reference only accepts a target whose tokens do.
fn one_image(text: usize, f: u64, h: u64, w: u64) -> Case {
    let tokens = f * h * w;
    assert_eq!(tokens % 4, 0, "a 2x2 slot cannot hold {tokens} tokens");
    let mut mask = vec![false; text];
    mask.extend(std::iter::repeat(true).take(tokens as usize / 4));
    (vec![(f, h, w)], mask)
}

/// A case built from explicit blocks, with `text` text-only slots leading.
fn blocks(text: usize, shapes: &[(u64, u64, u64)]) -> Case {
    let mut mask = vec![false; text];
    for &(f, h, w) in shapes {
        assert_eq!((f * h * w) % 4, 0, "a 2x2 slot cannot hold a {f}x{h}x{w}");
        mask.extend(std::iter::repeat(true).take((f * h * w) as usize / 4));
    }
    (shapes.to_vec(), mask)
}

fn cases() -> Vec<(&'static str, Case)> {
    vec![
        ("text + 2x4", one_image(2, 1, 2, 4)),
        ("text + 4x4", one_image(3, 1, 4, 4)),
        ("text + 3x4", one_image(1, 1, 3, 4)),
        ("text + 6x6", one_image(4, 1, 6, 6)),
        ("no text + 4x4", blocks(0, &[(1, 4, 4)])),
        // Text resuming after an image: the shared position advances by
        // `max(h, w)`, not by the token count, so this is the case that pins the
        // frame axis' step.
        ("text after the image", {
            let (shapes, mut mask) = one_image(2, 1, 3, 4);
            mask.extend([false; 2]);
            (shapes, mask)
        }),
        // Two image blocks with nothing between them: one run of `true` but two
        // blocks, which is the case that pins the block boundaries. The second
        // is larger, so a boundary taken from the run's length disagrees.
        ("two adjacent blocks", blocks(0, &[(1, 2, 2), (1, 2, 2)])),
        ("uneven adjacent blocks", blocks(0, &[(1, 2, 2), (1, 4, 4)])),
        ("text between blocks", {
            let mut case = blocks(2, &[(1, 2, 2), (1, 2, 2)]);
            case.1.insert(3, false);
            case
        }),
        // More than one frame in the target. The reference builds its grid with
        // `h * w` entries while `token_metadata` counts `f * h * w`, so this is
        // the shape it cannot serve; the point of the case is that this mirror
        // fails the same way.
        ("two frames", one_image(1, 2, 2, 2)),
        // A rectangle whose height and width differ in parity, so the two grid
        // ranges round differently.
        ("odd 3x5-ish", blocks(1, &[(1, 1, 12)])),
    ]
}

fn main() -> ExitCode {
    // The reference's index panic is an expected outcome here, not a crash; its
    // message would only be noise on the way to the comparison.
    std::panic::set_hook(Box::new(|_| {}));

    let mut failures = 0usize;
    for (name, (shapes, mask)) in cases() {
        // The joint mask is the slot mask expanded four-fold at image slots.
        let expanded: Vec<bool> = mask
            .iter()
            .flat_map(|&b| {
                let n = if b { 4 } else { 1 };
                std::iter::repeat(b).take(n)
            })
            .collect();

        failures += compare(
            name,
            "rope_indices",
            capture(|| cpu::policy::rope_indices(&shapes, &expanded)),
            capture(|| gpu::rope_indices(&shapes, &expanded)),
        );
        failures += compare(
            name,
            "token_metadata",
            capture(|| cpu::policy::token_metadata(&expanded, &shapes)),
            capture(|| gpu::token_metadata(&expanded, &shapes)),
        );

        // `attends` is derived from the metadata, so compare it over every pair
        // rather than over the table it is derived from.
        if let (Ok((ids, _)), Ok((ids_gpu, _))) = (
            cpu::policy::token_metadata(&expanded, &shapes),
            gpu::token_metadata(&expanded, &shapes),
        ) {
            for q in 0..expanded.len() {
                for kv in 0..expanded.len() {
                    if cpu::policy::attends(&ids, q, kv) != gpu::attends(&ids_gpu, q, kv) {
                        eprintln!("  attends {name}: ({q},{kv}) differs");
                        failures += 1;
                    }
                }
            }
        }
    }
    failures += check_timesteps();

    if failures > 0 {
        eprintln!("{failures} policy disagreement(s) between dit and dit_gpu");
        ExitCode::FAILURE
    } else {
        println!("every policy helper agrees with the CPU reference");
        ExitCode::SUCCESS
    }
}

fn compare<T: PartialEq + std::fmt::Debug>(
    case: &str,
    what: &str,
    want: Outcome<T>,
    got: Outcome<T>,
) -> usize {
    if want == got {
        println!("  {what:16} {case:24} ok ({})", describe(&got));
        0
    } else {
        eprintln!("  {what} {case}: differs\n    cpu {want:?}\n    gpu {got:?}");
        1
    }
}

fn describe<T>(o: &Outcome<T>) -> &'static str {
    match o {
        Outcome::Value(_) => "value",
        Outcome::Rejected => "rejected",
        Outcome::Panicked => "panic",
    }
}

fn check_timesteps() -> usize {
    let mut failures = 0;
    for t in [0.0f32, 0.001, 0.5, 0.7, 1.0, 3.25] {
        let mut want = vec![0f32; 256];
        let mut got = vec![0f32; 256];
        cpu::policy::temporal_timesteps(t, &mut want);
        gpu::temporal_timesteps(t, &mut got);
        if want != got {
            eprintln!("  temporal_timesteps({t}) differs");
            failures += 1;
        }
    }
    if failures == 0 {
        println!("  temporal_timesteps              ok");
    }
    failures
}
