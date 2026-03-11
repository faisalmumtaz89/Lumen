//! Output formatting for benchmark results.

use crate::results::BenchSummary;

/// Print results as a human-readable table.
pub fn print_table(summaries: &[BenchSummary]) {
    println!("{:<40} {:>10} {:>10} {:>10} {:>8} {:>10} {:>10} {:>10}",
        "Config", "TPOT(ms)", "p95(ms)", "stddev", "CV%", "BW(GiB/s)", "Stall%", "Time(ms)");
    println!("{}", "-".repeat(128));

    for s in summaries {
        let cache_hit = if s.results.is_empty() {
            0.0
        } else {
            s.results.iter().map(|r| r.weight_cache_hit_rate).sum::<f64>()
                / s.results.len() as f64 * 100.0
        };
        let _ = cache_hit; // available for verbose mode

        println!("{:<40} {:>10.2} {:>10.2} {:>10.2} {:>7.1}% {:>10.3} {:>10.1} {:>10.1}",
            s.label,
            s.median_tpot_ms(),
            s.p95_tpot_ms(),
            s.std_dev_tpot_ms(),
            s.cv_tpot_percent(),
            s.mean_bandwidth_gibs(),
            s.mean_stall_fraction() * 100.0,
            s.mean_total_time().as_secs_f64() * 1000.0,
        );
    }
}

/// Print results as JSON (hand-rolled, zero deps).
pub fn print_json(summaries: &[BenchSummary]) {
    println!("{{");
    println!("  \"lumen_version\": \"{}\",", env!("CARGO_PKG_VERSION"));
    println!("  \"os\": \"{}\",", escape_json(std::env::consts::OS));
    println!("  \"arch\": \"{}\",", escape_json(std::env::consts::ARCH));
    println!("  \"benchmarks\": [");
    for (i, s) in summaries.iter().enumerate() {
        println!("    {{");
        println!("      \"label\": \"{}\",", escape_json(&s.label));
        println!("      \"iterations\": {},", s.iterations);
        println!("      \"median_tpot_ms\": {:.2},", s.median_tpot_ms());
        println!("      \"mean_tpot_ms\": {:.2},", s.mean_tpot_ms());
        println!("      \"p95_tpot_ms\": {:.2},", s.p95_tpot_ms());
        println!("      \"std_dev_tpot_ms\": {:.4},", s.std_dev_tpot_ms());
        println!("      \"cv_tpot_percent\": {:.2},", s.cv_tpot_percent());
        println!("      \"mean_bandwidth_gibs\": {:.4},", s.mean_bandwidth_gibs());
        println!("      \"mean_stall_fraction\": {:.4},", s.mean_stall_fraction());
        println!("      \"mean_total_time_ms\": {:.2},", s.mean_total_time().as_secs_f64() * 1000.0);
        println!("      \"results\": [");
        for (j, r) in s.results.iter().enumerate() {
            println!("        {{");
            println!("          \"total_time_ms\": {:.2},", r.total_time.as_secs_f64() * 1000.0);
            println!("          \"prefill_time_ms\": {:.2},", r.prefill_time.as_secs_f64() * 1000.0);
            println!("          \"decode_time_ms\": {:.2},", r.decode_time.as_secs_f64() * 1000.0);
            println!("          \"tpot_ms\": {:.2},", r.tpot_ms);
            println!("          \"bytes_read\": {},", r.bytes_read);
            println!("          \"read_ops\": {},", r.read_ops);
            println!("          \"bandwidth_gibs\": {:.4},", r.bandwidth_gibs);
            println!("          \"weight_cache_hit_rate\": {:.4},", r.weight_cache_hit_rate);
            println!("          \"initial_residency\": {:.4},", r.initial_residency);
            println!("          \"final_residency\": {:.4},", r.final_residency);
            println!("          \"stall_fraction\": {:.4},", r.stall_fraction);
            println!("          \"prompt_tokens\": {},", r.prompt_tokens);
            println!("          \"generated_tokens\": {}", r.generated_tokens);
            if j + 1 < s.results.len() {
                println!("        }},");
            } else {
                println!("        }}");
            }
        }
        println!("      ]");
        if i + 1 < summaries.len() {
            println!("    }},");
        } else {
            println!("    }}");
        }
    }
    println!("  ]");
    println!("}}");
}

fn escape_json(s: &str) -> String {
    let mut result = String::with_capacity(s.len());
    for c in s.chars() {
        match c {
            '\\' => result.push_str("\\\\"),
            '"' => result.push_str("\\\""),
            '\n' => result.push_str("\\n"),
            '\r' => result.push_str("\\r"),
            '\t' => result.push_str("\\t"),
            '\x08' => result.push_str("\\b"),
            '\x0C' => result.push_str("\\f"),
            c if c < '\x20' => {
                // Control characters must be escaped as \u00XX per RFC 8259.
                result.push_str(&format!("\\u{:04x}", c as u32));
            }
            _ => result.push(c),
        }
    }
    result
}
