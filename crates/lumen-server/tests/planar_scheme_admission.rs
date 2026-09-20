//! The server refuses an artifact whose weights are a scheme with no serving
//! kernels, by name, before the weight provider opens — and starts its
//! admission unchanged for an artifact whose schemes do serve.
//!
//! Host-only, no GPU: the refusal is decided from the header and index.

use std::process::Command;

mod fixture;

fn workdir(tag: &str) -> std::path::PathBuf {
    let dir = std::env::temp_dir().join(format!(
        "lumen-admission-server-{tag}-{}",
        std::process::id()
    ));
    let _ = std::fs::remove_dir_all(&dir);
    std::fs::create_dir_all(&dir).unwrap();
    dir
}

#[test]
fn an_nvfp4_artifact_is_refused_by_name_at_startup() {
    assert!(
        cfg!(feature = "bin"),
        "this test spawns the server binary: run with --features lumen-server/bin"
    );
    let dir = workdir("nvfp4");
    let artifact = fixture::write_nvfp4_artifact(&dir);

    // The rule's own answer for this artifact, before the binary is asked.
    let lbc = lumen_format::reader::LbcFile::open(&artifact).unwrap();
    assert_eq!(
        lumen_format::serving_rules::unservable_scheme(&lbc),
        Some(lumen_format::QuantScheme::Nvfp4)
    );

    let out = Command::new(env!("CARGO_BIN_EXE_lumen-server"))
        .args([
            "--model",
            artifact.to_str().unwrap(),
            "--backend",
            "cpu",
            "--port",
            "0",
        ])
        .output()
        .expect("run lumen-server");
    let stderr = String::from_utf8_lossy(&out.stderr);
    assert_ne!(out.status.code(), Some(0), "the server started: {stderr}");
    assert!(
        stderr.contains("Nvfp4") && stderr.contains("no serving kernels for this scheme yet"),
        "the refusal does not name the scheme:\n{stderr}"
    );
    // It stopped at admission: the provider never opened, so the server
    // never bound a port.
    assert!(
        !stderr.contains("listening"),
        "the server got past admission:\n{stderr}"
    );
}

#[test]
fn an_existing_scheme_still_passes_admission() {
    let dir = workdir("q4");
    let artifact = fixture::write_q4_0_artifact(&dir);
    let lbc = lumen_format::reader::LbcFile::open(&artifact).unwrap();
    assert_eq!(lumen_format::serving_rules::unservable_scheme(&lbc), None);

    let out = Command::new(env!("CARGO_BIN_EXE_lumen-server"))
        .args([
            "--model",
            artifact.to_str().unwrap(),
            "--backend",
            "cpu",
            "--port",
            "0",
        ])
        .output()
        .expect("run lumen-server");
    let stderr = String::from_utf8_lossy(&out.stderr);
    assert!(
        !stderr.contains("no serving kernels for this scheme yet"),
        "a Q4_0 artifact was refused by the planar rule:\n{stderr}"
    );
}
