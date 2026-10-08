//! A registry model that is not downloaded is fetched at start with the
//! `lumen` beside the server (`lumen pull <spec> --yes`), and anything else
//! keeps its refusal. A stand-in `lumen` records what it was asked, so nothing
//! is downloaded. Host-only; needs the `bin` feature.

use std::os::unix::fs::PermissionsExt;
use std::path::{Path, PathBuf};
use std::process::Command;
use std::sync::{Mutex, MutexGuard};

/// Held by each test from start to end: a test writes the executables it then
/// runs, and a sibling thread forking meanwhile would hold them open for
/// writing, which makes running them fail ("Text file busy").
static SERIAL: Mutex<()> = Mutex::new(());

fn serial() -> MutexGuard<'static, ()> {
    SERIAL.lock().unwrap_or_else(|p| p.into_inner())
}

/// A folder holding a copy of `lumen-server`, an empty cache and, with
/// `lumen`, a stand-in `lumen` running that shell script.
fn install(tag: &str, lumen: Option<&str>) -> PathBuf {
    let dir = std::env::temp_dir().join(format!("lumen-fetch-{tag}-{}", std::process::id()));
    let _ = std::fs::remove_dir_all(&dir);
    std::fs::create_dir_all(dir.join("cache")).unwrap();
    std::fs::copy(env!("CARGO_BIN_EXE_lumen-server"), dir.join("lumen-server")).unwrap();
    if let Some(script) = lumen {
        let path = dir.join("lumen");
        std::fs::write(&path, format!("#!/bin/sh\n{script}\n")).unwrap();
        std::fs::set_permissions(&path, std::fs::Permissions::from_mode(0o755)).unwrap();
    }
    dir
}

fn command(dir: &Path, args: &[&str]) -> Command {
    let mut command = Command::new(dir.join("lumen-server"));
    command
        .args(args)
        .env("LUMEN_CACHE_DIR", dir.join("cache"))
        .env("https_proxy", "ftp://127.0.0.1:9")
        .env("HTTPS_PROXY", "ftp://127.0.0.1:9");
    command
}

/// The server's exit code and standard error.
fn server(dir: &Path, args: &[&str]) -> (Option<i32>, String) {
    let out = command(dir, args).output().expect("run lumen-server");
    (
        out.status.code(),
        String::from_utf8_lossy(&out.stderr).into_owned(),
    )
}

/// What the stand-in `lumen` was asked, in the cache the server gave it.
fn asked(dir: &Path) -> Option<String> {
    std::fs::read_to_string(dir.join("cache/args"))
        .ok()
        .map(|s| s.trim().to_owned())
}

const RECORD: &str = r#"echo "$@" > "$LUMEN_CACHE_DIR/args""#;

#[test]
fn a_missing_model_is_fetched_with_the_lumen_beside_the_server() {
    let _serial = serial();
    let dir = install("fetch", Some(&format!("{RECORD}\nexit 3")));
    let (code, stderr) = server(
        &dir,
        &[
            "qwen3.8-27b:q8_0",
            "--quant",
            "q4_k_m",
            "--backend-device",
            "2",
        ],
    );
    let args = asked(&dir);
    let _ = std::fs::remove_dir_all(&dir);
    assert_ne!(code, Some(0), "{stderr}");
    assert!(
        stderr.contains("qwen3.8-27b:q4_k_m is not downloaded; fetching it with `lumen pull`"),
        "{stderr}"
    );
    assert!(
        stderr.contains("could not download qwen3.8-27b:q4_k_m"),
        "{stderr}"
    );
    assert_eq!(
        args.as_deref(),
        Some("pull qwen3.8-27b:q4_k_m --yes --cuda-device 2")
    );
}

#[test]
fn a_fetched_model_is_then_loaded() {
    let _serial = serial();
    let dir = install(
        "loaded",
        Some(&format!(
            "{RECORD}\necho x > \"$LUMEN_CACHE_DIR/qwen3-5-9b-Q8_0.lbc\""
        )),
    );
    let (code, stderr) = server(&dir, &["qwen3.5-9b"]);
    let args = asked(&dir);
    let lbc = dir.join("cache/qwen3-5-9b-Q8_0.lbc");
    let _ = std::fs::remove_dir_all(&dir);
    assert_eq!(args.as_deref(), Some("pull qwen3.5-9b --yes"));
    assert!(
        stderr.contains(&format!("model: {}", lbc.display())),
        "{stderr}"
    );
    assert_ne!(code, Some(0), "the stand-in file is no model: {stderr}");
}

#[test]
fn without_lumen_beside_it_the_server_says_how_to_download() {
    let _serial = serial();
    let dir = install("alone", None);
    let (code, stderr) = server(&dir, &["qwen3.5-9b"]);
    let _ = std::fs::remove_dir_all(&dir);
    assert_ne!(code, Some(0), "{stderr}");
    assert!(
        stderr.contains("there is no `lumen` beside this lumen-server"),
        "{stderr}"
    );
    assert!(
        stderr.contains("run `lumen pull qwen3.5-9b` first"),
        "{stderr}"
    );
}

#[test]
fn what_the_registry_cannot_supply_is_refused_without_a_fetch() {
    let _serial = serial();
    let dir = install("refused", Some(RECORD));
    let unknown = server(&dir, &["other-model"]);
    let no_such_quant = server(&dir, &["qwen3.8-27b:q9"]);
    let no_file = server(&dir, &["--model", "/nonexistent/x.lbc"]);
    let args = asked(&dir);
    let _ = std::fs::remove_dir_all(&dir);
    assert!(unknown.1.contains("model not cached"), "{}", unknown.1);
    assert!(
        no_such_quant
            .1
            .contains("Run `lumen pull qwen3.8-27b:q9` first"),
        "{}",
        no_such_quant.1
    );
    assert!(no_file.1.contains("model file not found"), "{}", no_file.1);
    for (code, stderr) in [&unknown, &no_such_quant, &no_file] {
        assert_ne!(*code, Some(0), "{stderr}");
    }
    assert_eq!(args, None, "nothing was fetched");
}

/// The server keeps its default signal handling while it downloads: a signal
/// ends it as the signal does, and the download (killed by the kernel with its
/// parent) goes with it.
#[cfg(target_os = "linux")]
#[test]
fn a_signal_stops_the_server_and_its_download() {
    use std::os::unix::process::ExitStatusExt;
    let _serial = serial();
    for (signal, number) in [("TERM", 15), ("KILL", 9)] {
        let dir = install(
            "stop",
            Some("echo $$ > \"$LUMEN_CACHE_DIR/pid\"\nexec sleep 60"),
        );
        let mut server = command(&dir, &["qwen3.5-9b"])
            .spawn()
            .expect("run lumen-server");
        let pid_file = dir.join("cache/pid");
        let mut waited = 0;
        while !pid_file.exists() && waited < 200 {
            std::thread::sleep(std::time::Duration::from_millis(50));
            waited += 1;
        }
        let child = std::fs::read_to_string(&pid_file)
            .expect("the download started")
            .trim()
            .to_owned();
        let sent = Command::new("kill")
            .args([format!("-{signal}"), server.id().to_string()])
            .status()
            .unwrap();
        assert!(sent.success());
        let status = server.wait().unwrap();
        let mut alive = true;
        for _ in 0..40 {
            alive = Command::new("kill")
                .args(["-0", &child])
                .stderr(std::process::Stdio::null())
                .status()
                .unwrap()
                .success();
            if !alive {
                break;
            }
            std::thread::sleep(std::time::Duration::from_millis(50));
        }
        let _ = std::fs::remove_dir_all(&dir);
        assert_eq!(status.signal(), Some(number), "SIG{signal} ends the server");
        assert!(
            !alive,
            "the download ({child}) outlived the server on SIG{signal}"
        );
    }
}

#[cfg(feature = "image")]
#[test]
fn the_image_model_is_fetched_by_its_name_until_every_file_it_serves_is_cached() {
    let _serial = serial();
    let dir = install("image", Some(&format!("{RECORD}\nexit 3")));
    let lbi = dir.join("cache/qwen-image-2-1/lbi");
    std::fs::create_dir_all(&lbi).unwrap();
    std::fs::write(lbi.join("transformer.lbi"), b"x").unwrap();
    let (code, stderr) = server(&dir, &["qwen-image"]);
    let args = asked(&dir);
    let _ = std::fs::remove_dir_all(&dir);
    assert_ne!(code, Some(0), "{stderr}");
    assert!(stderr.contains("could not download qwen-image"), "{stderr}");
    assert_eq!(args.as_deref(), Some("pull qwen-image --yes"));
}
