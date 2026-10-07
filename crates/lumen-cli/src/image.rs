//! `lumen image`: one picture from the registry's image model, no server.

use std::path::{Path, PathBuf};

/// What `lumen image` was asked for.
#[derive(Debug, PartialEq)]
pub(crate) struct ImageArgs {
    pub prompt: String,
    /// The file to write, and whether it was named with `-o` (which
    /// overwrites) or defaulted to `image-<seed>.png` (which must be new).
    pub output: PathBuf,
    pub named: bool,
    pub width: usize,
    pub height: usize,
    pub steps: usize,
    pub seed: u64,
}

/// Parse `lumen image [OPTIONS] [--] "<prompt>"`: `-o <file>`, `--size WxH`,
/// `--steps N`, `--seed N`. Every token after `--` is the prompt, so a prompt
/// may begin with `-`. An option's value may not look like an option. `seed`
/// is the seed used when `--seed` is absent.
pub(crate) fn parse_image_args(args: &[String], seed: u64) -> Result<ImageArgs, String> {
    let mut prompt: Option<String> = None;
    let mut output: Option<PathBuf> = None;
    let mut size = "1024x1024".to_string();
    let mut steps = 40usize;
    let mut seed = seed;
    let mut options_ended = false;
    let mut i = 0;
    while i < args.len() {
        let value = |i: usize| -> Result<&str, String> {
            match args.get(i + 1).map(String::as_str) {
                Some(v) if !v.starts_with('-') => Ok(v),
                _ => Err(format!("{} requires a value", args[i])),
            }
        };
        let token = args[i].as_str();
        if options_ended || !token.starts_with('-') {
            if prompt.is_some() {
                return Err(format!("unexpected argument: {token}"));
            }
            prompt = Some(token.to_string());
            i += 1;
            continue;
        }
        match token {
            "--" => options_ended = true,
            "-o" | "--output" => {
                output = Some(PathBuf::from(value(i)?));
                i += 1;
            }
            "--size" => {
                size = value(i)?.to_string();
                i += 1;
            }
            "--steps" => {
                let text = value(i)?;
                steps = text
                    .parse()
                    .map_err(|_| format!("--steps {text:?} is not a number"))?;
                i += 1;
            }
            "--seed" => {
                let text = value(i)?;
                seed = text
                    .parse()
                    .map_err(|_| format!("--seed {text:?} is not a number"))?;
                i += 1;
            }
            "-h" | "--help" => return Err(String::new()),
            other => return Err(format!("unknown option: {other}")),
        }
        i += 1;
    }
    let prompt = prompt.ok_or_else(|| "a prompt is required".to_string())?;
    if prompt.trim().is_empty() {
        return Err("the prompt is empty".to_string());
    }
    let (width, height) = lumen_image::pipeline::parse_size(&size)?;
    lumen_image::pipeline::check_steps(steps)?;
    let named = output.is_some();
    Ok(ImageArgs {
        prompt,
        output: output.unwrap_or_else(|| PathBuf::from(format!("image-{seed}.png"))),
        named,
        width,
        height,
        steps,
        seed,
    })
}

/// The output file. Prepared before the generation, so a destination that
/// cannot be written is refused before any model loads. The picture is
/// written into a staging file this run creates exclusively beside the
/// destination (`.<name>.<pid>.<nanos>.part`), then published: renamed onto a
/// path named with `-o`, which replaces it whole, or linked to a defaulted
/// name, which fails rather than overwrite a file that appeared meanwhile (on
/// a file system without hard links, created anew instead, which also never
/// overwrites). Nothing is created at the destination before the picture is
/// complete. A run that stops before it has a picture removes only its own
/// staging file; once it has one, a failure to publish keeps the staging file
/// and names it, so a finished picture is never lost.
#[derive(Debug)]
pub(crate) struct Output {
    path: PathBuf,
    staging: PathBuf,
    file: Option<std::fs::File>,
    named: bool,
    keep: bool,
}

impl Output {
    pub(crate) fn prepare(path: &Path, named: bool) -> Result<Self, String> {
        let name = path
            .file_name()
            .ok_or_else(|| format!("{:?} does not name a file", path.display().to_string()))?;
        // Not following a link: a link at a defaulted name, dangling or not,
        // is a file already there.
        match std::fs::symlink_metadata(path) {
            Ok(m) if m.is_dir() => {
                return Err(format!("{} is a directory; name a file", path.display()))
            }
            // A device, pipe or socket, or a link to one, would be replaced,
            // not written to.
            Ok(_) if std::fs::metadata(path).is_ok_and(|m| !m.is_file()) => {
                return Err(format!(
                    "{} is not a regular file; name a file",
                    path.display()
                ))
            }
            Ok(_) if !named => {
                return Err(format!(
                    "{} already exists; name the file with -o",
                    path.display()
                ))
            }
            _ => {}
        }
        let dir = match path.parent() {
            Some(parent) if !parent.as_os_str().is_empty() => parent.to_path_buf(),
            _ => PathBuf::from("."),
        };
        let nanos = std::time::SystemTime::now()
            .duration_since(std::time::UNIX_EPOCH)
            .map_or(0, |d| d.as_nanos());
        let mut staged = std::ffi::OsString::from(".");
        staged.push(name);
        staged.push(format!(".{}.{nanos}.part", std::process::id()));
        let staging = dir.join(staged);
        let file = std::fs::OpenOptions::new()
            .write(true)
            .create_new(true)
            .open(&staging)
            .map_err(|e| format!("cannot create {}: {e}", staging.display()))?;
        Ok(Self {
            path: path.to_path_buf(),
            staging,
            file: Some(file),
            named,
            keep: false,
        })
    }

    #[cfg_attr(not(feature = "cuda"), allow(dead_code))]
    pub(crate) fn finish(mut self, bytes: &[u8]) -> Result<(), String> {
        use std::io::Write;
        let mut file = self.file.take().expect("an output is finished once");
        file.write_all(bytes)
            .map_err(|e| format!("cannot write {}: {e}", self.staging.display()))?;
        drop(file);
        // From here the staging file holds the finished picture.
        self.keep = true;
        let kept =
            |why: String| format!("{why}; the picture is kept in {}", self.staging.display());
        if self.named {
            return std::fs::rename(&self.staging, &self.path)
                .map_err(|e| kept(format!("cannot replace {}: {e}", self.path.display())));
        }
        match std::fs::hard_link(&self.staging, &self.path) {
            Ok(()) => {}
            Err(e) if e.kind() == std::io::ErrorKind::AlreadyExists => {
                return Err(kept(format!(
                    "{} appeared while the picture was made",
                    self.path.display()
                )))
            }
            // A file system without hard links: create the name anew, which
            // also refuses a file that is already there.
            Err(_) => write_new(&self.path, bytes).map_err(kept)?,
        }
        let _ = std::fs::remove_file(&self.staging);
        Ok(())
    }
}

/// Write `bytes` to a file at `path` that must not exist yet; a partly
/// written file is removed.
fn write_new(path: &Path, bytes: &[u8]) -> Result<(), String> {
    use std::io::Write;
    let mut file = std::fs::OpenOptions::new()
        .write(true)
        .create_new(true)
        .open(path)
        .map_err(|e| format!("cannot create {}: {e}", path.display()))?;
    file.write_all(bytes).map_err(|e| {
        let _ = std::fs::remove_file(path);
        format!("cannot write {}: {e}", path.display())
    })
}

impl Drop for Output {
    fn drop(&mut self) {
        if !self.keep {
            let _ = std::fs::remove_file(&self.staging);
        }
    }
}

/// Make the picture. Exits on failure.
pub(crate) fn image_cmd(args: &[String]) {
    let seed = std::time::SystemTime::now()
        .duration_since(std::time::UNIX_EPOCH)
        .map(|d| d.as_nanos() as u64)
        .unwrap_or(42);
    let args = match parse_image_args(args, seed) {
        Ok(args) => args,
        Err(e) if e.is_empty() => {
            crate::help::print_image_usage();
            return;
        }
        Err(e) => {
            eprintln!("Error: {e}");
            crate::help::print_image_usage();
            std::process::exit(1);
        }
    };
    let reg = crate::registry::load_registry();
    let entry = reg
        .list()
        .into_iter()
        .find(|e| e.checkpoint.is_some())
        .expect("the registry has the image model");
    if !cfg!(feature = "cuda") {
        eprintln!(
            "{} makes images on NVIDIA CUDA, and this lumen was built without CUDA.",
            entry.display_name
        );
        std::process::exit(1);
    }
    // A machine that cannot run the model is refused before a first run
    // downloads it.
    #[cfg(feature = "cuda")]
    if let Err(e) = lumen_image::pipeline::check_device() {
        eprintln!("Error: {e}");
        std::process::exit(1);
    }
    let output = Output::prepare(&args.output, args.named).unwrap_or_else(|e| {
        eprintln!("Error: {e}");
        std::process::exit(1);
    });
    // Downloaded and converted on first use, as `lumen run` fetches a model.
    // A complete cache is read as it is: its files are published whole, so
    // no lock is needed to read them, and the cache may be read-only.
    if !crate::cache::cached_image(&entry.key) {
        let checkpoint = entry
            .checkpoint
            .as_ref()
            .expect("the image model has a checkpoint");
        let dir = crate::cache::image_checkpoint_dir(&entry.key);
        if let Err(e) = crate::fetch_image(&dir, checkpoint, &entry.display_name, None) {
            drop(output);
            eprintln!("Error: {e}");
            std::process::exit(1);
        }
    }
    generate(entry, &args, output).unwrap_or_else(|e| {
        eprintln!("Error: {e}");
        std::process::exit(1);
    });
}

#[cfg(feature = "cuda")]
fn generate(
    entry: &crate::registry::ModelEntry,
    args: &ImageArgs,
    output: Output,
) -> Result<(), String> {
    use lumen_image::pipeline::{generate_gpu, GenerationRequest, GpuSources, PipelinePaths};

    let paths = PipelinePaths::from_roots(
        &crate::cache::image_lbi_dir(&entry.key),
        &crate::cache::image_checkpoint_dir(&entry.key),
    );
    paths.check(true).map_err(|e| e.to_string())?;
    let sources = GpuSources::open(&paths).map_err(|e| e.to_string())?;
    let request = GenerationRequest {
        prompt: &args.prompt,
        height: args.height,
        width: args.width,
        steps: args.steps,
        seed: args.seed,
        init_latents: None,
    };
    let started = std::time::Instant::now();
    let mut progress = |done: usize, total: usize| {
        if done > 0 && (done == total || done % 5 == 0) {
            eprintln!("  step {done}/{total}");
        }
        std::ops::ControlFlow::Continue(())
    };
    let image = generate_gpu(&sources, &request, &mut progress).map_err(|e| e.to_string())?;
    output.finish(&lumen_image::png::encode(&image))?;
    println!("Saved {}", args.output.display());
    println!(
        "{}x{} · seed {} · {} steps · {:.1} s",
        image.width,
        image.height,
        args.seed,
        args.steps,
        started.elapsed().as_secs_f32()
    );
    Ok(())
}

#[cfg(not(feature = "cuda"))]
fn generate(
    _entry: &crate::registry::ModelEntry,
    _args: &ImageArgs,
    _output: Output,
) -> Result<(), String> {
    unreachable!("refused before reaching the generation")
}

#[cfg(test)]
mod tests {
    use super::*;

    fn parse(words: &[&str]) -> Result<ImageArgs, String> {
        let args: Vec<String> = words.iter().map(|w| w.to_string()).collect();
        parse_image_args(&args, 7)
    }

    fn scratch(tag: &str) -> PathBuf {
        let dir = std::env::temp_dir().join(format!("lumen-image-{tag}-{}", std::process::id()));
        let _ = std::fs::remove_dir_all(&dir);
        std::fs::create_dir_all(&dir).unwrap();
        dir
    }

    #[test]
    fn defaults_name_the_file_by_the_seed() {
        let args = parse(&["A red apple"]).unwrap();
        assert_eq!(
            args,
            ImageArgs {
                prompt: "A red apple".to_string(),
                output: PathBuf::from("image-7.png"),
                named: false,
                width: 1024,
                height: 1024,
                steps: 40,
                seed: 7,
            }
        );
    }

    #[test]
    fn every_option_is_taken_in_any_order() {
        let args = parse(&[
            "--seed",
            "42",
            "--size",
            "1536x1024",
            "A red apple",
            "--steps",
            "20",
            "-o",
            "apple.png",
        ])
        .unwrap();
        assert_eq!(args.prompt, "A red apple");
        assert_eq!(args.output, PathBuf::from("apple.png"));
        assert!(args.named);
        assert_eq!((args.width, args.height), (1536, 1024));
        assert_eq!(args.steps, 20);
        assert_eq!(args.seed, 42);
    }

    #[test]
    fn a_double_dash_ends_the_options_and_values_may_not_look_like_options() {
        assert_eq!(
            parse(&["--", "--retro poster"]).unwrap().prompt,
            "--retro poster"
        );
        assert_eq!(parse(&["--", "--help"]).unwrap().prompt, "--help");
        assert_eq!(
            parse(&["-o", "a.png", "--", "-5 degrees"]).unwrap().prompt,
            "-5 degrees"
        );
        assert!(parse(&["cat", "-o", "--help"])
            .unwrap_err()
            .contains("-o requires a value"));
        assert!(parse(&["cat", "-o", "--size", "--steps", "1"])
            .unwrap_err()
            .contains("-o requires a value"));
        assert!(parse(&["cat", "--seed", "-1"])
            .unwrap_err()
            .contains("--seed requires a value"));
        assert!(parse(&["-x", "cat"])
            .unwrap_err()
            .contains("unknown option: -x"));
    }

    #[test]
    fn out_of_range_and_malformed_requests_are_refused() {
        assert!(parse(&[]).unwrap_err().contains("prompt is required"));
        assert!(parse(&["   "]).unwrap_err().contains("empty"));
        assert!(parse(&["A", "B"]).unwrap_err().contains("unexpected"));
        assert!(parse(&["A", "--size", "31x1024"])
            .unwrap_err()
            .contains("minimum"));
        assert!(parse(&["A", "--size", "4097x100"])
            .unwrap_err()
            .contains("maximum"));
        assert!(parse(&["A", "--size", "big"])
            .unwrap_err()
            .contains("not WxH"));
        assert!(parse(&["A", "--steps", "0"])
            .unwrap_err()
            .contains("at least 1"));
        assert!(parse(&["A", "--steps", "201"])
            .unwrap_err()
            .contains("maximum"));
        assert!(parse(&["A", "--steps", "x"])
            .unwrap_err()
            .contains("not a number"));
        assert!(parse(&["A", "--seed"])
            .unwrap_err()
            .contains("requires a value"));
        assert!(parse(&["A", "--fast"])
            .unwrap_err()
            .contains("unknown option"));
        assert_eq!(parse(&["--help"]).unwrap_err(), "");
    }

    fn names(dir: &Path) -> Vec<String> {
        let mut names: Vec<String> = std::fs::read_dir(dir)
            .unwrap()
            .map(|e| e.unwrap().file_name().to_string_lossy().into_owned())
            .collect();
        names.sort();
        names
    }

    #[test]
    fn a_link_at_the_default_name_is_a_file_already_there_even_when_dangling() {
        let dir = scratch("dangling");
        let path = dir.join("image-7.png");
        std::os::unix::fs::symlink(dir.join("nowhere"), &path).unwrap();
        let err = Output::prepare(&path, false).unwrap_err();
        assert!(err.contains("already exists"), "{err}");
        assert!(
            !dir.join("nowhere").exists(),
            "nothing written through the link"
        );
        std::fs::remove_dir_all(&dir).ok();
    }

    #[test]
    fn writing_a_new_file_never_overwrites() {
        let dir = scratch("write-new");
        let path = dir.join("p.png");
        write_new(&path, b"picture").unwrap();
        assert_eq!(std::fs::read(&path).unwrap(), b"picture");
        assert!(write_new(&path, b"other")
            .unwrap_err()
            .contains("cannot create"));
        assert_eq!(std::fs::read(&path).unwrap(), b"picture");
        std::fs::remove_dir_all(&dir).ok();
    }

    #[test]
    fn a_destination_that_cannot_be_written_is_refused_before_anything_runs() {
        let dir = scratch("refuse");
        let err = Output::prepare(&dir.join("missing/x.png"), true).unwrap_err();
        assert!(err.contains("cannot create"), "{err}");
        let err = Output::prepare(&dir, true).unwrap_err();
        assert!(err.contains("is a directory"), "{err}");
        let socket = dir.join("socket");
        let _listener = std::os::unix::net::UnixListener::bind(&socket).unwrap();
        let err = Output::prepare(&socket, true).unwrap_err();
        assert!(err.contains("is not a regular file"), "{err}");
        let link = dir.join("link");
        std::os::unix::fs::symlink(&socket, &link).unwrap();
        let err = Output::prepare(&link, true).unwrap_err();
        assert!(err.contains("is not a regular file"), "{err}");
        assert!(Output::prepare(Path::new(""), true)
            .unwrap_err()
            .contains("does not name a file"));
        let taken = dir.join("image-7.png");
        std::fs::write(&taken, b"earlier").unwrap();
        let err = Output::prepare(&taken, false).unwrap_err();
        assert!(err.contains("already exists"), "{err}");
        assert_eq!(std::fs::read(&taken).unwrap(), b"earlier");
        assert_eq!(
            names(&dir),
            ["image-7.png", "link", "socket"],
            "nothing else created"
        );
        std::fs::remove_dir_all(&dir).ok();
    }

    #[test]
    fn nothing_appears_at_the_destination_until_the_picture_is_complete() {
        let dir = scratch("staging");
        let path = dir.join("image-8.png");
        let output = Output::prepare(&path, false).unwrap();
        assert!(!path.exists(), "no placeholder before the picture exists");
        drop(output);
        assert!(names(&dir).is_empty(), "a run that stops leaves nothing");
        let output = Output::prepare(&path, false).unwrap();
        output.finish(b"picture").unwrap();
        assert_eq!(std::fs::read(&path).unwrap(), b"picture");
        assert_eq!(names(&dir), ["image-8.png"], "the staging file is gone");
        std::fs::remove_dir_all(&dir).ok();
    }

    #[test]
    fn a_default_name_taken_meanwhile_is_left_alone_and_a_named_path_is_replaced_whole() {
        let dir = scratch("publish");
        let path = dir.join("image-9.png");
        let output = Output::prepare(&path, false).unwrap();
        let staging = output.staging.clone();
        std::fs::write(&path, b"another run's picture").unwrap();
        let err = output.finish(b"mine").unwrap_err();
        assert!(err.contains("appeared") && err.contains("kept in"), "{err}");
        assert_eq!(std::fs::read(&path).unwrap(), b"another run's picture");
        assert_eq!(
            std::fs::read(&staging).unwrap(),
            b"mine",
            "the picture survives"
        );
        std::fs::remove_file(&staging).unwrap();
        assert_eq!(names(&dir), ["image-9.png"]);

        let named = dir.join("apple.png");
        std::fs::write(&named, b"previous").unwrap();
        let first = Output::prepare(&named, true).unwrap();
        let second = Output::prepare(&named, true).unwrap();
        assert_ne!(
            first.staging, second.staging,
            "each run stages in its own file"
        );
        assert_eq!(std::fs::read(&named).unwrap(), b"previous");
        first.finish(b"first picture").unwrap();
        second.finish(b"second picture").unwrap();
        assert_eq!(std::fs::read(&named).unwrap(), b"second picture");
        assert_eq!(names(&dir), ["apple.png", "image-9.png"]);
        std::fs::remove_dir_all(&dir).ok();
    }
}
