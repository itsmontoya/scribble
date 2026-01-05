// A small CLI utility to download known Whisper (ASR) and Whisper-VAD models
// into a target directory.

use anyhow::{Context, Result};
use clap::Parser;
use indicatif::{ProgressBar, ProgressStyle};
use reqwest::blocking::Client;
use std::fs;
use std::io::{Read, Write};
use std::path::{Path, PathBuf};

#[derive(Parser, Debug)]
#[command(name = "model-downloader")]
#[command(about = "Download Whisper and VAD models for Scribble", long_about = None)]
struct Args {
    /// List supported model names and exit.
    #[arg(long)]
    list: bool,

    /// Model name (examples: tiny, base.en, large-v3-turbo, silero-v6.2.0)
    ///
    /// We intentionally keep an allowlist of known-good model artifacts.
    #[arg(long, required_unless_present = "list")]
    name: Option<String>,

    /// Target directory to store models (created if missing).
    #[arg(long, default_value = "./models")]
    dir: PathBuf,
}

#[derive(Debug, Clone, Copy, PartialEq, Eq)]
enum ModelKind {
    Whisper,
    Vad,
}

/// Download source for a known model.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
struct ModelSpec {
    /// Whether this is a Whisper ASR model or a VAD model.
    kind: ModelKind,

    /// Friendly name users type (e.g. "large-v3-turbo").
    name: &'static str,

    /// Filename written to disk (e.g. "ggml-large-v3-turbo.bin").
    filename: &'static str,

    /// Full download URL.
    url: &'static str,
}

// -----------------------------------------------------------------------------
// Whisper models (allowlist)
//
// These URLs match whisper.cpp’s standard Hugging Face repo for GGML models.
// -----------------------------------------------------------------------------
static WHISPER_MODELS: &[ModelSpec] = &[
    // tiny
    ModelSpec {
        kind: ModelKind::Whisper,
        name: "tiny",
        filename: "ggml-tiny.bin",
        url: "https://huggingface.co/ggerganov/whisper.cpp/resolve/main/ggml-tiny.bin",
    },
    ModelSpec {
        kind: ModelKind::Whisper,
        name: "tiny.en",
        filename: "ggml-tiny.en.bin",
        url: "https://huggingface.co/ggerganov/whisper.cpp/resolve/main/ggml-tiny.en.bin",
    },
    ModelSpec {
        kind: ModelKind::Whisper,
        name: "tiny-q5_1",
        filename: "ggml-tiny-q5_1.bin",
        url: "https://huggingface.co/ggerganov/whisper.cpp/resolve/main/ggml-tiny-q5_1.bin",
    },
    ModelSpec {
        kind: ModelKind::Whisper,
        name: "tiny.en-q5_1",
        filename: "ggml-tiny.en-q5_1.bin",
        url: "https://huggingface.co/ggerganov/whisper.cpp/resolve/main/ggml-tiny.en-q5_1.bin",
    },
    ModelSpec {
        kind: ModelKind::Whisper,
        name: "tiny-q8_0",
        filename: "ggml-tiny-q8_0.bin",
        url: "https://huggingface.co/ggerganov/whisper.cpp/resolve/main/ggml-tiny-q8_0.bin",
    },
    // base
    ModelSpec {
        kind: ModelKind::Whisper,
        name: "base",
        filename: "ggml-base.bin",
        url: "https://huggingface.co/ggerganov/whisper.cpp/resolve/main/ggml-base.bin",
    },
    ModelSpec {
        kind: ModelKind::Whisper,
        name: "base.en",
        filename: "ggml-base.en.bin",
        url: "https://huggingface.co/ggerganov/whisper.cpp/resolve/main/ggml-base.en.bin",
    },
    ModelSpec {
        kind: ModelKind::Whisper,
        name: "base-q5_1",
        filename: "ggml-base-q5_1.bin",
        url: "https://huggingface.co/ggerganov/whisper.cpp/resolve/main/ggml-base-q5_1.bin",
    },
    ModelSpec {
        kind: ModelKind::Whisper,
        name: "base.en-q5_1",
        filename: "ggml-base.en-q5_1.bin",
        url: "https://huggingface.co/ggerganov/whisper.cpp/resolve/main/ggml-base.en-q5_1.bin",
    },
    ModelSpec {
        kind: ModelKind::Whisper,
        name: "base-q8_0",
        filename: "ggml-base-q8_0.bin",
        url: "https://huggingface.co/ggerganov/whisper.cpp/resolve/main/ggml-base-q8_0.bin",
    },
    // small
    ModelSpec {
        kind: ModelKind::Whisper,
        name: "small",
        filename: "ggml-small.bin",
        url: "https://huggingface.co/ggerganov/whisper.cpp/resolve/main/ggml-small.bin",
    },
    ModelSpec {
        kind: ModelKind::Whisper,
        name: "small.en",
        filename: "ggml-small.en.bin",
        url: "https://huggingface.co/ggerganov/whisper.cpp/resolve/main/ggml-small.en.bin",
    },
    ModelSpec {
        kind: ModelKind::Whisper,
        name: "small-q5_1",
        filename: "ggml-small-q5_1.bin",
        url: "https://huggingface.co/ggerganov/whisper.cpp/resolve/main/ggml-small-q5_1.bin",
    },
    ModelSpec {
        kind: ModelKind::Whisper,
        name: "small.en-q5_1",
        filename: "ggml-small.en-q5_1.bin",
        url: "https://huggingface.co/ggerganov/whisper.cpp/resolve/main/ggml-small.en-q5_1.bin",
    },
    ModelSpec {
        kind: ModelKind::Whisper,
        name: "small-q8_0",
        filename: "ggml-small-q8_0.bin",
        url: "https://huggingface.co/ggerganov/whisper.cpp/resolve/main/ggml-small-q8_0.bin",
    },
    // medium
    ModelSpec {
        kind: ModelKind::Whisper,
        name: "medium",
        filename: "ggml-medium.bin",
        url: "https://huggingface.co/ggerganov/whisper.cpp/resolve/main/ggml-medium.bin",
    },
    ModelSpec {
        kind: ModelKind::Whisper,
        name: "medium.en",
        filename: "ggml-medium.en.bin",
        url: "https://huggingface.co/ggerganov/whisper.cpp/resolve/main/ggml-medium.en.bin",
    },
    ModelSpec {
        kind: ModelKind::Whisper,
        name: "medium-q5_0",
        filename: "ggml-medium-q5_0.bin",
        url: "https://huggingface.co/ggerganov/whisper.cpp/resolve/main/ggml-medium-q5_0.bin",
    },
    ModelSpec {
        kind: ModelKind::Whisper,
        name: "medium.en-q5_0",
        filename: "ggml-medium.en-q5_0.bin",
        url: "https://huggingface.co/ggerganov/whisper.cpp/resolve/main/ggml-medium.en-q5_0.bin",
    },
    ModelSpec {
        kind: ModelKind::Whisper,
        name: "medium-q8_0",
        filename: "ggml-medium-q8_0.bin",
        url: "https://huggingface.co/ggerganov/whisper.cpp/resolve/main/ggml-medium-q8_0.bin",
    },
    // large
    ModelSpec {
        kind: ModelKind::Whisper,
        name: "large-v1",
        filename: "ggml-large-v1.bin",
        url: "https://huggingface.co/ggerganov/whisper.cpp/resolve/main/ggml-large-v1.bin",
    },
    ModelSpec {
        kind: ModelKind::Whisper,
        name: "large-v2",
        filename: "ggml-large-v2.bin",
        url: "https://huggingface.co/ggerganov/whisper.cpp/resolve/main/ggml-large-v2.bin",
    },
    ModelSpec {
        kind: ModelKind::Whisper,
        name: "large-v2-q5_0",
        filename: "ggml-large-v2-q5_0.bin",
        url: "https://huggingface.co/ggerganov/whisper.cpp/resolve/main/ggml-large-v2-q5_0.bin",
    },
    ModelSpec {
        kind: ModelKind::Whisper,
        name: "large-v2-q8_0",
        filename: "ggml-large-v2-q8_0.bin",
        url: "https://huggingface.co/ggerganov/whisper.cpp/resolve/main/ggml-large-v2-q8_0.bin",
    },
    ModelSpec {
        kind: ModelKind::Whisper,
        name: "large-v3",
        filename: "ggml-large-v3.bin",
        url: "https://huggingface.co/ggerganov/whisper.cpp/resolve/main/ggml-large-v3.bin",
    },
    ModelSpec {
        kind: ModelKind::Whisper,
        name: "large-v3-q5_0",
        filename: "ggml-large-v3-q5_0.bin",
        url: "https://huggingface.co/ggerganov/whisper.cpp/resolve/main/ggml-large-v3-q5_0.bin",
    },
    // turbo
    ModelSpec {
        kind: ModelKind::Whisper,
        name: "large-v3-turbo",
        filename: "ggml-large-v3-turbo.bin",
        url: "https://huggingface.co/ggerganov/whisper.cpp/resolve/main/ggml-large-v3-turbo.bin",
    },
    ModelSpec {
        kind: ModelKind::Whisper,
        name: "large-v3-turbo-q5_0",
        filename: "ggml-large-v3-turbo-q5_0.bin",
        url: "https://huggingface.co/ggerganov/whisper.cpp/resolve/main/ggml-large-v3-turbo-q5_0.bin",
    },
    ModelSpec {
        kind: ModelKind::Whisper,
        name: "large-v3-turbo-q8_0",
        filename: "ggml-large-v3-turbo-q8_0.bin",
        url: "https://huggingface.co/ggerganov/whisper.cpp/resolve/main/ggml-large-v3-turbo-q8_0.bin",
    },
];

// -----------------------------------------------------------------------------
// VAD models (allowlist)
// -----------------------------------------------------------------------------
static VAD_MODELS: &[ModelSpec] = &[
    ModelSpec {
        kind: ModelKind::Vad,
        name: "silero-v5.1.2",
        filename: "ggml-silero-v5.1.2.bin",
        url: "https://huggingface.co/ggml-org/whisper-vad/resolve/main/ggml-silero-v5.1.2.bin",
    },
    ModelSpec {
        kind: ModelKind::Vad,
        name: "silero-v6.2.0",
        filename: "ggml-silero-v6.2.0.bin",
        url: "https://huggingface.co/ggml-org/whisper-vad/resolve/main/ggml-silero-v6.2.0.bin",
    },
];

fn main() -> Result<()> {
    let args = Args::parse();

    if args.list {
        print_model_list();
        return Ok(());
    }

    let name = args.name.as_deref().expect("clap should require --name");

    fs::create_dir_all(&args.dir)
        .with_context(|| format!("failed to create target dir: {}", args.dir.display()))?;

    let spec = lookup_model(name).with_context(|| {
        format!("unknown model '{name}'. Run with --list to see supported models.")
    })?;

    let dest_path = args.dir.join(spec.filename);

    if dest_path.exists() {
        println!("✅ already exists: {}", dest_path.display());
        return Ok(());
    }

    println!(
        "⬇️  downloading {} ({})",
        spec.filename,
        match spec.kind {
            ModelKind::Whisper => "whisper",
            ModelKind::Vad => "vad",
        }
    );
    println!("    {}", spec.url);

    let client = Client::builder()
        .user_agent("scribble-model-downloader")
        .build()
        .context("failed to build HTTP client")?;

    download_to_path(&client, spec.url, &dest_path)?;

    println!("✅ saved: {}", dest_path.display());
    Ok(())
}

fn lookup_model(name: &str) -> Option<&'static ModelSpec> {
    WHISPER_MODELS
        .iter()
        .find(|m| m.name == name)
        .or_else(|| VAD_MODELS.iter().find(|m| m.name == name))
}

fn print_model_list() {
    print!("{}", model_list_string());
}

fn model_list_string() -> String {
    let mut out = String::new();

    out.push_str("Whisper models:\n");
    for m in WHISPER_MODELS {
        out.push_str("  - ");
        out.push_str(m.name);
        out.push('\n');
    }

    out.push('\n');
    out.push_str("VAD models:\n");
    for m in VAD_MODELS {
        out.push_str("  - ");
        out.push_str(m.name);
        out.push('\n');
    }

    out
}

/// Download a URL into `dest_path` safely:
/// - download to `dest_path.part`
/// - fsync + rename to final path
fn download_to_path(client: &Client, url: &str, dest_path: &Path) -> Result<()> {
    let resp = client
        .get(url)
        .send()
        .with_context(|| format!("request failed: {url}"))?
        .error_for_status()
        .with_context(|| format!("download failed (bad status): {url}"))?;

    let total = resp.content_length();
    download_to_path_with_reader(resp, total, dest_path)
}

fn download_to_path_with_reader<R: Read>(
    mut reader: R,
    total_bytes: Option<u64>,
    dest_path: &Path,
) -> Result<()> {
    let total = total_bytes.unwrap_or(0);

    let pb = if total > 0 {
        ProgressBar::new(total)
    } else {
        ProgressBar::new_spinner()
    };

    pb.set_style(
        ProgressStyle::with_template(
            "{spinner:.green} {bytes}/{total_bytes} {bar:40.cyan/blue} {eta}",
        )
        .unwrap()
        .progress_chars("#>-"),
    );

    let tmp_path = PathBuf::from(format!("{}.part", dest_path.display()));

    let result = (|| -> Result<()> {
        let mut file = fs::File::create(&tmp_path)
            .with_context(|| format!("failed to create temp file: {}", tmp_path.display()))?;

        let mut buf = [0u8; 64 * 1024];
        loop {
            let n = reader.read(&mut buf)?;
            if n == 0 {
                break;
            }
            file.write_all(&buf[..n])?;
            pb.inc(n as u64);
        }

        file.sync_all()?;
        pb.finish_and_clear();

        fs::rename(&tmp_path, dest_path)
            .with_context(|| format!("failed to move into place: {}", dest_path.display()))?;

        Ok(())
    })();

    if result.is_err() {
        let _ = fs::remove_file(&tmp_path);
        pb.finish_and_clear();
    }

    result
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn lookup_model_finds_whisper_and_vad_specs() {
        let whisper = lookup_model("tiny").expect("expected tiny model spec");
        assert_eq!(whisper.kind, ModelKind::Whisper);
        assert_eq!(whisper.filename, "ggml-tiny.bin");

        let vad = lookup_model("silero-v6.2.0").expect("expected silero model spec");
        assert_eq!(vad.kind, ModelKind::Vad);
        assert_eq!(vad.filename, "ggml-silero-v6.2.0.bin");

        assert!(lookup_model("definitely-not-a-model").is_none());
    }

    #[test]
    fn model_list_string_includes_sections_and_known_names() {
        let list = model_list_string();
        assert!(list.contains("Whisper models:\n"));
        assert!(list.contains("  - tiny\n"));
        assert!(list.contains("VAD models:\n"));
        assert!(list.contains("  - silero-v6.2.0\n"));
    }

    #[test]
    fn args_parse_requires_name_unless_list() {
        let err = Args::try_parse_from(["model-downloader"])
            .err()
            .expect("expected missing-args error");
        assert!(err.to_string().contains("--name"));

        let args =
            Args::try_parse_from(["model-downloader", "--list"]).expect("parse list params");
        assert!(args.list);
        assert!(args.name.is_none());
    }

    #[test]
    fn download_to_path_with_reader_writes_and_renames() -> anyhow::Result<()> {
        let dir = tempfile::tempdir()?;
        let dest_path = dir.path().join("model.bin");
        let tmp_path = PathBuf::from(format!("{}.part", dest_path.display()));

        let bytes = b"abc123".to_vec();
        download_to_path_with_reader(
            std::io::Cursor::new(bytes.clone()),
            Some(bytes.len() as u64),
            &dest_path,
        )?;

        assert!(dest_path.exists());
        assert!(!tmp_path.exists());
        assert_eq!(std::fs::read(&dest_path)?, bytes);
        Ok(())
    }

    struct ErrorAfterNBytes {
        bytes: Vec<u8>,
        fail_at: usize,
        pos: usize,
    }

    impl Read for ErrorAfterNBytes {
        fn read(&mut self, buf: &mut [u8]) -> std::io::Result<usize> {
            if self.pos >= self.fail_at {
                return Err(std::io::Error::new(
                    std::io::ErrorKind::Other,
                    "simulated read failure",
                ));
            }

            let remaining = &self.bytes[self.pos..];
            let n = remaining.len().min(buf.len());
            buf[..n].copy_from_slice(&remaining[..n]);
            self.pos += n;
            Ok(n)
        }
    }

    #[test]
    fn download_to_path_with_reader_cleans_up_part_file_on_error() -> anyhow::Result<()> {
        let dir = tempfile::tempdir()?;
        let dest_path = dir.path().join("model.bin");
        let tmp_path = PathBuf::from(format!("{}.part", dest_path.display()));

        let reader = ErrorAfterNBytes {
            bytes: b"abc123".to_vec(),
            fail_at: 1,
            pos: 0,
        };

        let err = download_to_path_with_reader(reader, Some(6), &dest_path).unwrap_err();
        assert!(err.to_string().contains("simulated read failure"));
        assert!(!dest_path.exists());
        assert!(!tmp_path.exists());
        Ok(())
    }
}
