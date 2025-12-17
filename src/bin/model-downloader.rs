// src/bin/model-downloader.rs
//
// A small CLI utility to download known Whisper (ASR) and Whisper-VAD models into a target dir.
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
    #[arg(long)]
    name: Option<String>,

    /// Target directory to store models (created if missing)
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
    kind: ModelKind,
    /// Friendly model name the user types (e.g. "large-v3-turbo", "silero-v6.2.0").
    name: &'static str,
    /// Exact filename written to disk (e.g. "ggml-large-v3-turbo.bin").
    filename: &'static str,
    /// Full download URL.
    url: &'static str,
}

// Whisper models (allowlist).
//
// These URLs match whisper.cpp’s standard Hugging Face repo for GGML models.
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
    // large v1/v2/v3
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
    // turbo variants
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

// VAD models (allowlist).
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

    let name = args
        .name
        .as_deref()
        .context("missing --name (or use --list)")?;

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

    download_to_path(spec.url, &dest_path)?;

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
    println!("Whisper models:");
    for m in WHISPER_MODELS {
        println!("  - {}", m.name);
    }
    println!();
    println!("VAD models:");
    for m in VAD_MODELS {
        println!("  - {}", m.name);
    }
}

/// Download a URL into `dest_path` safely:
/// - download to `dest_path.part`
/// - fsync + rename to final path
fn download_to_path(url: &str, dest_path: &Path) -> Result<()> {
    let client = Client::new();

    let mut resp = client
        .get(url)
        .send()
        .with_context(|| format!("request failed: {url}"))?
        .error_for_status()
        .with_context(|| format!("download failed (bad status): {url}"))?;

    let total = resp.content_length().unwrap_or(0);

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

    let tmp_path = dest_path.with_extension("part");
    let mut file = fs::File::create(&tmp_path)
        .with_context(|| format!("failed to create temp file: {}", tmp_path.display()))?;

    let mut buf = [0u8; 64 * 1024];
    loop {
        let n = resp.read(&mut buf)?;
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
}
