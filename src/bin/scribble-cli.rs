// src/bin/scribble-cli.rs

use anyhow::{Context, Result};
use clap::Parser;
use std::fs::File;
use std::io::{self, Read};

use scribble::opts::Opts;
use scribble::output_type::OutputType;
use scribble::scribble::Scribble;

fn main() -> Result<()> {
    let params = Params::parse();

    // Map CLI flags into library options.
    //
    // Keeping this mapping explicit helps:
    // - keep the library reusable (Opts is the contract)
    // - keep the CLI thin (just parsing + wiring)
    let opts = Opts {
        enable_translate_to_english: params.enable_translation_to_english,
        enable_voice_activity_detection: params.enable_voice_activity_detection,
        language: params.language.clone(),
        output_type: params.output_type,
        incremental_min_window_seconds: 1,
    };

    // Open an input source.
    // - File path → open directly.
    // - "-"       → stream stdin.
    //
    // Note: we pass `io::stdin()` (not `stdin().lock()`) to avoid non-Send lock guards.
    let input = open_input(&params.input)?;

    // Load the Whisper + VAD models (expensive).
    //
    // `Scribble::new` validates both model paths, so once this succeeds, we know the backend
    // is ready for repeated transcriptions.
    let mut scribble = Scribble::new(params.model_path, params.vad_model_path)?;

    // Stream transcription output to stdout.
    scribble
        .transcribe(input, io::stdout(), &opts)
        .context("transcription failed")?;

    Ok(())
}

/// Open an input source as a boxed reader.
///
/// We return `Box<dyn Read + Send + Sync>` because the decoder pipeline may require
/// those bounds for its internal threading/ownership model.
///
/// For stdin:
/// - We use `io::stdin()` directly (not a lock guard).
/// - This stays streaming-friendly and avoids temp files.
fn open_input(path: &str) -> Result<Box<dyn Read + Send + Sync>> {
    if path == "-" {
        Ok(Box::new(io::stdin()))
    } else {
        let file =
            File::open(path).with_context(|| format!("failed to open input file: {path}"))?;
        Ok(Box::new(file))
    }
}

/// CLI parameters for `scribble`.
#[derive(Parser, Debug)]
#[command(name = "scribble")]
#[command(about = "A transcription CLI (audio or video input)")]
struct Params {
    /// Path to a whisper.cpp model file (e.g. `ggml-large-v3.bin`).
    #[arg(short = 'm', long = "model", required = true)]
    pub model_path: String,

    /// Path to a Whisper-VAD model file.
    #[arg(short = 'v', long = "vad-model", required = true)]
    pub vad_model_path: String,

    /// Input media path (audio or video), or "-" to read from stdin.
    ///
    /// Examples:
    ///   scribble -i samples/sintel_trailer-480p.mp4 ...
    ///   cat samples/audio.mp3 | scribble -i - ...
    #[arg(short = 'i', long = "input", required = true)]
    pub input: String,

    /// Output format for transcription segments.
    #[arg(
        short = 'o',
        long = "output-type",
        value_enum,
        default_value_t = OutputType::Vtt
    )]
    pub output_type: OutputType,

    /// Enable voice activity detection (VAD).
    #[arg(long = "enable-vad", default_value_t = false)]
    pub enable_voice_activity_detection: bool,

    /// Translate speech to English.
    #[arg(
        short = 't',
        long = "enable-translation-to-english",
        default_value_t = false
    )]
    pub enable_translation_to_english: bool,

    /// Optional language hint (e.g. "en", "es").
    #[arg(short = 'l', long = "language")]
    pub language: Option<String>,
}
