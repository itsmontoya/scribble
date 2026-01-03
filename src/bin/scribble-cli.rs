// src/bin/scribble-cli.rs

use anyhow::{Context, Result};
use clap::Parser;
use std::fs::File;
use std::io::{self, Read};
use std::path::Path;

#[cfg(feature = "silero-onnx")]
use scribble::backends::silero::{SileroBackend, SileroConfig, SileroDecoder};
use scribble::opts::Opts;
use scribble::output_type::OutputType;
use scribble::scribble::Scribble;

#[derive(clap::ValueEnum, Debug, Clone, Copy)]
enum SileroDecoderArg {
    Greedy,
    Beam,
}

#[derive(clap::ValueEnum, Debug, Clone, Copy)]
enum BackendKind {
    Whisper,
    #[cfg(feature = "silero-onnx")]
    Silero,
}

fn main() -> Result<()> {
    let params = Params::parse();

    let backend = select_backend(params.backend, &params.model_path)?;

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

    match backend {
        BackendKind::Whisper => {
            // Load the Whisper + VAD models (expensive).
            //
            // `Scribble::new` validates the VAD model path, so once this succeeds,
            // we know VAD is available whenever it is enabled via `Opts`.
            let mut scribble = Scribble::new(params.model_path, params.vad_model_path)?;

            // Stream transcription output to stdout.
            scribble
                .transcribe(input, io::stdout(), &opts)
                .context("transcription failed")?;
        }
        #[cfg(feature = "silero-onnx")]
        BackendKind::Silero => {
            let mut cfg = SileroConfig::default();
            if let Some(path) = &params.silero_vocab {
                cfg = cfg.with_vocab_file(path)?;
            }

            // Sensible defaults so Silero doesn’t require “extra knobs” compared to Whisper.
            //
            // Users can still override via flags, but by default we aim for higher-quality decoding.
            cfg.decoder = SileroDecoder::BeamSearch;
            cfg.beam_width = 32;
            cfg.frame_top_k = 32;
            cfg.word_boundary_bonus = 0.2;
            cfg.word_boundary_min_gap = 10;

            if let Some(decoder) = params.silero_decoder {
                cfg.decoder = match decoder {
                    SileroDecoderArg::Greedy => SileroDecoder::Greedy,
                    SileroDecoderArg::Beam => SileroDecoder::BeamSearch,
                };
            }
            if let Some(beam) = params.silero_beam_width {
                cfg.beam_width = beam;
            }
            if let Some(k) = params.silero_top_k {
                cfg.frame_top_k = k;
            }
            if let Some(blank) = params.silero_blank_id {
                cfg.blank_id = blank;
            }
            if let Some(tok) = &params.silero_word_boundary_token {
                cfg.word_boundary_token = Some(tok.clone());
            }
            if let Some(bonus) = params.silero_word_bonus {
                cfg.word_boundary_bonus = bonus;
            }
            if let Some(gap) = params.silero_word_min_gap {
                cfg.word_boundary_min_gap = gap;
            }

            let backend = SileroBackend::with_config(&params.model_path, cfg)?;
            let mut scribble = Scribble::with_backend(backend, params.vad_model_path)?;

            scribble
                .transcribe(input, io::stdout(), &opts)
                .context("transcription failed")?;
        }
    }

    Ok(())
}

fn select_backend(backend: Option<BackendKind>, model_path: &str) -> Result<BackendKind> {
    if let Some(backend) = backend {
        return Ok(backend);
    }

    let model_path = Path::new(model_path);
    let is_onnx = model_path
        .extension()
        .and_then(|e| e.to_str())
        .is_some_and(|e| e.eq_ignore_ascii_case("onnx"));

    if is_onnx {
        #[cfg(feature = "silero-onnx")]
        {
            return Ok(BackendKind::Silero);
        }

        #[cfg(not(feature = "silero-onnx"))]
        {
            return Err(anyhow::anyhow!(
                "model looks like ONNX ({}) but this build does not include the Silero backend; rebuild with `--features silero-onnx`",
                model_path.display()
            ));
        }
    }

    Ok(BackendKind::Whisper)
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
    /// ASR backend implementation to use.
    ///
    /// - `whisper` (default): whisper.cpp via `whisper-rs` (expects a `ggml-*.bin` model).
    /// - `silero`: Silero ASR via ONNX (expects a `.onnx` model; requires `--features silero-onnx`).
    ///
    /// If omitted, the backend is auto-detected from `--model`:
    /// - `*.onnx` → `silero` (when enabled)
    /// - otherwise → `whisper`
    #[arg(long = "backend", value_enum)]
    pub backend: Option<BackendKind>,

    /// Path to a whisper.cpp model file (e.g. `ggml-large-v3.bin`).
    #[arg(short = 'm', long = "model", required = true)]
    pub model_path: String,

    /// Path to a Whisper VAD model file.
    #[arg(short = 'v', long = "vad-model", required = true)]
    pub vad_model_path: String,

    /// Silero vocab file (newline-delimited tokens) used to decode model outputs.
    ///
    /// Required for some Silero ASR exports (e.g. `v5_en`) where the output vocab is large (e.g. 999).
    #[arg(long = "silero-vocab")]
    pub silero_vocab: Option<String>,

    /// Silero decoder strategy.
    #[arg(long = "silero-decoder", value_enum)]
    pub silero_decoder: Option<SileroDecoderArg>,

    /// Beam width for `--silero-decoder beam`.
    #[arg(long = "silero-beam-width")]
    pub silero_beam_width: Option<usize>,

    /// Per-frame shortlist size for `--silero-decoder beam`.
    #[arg(long = "silero-top-k")]
    pub silero_top_k: Option<usize>,

    /// CTC blank token id (usually 0).
    #[arg(long = "silero-blank-id")]
    pub silero_blank_id: Option<usize>,

    /// Word-boundary token to bias beam search (often `" "` or `"|"`).
    #[arg(long = "silero-word-boundary-token")]
    pub silero_word_boundary_token: Option<String>,

    /// Log-space bonus added when emitting the word-boundary token (only for beam search).
    #[arg(long = "silero-word-bonus")]
    pub silero_word_bonus: Option<f32>,

    /// Minimum number of tokens between word-boundary tokens for the bonus to apply.
    #[arg(long = "silero-word-min-gap")]
    pub silero_word_min_gap: Option<usize>,

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
