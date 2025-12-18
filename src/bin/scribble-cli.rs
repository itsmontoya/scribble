use anyhow::Result;
use clap::Parser;
use std::fs::File;
use std::io;

use scribble::opts::Opts;
use scribble::output_type::OutputType;
use scribble::scribble::Scribble;

fn main() -> Result<()> {
    let params = Params::parse();

    // Map CLI flags into library options.
    // This keeps the library reusable and the CLI thin.
    let opts = Opts {
        enable_translate_to_english: params.enable_translation_to_english,
        enable_voice_activity_detection: params.enable_voice_activity_detection,

        // If this is `None`, we allow Whisper to auto-detect language.
        language: params.language,

        output_type: params.output_type,
    };

    // Load the Whisper model (expensive) and prepare for transcription.
    let mut scribble = Scribble::new(params.model_path, params.vad_model_path)?;

    // `File` implements `Read + Seek`, which our WAV reader requires.
    let input = File::open(&params.audio_path)?;

    // Stream transcription output directly to stdout.
    scribble.transcribe(input, io::stdout(), &opts)?;

    Ok(())
}

/// CLI parameters for `scribble`.
///
/// We keep these flags explicit and well-documented so usage is self-explanatory.
#[derive(Parser, Debug)]
#[command(name = "scribble")]
#[command(about = "A transcription CLI")]
struct Params {
    /// Path to a whisper.cpp model file (e.g. `ggml-base.en.bin`).
    #[arg(short = 'm', long = "model", required = true)]
    pub model_path: String,

    /// Path to a Whisper VAD model file.
    #[arg(short = 'v', long = "vad-model", required = true)]
    pub vad_model_path: String,

    /// Path to a mono 16kHz WAV file to transcribe.
    #[arg(short = 'a', long = "audio", required = true)]
    pub audio_path: String,

    /// Output format for transcription segments.
    #[arg(
        short = 'o',
        long = "output-type",
        value_enum,
        default_value_t = OutputType::Vtt
    )]
    pub output_type: OutputType,

    /// Enable voice activity detection (VAD) to suppress non-speech regions.
    #[arg(long = "enable-vad", default_value_t = false)]
    pub enable_voice_activity_detection: bool,

    /// Translate speech to English (not wired into the transcription pipeline yet).
    #[arg(
        short = 't',
        long = "enable-translation-to-english",
        default_value_t = false
    )]
    pub enable_translation_to_english: bool,

    /// Optional language hint (e.g. "en", "es"). If omitted, Whisper will auto-detect.
    #[arg(short = 'l', long = "language")]
    pub language: Option<String>,
}
