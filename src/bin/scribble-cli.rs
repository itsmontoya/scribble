use anyhow::{Context, Result};
use clap::Parser;
use std::fs::File;
use std::io::{self, Read, Seek, SeekFrom};

use scribble::opts::Opts;
use scribble::output_type::OutputType;
use scribble::scribble::Scribble;

fn main() -> Result<()> {
    let params = Params::parse();

    // Map CLI flags into library options.
    // We keep this mapping explicit so the library stays reusable and the CLI stays thin.
    let opts = Opts {
        enable_translate_to_english: params.enable_translation_to_english,
        enable_voice_activity_detection: params.enable_voice_activity_detection,
        language: params.language,
        output_type: params.output_type,
    };

    // Load the Whisper + VAD models (expensive).
    //
    // `Scribble::new` validates the VAD model path, so once we construct `Scribble`,
    // we know VAD is available whenever it is enabled via `Opts`.
    let mut scribble = Scribble::new(params.model_path, params.vad_model_path)?;

    // Open a seekable input source.
    // - File path → open directly.
    // - "-"       → buffer stdin into a temp file so the decoder can Seek.
    let input = open_seekable_input(&params.input)?;

    // Stream transcription output to stdout.
    scribble.transcribe(input, io::stdout(), &opts)?;

    Ok(())
}

/// Open an input source that supports `Read + Seek`.
///
/// Many container formats (MP4, MKV, etc.) and some decoders require seeking.
/// For file paths, `File` is already seekable.
/// For stdin, we spool into a temporary file and rewind it to the beginning.
///
/// Why a temp file (instead of an in-memory buffer)?
/// - It avoids large RAM usage for big inputs.
/// - It keeps stdin support compatible with decoders that require Seek.
///
/// Tradeoff:
/// - This adds disk I/O. If we later add a decode path that supports true streaming,
///   we can avoid spooling stdin entirely.
fn open_seekable_input(path: &str) -> Result<Box<dyn ReadSeekSendSync>> {
    if path == "-" {
        // Create a temp file to spool stdin.
        let mut tmp =
            tempfile::NamedTempFile::new().context("failed to create temp file for stdin")?;

        // Copy stdin into the temp file.
        //
        // Note: `io::copy` reads until EOF, so the caller controls termination (e.g. piping a file).
        io::copy(&mut io::stdin(), &mut tmp).context("failed to copy stdin to temp file")?;

        // Convert the temp file into a normal `File` handle and rewind so reads start at the beginning.
        let mut file = tmp.into_file();
        file.seek(SeekFrom::Start(0))
            .context("failed to rewind temp file")?;

        Ok(Box::new(file))
    } else {
        let file =
            File::open(path).with_context(|| format!("failed to open input file: {path}"))?;
        Ok(Box::new(file))
    }
}

/// Convenience trait alias for "a seekable reader we can send across threads".
///
/// We use a local trait alias instead of unstable `type` aliases for traits.
/// This keeps signatures readable without pulling in extra crates.
trait ReadSeekSendSync: Read + Seek + Send + Sync {}
impl<T> ReadSeekSendSync for T where T: Read + Seek + Send + Sync {}

/// CLI parameters for `scribble`.
#[derive(Parser, Debug)]
#[command(name = "scribble")]
#[command(about = "A transcription CLI (audio or video input)")]
struct Params {
    /// Path to a whisper.cpp model file (e.g. `ggml-large-v3.bin`).
    #[arg(short = 'm', long = "model", required = true)]
    pub model_path: String,

    /// Path to a Whisper VAD model file.
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

    /// Translate speech to English (not wired into the transcription pipeline yet).
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
