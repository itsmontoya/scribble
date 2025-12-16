use anyhow::Result;
use clap::Parser;

use std::io::{self, BufWriter};

use scribble::ctx::get_context;
use scribble::json_array_encoder::JsonArrayEncoder;
use scribble::logging::init_whisper_logging;
use scribble::output_type::OutputType;
use scribble::segment_encoder::SegmentEncoder;
use scribble::segments::write_segments;
use scribble::vad::apply_vad;
use scribble::vtt_encoder::VttEncoder;
use scribble::wav::get_samples_from_wav;

fn main() -> Result<()> {
    init_whisper_logging();
    let params = get_params()?;
    let ctx = get_context(&params.model_path)?;
    let (mut samples, spec) = get_samples_from_wav(&params.audio_path)?;
    if !apply_vad(&params.vad_model_path, &spec, &mut samples)? {
        return Ok(());
    }

    let stdout = io::stdout();
    let writer = BufWriter::new(stdout.lock());

    let mut encoder: Box<dyn SegmentEncoder> = match params.output_type {
        OutputType::Json => Box::new(JsonArrayEncoder::new(writer)),
        OutputType::Vtt => Box::new(VttEncoder::new(writer)),
    };

    write_segments(&ctx, &mut *encoder, &mut samples)?;
    Ok(())
}

#[derive(Parser, Debug)]
#[command(name = "scribble")]
#[command(about = "A transcription CLI")]
struct Params {
    #[arg(short = 'm', long = "model")]
    pub model_path: String,

    #[arg(short = 'v', long = "vad-model")]
    pub vad_model_path: String,

    #[arg(short = 'a', long = "audio")]
    pub audio_path: String,

    #[arg(
        short = 'o',
        long = "output-type",
        value_enum,
        default_value_t = OutputType::Vtt
    )]
    pub output_type: OutputType,

    #[arg(long = "enable-vad", default_value_t = false)]
    pub enable_voice_activity_detection: bool,

    #[arg(
        short = 't',
        long = "enable-translation-to-english",
        default_value_t = false
    )]
    pub enable_translation_to_english: bool,
}

fn get_params() -> Result<Params> {
    Ok(Params::parse())
}
