//! High-level API for running transcriptions with Scribble.
//!
//! We expose a single, ergonomic entry point (`Scribble`) that wraps the lower-level
//! Whisper, VAD, decoding, and encoding logic.
//!
//! The intent is:
//! - We load the Whisper model once (expensive).
//! - We load the VAD model once (also somewhat expensive).
//! - We reuse both contexts to transcribe multiple inputs.
//! - Callers choose output format and behavior via `Opts`.
//!
//! This module is deliberately “high level”: it wires up decoding → (optional) VAD → Whisper → encoder,
//! while keeping the lower-level pieces testable in their own modules.

use std::io::{BufWriter, Read, Write};
use std::path::Path;

use anyhow::{Context, Result, ensure};
use hound::WavSpec;
use whisper_rs::{WhisperContext, WhisperVadContext, WhisperVadContextParams};

use crate::audio_pipeline::WHISPER_SAMPLE_RATE;
use crate::ctx::get_context;
use crate::decoder::{SamplesSink, StreamDecodeOpts, decode_to_whisper_stream_from_read};
use crate::incremental::BufferedSegmentTranscriber;
use crate::json_array_encoder::JsonArrayEncoder;
use crate::logging::init_whisper_logging;
use crate::opts::Opts;
use crate::output_type::OutputType;
use crate::segment_encoder::SegmentEncoder;
use crate::segments::write_segments;
use crate::vad::to_speech_only;
use crate::vtt_encoder::VttEncoder;

/// The main high-level transcription entry point.
///
/// `Scribble` owns the long-lived resources required for transcription:
/// - a `WhisperContext` (loaded model + runtime state)
/// - a `WhisperVadContext` (loaded VAD model + runtime state)
/// - the VAD model path (kept for diagnostics)
///
/// Typical usage:
/// - Construct once (model loading happens here).
/// - Call `transcribe` many times with different inputs and outputs.
///
/// Note: `transcribe` takes `&mut self` because whisper_rs’s VAD context requires mutable access
/// to run inference (`segments_from_samples` takes `&mut self`).
pub struct Scribble {
    ctx: WhisperContext,
    vad_model_path: String,
    vad_ctx: WhisperVadContext,
}

impl Scribble {
    /// Create a new `Scribble` instance by loading Whisper + VAD models from disk.
    ///
    /// We fail fast if the VAD model path is missing or invalid. This keeps invariants simple:
    /// once `Scribble::new` succeeds, we know VAD is available whenever enabled via `Opts`.
    pub fn new(model_path: impl AsRef<str>, vad_model_path: impl Into<String>) -> Result<Self> {
        // We keep whisper logs quiet by default so callers fully control stdout/stderr.
        // This function is idempotent (safe to call multiple times).
        init_whisper_logging();

        // We require a valid VAD model path up front so we can provide a clear error early.
        let vad_model_path = vad_model_path.into();
        ensure!(
            !vad_model_path.trim().is_empty(),
            "VAD model path must be provided"
        );

        let vad_path = Path::new(&vad_model_path);
        ensure!(
            vad_path.exists(),
            "VAD model not found at '{}'",
            vad_model_path
        );
        ensure!(
            vad_path.is_file(),
            "VAD model path is not a file: '{}'",
            vad_model_path
        );

        // Load the Whisper model once (this is the expensive part).
        let ctx = get_context(model_path.as_ref())?;

        // Load the VAD model once so repeated transcriptions don't re-initialize it.
        let vad_ctx_params = WhisperVadContextParams::default();
        let vad_ctx = WhisperVadContext::new(&vad_model_path, vad_ctx_params)
            .with_context(|| format!("failed to load VAD model from '{}'", vad_model_path))?;

        Ok(Self {
            ctx,
            vad_model_path,
            vad_ctx,
        })
    }

    /// Transcribe an input stream and write the result to an output writer.
    ///
    /// We accept a generic `Read` input rather than a filename so callers can pass:
    /// - `File`
    /// - stdin
    /// - sockets / HTTP bodies
    /// - any other byte stream
    ///
    /// We decode audio into a mono 16kHz stream (whisper.cpp’s expected format) using the
    /// `decoder` module, optionally apply VAD, and then run Whisper and encode segments.
    ///
    /// Note: The `Send + Sync + 'static` bounds mirror the decoder API. If you later relax
    /// those constraints in `decoder`, we can simplify these bounds too.
    pub fn transcribe<R, W>(&mut self, r: R, w: W, opts: &Opts) -> Result<()>
    where
        R: Read + Send + Sync + 'static,
        W: Write,
    {
        // Buffer output for efficiency (especially important for stdout).
        let writer = BufWriter::new(w);

        // Select an encoder based on the requested output type.
        // We keep this explicit (no trait objects) to avoid lifetime surprises.
        match opts.output_type {
            OutputType::Json => {
                let mut encoder = JsonArrayEncoder::new(writer);
                let run_res = self.transcribe_with_encoder(r, opts, &mut encoder);
                merge_run_and_close(run_res, encoder.close())
            }
            OutputType::Vtt => {
                let mut encoder = VttEncoder::new(writer);
                let run_res = self.transcribe_with_encoder(r, opts, &mut encoder);
                merge_run_and_close(run_res, encoder.close())
            }
        }
    }

    fn transcribe_with_encoder<R, E>(&mut self, r: R, opts: &Opts, encoder: &mut E) -> Result<()>
    where
        R: Read + Send + Sync + 'static,
        E: SegmentEncoder,
    {
        if opts.enable_voice_activity_detection {
            let mut samples = decode_all_samples(r)?;
            if samples.is_empty() {
                return Ok(());
            }

            let spec = whisper_wav_spec();
            let found_speech = to_speech_only(&mut self.vad_ctx, &spec, &mut samples)?;
            if found_speech {
                write_segments(&self.ctx, opts, encoder, &samples)?;
            }
            return Ok(());
        }

        let mut transcriber = BufferedSegmentTranscriber::new(&self.ctx, opts, encoder);
        decode_to_whisper_stream_from_read(r, StreamDecodeOpts::default(), &mut transcriber)?;
        transcriber.finish()
    }

    /// Access the underlying Whisper context.
    ///
    /// This is primarily intended for advanced or experimental use-cases.
    pub fn context(&self) -> &WhisperContext {
        &self.ctx
    }

    /// Access the underlying VAD context.
    ///
    /// This is primarily intended for advanced or experimental use-cases.
    pub fn vad_context(&self) -> &WhisperVadContext {
        &self.vad_ctx
    }

    /// Access the configured VAD model path.
    ///
    /// This is primarily intended for diagnostics and debugging.
    pub fn vad_model_path(&self) -> &str {
        &self.vad_model_path
    }
}

/// A simple `SamplesSink` implementation that appends decoded chunks into a `Vec<f32>`.
///
/// We keep this type private to the module because it's an implementation detail of `transcribe`.
struct VecSamplesSink<'a> {
    out: &'a mut Vec<f32>,
}

impl<'a> VecSamplesSink<'a> {
    /// Create a sink that appends decoded samples into `out`.
    fn new(out: &'a mut Vec<f32>) -> Self {
        Self { out }
    }
}

impl<'a> SamplesSink for VecSamplesSink<'a> {
    /// Receive a chunk of mono 16kHz samples.
    ///
    /// Returning `Ok(true)` tells the decoder to keep going.
    fn on_samples(&mut self, samples_16k_mono: &[f32]) -> Result<bool> {
        // We append chunks as they arrive. This preserves ordering and builds the full buffer.
        self.out.extend_from_slice(samples_16k_mono);
        Ok(true)
    }
}

fn whisper_wav_spec() -> WavSpec {
    WavSpec {
        channels: 1,
        sample_rate: WHISPER_SAMPLE_RATE,
        bits_per_sample: 16,
        sample_format: hound::SampleFormat::Int,
    }
}

fn decode_all_samples<R>(r: R) -> Result<Vec<f32>>
where
    R: Read + Send + Sync + 'static,
{
    let mut samples = Vec::<f32>::new();
    let mut sink = VecSamplesSink::new(&mut samples);
    decode_to_whisper_stream_from_read(r, StreamDecodeOpts::default(), &mut sink)?;
    Ok(samples)
}

fn merge_run_and_close(run_res: Result<()>, close_res: Result<()>) -> Result<()> {
    match (run_res, close_res) {
        (Ok(()), Ok(())) => Ok(()),
        (Ok(()), Err(close_err)) => Err(close_err),
        (Err(err), Ok(())) => Err(err),
        (Err(err), Err(close_err)) => Err(err.context(close_err)),
    }
}
