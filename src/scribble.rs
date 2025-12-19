//! High-level API for running transcriptions with Scribble.
//!
//! We expose a single, ergonomic entry point (`Scribble`) that wraps the lower-level
//! Whisper, VAD, and encoding logic.
//!
//! The intent is:
//! - We load the Whisper model once (expensive).
//! - We reuse the context to transcribe multiple inputs.
//! - Callers choose an output format via options.

use anyhow::Result;
use hound::WavSpec;
use std::io::{BufWriter, Read, Seek, Write};
use std::path::Path;
use whisper_rs::{WhisperContext, WhisperVadContext, WhisperVadContextParams, WhisperVadParams};

use crate::ctx::get_context;
use crate::json_array_encoder::JsonArrayEncoder;
use crate::logging::init_whisper_logging;
use crate::opts::Opts;
use crate::output_type::OutputType;
use crate::segments::write_segments;
use crate::vad::apply_vad;
use crate::vtt_encoder::VttEncoder;
use crate::wav::get_samples_from_wav_reader;

/// The main high-level transcription entry point.
///
/// `Scribble` owns the long-lived resources required for transcription:
/// - a `WhisperContext` (loaded model + runtime state)
/// - a `WhisperVadContext` (loaded VAD model + runtime state)
/// - the VAD model path (kept so configuration is inspectable/debuggable)
///
/// Typical usage:
/// - Construct once (model load happens here).
/// - Call `transcribe` many times with different inputs and outputs.
pub struct Scribble {
    ctx: WhisperContext,
    vad_model_path: String,
    vad_ctx: WhisperVadContext,
}

impl Scribble {
    /// Create a new `Scribble` instance by loading Whisper + VAD models from disk.
    ///
    /// Model loading is expensive, so we expect `Scribble::new` to be called sparingly
    /// and reused across multiple transcriptions.
    ///
    /// VAD model path is required. We validate it here so once construction succeeds,
    /// the rest of the library can assume VAD is available when enabled via `Opts`.
    pub fn new(model_path: impl AsRef<str>, vad_model_path: impl Into<String>) -> Result<Self> {
        // We keep whisper logs quiet by default so callers fully control output.
        // This function is idempotent (safe to call multiple times).
        init_whisper_logging();

        // We require a valid VAD model path up front so we can fail fast with a clear error.
        let vad_model_path = vad_model_path.into();
        let vad_path = Path::new(&vad_model_path);

        if vad_model_path.trim().is_empty() {
            anyhow::bail!("VAD model path must be provided");
        }
        if !vad_path.exists() {
            anyhow::bail!("VAD model not found at '{}'", vad_model_path);
        }
        if !vad_path.is_file() {
            anyhow::bail!("VAD model path is not a file: '{}'", vad_model_path);
        }

        // Load the Whisper model once (expensive).
        let ctx = get_context(model_path.as_ref())?;

        // Create a VAD context from the model on disk.
        // We do this once so repeated transcriptions don't re-initialize it.
        let vad_ctx_params = WhisperVadContextParams::default();
        let vad_ctx = WhisperVadContext::new(&vad_model_path, vad_ctx_params)?;

        Ok(Self {
            ctx,
            vad_model_path,
            vad_ctx,
        })
    }

    /// Transcribe a WAV stream and write the result to an output writer.
    ///
    /// Reader requirements:
    /// - `Read`: to consume WAV bytes
    /// - `Seek`: required by `hound::WavReader` to parse headers correctly
    ///
    /// Output is streamed directly into the provided writer.
    pub fn transcribe<R, W>(&mut self, r: R, w: W, opts: &Opts) -> Result<()>
    where
        R: Read + Seek,
        W: Write,
    {
        // Read and normalize audio samples from the WAV stream.
        // This enforces mono, 16 kHz audio to match whisper.cpp expectations.
        let (mut samples, spec) = get_samples_from_wav_reader(r)?;

        // Optionally apply VAD in-place by zeroing out non-speech regions.
        // If VAD finds no speech, we exit successfully with no output.
        if opts.enable_voice_activity_detection {
            let found_speech = self.process_vad(&spec, &mut samples)?;
            if !found_speech {
                return Ok(());
            }
        }

        // Buffer output for efficiency (especially important for stdout).
        let writer = BufWriter::new(w);

        // Select an encoder based on the requested output type.
        // We avoid trait objects here to keep lifetimes simple and explicit.
        match opts.output_type {
            OutputType::Json => {
                let mut encoder = JsonArrayEncoder::new(writer);
                write_segments(&self.ctx, opts, &mut encoder, &samples)?;
            }
            OutputType::Vtt => {
                let mut encoder = VttEncoder::new(writer);
                write_segments(&self.ctx, opts, &mut encoder, &samples)?;
            }
        }

        Ok(())
    }

    /// Run VAD to identify speech segments, then apply our in-place masking policy.
    fn process_vad(&mut self, spec: &WavSpec, samples: &mut [f32]) -> Result<bool> {
        // We start from defaults and then apply the specific policy we want.
        let mut vad_params = WhisperVadParams::default();

        // We cap max speech duration to keep VAD from producing extremely long segments.
        // (This value is in milliseconds in whisper_rs.)
        vad_params.set_max_speech_duration(15000.0);

        // Run VAD to produce segments from our sample buffer.
        let segments = self.vad_ctx.segments_from_samples(vad_params, samples)?;

        // Apply our masking policy in-place.
        apply_vad(spec, &segments, samples)
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
