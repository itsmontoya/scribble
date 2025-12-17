//! High-level API for running transcriptions with Scribble.
//!
//! We expose a single, ergonomic entry point (`Scribble`) that wraps the lower-level
//! Whisper, VAD, and encoding logic.
//!
//! The intent is:
//! - We load the Whisper model once (expensive).
//! - We reuse the context to transcribe multiple inputs.
//! - Callers choose an output format via options.

use anyhow::{Result, anyhow};
use std::io::{BufWriter, Read, Seek, Write};
use whisper_rs::WhisperContext;

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
/// - a VAD model path (used only when VAD is enabled)
///
/// Typical usage:
/// - Construct once (model load happens here).
/// - Call `transcribe` many times with different inputs and outputs.
pub struct Scribble {
    ctx: WhisperContext,
    vad_model_path: String,
}

impl Scribble {
    /// Create a new `Scribble` instance by loading a Whisper model from disk.
    ///
    /// Model loading is expensive, so we expect `Scribble::new` to be called sparingly
    /// and reused across multiple transcriptions.
    pub fn new(model_path: impl AsRef<str>, vad_model_path: impl Into<String>) -> Result<Self> {
        // We keep whisper logs quiet by default so callers fully control output.
        // This function is idempotent (safe to call multiple times).
        init_whisper_logging();

        let ctx = get_context(model_path.as_ref())?;

        Ok(Self {
            ctx,
            vad_model_path: vad_model_path.into(),
        })
    }

    /// Transcribe a WAV stream and write the result to an output writer.
    ///
    /// Reader requirements:
    /// - `Read`: to consume WAV bytes
    /// - `Seek`: required by `hound::WavReader` to parse headers correctly
    ///
    /// Output is streamed directly into the provided writer.
    pub fn transcribe<R, W>(&self, r: R, w: W, opts: &Opts) -> Result<()>
    where
        R: Read + Seek,
        W: Write,
    {
        // Validate option combinations early so failures are obvious and actionable.
        if opts.enable_voice_activity_detection && self.vad_model_path.is_empty() {
            return Err(anyhow!(
                "voice activity detection is enabled, but no VAD model path was provided"
            ));
        }

        // Read and normalize audio samples from the WAV stream.
        // This enforces mono, 16 kHz audio to match whisper.cpp expectations.
        let (mut samples, spec) = get_samples_from_wav_reader(r)?;

        // Optionally apply VAD in-place by zeroing out non-speech regions.
        // If VAD finds no speech, we exit successfully with no output.
        if opts.enable_voice_activity_detection {
            let found_speech = apply_vad(&self.vad_model_path, &spec, &mut samples)?;
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

    /// Access the underlying Whisper context.
    ///
    /// This is primarily intended for advanced or experimental use-cases.
    pub fn context(&self) -> &WhisperContext {
        &self.ctx
    }
}
