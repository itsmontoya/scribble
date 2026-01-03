use std::path::Path;

use anyhow::{Context, Result, ensure};
use hound::WavSpec;
use whisper_rs::{WhisperContext, WhisperVadContext, WhisperVadContextParams};

use crate::audio_pipeline::TARGET_SAMPLE_RATE;
use crate::backend::{Backend, BackendStream};
use crate::decoder::SamplesSink;
use crate::opts::Opts;
use crate::segment_encoder::SegmentEncoder;

mod ctx;
mod incremental;
mod logging;
mod segments;
mod token;
mod vad;

use incremental::BufferedSegmentTranscriber;
use segments::emit_segments;
use vad::to_speech_only;

/// Built-in backend powered by `whisper-rs` / `whisper.cpp`.
pub struct WhisperBackend {
    ctx: WhisperContext,
    vad_ctx: WhisperVadContext,
    vad_model_path: String,
}

/// Streaming state for [`WhisperBackend`].
pub struct WhisperStream<'a> {
    inner: BufferedSegmentTranscriber<'a>,
}

impl SamplesSink for WhisperStream<'_> {
    fn on_samples(&mut self, samples_16k_mono: &[f32]) -> Result<bool> {
        self.inner.on_samples(samples_16k_mono)
    }
}

impl BackendStream for WhisperStream<'_> {
    fn finish(&mut self) -> Result<()> {
        self.inner.finish()
    }
}

impl WhisperBackend {
    /// Load a whisper.cpp model and a Whisper VAD model from disk and initialize a backend.
    pub fn new(model_path: &str, vad_model_path: &str) -> Result<Self> {
        ensure!(!model_path.trim().is_empty(), "model path must be provided");
        ensure!(
            !vad_model_path.trim().is_empty(),
            "VAD model path must be provided"
        );

        let vad_path = Path::new(vad_model_path);
        ensure!(vad_path.exists(), "VAD model not found at '{}'", vad_model_path);
        ensure!(
            vad_path.is_file(),
            "VAD model path is not a file: '{}'",
            vad_model_path
        );

        let ctx = ctx::get_context(model_path)?;

        // Load the VAD model once so repeated transcriptions don't re-initialize it.
        let vad_ctx_params = WhisperVadContextParams::default();
        let vad_ctx = WhisperVadContext::new(vad_model_path, vad_ctx_params)
            .with_context(|| format!("failed to load VAD model from '{}'", vad_model_path))?;

        Ok(Self {
            ctx,
            vad_ctx,
            vad_model_path: vad_model_path.to_owned(),
        })
    }

    /// Access the underlying Whisper context.
    pub fn context(&self) -> &WhisperContext {
        &self.ctx
    }

    /// Access the configured VAD model path.
    pub fn vad_model_path(&self) -> &str {
        &self.vad_model_path
    }
}

impl Backend for WhisperBackend {
    type Stream<'a>
        = WhisperStream<'a>
    where
        Self: 'a;

    fn requires_full_audio(&self, opts: &Opts) -> bool {
        opts.enable_voice_activity_detection
    }

    fn transcribe_full(
        &mut self,
        opts: &Opts,
        encoder: &mut dyn SegmentEncoder,
        samples: &[f32],
    ) -> Result<()> {
        if samples.is_empty() {
            return Ok(());
        }

        if !opts.enable_voice_activity_detection {
            return emit_segments(&self.ctx, opts, samples, &mut |seg| encoder.write_segment(seg));
        }

        // NOTE: VAD is currently **not** supported in the streaming/incremental flow.
        //
        // Today we run VAD by decoding the *entire* input into a contiguous buffer,
        // applying VAD in-place, then running a single Whisper pass.
        let mut samples = samples.to_vec();

        let spec = WavSpec {
            channels: 1,
            sample_rate: TARGET_SAMPLE_RATE,
            bits_per_sample: 16,
            sample_format: hound::SampleFormat::Int,
        };

        let found_speech = to_speech_only(&mut self.vad_ctx, &spec, &mut samples)?;
        if !found_speech {
            return Ok(());
        }

        emit_segments(&self.ctx, opts, &samples, &mut |seg| encoder.write_segment(seg))
    }

    fn create_stream<'a>(
        &'a mut self,
        opts: &'a Opts,
        encoder: &'a mut dyn SegmentEncoder,
    ) -> Result<Self::Stream<'a>> {
        ensure!(
            !opts.enable_voice_activity_detection,
            "streaming transcription does not currently support VAD; disable `enable_voice_activity_detection` or run in full-audio mode"
        );
        Ok(WhisperStream {
            inner: BufferedSegmentTranscriber::new(&self.ctx, opts, encoder),
        })
    }
}

