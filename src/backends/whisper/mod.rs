use std::path::Path;

use anyhow::{Result, ensure};
use whisper_rs::WhisperContext;

use crate::backend::{Backend, BackendStream};
use crate::decoder::SamplesSink;
use crate::opts::Opts;
use crate::segment_encoder::SegmentEncoder;

mod ctx;
mod incremental;
mod logging;
mod segments;
mod token;

use incremental::BufferedSegmentTranscriber;
use segments::emit_segments;

/// Built-in backend powered by `whisper-rs` / `whisper.cpp`.
pub struct WhisperBackend {
    ctx: WhisperContext,
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

        Ok(Self {
            ctx,
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

    fn transcribe_full(
        &mut self,
        opts: &Opts,
        encoder: &mut dyn SegmentEncoder,
        samples: &[f32],
    ) -> Result<()> {
        if samples.is_empty() {
            return Ok(());
        }

        // VAD workflow is temporarily disabled while the streaming-focused version is reworked.
        let _ = opts.enable_voice_activity_detection;
        emit_segments(&self.ctx, opts, samples, &mut |seg| encoder.write_segment(seg))
    }

    fn create_stream<'a>(
        &'a mut self,
        opts: &'a Opts,
        encoder: &'a mut dyn SegmentEncoder,
    ) -> Result<Self::Stream<'a>> {
        // VAD workflow is temporarily disabled while the streaming-focused version is reworked.
        let _ = opts.enable_voice_activity_detection;
        Ok(WhisperStream {
            inner: BufferedSegmentTranscriber::new(&self.ctx, opts, encoder),
        })
    }
}
