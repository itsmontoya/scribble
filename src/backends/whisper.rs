use anyhow::Result;
use whisper_rs::WhisperContext;

use crate::backend::{Backend, BackendStream};
use crate::ctx::get_context;
use crate::decoder::SamplesSink;
use crate::incremental::BufferedSegmentTranscriber;
use crate::logging::init_whisper_logging;
use crate::opts::Opts;
use crate::segment_encoder::SegmentEncoder;
use crate::segments::emit_segments;

/// Built-in backend powered by `whisper-rs` / `whisper.cpp`.
pub struct WhisperBackend {
    ctx: WhisperContext,
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
    /// Load a whisper.cpp model from disk and initialize a backend.
    pub fn new(model_path: impl AsRef<str>) -> Result<Self> {
        // Whisper can be very chatty; keep it quiet by default.
        // This function is idempotent (safe to call multiple times).
        init_whisper_logging();

        let ctx = get_context(model_path.as_ref())?;
        Ok(Self { ctx })
    }

    /// Access the underlying Whisper context.
    pub fn context(&self) -> &WhisperContext {
        &self.ctx
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
        samples_16k_mono: &[f32],
    ) -> Result<()> {
        emit_segments(&self.ctx, opts, samples_16k_mono, &mut |seg| {
            encoder.write_segment(seg)
        })
    }

    fn create_stream<'a>(
        &'a mut self,
        opts: &'a Opts,
        encoder: &'a mut dyn SegmentEncoder,
    ) -> Result<Self::Stream<'a>> {
        Ok(WhisperStream {
            inner: BufferedSegmentTranscriber::new(&self.ctx, opts, encoder),
        })
    }
}
