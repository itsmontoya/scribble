use anyhow::Result;

use crate::decoder::SamplesSink;
use crate::opts::Opts;
use crate::segment_encoder::SegmentEncoder;

/// Pluggable ASR backend used by [`crate::scribble::Scribble`].
///
/// A backend is responsible for turning 16kHz mono `f32` samples into `Segment`s written via a
/// [`SegmentEncoder`].
///
/// Backends may choose to implement streaming/incremental emission by returning a stream object
/// that implements [`BackendStream`].
pub trait Backend {
    /// Streaming transcription state for this backend.
    ///
    /// The lifetime typically ties together:
    /// - the backend borrow (`&'a mut self`)
    /// - the options borrow (`&'a Opts`)
    /// - the encoder borrow (`&'a mut dyn SegmentEncoder`)
    type Stream<'a>: BackendStream + 'a
    where
        Self: 'a;

    /// Run a non-streaming transcription pass over a contiguous sample buffer.
    ///
    /// Backends should not call `encoder.close()`; the caller is responsible for encoder lifecycle.
    fn transcribe_full(
        &mut self,
        opts: &Opts,
        encoder: &mut dyn SegmentEncoder,
        samples_16k_mono: &[f32],
    ) -> Result<()>;

    /// Create a streaming transcriber that accepts samples incrementally.
    ///
    /// Backends should not call `encoder.close()`; the caller is responsible for encoder lifecycle.
    fn create_stream<'a>(
        &'a mut self,
        opts: &'a Opts,
        encoder: &'a mut dyn SegmentEncoder,
    ) -> Result<Self::Stream<'a>>;
}

/// Streaming transcription interface returned by [`Backend::create_stream`].
pub trait BackendStream: SamplesSink {
    /// Flush and emit any final segments.
    fn finish(&mut self) -> Result<()>;
}
