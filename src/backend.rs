use crate::Result;
use crate::opts::Opts;
use crate::segment_encoder::SegmentEncoder;

/// Pluggable ASR backend used by [`crate::Scribble`].
///
/// A backend is responsible for turning mono `f32` samples at Scribble's target sample rate into `Segment`s written via a
/// [`SegmentEncoder`].
///
/// Backends may choose to implement streaming/incremental emission by returning a stream object
/// that implements [`BackendStream`].
pub trait Backend {
    /// Streaming transcription state for this backend.
    ///
    /// The lifetime typically ties together:
    /// - the backend borrow (`&'a self`)
    /// - the options borrow (`&'a Opts`)
    /// - the encoder borrow (`&'a mut dyn SegmentEncoder`)
    type Stream<'a>: BackendStream + 'a
    where
        Self: 'a;

    /// Run a non-streaming transcription pass over a contiguous sample buffer.
    ///
    /// Backends should not call `encoder.close()`; the caller is responsible for encoder lifecycle.
    fn transcribe_full(
        &self,
        opts: &Opts,
        encoder: &mut dyn SegmentEncoder,
        samples: &[f32],
    ) -> Result<()>;

    /// Create a streaming transcriber that accepts samples incrementally.
    ///
    /// Backends should not call `encoder.close()`; the caller is responsible for encoder lifecycle.
    fn create_stream<'a>(
        &'a self,
        opts: &'a Opts,
        encoder: &'a mut dyn SegmentEncoder,
    ) -> Result<Self::Stream<'a>>;
}

/// Streaming transcription interface returned by [`Backend::create_stream`].
pub trait BackendStream {
    /// Consume a new chunk of mono `f32` samples at Scribble's target sample rate.
    ///
    /// Returning `Ok(false)` signals "stop early".
    fn on_samples(&mut self, samples_16k_mono: &[f32]) -> Result<bool>;

    /// Flush and emit any final segments.
    fn finish(&mut self) -> Result<()>;
}
