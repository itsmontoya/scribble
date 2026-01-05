use crate::Result;
use crate::segments::Segment;

/// A streaming encoder for transcription segments.
///
/// We use this trait to decouple *segment production* (Whisper/VAD/etc.)
/// from *segment presentation* (JSON, VTT, plain text, ...).
///
/// Lifecycle:
/// - We call `write_segment` zero or more times.
/// - We call `close` exactly once when we’re done.
/// - Implementations should treat `close()` as idempotent (safe to call multiple times).
///
/// Streaming considerations:
/// - Implementations should write incrementally and avoid buffering the full output
///   whenever possible.
/// - `close()` is where we finalize output (e.g., write trailing brackets, flush writers).
pub trait SegmentEncoder {
    /// Encode and write a single segment.
    fn write_segment(&mut self, seg: &Segment) -> Result<()>;

    /// Finalize the encoded output and flush any buffered data.
    ///
    /// We prefer `close()` to be idempotent so callers can safely call it in cleanup paths.
    fn close(&mut self) -> Result<()>;
}
