use crate::Result;
use crate::segments::Segment;

/// A streaming encoder for transcription segments.
///
/// Decouples *segment production* (Whisper/VAD/etc.) from *segment presentation*
/// (JSON, VTT, plain text, ...).
///
/// Lifecycle:
/// - Call `write_segment` zero or more times.
/// - Call `close` exactly once when done.
/// - Implementations should treat `close()` as idempotent (safe to call multiple times).
///
/// Streaming considerations:
/// - Implementations should write incrementally and avoid buffering the full output
///   whenever possible.
/// - `close()` is where output is finalized (e.g., write trailing brackets, flush writers).
pub trait SegmentEncoder {
    /// Encode and write a single segment.
    fn write_segment(&mut self, seg: &Segment) -> Result<()>;

    /// Finalize the encoded output and flush any buffered data.
    ///
    /// Prefer `close()` to be idempotent so callers can safely call it in cleanup paths.
    fn close(&mut self) -> Result<()>;
}
