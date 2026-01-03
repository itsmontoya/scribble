use crate::token::Token;
use serde::Serialize;

/// A single transcription segment produced by an ASR backend.
///
/// We keep this struct format-focused:
/// - timestamps are in seconds (f32 is sufficient for typical subtitle timing)
/// - `text` is the raw segment text returned by the backend
/// - `language_code` is included for forward compatibility (see notes below)
#[derive(Debug, Serialize, Clone)]
pub struct Segment {
    pub start_seconds: f32,
    pub end_seconds: f32,
    pub text: String,

    /// Tokens that make up this segment.
    ///
    /// We include token-level timing and probabilities so consumers can build
    /// detailed overlays or custom renderers without re-tokenizing.
    pub tokens: Vec<Token>,

    /// Language of the segment as a short code (e.g. "en", "es").
    ///
    /// Today we default this because we are not yet running per-segment language detection.
    /// When we add language detection, we can populate this field more accurately.
    pub language_code: String,

    /// True if the next segment begins a new speaker turn.
    ///
    /// Populated from `WhisperSegment::next_segment_speaker_turn()` so downstream
    /// encoders/UIs can insert speaker breaks without re-deriving this signal.
    pub next_speaker_turn: bool,
}
