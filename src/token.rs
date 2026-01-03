use serde::Serialize;

/// A single token produced by an ASR backend.
#[derive(Debug, Serialize, Clone)]
pub struct Token {
    /// Start time in seconds (whisper returns centiseconds).
    pub start_seconds: f32,
    /// End time in seconds (whisper returns centiseconds).
    pub end_seconds: f32,
    /// Token text.
    pub text: String,
    /// Probability assigned to this token.
    pub probability: f32,
}

pub(crate) fn centiseconds_to_seconds(value: i64) -> f32 {
    if value < 0 { 0.0 } else { value as f32 / 100.0 }
}
