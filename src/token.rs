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

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn centiseconds_to_seconds_clamps_negative_values() {
        assert_eq!(centiseconds_to_seconds(-1), 0.0);
        assert_eq!(centiseconds_to_seconds(-100), 0.0);
    }

    #[test]
    fn centiseconds_to_seconds_converts_to_fractional_seconds() {
        assert_eq!(centiseconds_to_seconds(0), 0.0);
        assert_eq!(centiseconds_to_seconds(1), 0.01);
        assert_eq!(centiseconds_to_seconds(250), 2.5);
    }
}
