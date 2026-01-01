use serde::Serialize;

use anyhow::{Context, Result};
use whisper_rs::WhisperSegment;

/// A single token produced by Whisper.
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

/// Extract tokens (with timing and probabilities) from a Whisper segment.
pub fn tokens_from_segment(segment: &WhisperSegment) -> Result<Vec<Token>> {
    let token_count = segment.n_tokens();
    let mut tokens = Vec::with_capacity(token_count as usize);

    for token_idx in 0..token_count {
        let token = segment
            .get_token(token_idx)
            .context("failed to get token from segment")?;

        let data = token.token_data();
        let text = token
            .to_str()
            .with_context(|| format!("failed to get token text at index {token_idx}"))?
            .to_owned();

        tokens.push(Token {
            // whisper uses -1 for unknown; clamp to 0 so consumers don't see -0.01s
            start_seconds: centiseconds_to_seconds(data.t0),
            end_seconds: centiseconds_to_seconds(data.t1),
            text,
            probability: data.p,
        });
    }

    Ok(tokens)
}

fn centiseconds_to_seconds(value: i64) -> f32 {
    if value < 0 { 0.0 } else { value as f32 / 100.0 }
}
