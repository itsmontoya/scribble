use anyhow::{Context, Result};
use whisper_rs::WhisperSegment;

use crate::token::{Token, centiseconds_to_seconds};

pub(super) fn tokens_from_segment(segment: &WhisperSegment) -> Result<Vec<Token>> {
    let token_count = segment.n_tokens();
    let token_count_usize = usize::try_from(token_count)
        .with_context(|| format!("segment reported negative token count: {token_count}"))?;
    let mut tokens = Vec::with_capacity(token_count_usize);

    for token_idx in 0..token_count_usize {
        let token_idx_i32 = token_idx as i32;
        let token = segment
            .get_token(token_idx_i32)
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

