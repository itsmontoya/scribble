use anyhow::{Context, Result};
use whisper_rs::{FullParams, SamplingStrategy, WhisperContext, WhisperSegment, WhisperState};

use crate::opts::Opts;
use crate::segments::Segment;
use crate::token::centiseconds_to_seconds;

use super::token::tokens_from_segment;

/// Our current placeholder language code.
///
/// Prefers `"und"` (“undetermined”) over `"none"` because it’s a common convention
/// in language tagging systems and makes the meaning obvious.
const DEFAULT_LANGUAGE_CODE: &str = "und";

pub(super) fn emit_segments(
    ctx: &WhisperContext,
    opts: &Opts,
    samples: &[f32],
    on_segment: &mut dyn FnMut(&Segment) -> Result<()>,
) -> Result<()> {
    let state = run_whisper_full(ctx, opts, samples)?;
    for whisper_segment in state.as_iter() {
        let segment = to_segment(whisper_segment)?;
        on_segment(&segment)?;
    }
    Ok(())
}

pub(super) fn to_segment(segment: WhisperSegment) -> Result<Segment> {
    let text = segment
        .to_str()
        .context("failed to get segment text")?
        .to_owned();

    let tokens = tokens_from_segment(&segment)?;

    // Prefer token-derived timing when available to avoid long segments that include
    // leading/trailing silence. Fall back to whisper’s segment-level timestamps when token
    // timing is unavailable.
    let (start_seconds, end_seconds) = segment_seconds_from_tokens_or_fallback(&segment, &tokens);

    Ok(Segment {
        start_seconds,
        end_seconds,
        text,
        tokens,
        language_code: DEFAULT_LANGUAGE_CODE.to_owned(),
        next_speaker_turn: segment.next_segment_speaker_turn(),
    })
}

fn segment_seconds_from_tokens_or_fallback(
    segment: &WhisperSegment,
    tokens: &[crate::token::Token],
) -> (f32, f32) {
    let mut min_start: Option<f32> = None;
    let mut max_end: Option<f32> = None;

    for token in tokens {
        // Filter out whisper special/control tokens (commonly formatted like `[_BEG_]`, `[_TT_50]`).
        if token.text.starts_with("[_") && token.text.ends_with("_]") {
            continue;
        }

        // Skip tokens with unknown timestamps (whisper uses -1, clamped to 0.0).
        if token.start_seconds <= 0.0 && token.end_seconds <= 0.0 {
            continue;
        }

        min_start = Some(min_start.map_or(token.start_seconds, |v| v.min(token.start_seconds)));
        max_end = Some(max_end.map_or(token.end_seconds, |v| v.max(token.end_seconds)));
    }

    match (min_start, max_end) {
        (Some(s), Some(e)) if e >= s => (s, e),
        _ => (
            centiseconds_to_seconds(segment.start_timestamp()),
            centiseconds_to_seconds(segment.end_timestamp()),
        ),
    }
}

fn build_full_params(opts: &Opts) -> FullParams<'_, '_> {
    let mut params = FullParams::new(SamplingStrategy::BeamSearch {
        beam_size: 5,
        patience: 1.0,
    });

    params.set_n_threads(num_cpus::get() as i32);
    params.set_translate(opts.enable_translate_to_english);
    params.set_language(opts.language.as_deref());
    params.set_no_context(true);
    params.set_single_segment(false);

    params.set_print_progress(false);
    params.set_print_special(false);
    params.set_print_realtime(false);
    params.set_print_timestamps(false);

    params.set_token_timestamps(true);

    params
}

pub(super) fn run_whisper_full(
    ctx: &WhisperContext,
    opts: &Opts,
    samples: &[f32],
) -> Result<WhisperState> {
    let params = build_full_params(opts);

    let mut state = ctx
        .create_state()
        .context("failed to create whisper state")?;

    state
        .full(params, samples)
        .context("failed to run whisper full()")?;

    Ok(state)
}
