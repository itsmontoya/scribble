use anyhow::{Context, Result};
use serde::Serialize;
use whisper_rs::{FullParams, SamplingStrategy, WhisperContext, WhisperSegment, WhisperState};

use crate::segment_encoder::SegmentEncoder;

/// A single transcription segment produced by Whisper.
///
/// We keep this struct format-focused:
/// - timestamps are in seconds (f32 is sufficient for typical subtitle timing)
/// - `text` is the raw segment text returned by Whisper
/// - `language_code` is included for forward compatibility (see notes below)
#[derive(Debug, Serialize, Clone)]
pub struct Segment {
    pub start_seconds: f32,
    pub end_seconds: f32,
    pub text: String,

    /// Language of the segment as a short code (e.g. "en", "es").
    ///
    /// Today we default this because we are not yet running per-segment language detection.
    /// When we add language detection, we can populate this field more accurately.
    pub language_code: String,
}

/// Our current placeholder language code.
///
/// We prefer `"und"` (“undetermined”) over `"none"` because it’s a common convention
/// in language tagging systems and makes the meaning obvious.
const DEFAULT_LANGUAGE_CODE: &str = "und";

/// Run whisper on the provided samples and stream segments into the given encoder.
///
/// Why this exists:
/// - We want one “happy path” that: runs whisper, converts segments into our `Segment` type,
///   and hands them off to an output encoder.
/// - Keeping this flow centralized makes it easier to add features later (timestamps modes,
///   language detection, alternative sampling strategies, etc.).
///
/// Encoder lifecycle:
/// - We call `write_segment` for each segment.
/// - We *always attempt* to call `close()` at the end, even if a write fails, so we can
///   produce well-formed output where possible (e.g. closing `]` for JSON arrays).
pub fn write_segments(
    ctx: &WhisperContext,
    encoder: &mut dyn SegmentEncoder,
    samples: &[f32],
) -> Result<()> {
    let state = run_whisper_full(ctx, samples)?;

    // We want “try our best to close” semantics.
    // If a segment write fails, we stop writing further segments, but we still try to close
    // the encoder to finalize output.
    let mut first_err: Option<anyhow::Error> = None;

    for whisper_segment in state.as_iter() {
        let segment = to_segment(whisper_segment)?;

        if let Err(err) = encoder.write_segment(&segment) {
            first_err = Some(err);
            break;
        }
    }

    // Always attempt to close, even if we encountered an earlier error.
    let close_res = encoder.close();

    match (first_err, close_res) {
        (None, Ok(())) => Ok(()),
        (None, Err(close_err)) => Err(close_err),
        (Some(err), Ok(())) => Err(err),
        (Some(err), Err(close_err)) => Err(err.context(close_err)),
    }
}

/// Convert a `WhisperSegment` from whisper.rs into our serializable `Segment` type.
///
/// Notes:
/// - whisper timestamps are milliseconds, so we convert to seconds for output.
/// - whisper returns text via `to_str()`, which can fail, so we attach context.
pub fn to_segment(segment: WhisperSegment) -> Result<Segment> {
    // ms → seconds (whisper timestamps are milliseconds)
    let start_seconds = segment.start_timestamp() as f32 / 1000.0;
    let end_seconds = segment.end_timestamp() as f32 / 1000.0;

    let text = segment
        .to_str()
        .context("failed to get segment text")?
        .to_owned();

    Ok(Segment {
        start_seconds,
        end_seconds,
        text,
        language_code: DEFAULT_LANGUAGE_CODE.to_owned(),
    })
}

/// Build the whisper “full” parameters we use for transcription.
///
/// We keep parameter construction in one function so it’s easy to reason about and adjust.
///
/// Current choices:
/// - Beam search for a bit more stability than greedy decoding.
/// - `no_context = true` because we currently want segments to stand alone.
///   (If we later want better continuity across segments, we can revisit this.)
fn build_full_params() -> FullParams<'static, 'static> {
    let mut params = FullParams::new(SamplingStrategy::BeamSearch {
        beam_size: 5,
        patience: 1.0,
    });

    // We default to using all CPUs. If we later want to expose this as a CLI flag,
    // we can plumb it through here.
    params.set_n_threads(num_cpus::get() as i32);

    // Transcription (not translation). If we add translation output types later,
    // this can become configurable.
    params.set_translate(false);

    // Let whisper auto-detect language for the run (we’re not doing per-segment language yet).
    params.set_language(None);

    // Keep segments independent for now.
    params.set_no_context(true);

    // Allow multiple segments.
    params.set_single_segment(false);

    // We silence whisper’s internal printing; our crate controls output via encoders.
    params.set_print_progress(false);
    params.set_print_special(false);
    params.set_print_realtime(false);
    params.set_print_timestamps(false);

    params
}

/// Run whisper’s `full()` pipeline over the provided samples and return the resulting state.
///
/// We return the `WhisperState` so callers can iterate segments without forcing us
/// to allocate a separate list upfront.
fn run_whisper_full(ctx: &WhisperContext, samples: &[f32]) -> Result<WhisperState> {
    let params = build_full_params();

    let mut state = ctx
        .create_state()
        .context("failed to create whisper state")?;

    state
        .full(params, samples)
        .context("failed to run whisper full()")?;

    Ok(state)
}
