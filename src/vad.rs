// src/vad.rs

use anyhow::{Result, anyhow};
use hound::WavSpec;
use whisper_rs::{WhisperVadContext, WhisperVadParams, WhisperVadSegments};

/// Run VAD to identify speech segments, then replace the sample buffer with speech-only audio.
///
/// What this does:
/// - Runs VAD on the input buffer and gets a list of speech time ranges.
/// - Applies a policy (padding, minimum duration, and gap merging).
/// - Concatenates only the speech regions into a new buffer.
/// - Replaces `samples` with that speech-only buffer.
///
/// Return value:
/// - `Ok(true)`  => at least one speech segment survived filtering and `samples` was replaced
/// - `Ok(false)` => no speech was detected (or everything was filtered out)
///
/// Design note:
/// - This *does* change buffer length (unlike our "zero-out non-speech" approach).
/// - That means timestamps from Whisper will refer to the new, speech-only timeline.
///   That's fine for "just get me the words" pipelines, but not ideal if you need
///   timestamps aligned to the original media.
pub fn to_speech_only(
    ctx: &mut WhisperVadContext,
    spec: &WavSpec,
    samples: &mut Vec<f32>,
) -> Result<bool> {
    to_speech_only_with_policy(ctx, spec, samples, DEFAULT_VAD_POLICY)
}

/// Same as [`to_speech_only`] but allows a custom policy.
pub fn to_speech_only_with_policy(
    ctx: &mut WhisperVadContext,
    spec: &WavSpec,
    samples: &mut Vec<f32>,
    policy: VadPolicy,
) -> Result<bool> {
    // Build VAD parameters from defaults and apply our policy knobs.
    let mut vad_params = WhisperVadParams::default();

    // We cap max speech duration to keep VAD from producing extremely long segments.
    // (This value is in milliseconds in whisper_rs.)
    vad_params.set_max_speech_duration(15_000.0);

    // Some whisper_rs versions expose these setters. If yours doesn't, remove them
    // and keep the filtering logic in `extract_speech_with_policy`.
    vad_params.set_threshold(policy.threshold);
    vad_params.set_min_speech_duration(policy.min_speech_ms as i32);

    // Run VAD to produce segments from our sample buffer.
    let segments = ctx.segments_from_samples(vad_params, samples)?;

    // Extract speech windows (with padding/merge policy) and replace the full buffer.
    let Some(speech_only) = extract_speech_with_policy(spec, &segments, samples, policy)? else {
        return Ok(false);
    };

    *samples = speech_only;
    Ok(!samples.is_empty())
}

/// Extract speech samples from `samples` according to `segments` and `policy`.
///
/// Returns:
/// - `Ok(Some(vec))` when one or more ranges are selected
/// - `Ok(None)` when no ranges are selected
fn extract_speech_with_policy(
    spec: &WavSpec,
    segments: &WhisperVadSegments,
    samples: &[f32],
    policy: VadPolicy,
) -> Result<Option<Vec<f32>>> {
    let n = segments.num_segments();
    if n == 0 {
        // No speech detected.
        return Ok(None);
    }

    // Convert VAD timestamps into sample indices using our WAV sample rate.
    let sample_rate = spec.sample_rate as f32;

    // Convert policy values from ms → samples once.
    let pre_pad_samples = ms_to_samples(policy.pre_pad_ms, sample_rate);
    let post_pad_samples = ms_to_samples(policy.post_pad_ms, sample_rate);
    let min_speech_samples = ms_to_samples(policy.min_speech_ms, sample_rate);
    let gap_merge_samples = ms_to_samples(policy.gap_merge_ms, sample_rate);

    // Collect padded ranges, then merge overlaps / near-gaps.
    //
    // Invariant: `ranges` stays sorted and non-overlapping after the merge step.
    let mut ranges: Vec<(usize, usize)> = Vec::new();

    for i in 0..n {
        let (mut start_idx, mut end_idx) =
            segment_sample_indexes(segments, i, sample_rate, samples.len())?;

        // Drop very short speech segments according to policy.
        let dur = end_idx.saturating_sub(start_idx);
        if dur < min_speech_samples {
            continue;
        }

        // Apply padding in samples.
        start_idx = start_idx.saturating_sub(pre_pad_samples);
        end_idx = (end_idx + post_pad_samples).min(samples.len());

        if start_idx >= end_idx {
            continue;
        }

        // Merge with previous if overlapping or gap is small.
        if let Some((_, prev_end)) = ranges.last_mut() {
            let gap = start_idx.saturating_sub(*prev_end);
            if start_idx <= *prev_end || gap <= gap_merge_samples {
                *prev_end = (*prev_end).max(end_idx);
                continue;
            }
        }

        ranges.push((start_idx, end_idx));
    }

    if ranges.is_empty() {
        return Ok(None);
    }

    // Build the concatenated output buffer.
    //
    // We precompute capacity to minimize reallocations while we extend.
    let total_len: usize = ranges.iter().map(|(s, e)| e - s).sum();
    let mut out = Vec::with_capacity(total_len);

    for (s, e) in ranges {
        out.extend_from_slice(&samples[s..e]);
    }

    Ok(Some(out))
}

/// Convert milliseconds → number of samples at `sample_rate`.
///
/// We round to the nearest sample so padding is stable across rates.
fn ms_to_samples(ms: u32, sample_rate: f32) -> usize {
    ((ms as f32 / 1000.0) * sample_rate).round() as usize
}

/// Convert the i'th VAD segment into `(start_idx, end_idx)` sample indices.
///
/// whisper_rs VAD timestamps are in centiseconds (10ms units), so we convert:
/// - centiseconds → seconds
/// - seconds → samples
///
/// Index rounding policy:
/// - We `floor()` the start index so we include the first speech sample.
/// - We `ceil()` the end index so we include the last speech sample.
///
/// We clamp indices into `[0 .. samples_len]` so slicing is always safe.
fn segment_sample_indexes(
    segments: &WhisperVadSegments,
    i: i32,
    sample_rate: f32,
    samples_len: usize,
) -> Result<(usize, usize)> {
    // Timestamps are in centiseconds (10ms units).
    let start_cs = segments
        .get_segment_start_timestamp(i)
        .ok_or_else(|| anyhow!("missing start timestamp for VAD segment {i}"))?;

    let end_cs = segments
        .get_segment_end_timestamp(i)
        .ok_or_else(|| anyhow!("missing end timestamp for VAD segment {i}"))?;

    // Convert to seconds.
    let start_sec = start_cs / 100.0;
    let end_sec = end_cs / 100.0;

    // Convert seconds → sample indices.
    let mut start_idx = (start_sec * sample_rate).floor() as usize;
    let mut end_idx = (end_sec * sample_rate).ceil() as usize;

    // Clamp into range so slicing is safe.
    start_idx = start_idx.min(samples_len);
    end_idx = end_idx.min(samples_len);

    // Be defensive: ensure we never produce an inverted range.
    if end_idx < start_idx {
        end_idx = start_idx;
    }

    Ok((start_idx, end_idx))
}

/// Policy knobs for speech extraction.
///
/// These values are intentionally simple and expressed in human-friendly units (ms),
/// and are converted to sample counts using the current WAV sample rate.
#[derive(Debug, Clone, Copy)]
pub struct VadPolicy {
    /// VAD confidence threshold (higher = more conservative).
    pub threshold: f32,

    /// Padding to include before each speech segment.
    pub pre_pad_ms: u32,

    /// Padding to include after each speech segment.
    pub post_pad_ms: u32,

    /// Drop speech segments shorter than this duration.
    pub min_speech_ms: u32,

    /// Merge adjacent segments if the gap between them is <= this duration.
    pub gap_merge_ms: u32,

    pub non_speech_gain: f32,
}

/// Whisper-friendly, conservative policy.
///
/// Optimized for transcription quality over aggressiveness:
/// - generous padding reduces clipped words
/// - small gap merge reduces over-fragmentation
pub const DEFAULT_VAD_POLICY: VadPolicy = VadPolicy {
    threshold: 0.2,
    pre_pad_ms: 600,
    post_pad_ms: 1200,
    min_speech_ms: 40,
    gap_merge_ms: 500,
    non_speech_gain: 0.1,
};
