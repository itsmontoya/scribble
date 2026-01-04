use anyhow::{Result, anyhow};
use whisper_rs::{WhisperVadContext, WhisperVadParams, WhisperVadSegments};

/// Voice Activity Detection (VAD) helpers.
///
/// Current behavior:
/// - Run VAD to identify speech time ranges.
/// - Convert those ranges into sample index ranges (with padding / filtering / merging).
/// - Keep the original buffer length.
/// - Apply a gain to **non-speech** regions (0.0 = mute, 1.0 = unchanged).
///
/// Why this design:
/// - Preserves timeline alignment with the original media (useful for timestamps).
/// - Lets you keep faint room tone if desired (via `non_speech_gain`).
pub fn to_speech_only_with_policy(
    ctx: &mut WhisperVadContext,
    sample_rate_hz: u32,
    samples: &mut [f32],
    policy: VadPolicy,
) -> Result<bool> {
    // Build VAD parameters from defaults and apply our policy knobs.
    let mut vad_params = WhisperVadParams::default();

    // Cap max speech duration to avoid producing extremely long segments.
    // (This value is in seconds in whisper_rs / whisper.cpp.)
    vad_params.set_max_speech_duration(15.0);

    vad_params.set_threshold(policy.threshold);
    vad_params.set_min_speech_duration(policy.min_speech_ms as i32);

    // Run VAD to produce segments from our sample buffer.
    let segments = ctx.segments_from_samples(vad_params, samples)?;

    // Convert segments -> merged/filtered/padded sample ranges.
    let Some(ranges) = speech_ranges_with_policy(sample_rate_hz, &segments, samples, policy)?
    else {
        return Ok(false);
    };

    // Attenuate non-speech regions in-place (preserves buffer length).
    apply_non_speech_gain_in_place(samples, &ranges, policy.non_speech_gain);
    Ok(true)
}

/// Compute speech ranges (sample indices) according to `segments` and `policy`.
///
/// Returns:
/// - `Ok(Some(ranges))` when one or more ranges are selected
/// - `Ok(None)` when no ranges are selected
fn speech_ranges_with_policy(
    sample_rate_hz: u32,
    segments: &WhisperVadSegments,
    samples: &[f32],
    policy: VadPolicy,
) -> Result<Option<Vec<(usize, usize)>>> {
    let n = segments.num_segments();
    if n == 0 {
        return Ok(None);
    }

    let sample_rate = sample_rate_hz as f32;

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

    Ok(Some(ranges))
}

/// Apply gain to non-speech regions in-place, keeping speech untouched.
///
/// - `ranges` must be sorted and non-overlapping (the builder guarantees this).
/// - `gain` is clamped to [0.0, 1.0]
fn apply_non_speech_gain_in_place(samples: &mut [f32], ranges: &[(usize, usize)], gain: f32) {
    let gain = gain.clamp(0.0, 1.0);

    // If gain == 1.0, no change needed.
    if (gain - 1.0).abs() < f32::EPSILON {
        return;
    }

    let mut cursor = 0usize;

    for &(s, e) in ranges {
        // Defensively clamp to the buffer in case callers ever pass bad ranges.
        let s = s.min(samples.len());
        let e = e.min(samples.len());

        // Attenuate the gap before speech.
        if s > cursor {
            scale_samples(&mut samples[cursor..s], gain);
        }

        // Advance cursor to the end of this speech region.
        cursor = cursor.max(e);
    }

    // Attenuate everything after the last speech segment.
    if cursor < samples.len() {
        scale_samples(&mut samples[cursor..], gain);
    }
}

/// Multiply all samples by a gain factor.
fn scale_samples(buf: &mut [f32], gain: f32) {
    if gain == 0.0 {
        buf.fill(0.0);
        return;
    }

    for s in buf.iter_mut() {
        *s *= gain;
    }
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

    // Convert to seconds. (centiseconds = 1/100s)
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

/// Policy knobs for VAD range selection and non-speech handling.
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

    /// Merge speech segments separated by less than this gap.
    pub gap_merge_ms: u32,

    /// Gain applied to non-speech regions (0.0 = mute, 1.0 = unchanged).
    pub non_speech_gain: f32,
}

/// Default policy tuned for "keep speech, drop/attenuate silence".
pub const DEFAULT_VAD_POLICY: VadPolicy = VadPolicy {
    threshold: 0.5,
    pre_pad_ms: 250,
    post_pad_ms: 250,
    min_speech_ms: 250,
    gap_merge_ms: 300,
    non_speech_gain: 0.0,
};
