use anyhow::{Result, anyhow};
use hound::WavSpec;
use whisper_rs::{
    WhisperVadContext, WhisperVadContextParams, WhisperVadParams, WhisperVadSegments,
};

/// Apply voice activity detection (VAD) to the audio buffer in-place.
///
/// What we do:
/// - We run VAD and get back a list of "speech segments" (time ranges).
/// - We keep samples inside those speech segments untouched.
/// - We zero out everything outside speech segments (leading silence, gaps, trailing silence).
///
/// Return value:
/// - `Ok(true)`  => VAD found at least one speech segment and we modified the buffer
/// - `Ok(false)` => no speech segments were found (caller may choose to stop early)
///
/// Why this design:
/// - Many downstream pipelines prefer the original buffer length (timestamps remain stable),
///   so we zero out non-speech rather than physically removing samples.
pub fn apply_vad(vad_model_path: &str, spec: &WavSpec, samples: &mut [f32]) -> Result<bool> {
    // We start from defaults and then apply the specific policy we want.
    let mut vad_params = WhisperVadParams::default();

    // We cap max speech duration to keep VAD from producing extremely long segments.
    // (This value is in milliseconds in whisper_rs.)
    vad_params.set_max_speech_duration(15000.0);

    // Create a VAD context from the model on disk.
    // This may be somewhat expensive; if we later do streaming or batch processing,
    // we may want to reuse the context rather than recreate it per call.
    let vad_ctx_params = WhisperVadContextParams::default();
    let mut vad_ctx = WhisperVadContext::new(vad_model_path, vad_ctx_params)?;

    // Run VAD to produce segments from our sample buffer.
    let segments = vad_ctx.segments_from_samples(vad_params, samples)?;

    let n = segments.num_segments();
    if n == 0 {
        // No speech detected. We return `false` so callers can exit early.
        return Ok(false);
    }

    // Convert VAD timestamps into sample indices using our WAV sample rate.
    let sample_rate = spec.sample_rate as f32;

    // We track the end of the last speech region so we can zero out gaps between speech.
    let mut last_end_idx: usize = 0;

    for i in 0..n {
        let (start_idx, end_idx) =
            segment_sample_indexes(&segments, i, sample_rate, samples.len())?;

        // Zero the gap before this speech segment: [last_end_idx .. start_idx)
        if start_idx > last_end_idx {
            zero_out_samples(&mut samples[last_end_idx..start_idx]);
        }

        // Keep speech untouched: [start_idx .. end_idx)
        //
        // We only advance `last_end_idx` forward (it should already be monotonic, but this
        // makes the logic robust if VAD ever returns overlapping or out-of-order segments).
        last_end_idx = last_end_idx.max(end_idx);
    }

    // Zero everything after the final speech segment: [last_end_idx .. end)
    if last_end_idx < samples.len() {
        zero_out_samples(&mut samples[last_end_idx..]);
    }

    Ok(true)
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

    Ok((start_idx, end_idx))
}

/// Zero out the provided samples in-place.
///
/// We do this (instead of removing samples) so the buffer length stays the same,
/// which keeps timestamps stable for downstream processing.
fn zero_out_samples(samples: &mut [f32]) {
    for s in samples {
        *s = 0.0;
    }
}
