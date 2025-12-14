use anyhow::{Result, anyhow};
use hound::WavSpec;
use whisper_rs::{
    WhisperVadContext, WhisperVadContextParams, WhisperVadParams, WhisperVadSegments,
};

pub fn apply_vad(vad_model_path: &str, spec: &WavSpec, samples: &mut [f32]) -> Result<bool> {
    let mut vad_params = WhisperVadParams::default();
    vad_params.set_max_speech_duration(15000.0);

    let vad_ctx_params: WhisperVadContextParams = WhisperVadContextParams::default();
    let mut vad_ctx = WhisperVadContext::new(vad_model_path, vad_ctx_params)?;
    let segments = vad_ctx.segments_from_samples(vad_params, samples)?;

    let sample_rate = spec.sample_rate as f32;
    let mut last_end_idx: usize = 0;

    let n = segments.num_segments();

    if n == 0 {
        return Ok(false);
    }

    for i in 0..n {
        let (start_idx, end_idx) = get_indexes(samples, &segments, i, sample_rate)?;
        // zero *before* this speech segment (gap from last_end_idx..start_idx)
        if start_idx > last_end_idx {
            zero_out_samples(&mut samples[last_end_idx..start_idx]);
        }

        // keep [start_idx..end_idx] as-is (speech)
        if end_idx > last_end_idx {
            last_end_idx = end_idx;
        }
    }

    // zero everything *after* the last speech segment
    if last_end_idx < samples.len() {
        zero_out_samples(&mut samples[last_end_idx..]);
    }

    Ok(true)
}

fn get_indexes(
    samples: &mut [f32],
    segments: &WhisperVadSegments,
    i: i32,
    sample_rate: f32,
) -> Result<(usize, usize)> {
    // timestamps in *centiseconds* (10ms units)
    let start_cs = segments
        .get_segment_start_timestamp(i)
        .ok_or_else(|| anyhow!("missing start timestamp for segment {i}"))?;

    let end_cs = segments
        .get_segment_end_timestamp(i)
        .ok_or_else(|| anyhow!("missing end timestamp for segment {i}"))?;

    // convert to seconds
    let start_sec = start_cs / 100.0;
    let end_sec = end_cs / 100.0;

    // seconds -> sample indices
    let mut start_idx = (start_sec * sample_rate).floor() as usize;
    let mut end_idx = (end_sec * sample_rate).ceil() as usize;

    // clamp into range
    let len = samples.len();
    start_idx = start_idx.min(len);
    end_idx = end_idx.min(len);
    return Ok((start_idx, end_idx));
}

fn zero_out_samples(samples: &mut [f32]) {
    for s in samples {
        *s = 0.0;
    }
}
