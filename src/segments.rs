use anyhow::{Context, Result};
use serde::Serialize;
use whisper_rs::{FullParams, SamplingStrategy, WhisperContext, WhisperSegment, WhisperState};

#[derive(Debug, Serialize, Clone)]
pub struct Segment {
    pub start_seconds: f32,
    pub end_seconds: f32,
    pub text: String,
    pub language_code: String,
}

pub fn get_segments(ctx: &WhisperContext, samples: &mut [f32]) -> Result<Vec<Segment>> {
    let state = process(ctx, samples)?;
    let mut segments: Vec<Segment> = Vec::new();
    for segment in state.as_iter() {
        let s = get_segment(segment)?;
        segments.push(s);
    }

    Ok(segments)
}

pub fn get_segment(segment: WhisperSegment) -> Result<Segment> {
    // ms → seconds (whisper timestamps are ms)
    let start_sec = segment.start_timestamp() as f32 / 1000.0;
    let end_sec = segment.end_timestamp() as f32 / 1000.0;
    let text = segment
        .to_str() // Result<&str, WhisperError>
        .context("failed to get segment text")? // -> &str or bail
        .to_owned(); // &str -> String

    let s = Segment {
        start_seconds: start_sec,
        end_seconds: end_sec,
        text,
        language_code: "none".to_owned(),
    };

    Ok(s)
}

fn get_params() -> FullParams<'static, 'static> {
    let mut params = FullParams::new(SamplingStrategy::BeamSearch {
        beam_size: 5,
        patience: 1.0,
    });

    params.set_n_threads(num_cpus::get() as i32);
    params.set_translate(false);
    params.set_language(None);
    params.set_no_context(true);
    params.set_single_segment(false);
    params.set_print_progress(false);
    params.set_print_special(false);
    params.set_print_realtime(false);
    params.set_print_timestamps(false);
    return params;
}

fn process(ctx: &WhisperContext, samples: &mut [f32]) -> Result<WhisperState> {
    let params = get_params();

    // create_state() already returns a Result, so propagate using ? instead of expect()
    let mut state = ctx
        .create_state()
        .context("failed to create whisper state")?;

    // whisper_rs::WhisperState::full() returns Result<(), WhisperError>
    // so again, propagate properly using ?
    state
        .full(params, samples)
        .context("failed to run whisper full()")?;

    Ok(state)
}
