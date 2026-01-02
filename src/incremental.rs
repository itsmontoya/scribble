//! Incremental, segment-driven transcription over a growing sample buffer.
//!
//! This module provides a small adapter that:
//! - buffers decoded 16 kHz mono samples
//! - runs Whisper once the buffer reaches a minimum duration
//! - emits the first completed segment
//! - advances the buffer by that segment’s end timestamp

use anyhow::{Context, Result, ensure};
use std::sync::OnceLock;
use std::time::Instant;
use whisper_rs::WhisperContext;

use crate::audio_pipeline::WHISPER_SAMPLE_RATE;
use crate::decoder::SamplesSink;
use crate::opts::Opts;
use crate::segment_encoder::SegmentEncoder;
use crate::segments::{run_whisper_full, to_segment};

/// Default minimum buffer duration before running Whisper.
///
/// This is intentionally conservative; the buffer grows until Whisper produces at least one
/// segment, at which point we emit that segment and advance.
const DEFAULT_MIN_BUFFER_SECONDS: usize = 1;

/// Maximum buffer size before we force progress.
///
/// If Whisper keeps returning <= 1 segment, we keep accumulating audio until this cap and then
/// emit whatever we have to avoid unbounded memory growth.
const DEFAULT_MAX_BUFFER_SECONDS: usize = 30;

/// Max exponential backoff when Whisper makes no progress (in multiples of `min_window_samples`).
const MAX_BACKOFF_SHIFT: u32 = 4; // up to 16x

/// A streaming `SamplesSink` that incrementally emits Whisper segments as audio arrives.
pub(crate) struct BufferedSegmentTranscriber<'a> {
    ctx: &'a WhisperContext,
    opts: &'a Opts,
    encoder: &'a mut dyn SegmentEncoder,

    min_window_samples: usize,
    max_window_samples: usize,
    next_infer_at_samples: usize,
    no_progress_runs: u32,

    // Backing buffer for decoded samples. We keep an index (`head`) instead of draining on every
    // segment so advancing is cheap; we occasionally compact to keep memory usage reasonable.
    samples: Vec<f32>,
    head: usize,

    // Total number of samples we have advanced past since the beginning of the stream.
    advanced_samples: usize,
}

impl<'a> BufferedSegmentTranscriber<'a> {
    pub(crate) fn new(
        ctx: &'a WhisperContext,
        opts: &'a Opts,
        encoder: &'a mut dyn SegmentEncoder,
    ) -> Self {
        let min_window_samples = WHISPER_SAMPLE_RATE as usize * DEFAULT_MIN_BUFFER_SECONDS;
        let max_window_samples = WHISPER_SAMPLE_RATE as usize * DEFAULT_MAX_BUFFER_SECONDS;
        Self {
            ctx,
            opts,
            encoder,
            min_window_samples,
            max_window_samples,
            next_infer_at_samples: min_window_samples,
            no_progress_runs: 0,
            samples: Vec::new(),
            head: 0,
            advanced_samples: 0,
        }
    }

    pub(crate) fn finish(&mut self) -> Result<()> {
        while self.process_available(true)? == Progress::Advanced {}
        self.samples.clear();
        self.head = 0;
        Ok(())
    }

    fn window(&self) -> &[f32] {
        &self.samples[self.head..]
    }

    fn window_len(&self) -> usize {
        self.samples.len().saturating_sub(self.head)
    }

    fn maybe_compact(&mut self) {
        if self.head == 0 {
            return;
        }

        // Compact when either:
        // - we’ve consumed at least 1s of audio, or
        // - the head is past half the buffer
        let should_compact =
            self.head >= WHISPER_SAMPLE_RATE as usize || self.head >= self.samples.len() / 2;
        if should_compact {
            self.samples.drain(..self.head);
            self.head = 0;
        }
    }

    fn process_available(&mut self, end_of_stream: bool) -> Result<Progress> {
        let win_len = self.window_len();
        if win_len == 0 {
            return Ok(Progress::NoOp);
        }

        if !end_of_stream && win_len < self.min_window_samples {
            return Ok(Progress::NoOp);
        }

        // Avoid re-running Whisper on every tiny incoming chunk: only re-run once the window
        // has grown by ~1s since the last attempt (or when flushing).
        let force_flush = end_of_stream || win_len >= self.max_window_samples;
        if !force_flush && win_len < self.next_infer_at_samples {
            return Ok(Progress::NoOp);
        }

        let t0 = Instant::now();
        if debug_incremental() {
            eprintln!(
                "[scribble][incremental] infer_start win_len={}samp advanced={}samp eos={} force_flush={}",
                win_len, self.advanced_samples, end_of_stream, force_flush
            );
        }
        let state = run_whisper_full(self.ctx, self.opts, self.window())?;
        let n_segments_i32 = state.full_n_segments();
        if debug_incremental() {
            eprintln!(
                "[scribble][incremental] infer_done elapsed_ms={} n_segments={}",
                t0.elapsed().as_millis(),
                n_segments_i32
            );
        }
        if n_segments_i32 <= 0 {
            if !force_flush {
                self.no_progress_runs = self.no_progress_runs.saturating_add(1);
                self.next_infer_at_samples = next_infer_threshold(
                    win_len,
                    self.min_window_samples,
                    self.max_window_samples,
                    self.no_progress_runs,
                );
                if debug_incremental() {
                    eprintln!(
                        "[scribble][incremental] no_progress runs={} next_infer_at={}samp",
                        self.no_progress_runs, self.next_infer_at_samples
                    );
                }
            }
            return Ok(Progress::NoOp);
        }
        let n_segments: usize = n_segments_i32
            .try_into()
            .context("whisper returned a negative segment count")?;

        // Finalization rule:
        // - If Whisper produced >= 2 segments, we treat all but the last as “final”.
        // - At end-of-stream (or max-buffer cap), we flush everything we have.
        let emit_count = if force_flush {
            n_segments
        } else if n_segments >= 2 {
            n_segments - 1
        } else {
            0
        };

        if debug_incremental() {
            eprintln!(
                "[scribble][incremental] finalize emit_count={} (n_segments={})",
                emit_count, n_segments
            );
        }
        if emit_count == 0 {
            self.no_progress_runs = self.no_progress_runs.saturating_add(1);
            self.next_infer_at_samples = next_infer_threshold(
                win_len,
                self.min_window_samples,
                self.max_window_samples,
                self.no_progress_runs,
            );
            if debug_incremental() {
                eprintln!(
                    "[scribble][incremental] no_progress runs={} next_infer_at={}samp",
                    self.no_progress_runs, self.next_infer_at_samples
                );
            }
            return Ok(Progress::NoOp);
        }

        let offset_seconds = self.advanced_samples as f32 / WHISPER_SAMPLE_RATE as f32;

        for segment_idx in 0..emit_count {
            let whisper_segment = state
                .get_segment(segment_idx as i32)
                .with_context(|| format!("whisper segment {segment_idx} was missing"))?;

            let mut segment = to_segment(whisper_segment)?;
            apply_time_offset(&mut segment, offset_seconds);
            self.encoder.write_segment(&segment)?;
        }

        // Advance by the end timestamp of the last emitted segment.
        let last_emitted_idx = emit_count - 1;
        let last_emitted = state
            .get_segment(last_emitted_idx as i32)
            .with_context(|| format!("whisper segment {last_emitted_idx} was missing"))?;
        let end_samples = segment_end_samples(last_emitted.end_timestamp(), win_len)?;
        if debug_incremental() {
            eprintln!(
                "[scribble][incremental] advance end_samples={} new_advanced={}samp",
                end_samples,
                self.advanced_samples + end_samples
            );
        }

        self.head += end_samples;
        self.advanced_samples += end_samples;
        self.maybe_compact();
        self.no_progress_runs = 0;

        // After emitting (and advancing), wait for more audio before running Whisper again,
        // unless we are flushing (finish() will call us again until no progress).
        if !force_flush {
            self.next_infer_at_samples = self.window_len() + self.min_window_samples;
        } else {
            self.next_infer_at_samples = self.min_window_samples;
        }

        Ok(Progress::Advanced)
    }
}

impl SamplesSink for BufferedSegmentTranscriber<'_> {
    fn on_samples(&mut self, samples_16k_mono: &[f32]) -> Result<bool> {
        self.samples.extend_from_slice(samples_16k_mono);
        let _ = self.process_available(false)?;
        Ok(true)
    }
}

#[derive(Debug, Clone, Copy, PartialEq, Eq)]
enum Progress {
    NoOp,
    Advanced,
}

fn debug_incremental() -> bool {
    static ENABLED: OnceLock<bool> = OnceLock::new();
    *ENABLED.get_or_init(|| std::env::var_os("SCRIBBLE_DEBUG_INCREMENTAL").is_some())
}

fn next_infer_threshold(
    current_len: usize,
    min_window_samples: usize,
    max_window_samples: usize,
    no_progress_runs: u32,
) -> usize {
    let shift = no_progress_runs.saturating_sub(1).min(MAX_BACKOFF_SHIFT);
    let step = min_window_samples.saturating_mul(1usize << shift);
    let proposed = current_len.saturating_add(step);
    // Don't schedule beyond our forced flush boundary; if we get there, we'll force_flush.
    proposed.min(max_window_samples)
}

fn segment_end_samples(end_timestamp_cs: i64, available_samples: usize) -> Result<usize> {
    ensure!(
        end_timestamp_cs >= 0,
        "whisper returned negative end timestamp: {end_timestamp_cs}"
    );
    let end_timestamp_cs: usize = end_timestamp_cs
        .try_into()
        .context("whisper end timestamp did not fit in usize")?;

    // Whisper timestamps are centiseconds (1/100s).
    let mut end_samples = end_timestamp_cs.saturating_mul(WHISPER_SAMPLE_RATE as usize) / 100;

    // Avoid infinite loops if whisper returns a degenerate segment.
    if end_samples == 0 {
        end_samples = 1;
    }
    if end_samples > available_samples {
        end_samples = available_samples;
    }

    Ok(end_samples)
}

fn apply_time_offset(segment: &mut crate::segments::Segment, offset_seconds: f32) {
    segment.start_seconds += offset_seconds;
    segment.end_seconds += offset_seconds;
    for token in &mut segment.tokens {
        token.start_seconds += offset_seconds;
        token.end_seconds += offset_seconds;
    }
}
