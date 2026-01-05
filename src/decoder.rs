// src/decoder.rs

//! Stream-decode media (audio/video containers) into mono `f32` at Scribble's target sample rate,
//! emitting fixed-size chunks via a callback.
//!
//! This module is intentionally small and orchestration-focused:
//! - `demux` handles probing + packet iteration
//! - `decode` handles codec decoding
//! - `audio_pipeline` handles PCM normalization (downmix + resample) + chunking
//!
//! Current mode: **unseekable** (`Read` only) via `ReadOnlySource`.
//! This works well for stdin / sockets / HTTP bodies and stream-friendly container layouts.
//! If you later want to support seekable inputs (many MP4/MOV files), add a
//! `decode_to_stream_from_reader(Read + Seek)` variant using a seekable `MediaSource`.

use std::io::Read;
use std::sync::Mutex;

use anyhow::{Context, Result};
use symphonia::core::io::{MediaSource, ReadOnlySource};

use crate::audio_pipeline::AudioPipeline;
use crate::decode::{decode_packet_and_then, make_decoder_for_track};
use crate::demux::{next_packet, probe_source_and_pick_default_track};

/// Consumer callback for decoded samples.
///
/// The sink receives **mono** `f32` samples at Scribble's target sample rate.
/// Returning `Ok(false)` signals "stop decoding early".
pub trait SamplesSink {
    fn on_samples(&mut self, samples_16k_mono: &[f32]) -> Result<bool>;
}

/// Streaming decode configuration.
#[derive(Debug, Clone)]
pub struct StreamDecodeOpts {
    /// Chunk size *after* resampling (target-rate frames).
    ///
    /// Examples:
    /// - 320  = 20ms
    /// - 1600 = 100ms
    pub target_chunk_frames: usize,

    /// Optional container hint (e.g. "mp4", "ts", "webm", "mkv", "ogg").
    /// This can improve probing, especially for unseekable streams.
    pub hint_extension: Option<String>,
}

impl Default for StreamDecodeOpts {
    fn default() -> Self {
        Self {
            target_chunk_frames: 1024,
            hint_extension: None,
        }
    }
}

/// Decode an unseekable input stream and emit normalized chunks into `sink`.
///
/// This is ideal for true streaming sources (stdin, sockets, live HTTP bodies).
/// Some container layouts (notably certain MP4/MOV files) may still require seeking
/// to locate metadata (`moov` at the end) and will fail in this mode.
pub fn decode_to_stream_from_read<R>(
    reader: R,
    opts: StreamDecodeOpts,
    sink: &mut dyn SamplesSink,
) -> Result<()>
where
    R: Read + Send + 'static,
{
    // Symphonia's `MediaSource` is `Read + Send + Sync`. We only need to *move* the reader to the
    // decode thread (not share it concurrently), so we wrap it in a mutex to satisfy `Sync`.
    let source = ReadOnlySource::new(LockedRead::new(reader));
    decode_impl(Box::new(source), opts, sink)
}

/// Shared implementation that takes an abstract Symphonia `MediaSource`.
fn decode_impl(
    source: Box<dyn MediaSource>,
    opts: StreamDecodeOpts,
    sink: &mut dyn SamplesSink,
) -> Result<()> {
    let (mut format, track) =
        probe_source_and_pick_default_track(source, opts.hint_extension.as_deref())?;

    let mut decoder = make_decoder_for_track(&track)?;
    let mut pipeline = AudioPipeline::new();

    loop {
        let Some(packet) = next_packet(&mut format)? else {
            break;
        };

        // Ignore packets from non-audio tracks.
        if packet.track_id() != track.id {
            continue;
        }

        // Decode packet → normalized audio pipeline → emit chunks.
        //
        // `decode_packet_and_then` returns `Ok(false)` for recoverable cases
        // (e.g. bad frames / IO end). We keep iterating.
        decode_packet_and_then(&mut decoder, &packet, |decoded| {
            pipeline
                .push_decoded_and_emit(&decoded, opts.target_chunk_frames, |chunk| {
                    sink.on_samples(chunk)
                })
                .context("audio pipeline failed while processing decoded samples")
        })?;
    }

    // Flush any buffered resampler tail.
    pipeline
        .finalize(opts.target_chunk_frames, |chunk| sink.on_samples(chunk))
        .context("audio pipeline failed during finalize")?;

    Ok(())
}

struct LockedRead<R> {
    inner: Mutex<R>,
}

impl<R> LockedRead<R> {
    fn new(inner: R) -> Self {
        Self {
            inner: Mutex::new(inner),
        }
    }
}

impl<R: Read> Read for LockedRead<R> {
    fn read(&mut self, buf: &mut [u8]) -> std::io::Result<usize> {
        self.inner
            .lock()
            .map_err(|_| std::io::Error::other("decoder input mutex poisoned"))?
            .read(buf)
    }
}
