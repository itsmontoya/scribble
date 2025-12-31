// src/demux.rs

//! Demux helpers for Symphonia.
//!
//! This module keeps container probing and packet iteration logic isolated from the
//! rest of the decode/transcode pipeline.
//!
//! Responsibilities:
//! - Probe a `MediaSource` and select a reasonable default audio track
//! - Provide a `next_packet` helper that treats IO errors as end-of-stream

use anyhow::{Context, Result, anyhow};
use symphonia::core::codecs::CODEC_TYPE_NULL;
use symphonia::core::errors::Error as SymphoniaError;
use symphonia::core::formats::{FormatOptions, FormatReader, Packet, Track};
use symphonia::core::io::{MediaSource, MediaSourceStream, MediaSourceStreamOptions};
use symphonia::core::meta::MetadataOptions;
use symphonia::core::probe::Hint;

/// Probe the container and pick a default audio track.
///
/// Track selection policy:
/// - choose the first track that looks decodable (codec != NULL)
/// - and has a known sample rate (required for resampling decisions downstream)
///
/// `hint_extension` can improve probe accuracy for ambiguous/unseekable inputs
/// (e.g. "mp4", "ts", "webm", "mkv", "ogg").
pub fn probe_source_and_pick_default_track(
    source: Box<dyn MediaSource>,
    hint_extension: Option<&str>,
) -> Result<(Box<dyn FormatReader>, Track)> {
    let mss_opts = MediaSourceStreamOptions {
        // Symphonia expects a power-of-two buffer > 32KiB for good probing behavior.
        buffer_len: 256 * 1024,
    };

    let mss = MediaSourceStream::new(source, mss_opts);

    let mut hint = Hint::new();
    if let Some(ext) = hint_extension {
        hint.with_extension(ext);
    }

    let format_opts: FormatOptions = Default::default();
    let metadata_opts: MetadataOptions = Default::default();

    let probed = symphonia::default::get_probe()
        .format(&hint, mss, &format_opts, &metadata_opts)
        .map_err(|e| anyhow!(e))
        .context("failed to probe media stream")?;

    let format = probed.format;

    let track = format
        .tracks()
        .iter()
        .find(|t| t.codec_params.codec != CODEC_TYPE_NULL && t.codec_params.sample_rate.is_some())
        .cloned()
        .ok_or_else(|| anyhow!("no audio track found"))?;

    Ok((format, track))
}

/// Read the next packet, treating IO errors as "end of stream".
///
/// This makes decode loops simpler and streaming-friendly:
/// - `Ok(None)` means EOF or stream ended
/// - other errors are surfaced with context
pub fn next_packet(format: &mut Box<dyn FormatReader>) -> Result<Option<Packet>> {
    match format.next_packet() {
        Ok(p) => Ok(Some(p)),
        Err(SymphoniaError::IoError(_)) => Ok(None),
        Err(e) => Err(anyhow!(e)).context("failed reading packet"),
    }
}
