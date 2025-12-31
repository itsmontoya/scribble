// src/decode.rs

//! Decoder helpers built on top of Symphonia.
//!
//! This module isolates codec-level concerns:
//! - constructing a decoder for a selected audio track
//! - decoding packets into PCM buffers
//! - handling Symphonia’s error model in a predictable, streaming-friendly way
//!
//! By keeping this logic here, higher-level pipelines can focus on
//! demux → resample → VAD → Whisper, without worrying about codec edge cases.

use anyhow::{Context, Result, anyhow};
use symphonia::core::audio::AudioBufferRef;
use symphonia::core::codecs::{Decoder, DecoderOptions};
use symphonia::core::errors::Error as SymphoniaError;
use symphonia::core::formats::{Packet, Track};

/// Create a decoder for the given audio track.
///
/// This uses Symphonia’s default codec registry and options.
///
/// Fails if:
/// - the codec is unsupported
/// - the codec parameters are invalid
pub fn make_decoder_for_track(track: &Track) -> Result<Box<dyn Decoder>> {
    let decoder_opts: DecoderOptions = Default::default();

    symphonia::default::get_codecs()
        .make(&track.codec_params, &decoder_opts)
        .map_err(|e| anyhow!(e))
        .context("failed to create decoder for audio track")
}

/// Decode a packet and immediately hand the decoded buffer to a callback.
///
/// This helper centralizes Symphonia’s decode error handling so callers don’t
/// have to repeat it in every decode loop.
///
/// Return value semantics:
/// - `Ok(true)`  → a decoded audio buffer was produced and `on_decoded` ran
/// - `Ok(false)` → packet was skipped or stream ended (recoverable condition)
/// - `Err(_)`    → fatal decoder error
///
/// Error handling policy:
/// - `DecodeError` → skip bad frame (common with some codecs)
/// - `IoError`     → treat as end-of-stream (streaming-friendly)
/// - other errors  → bubble up with context
pub fn decode_packet_and_then(
    decoder: &mut Box<dyn Decoder>,
    packet: &Packet,
    mut on_decoded: impl FnMut(AudioBufferRef<'_>) -> Result<()>,
) -> Result<bool> {
    match decoder.decode(packet) {
        Ok(buf) => {
            on_decoded(buf)?;
            Ok(true)
        }

        // Recoverable: corrupted frame, but decoding can continue.
        Err(SymphoniaError::DecodeError(_)) => Ok(false),

        // Treat IO errors as graceful end-of-stream.
        Err(SymphoniaError::IoError(_)) => Ok(false),

        // Anything else is considered fatal.
        Err(e) => Err(anyhow!(e)).context("decoder failure"),
    }
}
