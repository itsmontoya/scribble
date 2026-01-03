use anyhow::{Context, Result};
use hound::{WavReader, WavSpec};
use std::io::{Read, Seek};

use crate::audio_pipeline::TARGET_SAMPLE_RATE;

/// Load WAV audio from a reader and return normalized audio samples.
///
/// What we return:
/// - A `Vec<f32>` containing mono audio samples normalized to `[-1.0, 1.0]`
/// - The associated `WavSpec` so callers still have access to metadata
///
/// Format requirements:
/// - Mono (1 channel)
/// - Scribble's target sample rate
///
/// Why we enforce this:
/// - enforcing constraints here keeps downstream transcription simple and predictable
pub fn get_samples_from_wav_reader<R>(reader: R) -> Result<(Vec<f32>, WavSpec)>
where
    R: Read + Seek,
{
    // Create a WAV reader from the provided input.
    let mut reader = WavReader::new(reader).context("failed to read WAV data from reader")?;
    let spec = reader.spec();

    // We require mono audio.
    if spec.channels != 1 {
        anyhow::bail!(
            "expected mono WAV (1 channel), got {} channels",
            spec.channels
        );
    }

    // We require the target sample rate.
    if spec.sample_rate != TARGET_SAMPLE_RATE {
        anyhow::bail!(
            "expected {} Hz sample rate, got {} Hz",
            TARGET_SAMPLE_RATE,
            spec.sample_rate
        );
    }

    // Read samples and normalize from i16 PCM to f32 in [-1.0, 1.0].
    //
    // Most ASR backends expect audio in this normalized floating-point format.
    let mut samples = Vec::new();
    for sample in reader.samples::<i16>() {
        let pcm = sample?;
        let normalized = pcm as f32 / i16::MAX as f32;
        samples.push(normalized);
    }

    Ok((samples, spec))
}
