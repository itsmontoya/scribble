use anyhow::{Context, Result};
use hound::{WavReader, WavSpec};
use std::io::{Read, Seek};

/// Load WAV audio from a reader and return normalized audio samples.
///
/// What we return:
/// - A `Vec<f32>` containing mono audio samples normalized to `[-1.0, 1.0]`
/// - The associated `WavSpec` so callers still have access to metadata
///
/// Format requirements:
/// - Mono (1 channel)
/// - 16 kHz sample rate
///
/// Why we enforce this:
/// - whisper.cpp is tuned specifically for 16 kHz mono audio
/// - enforcing constraints here keeps the rest of the pipeline simple and predictable
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

    // We require a 16 kHz sample rate.
    if spec.sample_rate != 16_000 {
        anyhow::bail!(
            "expected 16 kHz sample rate, got {} Hz (whisper.cpp is tuned for 16 kHz)",
            spec.sample_rate
        );
    }

    // Read samples and normalize from i16 PCM to f32 in [-1.0, 1.0].
    //
    // whisper_rs expects audio in this normalized floating-point format.
    let mut samples = Vec::new();
    for sample in reader.samples::<i16>() {
        let pcm = sample?;
        let normalized = pcm as f32 / i16::MAX as f32;
        samples.push(normalized);
    }

    Ok((samples, spec))
}
