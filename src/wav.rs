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
#[cfg(test)]
pub(crate) fn get_samples_from_wav_reader<R>(reader: R) -> Result<(Vec<f32>, WavSpec)>
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

#[cfg(test)]
mod tests {
    use super::*;
    use std::io::Cursor;

    fn wav_bytes(spec: WavSpec, samples: &[i16]) -> Vec<u8> {
        let mut buf = Vec::new();
        {
            let mut writer =
                hound::WavWriter::new(Cursor::new(&mut buf), spec).expect("create WAV writer");
            for s in samples {
                writer.write_sample(*s).expect("write sample");
            }
            writer.finalize().expect("finalize WAV");
        }
        buf
    }

    #[test]
    fn wav_rejects_stereo_input() {
        let spec = WavSpec {
            channels: 2,
            sample_rate: TARGET_SAMPLE_RATE,
            bits_per_sample: 16,
            sample_format: hound::SampleFormat::Int,
        };
        let bytes = wav_bytes(spec, &[0i16; 16]);
        let err = get_samples_from_wav_reader(Cursor::new(bytes)).unwrap_err();
        assert!(err.to_string().contains("expected mono WAV"));
    }

    #[test]
    fn wav_rejects_wrong_sample_rate() {
        let spec = WavSpec {
            channels: 1,
            sample_rate: TARGET_SAMPLE_RATE + 1,
            bits_per_sample: 16,
            sample_format: hound::SampleFormat::Int,
        };
        let bytes = wav_bytes(spec, &[0i16; 16]);
        let err = get_samples_from_wav_reader(Cursor::new(bytes)).unwrap_err();
        assert!(err.to_string().contains("expected 16000 Hz"));
    }

    #[test]
    fn wav_normalizes_i16_pcm_samples() -> anyhow::Result<()> {
        let spec = WavSpec {
            channels: 1,
            sample_rate: TARGET_SAMPLE_RATE,
            bits_per_sample: 16,
            sample_format: hound::SampleFormat::Int,
        };
        let bytes = wav_bytes(spec, &[i16::MIN, -1, 0, 1, i16::MAX]);

        let (samples, got_spec) = get_samples_from_wav_reader(Cursor::new(bytes))?;
        assert_eq!(got_spec.channels, 1);
        assert_eq!(got_spec.sample_rate, TARGET_SAMPLE_RATE);
        assert_eq!(samples.len(), 5);
        assert!(samples[2].abs() < f32::EPSILON);
        assert!(samples[4] <= 1.0);
        Ok(())
    }
}
