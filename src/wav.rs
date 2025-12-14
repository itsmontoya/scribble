use anyhow::{Context, Result};
use hound::{WavReader, WavSpec};

pub fn get_samples_from_wav(audio_path: &String) -> Result<(Vec<f32>, WavSpec)> {
    let mut reader = WavReader::open(&audio_path)
        .with_context(|| format!("failed to open wav file {}", audio_path))?;

    let spec = reader.spec();
    if spec.channels != 1 {
        anyhow::bail!("expected mono WAV, got {} channels", spec.channels);
    }

    if spec.sample_rate != 16_000 {
        anyhow::bail!(
            "expected 16 kHz, got {} – whisper.cpp is tuned for 16 kHz",
            spec.sample_rate
        );
    }

    let mut samples = Vec::new();
    for s in reader.samples::<i16>() {
        let v = s? as f32 / i16::MAX as f32;
        samples.push(v);
    }

    Ok((samples, spec))
}
