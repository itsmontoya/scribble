//! Audio normalization pipeline for Scribble.
//!
//! Responsibilities:
//! - Convert Symphonia-decoded PCM into interleaved `f32`
//! - Downmix to mono
//! - Resample to Whisper’s expected 16 kHz (when needed)
//! - Emit fixed-size chunks via a callback (incremental consumption)
//!
//! Notes:
//! - This pipeline is intentionally allocation-conscious, but favors clarity first.
//! - `finalize()` should be called at end-of-stream to flush any remaining resampler input.

use anyhow::{Context, Result, anyhow, bail};
use rubato::{Resampler, SincFixedIn, WindowFunction};
use symphonia::core::audio::{AudioBufferRef, SampleBuffer};

/// Whisper expects mono audio sampled at 16 kHz.
pub const WHISPER_SAMPLE_RATE: u32 = 16_000;

/// A small stateful pipeline that converts decoded audio into mono 16 kHz `f32` chunks.
pub struct AudioPipeline {
    // Scratch buffer used to copy decoded PCM into an interleaved `Vec<f32>`.
    sample_buf_f32: Option<SampleBuffer<f32>>,

    // Lazily initialized resampler (only needed when the source sample rate != 16 kHz).
    resampler: Option<SincFixedIn<f32>>,

    // Accumulator for mono source samples before feeding full blocks into rubato.
    mono_src_acc: Vec<f32>,

    // Reusable mono input channel buffer for rubato (we pass a single channel Vec).
    resample_in_chan: Vec<f32>,

    // Reusable mono output channel buffer from rubato (avoids cloning the output Vec).
    resample_out_chan: Vec<f32>,
}

impl AudioPipeline {
    /// Create a new audio pipeline with empty internal buffers.
    pub fn new() -> Self {
        Self {
            sample_buf_f32: None,
            resampler: None,
            mono_src_acc: Vec::new(),
            resample_in_chan: Vec::new(),
            resample_out_chan: Vec::new(),
        }
    }

    /// Push a decoded Symphonia buffer through the pipeline and emit 16 kHz mono chunks.
    ///
    /// The `emit` callback receives mono 16 kHz `f32` samples.
    /// Returning `Ok(false)` signals “stop early”.
    pub fn push_decoded_and_emit(
        &mut self,
        decoded: &AudioBufferRef<'_>,
        target_chunk_frames_16k: usize,
        mut emit: impl FnMut(&[f32]) -> Result<bool>,
    ) -> Result<()> {
        let (interleaved, src_rate, channels) =
            decoded_to_interleaved_f32(decoded, &mut self.sample_buf_f32)?;

        let mono_src = downmix_to_mono(&interleaved, channels);

        // Fast path: already 16 kHz.
        if src_rate == WHISPER_SAMPLE_RATE {
            emit_mono_chunks(&mono_src, target_chunk_frames_16k, &mut emit)?;
            return Ok(());
        }

        // Slow path: resample to 16 kHz.
        self.ensure_resampler(src_rate)?;
        self.push_and_flush_resampler(&mono_src, target_chunk_frames_16k, &mut emit)?;
        Ok(())
    }

    /// Flush remaining buffered samples at end-of-stream.
    ///
    /// If resampling was never needed, this is a no-op.
    pub fn finalize(
        &mut self,
        target_chunk_frames_16k: usize,
        mut emit: impl FnMut(&[f32]) -> Result<bool>,
    ) -> Result<()> {
        let Some(rs) = self.resampler.as_mut() else {
            return Ok(());
        };

        if self.mono_src_acc.is_empty() {
            return Ok(());
        }

        // rubato expects exact block sizes; pad the remainder with zeros.
        let in_max = rs.input_frames_max();
        let rem = self.mono_src_acc.len() % in_max;
        if rem != 0 {
            self.mono_src_acc
                .resize(self.mono_src_acc.len() + (in_max - rem), 0.0);
        }

        while !self.mono_src_acc.is_empty() {
            // Drain exactly one input block.
            let block: Vec<f32> = self.mono_src_acc.drain(..in_max).collect();

            let out = self.resample_block_into_out(&block)?;
            emit_mono_chunks(out, target_chunk_frames_16k, &mut emit)?;
        }

        Ok(())
    }

    fn ensure_resampler(&mut self, src_rate: u32) -> Result<()> {
        if self.resampler.is_some() {
            return Ok(());
        }

        // How many source frames we feed rubato per `process()` call.
        // Tradeoff: larger chunks = better throughput; smaller chunks = lower latency.
        let in_chunk_src_frames = 2048;

        let rs = SincFixedIn::<f32>::new(
            WHISPER_SAMPLE_RATE as f64 / src_rate as f64,
            2.0,
            rubato::SincInterpolationParameters {
                sinc_len: 256,
                f_cutoff: 0.95,
                interpolation: rubato::SincInterpolationType::Linear,
                oversampling_factor: 256,
                window: WindowFunction::BlackmanHarris2,
            },
            in_chunk_src_frames,
            1, // mono
        )
        .map_err(|e| anyhow!(e))
        .context("failed to init resampler")?;

        self.resampler = Some(rs);
        Ok(())
    }

    fn push_and_flush_resampler(
        &mut self,
        mono_src: &[f32],
        target_chunk_frames_16k: usize,
        emit: &mut impl FnMut(&[f32]) -> Result<bool>,
    ) -> Result<()> {
        self.mono_src_acc.extend_from_slice(mono_src);

        loop {
            let in_max = self
                .resampler
                .as_ref()
                .expect("resampler must be initialized before flushing")
                .input_frames_max();

            if self.mono_src_acc.len() < in_max {
                break;
            }

            let block: Vec<f32> = self.mono_src_acc.drain(..in_max).collect();
            let out = self.resample_block_into_out(&block)?;

            // `out` is a borrowed view into `self.resample_out_chan`.
            // Emit in fixed-size chunks.
            for chunk in out.chunks(target_chunk_frames_16k) {
                if !emit(chunk)? {
                    return Ok(());
                }
            }
        }

        Ok(())
    }

    /// Resample one mono block and return a borrowed view of the mono 16 kHz output.
    ///
    /// This avoids allocating/cloning the output for every block; the returned slice is
    /// valid until the next call to `resample_block_into_out`.
    fn resample_block_into_out(&mut self, mono_src_block: &[f32]) -> Result<&[f32]> {
        let rs = self
            .resampler
            .as_mut()
            .ok_or_else(|| anyhow!("resampler not initialized"))?;

        // Build rubato’s expected `Vec<Vec<f32>>` input (one channel for mono).
        self.resample_in_chan.clear();
        self.resample_in_chan.extend_from_slice(mono_src_block);

        let input = vec![self.resample_in_chan.clone()];
        let out = rs
            .process(&input, None)
            .map_err(|e| anyhow!(e))
            .context("resampler process failed")?;

        if out.len() != 1 {
            bail!("expected mono output from resampler");
        }

        self.resample_out_chan = out[0].clone();
        Ok(&self.resample_out_chan)
    }
}

fn decoded_to_interleaved_f32(
    decoded: &AudioBufferRef<'_>,
    sample_buf_f32: &mut Option<SampleBuffer<f32>>,
) -> Result<(Vec<f32>, u32, usize)> {
    ensure_sample_buffer(decoded, sample_buf_f32);

    let buf = sample_buf_f32
        .as_mut()
        .ok_or_else(|| anyhow!("sample buffer not initialized"))?;

    // Copy decoded PCM into our interleaved scratch buffer.
    buf.copy_interleaved_ref(decoded.clone());

    let src_rate = decoded.spec().rate;
    let channels = decoded.spec().channels.count();
    if channels == 0 {
        bail!("decoded audio had zero channels");
    }

    Ok((buf.samples().to_vec(), src_rate, channels))
}

fn ensure_sample_buffer(
    decoded: &AudioBufferRef<'_>,
    sample_buf_f32: &mut Option<SampleBuffer<f32>>,
) {
    if sample_buf_f32.is_some() {
        return;
    }

    let spec = *decoded.spec();
    let duration = decoded.capacity() as u64;
    *sample_buf_f32 = Some(SampleBuffer::<f32>::new(duration, spec));
}

/// Downmix interleaved samples into mono by averaging channels.
///
/// Policy: equal-weight average across channels (simple, predictable).
fn downmix_to_mono(interleaved: &[f32], channels: usize) -> Vec<f32> {
    if channels == 1 {
        return interleaved.to_vec();
    }

    let frames = interleaved.len() / channels;
    let mut mono = Vec::with_capacity(frames);

    for f in 0..frames {
        let base = f * channels;
        let mut acc = 0.0;
        for c in 0..channels {
            acc += interleaved[base + c];
        }
        mono.push(acc / channels as f32);
    }

    mono
}

/// Emit mono 16 kHz samples to the callback in fixed-size chunks.
fn emit_mono_chunks(
    mono_16k: &[f32],
    chunk_frames: usize,
    emit: &mut impl FnMut(&[f32]) -> Result<bool>,
) -> Result<()> {
    for chunk in mono_16k.chunks(chunk_frames) {
        if !emit(chunk)? {
            break;
        }
    }
    Ok(())
}
