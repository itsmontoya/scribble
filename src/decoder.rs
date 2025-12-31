//! Stream-decode audio from media (audio or video containers) into Whisper-friendly
//! mono `f32` @ 16kHz, delivered in chunks via a callback.
//!
//! We support two input shapes:
//! - `Read + Seek` (most robust; supports containers that require seeking).
//! - `Read` only (true streaming; works for stream-friendly container layouts).
//!
//! Design overview:
//! - Symphonia demuxes + decodes packets into PCM frames.
//! - We normalize to interleaved `f32`, downmix to mono, then resample to 16 kHz if needed.
//! - We emit fixed-size chunks to a `SamplesSink` so callers can process incrementally.

use std::io::Read;

use anyhow::{Context, Result, anyhow, bail};
use rubato::{Resampler, SincFixedIn, WindowFunction};

use symphonia::core::audio::{AudioBufferRef, SampleBuffer};
use symphonia::core::codecs::{CODEC_TYPE_NULL, Decoder, DecoderOptions};
use symphonia::core::errors::Error as SymphoniaError;
use symphonia::core::formats::{FormatOptions, FormatReader, Packet, Track};
use symphonia::core::io::{
    MediaSource, MediaSourceStream, MediaSourceStreamOptions, ReadOnlySource,
};
use symphonia::core::meta::MetadataOptions;
use symphonia::core::probe::Hint;

/// Whisper expects mono audio sampled at 16 kHz.
pub const WHISPER_SAMPLE_RATE: u32 = 16_000;

/// Consumer callback for decoded samples.
///
/// The sink receives **mono 16 kHz** `f32` samples.
/// Returning `Ok(false)` signals "stop decoding early" (useful for streaming).
pub trait SamplesSink {
    fn on_samples(&mut self, samples_16k_mono: &[f32]) -> Result<bool>;
}

/// Streaming decode configuration.
#[derive(Debug, Clone)]
pub struct StreamDecodeOpts {
    /// Chunk size *after* resampling (16kHz frames).
    ///
    /// Examples:
    /// - 320  = 20ms
    /// - 1600 = 100ms
    pub target_chunk_frames_16k: usize,

    /// Optional container hint (e.g. "mp4", "ts", "webm", "mkv", "ogg").
    /// This can help probing, especially for unseekable streams.
    pub hint_extension: Option<String>,
}

impl Default for StreamDecodeOpts {
    fn default() -> Self {
        Self {
            target_chunk_frames_16k: 1024,
            hint_extension: None,
        }
    }
}

/// Decode an unseekable stream and emit Whisper-friendly chunks into `sink`.
///
/// This is the best mode for true streaming (stdin, sockets, live HTTP bodies),
/// but some container layouts may still require seeking to locate metadata.
pub fn decode_to_whisper_stream_from_read<R>(
    reader: R,
    opts: StreamDecodeOpts,
    sink: &mut dyn SamplesSink,
) -> Result<()>
where
    R: Read + Send + Sync + 'static,
{
    let source = ReadOnlySource::new(reader);
    decode_impl(Box::new(source), opts, sink)
}

/// Shared implementation for both input modes.
///
/// We take a boxed `MediaSource` so we can reuse *all* decode logic without duplicating it.
fn decode_impl(
    source: Box<dyn MediaSource>,
    opts: StreamDecodeOpts,
    sink: &mut dyn SamplesSink,
) -> Result<()> {
    let (mut format, track) = probe_source_and_pick_default_track(source, &opts)?;
    let mut decoder = make_decoder_for_track(&track)?;

    // Mutable state shared across packets (buffers + resampler accumulator).
    let mut state = DecodeState::new(track.id, opts);

    decode_loop(&mut format, &mut decoder, &mut state, sink)?;
    finalize_stream(&mut state, sink)?;

    Ok(())
}

/// Internal mutable state shared across the decode loop.
struct DecodeState {
    track_id: u32,
    opts: StreamDecodeOpts,

    // Reusable scratch buffer Symphonia uses to copy decoded PCM into an interleaved vec.
    sample_buf_f32: Option<SampleBuffer<f32>>,

    // Resampler from source sample rate → 16 kHz (initialized lazily if needed).
    resampler: Option<SincFixedIn<f32>>,

    // Accumulator for mono source samples before feeding blocks into the resampler.
    mono_src_acc: Vec<f32>,

    // Reused input channel vec for rubato (avoid vec![mono.to_vec()] each call).
    resample_in_chan: Vec<f32>,
}

impl DecodeState {
    fn new(track_id: u32, opts: StreamDecodeOpts) -> Self {
        Self {
            track_id,
            opts,
            sample_buf_f32: None,
            resampler: None,
            mono_src_acc: Vec::new(),
            resample_in_chan: Vec::new(),
        }
    }
}

/// Probe the container and pick a default audio track.
fn probe_source_and_pick_default_track(
    source: Box<dyn MediaSource>,
    opts: &StreamDecodeOpts,
) -> Result<(Box<dyn FormatReader>, Track)> {
    let mss_opts = MediaSourceStreamOptions {
        // Symphonia requires a power-of-two buffer > 32KiB for good probing behavior.
        buffer_len: 256 * 1024,
    };

    let mss = MediaSourceStream::new(source, mss_opts);

    // Optional hint can speed up probing and improve success on tricky streams.
    let mut hint = Hint::new();
    if let Some(ext) = &opts.hint_extension {
        hint.with_extension(ext);
    }

    let format_opts: FormatOptions = Default::default();
    let metadata_opts: MetadataOptions = Default::default();

    let probed = symphonia::default::get_probe()
        .format(&hint, mss, &format_opts, &metadata_opts)
        .map_err(|e| anyhow!(e))
        .context("failed to probe media stream")?;

    let format = probed.format;

    // Pick the first track that looks like decodable audio.
    let track = format
        .tracks()
        .iter()
        .find(|t| t.codec_params.codec != CODEC_TYPE_NULL && t.codec_params.sample_rate.is_some())
        .cloned()
        .ok_or_else(|| anyhow!("no audio track found"))?;

    Ok((format, track))
}

/// Create an audio decoder for the selected track.
fn make_decoder_for_track(track: &Track) -> Result<Box<dyn Decoder>> {
    let decoder_opts: DecoderOptions = Default::default();
    symphonia::default::get_codecs()
        .make(&track.codec_params, &decoder_opts)
        .map_err(|e| anyhow!(e))
        .context("failed to create decoder")
}

/// Main packet → PCM → mono → resample → chunk emission loop.
fn decode_loop(
    format: &mut Box<dyn FormatReader>,
    decoder: &mut Box<dyn Decoder>,
    state: &mut DecodeState,
    sink: &mut dyn SamplesSink,
) -> Result<()> {
    loop {
        let packet = match next_packet(format)? {
            Some(p) => p,
            None => break,
        };

        // Ignore packets from non-audio tracks.
        if packet.track_id() != state.track_id {
            continue;
        }

        // Decode to PCM.
        let decoded = match decoder.decode(&packet) {
            Ok(buf) => buf,
            // Some codecs emit recoverable errors for "bad" frames. Skipping is fine.
            Err(SymphoniaError::DecodeError(_)) => continue,
            // IO error means the stream ended (or got cut off). We exit gracefully.
            Err(SymphoniaError::IoError(_)) => break,
            Err(e) => return Err(anyhow!(e)).context("decoder failure"),
        };

        // Convert decoded PCM into interleaved f32 samples.
        let (interleaved, src_rate, channels) =
            decoded_to_interleaved_f32(&decoded, &mut state.sample_buf_f32)?;

        // Downmix multi-channel audio into mono for Whisper.
        let mono_src = downmix_to_mono(&interleaved, channels);

        // Fast path: already 16 kHz.
        if src_rate == WHISPER_SAMPLE_RATE {
            emit_mono_chunks(&mono_src, state.opts.target_chunk_frames_16k, sink)?;
            continue;
        }

        // Slow path: resample to 16 kHz.
        ensure_resampler(state, src_rate)?;
        push_and_flush_resampler(state, &mono_src, sink)?;
    }

    Ok(())
}

/// Read the next packet, treating IO errors as "end of stream".
fn next_packet(format: &mut Box<dyn FormatReader>) -> Result<Option<Packet>> {
    match format.next_packet() {
        Ok(p) => Ok(Some(p)),
        Err(SymphoniaError::IoError(_)) => Ok(None),
        Err(e) => Err(anyhow!(e)).context("failed reading packet"),
    }
}

/// Convert a decoded Symphonia buffer into an owned interleaved `Vec<f32>`.
///
/// Returns:
/// - interleaved samples
/// - source sample rate
/// - channel count
fn decoded_to_interleaved_f32(
    decoded: &AudioBufferRef<'_>,
    sample_buf_f32: &mut Option<SampleBuffer<f32>>,
) -> Result<(Vec<f32>, u32, usize)> {
    ensure_sample_buffer(decoded, sample_buf_f32);

    let buf = sample_buf_f32
        .as_mut()
        .ok_or_else(|| anyhow!("sample buffer not initialized"))?;

    // Copy decoded audio into our reusable interleaved `SampleBuffer<f32>`.
    buf.copy_interleaved_ref(decoded.clone());

    let src_rate = decoded.spec().rate;
    let channels = decoded.spec().channels.count();

    if channels == 0 {
        bail!("decoded audio had zero channels");
    }

    Ok((buf.samples().to_vec(), src_rate, channels))
}

/// Initialize the reusable `SampleBuffer<f32>` if we don't have one yet.
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
/// We keep the policy simple and predictable: average all channels equally.
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

/// Emit mono 16 kHz samples to the sink in fixed-size chunks.
fn emit_mono_chunks(
    mono_16k: &[f32],
    chunk_frames: usize,
    sink: &mut dyn SamplesSink,
) -> Result<()> {
    for chunk in mono_16k.chunks(chunk_frames) {
        if !sink.on_samples(chunk)? {
            break;
        }
    }
    Ok(())
}

/// Lazily initialize a resampler if the source sample rate is not 16 kHz.
///
/// Note:
/// - This resampler is configured for **mono** input (channels = 1).
/// - The tuning values here bias toward good quality without being absurdly expensive.
fn ensure_resampler(state: &mut DecodeState, src_rate: u32) -> Result<()> {
    if state.resampler.is_some() {
        return Ok(());
    }

    // How many source frames we feed rubato per `process()` call.
    //
    // Tradeoff:
    // - larger chunks are more efficient
    // - smaller chunks reduce latency
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

    state.resampler = Some(rs);
    Ok(())
}

/// Accumulate source samples and flush full resampler blocks into the sink.
fn push_and_flush_resampler(
    state: &mut DecodeState,
    mono_src: &[f32],
    sink: &mut dyn SamplesSink,
) -> Result<()> {
    state.mono_src_acc.extend_from_slice(mono_src);

    loop {
        let in_max = match state.resampler.as_ref() {
            Some(rs) => rs.input_frames_max(),
            None => bail!("resampler not initialized"),
        };

        if state.mono_src_acc.len() < in_max {
            break;
        }

        // Drain exactly one input block for rubato.
        let block: Vec<f32> = state.mono_src_acc.drain(..in_max).collect();

        let out = resample_block(state, &block)?;

        for chunk in out.chunks(state.opts.target_chunk_frames_16k) {
            if !sink.on_samples(chunk)? {
                return Ok(());
            }
        }
    }

    Ok(())
}

/// Resample one mono block and return the mono 16 kHz output.
fn resample_block(state: &mut DecodeState, mono_src_block: &[f32]) -> Result<Vec<f32>> {
    let rs = state
        .resampler
        .as_mut()
        .ok_or_else(|| anyhow!("resampler not initialized"))?;

    // Reuse the channel vec to avoid allocating mono_src_block.to_vec() into a fresh Vec each call.
    state.resample_in_chan.clear();
    state.resample_in_chan.extend_from_slice(mono_src_block);

    let input = vec![state.resample_in_chan.clone()];
    let out = rs
        .process(&input, None)
        .map_err(|e| anyhow!(e))
        .context("resampler process failed")?;

    if out.len() != 1 {
        bail!("expected mono output from resampler");
    }

    Ok(out[0].clone())
}

/// Flush remaining resampler input at end-of-stream.
///
/// rubato expects to be fed exact block sizes. We pad the tail with zeros to complete
/// a final block, then drain until empty.
fn finalize_stream(state: &mut DecodeState, sink: &mut dyn SamplesSink) -> Result<()> {
    let Some(rs) = state.resampler.as_mut() else {
        // No resampling happened.
        return Ok(());
    };

    if state.mono_src_acc.is_empty() {
        return Ok(());
    }

    let in_max = rs.input_frames_max();
    let rem = state.mono_src_acc.len() % in_max;

    // Pad the remainder so we can feed complete blocks.
    if rem != 0 {
        state
            .mono_src_acc
            .resize(state.mono_src_acc.len() + (in_max - rem), 0.0);
    }

    while !state.mono_src_acc.is_empty() {
        let block: Vec<f32> = state.mono_src_acc.drain(..in_max).collect();
        let out = resample_block(state, &block)?;

        for chunk in out.chunks(state.opts.target_chunk_frames_16k) {
            if !sink.on_samples(chunk)? {
                return Ok(());
            }
        }
    }

    Ok(())
}
