//! Stream-decode audio from media (audio or video containers) into Whisper-friendly
//! mono f32 @ 16kHz, delivered in chunks via a callback.
//!
//! Input is any `Read + Seek` source (file, buffer, etc).

use std::io::{Read, Seek, SeekFrom};

use anyhow::{Context, Result, anyhow, bail};
use rubato::{Resampler, SincFixedIn, WindowFunction};

use symphonia::core::audio::{AudioBufferRef, SampleBuffer};
use symphonia::core::codecs::{CODEC_TYPE_NULL, Decoder, DecoderOptions};
use symphonia::core::errors::Error as SymphoniaError;
use symphonia::core::formats::{FormatOptions, FormatReader, Track};
use symphonia::core::io::{MediaSource, MediaSourceStream, MediaSourceStreamOptions};
use symphonia::core::meta::MetadataOptions;
use symphonia::core::probe::Hint;

pub const WHISPER_SAMPLE_RATE: u32 = 16_000;

pub trait SamplesSink {
    fn on_samples(&mut self, samples_16k_mono: &[f32]) -> Result<bool>;
}

pub struct StreamDecodeOpts {
    /// Chunk size *after* resampling (16kHz frames).
    /// 320 = 20ms, 1600 = 100ms
    pub target_chunk_frames_16k: usize,
}

impl Default for StreamDecodeOpts {
    fn default() -> Self {
        Self {
            target_chunk_frames_16k: 1024,
        }
    }
}

pub fn decode_to_whisper_stream_from_reader<R>(
    reader: R,
    opts: StreamDecodeOpts,
    sink: &mut dyn SamplesSink,
) -> Result<()>
where
    R: Read + Seek + Send + Sync + 'static,
{
    let (mut format, track) = probe_reader_and_pick_default_track(reader)?;
    let mut decoder = make_decoder_for_track(&track)?;
    let mut state = DecodeState::new(track.id, opts);

    decode_loop(&mut format, &mut decoder, &mut state, sink)?;
    finalize_stream(&mut state, sink)?;
    Ok(())
}

struct DecodeState {
    track_id: u32,
    opts: StreamDecodeOpts,

    sample_buf_f32: Option<SampleBuffer<f32>>,
    resampler: Option<SincFixedIn<f32>>,
    mono_src_acc: Vec<f32>,
}

impl DecodeState {
    fn new(track_id: u32, opts: StreamDecodeOpts) -> Self {
        Self {
            track_id,
            opts,
            sample_buf_f32: None,
            resampler: None,
            mono_src_acc: Vec::new(),
        }
    }
}

struct ReadSeekMediaSource<R> {
    inner: R,
    len: Option<u64>,
}

impl<R> ReadSeekMediaSource<R> {
    fn new(mut inner: R) -> Result<Self>
    where
        R: Seek,
    {
        // Try to determine byte length once up front, then restore position.
        // Some Symphonia probing logic relies on this to confidently treat the
        // stream as seekable.
        let len = {
            let cur = inner.seek(SeekFrom::Current(0)).ok();
            let end = inner.seek(SeekFrom::End(0)).ok();
            if let Some(cur) = cur {
                let _ = inner.seek(SeekFrom::Start(cur));
            }
            end
        };

        Ok(Self { inner, len })
    }
}

impl<R: Read> Read for ReadSeekMediaSource<R> {
    fn read(&mut self, buf: &mut [u8]) -> std::io::Result<usize> {
        self.inner.read(buf)
    }
}

impl<R: Seek> Seek for ReadSeekMediaSource<R> {
    fn seek(&mut self, pos: SeekFrom) -> std::io::Result<u64> {
        self.inner.seek(pos)
    }
}

impl<R: Read + Seek + Send + Sync> MediaSource for ReadSeekMediaSource<R> {
    fn is_seekable(&self) -> bool {
        true
    }

    fn byte_len(&self) -> Option<u64> {
        self.len
    }
}

fn probe_reader_and_pick_default_track<R>(reader: R) -> Result<(Box<dyn FormatReader>, Track)>
where
    R: Read + Seek + Send + Sync + 'static,
{
    let source = ReadSeekMediaSource::new(reader)?;
    let mss_opts = MediaSourceStreamOptions {
        // Must be > 32KiB and power-of-two to enable proper buffering for probing.
        buffer_len: 256 * 1024,
    };

    let mss = MediaSourceStream::new(Box::new(source), mss_opts);

    let hint = Hint::new();
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

fn make_decoder_for_track(track: &Track) -> Result<Box<dyn Decoder>> {
    let decoder_opts: DecoderOptions = Default::default();
    symphonia::default::get_codecs()
        .make(&track.codec_params, &decoder_opts)
        .map_err(|e| anyhow!(e))
        .context("failed to create decoder")
}

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

        if packet.track_id() != state.track_id {
            continue;
        }

        let decoded = match decoder.decode(&packet) {
            Ok(buf) => buf,
            Err(SymphoniaError::DecodeError(_)) => continue,
            Err(SymphoniaError::IoError(_)) => break,
            Err(e) => return Err(anyhow!(e)).context("decoder failure"),
        };

        let (interleaved, src_rate, channels) =
            decoded_to_interleaved_f32(&decoded, &mut state.sample_buf_f32)?;

        let mono_src = downmix_to_mono(&interleaved, channels);

        if src_rate == WHISPER_SAMPLE_RATE {
            emit_mono_chunks(&mono_src, state.opts.target_chunk_frames_16k, sink)?;
            continue;
        }

        ensure_resampler(state, src_rate)?;
        push_and_flush_resampler(state, &mono_src, sink)?;
    }

    Ok(())
}

fn next_packet(
    format: &mut Box<dyn FormatReader>,
) -> Result<Option<symphonia::core::formats::Packet>> {
    match format.next_packet() {
        Ok(p) => Ok(Some(p)),
        Err(SymphoniaError::IoError(_)) => Ok(None),
        Err(e) => Err(anyhow!(e)).context("failed reading packet"),
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

fn ensure_resampler(state: &mut DecodeState, src_rate: u32) -> Result<()> {
    if state.resampler.is_some() {
        return Ok(());
    }

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
        1,
    )
    .map_err(|e| anyhow!(e))
    .context("failed to init resampler")?;

    state.resampler = Some(rs);
    Ok(())
}

fn push_and_flush_resampler(
    state: &mut DecodeState,
    mono_src: &[f32],
    sink: &mut dyn SamplesSink,
) -> Result<()> {
    state.mono_src_acc.extend_from_slice(mono_src);

    loop {
        let in_max = state.resampler.as_ref().unwrap().input_frames_max();
        if state.mono_src_acc.len() < in_max {
            break;
        }

        let block: Vec<f32> = state.mono_src_acc.drain(..in_max).collect();
        let out = resample_block(state.resampler.as_mut().unwrap(), &block)?;

        for chunk in out.chunks(state.opts.target_chunk_frames_16k) {
            if !sink.on_samples(chunk)? {
                return Ok(());
            }
        }
    }

    Ok(())
}

fn resample_block(rs: &mut SincFixedIn<f32>, mono_src_block: &[f32]) -> Result<Vec<f32>> {
    let out = rs
        .process(&vec![mono_src_block.to_vec()], None)
        .map_err(|e| anyhow!(e))
        .context("resampler process failed")?;

    if out.len() != 1 {
        bail!("expected mono output from resampler");
    }

    Ok(out[0].clone())
}

fn finalize_stream(state: &mut DecodeState, sink: &mut dyn SamplesSink) -> Result<()> {
    let Some(rs) = state.resampler.as_mut() else {
        return Ok(());
    };

    if state.mono_src_acc.is_empty() {
        return Ok(());
    }

    let in_max = rs.input_frames_max();
    let rem = state.mono_src_acc.len() % in_max;

    if rem != 0 {
        state
            .mono_src_acc
            .resize(state.mono_src_acc.len() + (in_max - rem), 0.0);
    }

    while !state.mono_src_acc.is_empty() {
        let block: Vec<f32> = state.mono_src_acc.drain(..in_max).collect();
        let out = resample_block(rs, &block)?;

        for chunk in out.chunks(state.opts.target_chunk_frames_16k) {
            if !sink.on_samples(chunk)? {
                return Ok(());
            }
        }
    }

    Ok(())
}
