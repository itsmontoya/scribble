//! High-level API for running transcriptions with Scribble.
//!
//! We expose a single, ergonomic entry point (`Scribble`) that wraps the lower-level
//! backend, decoding, and encoding logic.
//!
//! The intent is:
//! - We initialize a backend once (often expensive).
//! - We reuse that backend to transcribe multiple inputs.
//! - Callers choose output format and behavior via `Opts`.
//!
//! This module is deliberately “high level”: it wires up decoding → backend → encoder,
//! while keeping the lower-level pieces testable in their own modules.

use std::io::{BufWriter, Read, Write};
use std::sync::mpsc;

use anyhow::{Result, anyhow};

use crate::backend::{Backend, BackendStream};
use crate::backends::whisper::WhisperBackend;
use crate::decoder::{SamplesSink, StreamDecodeOpts, decode_to_stream_from_read};
use crate::json_array_encoder::JsonArrayEncoder;
use crate::opts::Opts;
use crate::output_type::OutputType;
use crate::segment_encoder::SegmentEncoder;
use crate::vad::{VadProcessor, VadStream};
use crate::vtt_encoder::VttEncoder;

/// The main high-level transcription entry point.
///
/// `Scribble` owns the long-lived resources required for transcription:
/// - a backend instance (model(s) + runtime state)
///
/// Typical usage:
/// - Construct once (model loading happens here).
/// - Call `transcribe` many times with different inputs and outputs.
pub struct Scribble<B: Backend> {
    backend: B,
    vad: Option<VadProcessor>,
}

impl Scribble<WhisperBackend> {
    /// Create a new `Scribble` instance using the built-in Whisper backend.
    pub fn new(model_path: impl AsRef<str>, vad_model_path: impl AsRef<str>) -> Result<Self> {
        let backend = WhisperBackend::new(model_path.as_ref(), vad_model_path.as_ref())?;
        let vad = Some(VadProcessor::new(vad_model_path.as_ref())?);
        Ok(Self { backend, vad })
    }
}

impl<B: Backend> Scribble<B> {
    /// Create a new `Scribble` instance using a custom backend.
    pub fn with_backend(backend: B) -> Self {
        Self { backend, vad: None }
    }

    /// Transcribe an input stream and write the result to an output writer.
    ///
    /// We accept a generic `Read` input rather than a filename so callers can pass:
    /// - `File`
    /// - stdin
    /// - sockets / HTTP bodies
    /// - any other byte stream
    ///
    /// We decode audio into a mono 16kHz stream (whisper.cpp’s expected format) using the
    /// `decoder` module, optionally apply VAD, and then run Whisper and encode segments.
    ///
    /// Note: The `Send + Sync + 'static` bounds mirror the decoder API. If you later relax
    /// those constraints in `decoder`, we can simplify these bounds too.
    pub fn transcribe<R, W>(&mut self, r: R, w: W, opts: &Opts) -> Result<()>
    where
        R: Read + Send + Sync + 'static,
        W: Write,
    {
        // Buffer output for efficiency (especially important for stdout).
        let writer = BufWriter::new(w);

        // Select an encoder based on the requested output type.
        // We keep this explicit (no trait objects) to avoid lifetime surprises.
        match opts.output_type {
            OutputType::Json => {
                let mut encoder = JsonArrayEncoder::new(writer);
                let run_res = self.transcribe_with_encoder(r, opts, &mut encoder);
                merge_run_and_close(run_res, encoder.close())
            }
            OutputType::Vtt => {
                let mut encoder = VttEncoder::new(writer);
                let run_res = self.transcribe_with_encoder(r, opts, &mut encoder);
                merge_run_and_close(run_res, encoder.close())
            }
        }
    }

    fn transcribe_with_encoder<R, E>(&mut self, r: R, opts: &Opts, encoder: &mut E) -> Result<()>
    where
        R: Read + Send + Sync + 'static,
        E: SegmentEncoder,
    {
        // Streaming-friendly path: run decode on a separate thread so reading/decoding can
        // continue even while Whisper inference is running.
        let (tx, rx) = mpsc::sync_channel::<Vec<f32>>(512);
        let decode_opts = StreamDecodeOpts::default();
        let emit_frames = decode_opts.target_chunk_frames;

        let decode_handle = std::thread::spawn(move || -> Result<()> {
            let mut sink = ChannelSamplesSink { tx };
            decode_to_stream_from_read(r, decode_opts, &mut sink)
        });

        let mut stream = self.backend.create_stream(opts, encoder)?;

        if opts.enable_voice_activity_detection {
            let vad = self
                .vad
                .as_mut()
                .ok_or_else(|| anyhow!("VAD is enabled, but no VAD processor is configured"))?;

            // Feed the backend incrementally, but buffer enough audio for VAD to be meaningful.
            let mut vad_stream = VadStream::new(vad);

            while let Ok(chunk) = rx.recv() {
                vad_stream.push(&chunk)?;

                while let Some(out_chunk) = vad_stream.peek_chunk(emit_frames) {
                    let _ = stream.on_samples(out_chunk)?;
                    vad_stream.consume_chunk(emit_frames);
                }
            }

            // Flush remaining buffered audio through VAD and emit whatever is left.
            vad_stream.flush()?;

            while let Some(out_chunk) = vad_stream.peek_chunk(emit_frames) {
                let _ = stream.on_samples(out_chunk)?;
                vad_stream.consume_chunk(emit_frames);
            }

            if let Some(rem) = vad_stream.peek_remainder() {
                let _ = stream.on_samples(rem)?;
                vad_stream.consume_remainder();
            }
        } else {
            // Consume decoded chunks as they arrive. This can run Whisper and emit segments while the
            // decode thread continues reading.
            while let Ok(chunk) = rx.recv() {
                let _ = stream.on_samples(&chunk)?;
            }
        }

        // Ensure we flush any final segments.
        let finish_res = stream.finish();

        // If decode failed, surface that error (but still prefer a transcription error if present).
        let decode_res: Result<()> = match decode_handle.join() {
            Ok(res) => res,
            Err(_) => Err(anyhow::anyhow!("decoder thread panicked")),
        };

        match (finish_res, decode_res) {
            (Ok(()), Ok(())) => Ok(()),
            (Ok(()), Err(err)) => Err(err),
            (Err(err), Ok(())) => Err(err),
            (Err(err), Err(decode_err)) => Err(err.context(decode_err)),
        }
    }

    /// Access the configured backend.
    pub fn backend(&self) -> &B {
        &self.backend
    }

    /// Access the configured backend mutably.
    pub fn backend_mut(&mut self) -> &mut B {
        &mut self.backend
    }
}

fn merge_run_and_close(run_res: Result<()>, close_res: Result<()>) -> Result<()> {
    match (run_res, close_res) {
        (Ok(()), Ok(())) => Ok(()),
        (Ok(()), Err(close_err)) => Err(close_err),
        (Err(err), Ok(())) => Err(err),
        (Err(err), Err(close_err)) => Err(err.context(close_err)),
    }
}

struct ChannelSamplesSink {
    tx: mpsc::SyncSender<Vec<f32>>,
}

impl SamplesSink for ChannelSamplesSink {
    fn on_samples(&mut self, samples_16k_mono: &[f32]) -> Result<bool> {
        // Copy into an owned buffer so the decoder thread can send it across threads safely.
        // Whisper inference runs on the receiver side.
        let buf = samples_16k_mono.to_vec();
        self.tx
            .send(buf)
            .map_err(|_| anyhow::anyhow!("decoder output channel disconnected"))?;
        Ok(true)
    }
}
