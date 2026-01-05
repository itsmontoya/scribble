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

use anyhow::anyhow;

use crate::Result;
use crate::backend::{Backend, BackendStream};
use crate::backends::whisper::WhisperBackend;
use crate::decoder::{SamplesSink, StreamDecodeOpts, decode_to_stream_from_read};
use crate::json_array_encoder::JsonArrayEncoder;
use crate::opts::Opts;
use crate::output_type::OutputType;
use crate::samples_rx::SamplesRx;
use crate::segment_encoder::SegmentEncoder;
use crate::vad::{VadProcessor, VadStreamReceiver};
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
    vad_model_path: Option<String>,
}

impl Scribble<WhisperBackend> {
    /// Create a new `Scribble` instance using the built-in Whisper backend.
    ///
    /// The default model used for transcription is the first successfully loaded model. Callers
    /// can select a specific model per transcription via `Opts::model_key`.
    pub fn new<I, P>(model_paths: I, vad_model_path: impl AsRef<str>) -> Result<Self>
    where
        I: IntoIterator<Item = P>,
        P: AsRef<str>,
    {
        let vad_model_path = vad_model_path.as_ref();
        let backend = WhisperBackend::new(model_paths, vad_model_path)?;
        Ok(Self {
            backend,
            vad_model_path: Some(vad_model_path.to_owned()),
        })
    }
}

impl<B: Backend> Scribble<B> {
    /// Create a new `Scribble` instance using a custom backend.
    pub fn with_backend(backend: B) -> Self {
        Self {
            backend,
            vad_model_path: None,
        }
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
    /// Note: The `Send + 'static` bounds mirror the decoder API.
    pub fn transcribe<R, W>(&self, r: R, w: W, opts: &Opts) -> Result<()>
    where
        R: Read + Send + 'static,
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

    fn transcribe_with_encoder<R, E>(&self, r: R, opts: &Opts, encoder: &mut E) -> Result<()>
    where
        R: Read + Send + 'static,
        E: SegmentEncoder,
    {
        let backend = &self.backend;
        let vad = Self::get_vad(self.vad_model_path.as_deref(), opts)?;

        // We start decoding on a dedicated thread so we can overlap I/O + decode with backend
        // inference. The main thread stays responsible for orchestration and error plumbing.
        let (mut rx, decode_handle) = Self::get_samples_rx(r, opts, vad)?;

        let mut stream = backend.create_stream(opts, encoder)?;

        // Consume decoded chunks as they arrive. This can run Whisper and emit segments while the
        // decode thread continues reading.
        while let Ok(chunk) = rx.recv() {
            let _ = stream.on_samples(&chunk)?;
        }

        // We always ask the backend stream to finish so it can flush any buffered segments.
        let finish_res = stream.finish();

        // If decode failed, we surface that error (but still prefer a backend/transcription error
        // if both happened). This keeps failure reporting stable and unsurprising.
        let decode_res: Result<()> = match decode_handle.join() {
            Ok(res) => res,
            Err(_) => Err(anyhow::anyhow!("audio decoder thread panicked").into()),
        };

        match (finish_res, decode_res) {
            (Ok(()), Ok(())) => Ok(()),
            (Ok(()), Err(err)) => Err(err),
            (Err(err), Ok(())) => Err(err),
            (Err(err), Err(decode_err)) => Err(anyhow::Error::from(err)
                .context(format!("{decode_err:#}"))
                .into()),
        }
    }

    fn get_vad(vad_model_path: Option<&str>, opts: &Opts) -> Result<Option<VadProcessor>> {
        if !opts.enable_voice_activity_detection {
            return Ok(None);
        }

        let Some(vad_model_path) = vad_model_path else {
            return Err(anyhow!(
                "VAD is enabled, but no VAD model path is configured for this Scribble instance"
            )
            .into());
        };

        // We initialize a fresh VAD context per transcription.
        //
        // Rationale:
        // - The upstream `whisper-rs` VAD bindings currently do not mark the VAD context as
        //   `Send`/`Sync`, making it awkward to store inside long-lived, shared server state.
        // - Even when upstream adds the necessary guarantees, we may want per-request state
        //   (or a pool) to enable true concurrency without a global lock.
        //
        // If/when the underlying library provides a clearly shareable VAD context, we can revisit
        // this to reuse or pool VAD processors for lower latency and less per-request overhead.
        Ok(Some(VadProcessor::new(vad_model_path)?))
    }

    fn get_samples_rx<R>(
        r: R,
        opts: &Opts,
        vad: Option<VadProcessor>,
    ) -> Result<(SamplesRx, std::thread::JoinHandle<Result<()>>)>
    where
        R: Read + Send + 'static,
    {
        // We use a bounded channel to keep memory usage predictable if the backend is slower than
        // decoding. This also makes backpressure explicit rather than relying on unbounded queues.
        let (tx, rx) = mpsc::sync_channel::<Vec<f32>>(512);
        let decode_opts = StreamDecodeOpts::default();
        let emit_frames = decode_opts.target_chunk_frames;

        let decode_handle = std::thread::spawn(move || -> Result<()> {
            let mut sink = ChannelSamplesSink { tx };
            decode_to_stream_from_read(r, decode_opts, &mut sink).map_err(Into::into)
        });

        let rx = if opts.enable_voice_activity_detection {
            // We wrap the decoder receiver so the main loop can stay unchanged when VAD is enabled.
            // `VadStreamReceiver` is responsible for buffering and flushing VAD state.
            let vad = vad.ok_or_else(|| anyhow!("VAD is enabled, but VAD failed to initialize"))?;
            let vad_rx = VadStreamReceiver::new(rx, vad, emit_frames);
            SamplesRx::Vad(vad_rx)
        } else {
            SamplesRx::Plain(rx)
        };

        Ok((rx, decode_handle))
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
        (Err(err), Err(close_err)) => Err(anyhow::Error::from(err)
            .context(format!("{close_err:#}"))
            .into()),
    }
}

struct ChannelSamplesSink {
    tx: mpsc::SyncSender<Vec<f32>>,
}

impl SamplesSink for ChannelSamplesSink {
    fn on_samples(&mut self, samples_16k_mono: &[f32]) -> anyhow::Result<bool> {
        // Copy into an owned buffer so the decoder thread can send it across threads safely.
        // Whisper inference runs on the receiver side.
        let buf = samples_16k_mono.to_vec();
        self.tx
            .send(buf)
            .map_err(|_| anyhow::anyhow!("decoder output channel disconnected"))?;
        Ok(true)
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    struct DummyBackend;

    struct DummyStream;

    impl BackendStream for DummyStream {
        fn on_samples(&mut self, _samples_16k_mono: &[f32]) -> Result<bool> {
            Ok(true)
        }

        fn finish(&mut self) -> Result<()> {
            Ok(())
        }
    }

    impl Backend for DummyBackend {
        type Stream<'a>
            = DummyStream
        where
            Self: 'a;

        fn transcribe_full(
            &self,
            _opts: &Opts,
            _encoder: &mut dyn SegmentEncoder,
            _samples: &[f32],
        ) -> Result<()> {
            Ok(())
        }

        fn create_stream<'a>(
            &'a self,
            _opts: &'a Opts,
            _encoder: &'a mut dyn SegmentEncoder,
        ) -> Result<Self::Stream<'a>> {
            Ok(DummyStream)
        }
    }

    fn default_opts(output_type: OutputType) -> Opts {
        Opts {
            model_key: None,
            enable_translate_to_english: false,
            enable_voice_activity_detection: false,
            language: None,
            output_type,
            incremental_min_window_seconds: 1,
        }
    }

    #[test]
    fn merge_run_and_close_prefers_run_error() {
        let run_err = anyhow::anyhow!("run failed");
        let close_err = anyhow::anyhow!("close failed");
        let err = merge_run_and_close(Err(run_err.into()), Err(close_err.into())).unwrap_err();
        let s = err.to_string();
        assert!(s.contains("close failed"));
        assert!(s.contains("run failed"));
    }

    #[test]
    fn merge_run_and_close_surfaces_close_error_when_run_ok() {
        let close_err = anyhow::anyhow!("close failed");
        let err = merge_run_and_close(Ok(()), Err(close_err.into())).unwrap_err();
        assert!(err.to_string().contains("close failed"));
    }

    #[test]
    fn get_vad_returns_none_when_disabled() -> anyhow::Result<()> {
        let scribble = Scribble::with_backend(DummyBackend);
        let opts = default_opts(OutputType::Json);
        let vad = Scribble::<DummyBackend>::get_vad(scribble.vad_model_path.as_deref(), &opts)?;
        assert!(vad.is_none());
        Ok(())
    }

    #[test]
    fn get_vad_errors_when_enabled_but_missing_path() {
        let scribble = Scribble::with_backend(DummyBackend);
        let mut opts = default_opts(OutputType::Json);
        opts.enable_voice_activity_detection = true;

        let err =
            Scribble::<DummyBackend>::get_vad(scribble.vad_model_path.as_deref(), &opts).err();
        let err = err.expect("expected get_vad() to error");
        assert!(err.to_string().contains("no VAD model path"));
    }

    struct NoopEncoder;

    impl SegmentEncoder for NoopEncoder {
        fn write_segment(&mut self, _seg: &crate::segments::Segment) -> Result<()> {
            Ok(())
        }

        fn close(&mut self) -> Result<()> {
            Ok(())
        }
    }

    struct FinishErrBackend;

    struct FinishErrStream;

    impl BackendStream for FinishErrStream {
        fn on_samples(&mut self, _samples_16k_mono: &[f32]) -> Result<bool> {
            Ok(true)
        }

        fn finish(&mut self) -> Result<()> {
            Err(anyhow::anyhow!("finish failed").into())
        }
    }

    impl Backend for FinishErrBackend {
        type Stream<'a>
            = FinishErrStream
        where
            Self: 'a;

        fn transcribe_full(
            &self,
            _opts: &Opts,
            _encoder: &mut dyn SegmentEncoder,
            _samples: &[f32],
        ) -> Result<()> {
            Ok(())
        }

        fn create_stream<'a>(
            &'a self,
            _opts: &'a Opts,
            _encoder: &'a mut dyn SegmentEncoder,
        ) -> Result<Self::Stream<'a>> {
            Ok(FinishErrStream)
        }
    }

    #[test]
    fn get_samples_rx_errors_when_vad_enabled_but_missing_processor() {
        let mut opts = default_opts(OutputType::Json);
        opts.enable_voice_activity_detection = true;

        let input = std::io::Cursor::new(Vec::<u8>::new());
        let err = Scribble::<DummyBackend>::get_samples_rx(input, &opts, None)
            .err()
            .expect("expected get_samples_rx() to error");
        assert!(err.to_string().contains("VAD failed to initialize"));
    }

    #[test]
    fn transcribe_with_encoder_surfaces_decoder_error_when_finish_ok() {
        let scribble = Scribble::with_backend(DummyBackend);
        let opts = default_opts(OutputType::Json);
        let input = std::io::Cursor::new(Vec::<u8>::new());
        let mut encoder = NoopEncoder;

        let err = scribble
            .transcribe_with_encoder(input, &opts, &mut encoder)
            .unwrap_err();
        assert!(err.to_string().contains("failed to probe media stream"));
    }

    #[test]
    fn transcribe_with_encoder_surfaces_finish_error_when_decode_ok() -> anyhow::Result<()> {
        let scribble = Scribble::with_backend(FinishErrBackend);
        let opts = default_opts(OutputType::Json);
        let input = std::fs::File::open("tests/fixtures/jfk.wav")?;
        let mut encoder = NoopEncoder;

        let err = scribble
            .transcribe_with_encoder(input, &opts, &mut encoder)
            .unwrap_err();
        assert!(err.to_string().contains("finish failed"));
        Ok(())
    }

    #[test]
    fn transcribe_with_encoder_prefers_finish_error_when_both_fail() {
        let scribble = Scribble::with_backend(FinishErrBackend);
        let opts = default_opts(OutputType::Json);
        let input = std::io::Cursor::new(Vec::<u8>::new());
        let mut encoder = NoopEncoder;

        let err = scribble
            .transcribe_with_encoder(input, &opts, &mut encoder)
            .unwrap_err();
        let s = err.to_string();
        assert!(s.contains("finish failed"));
        assert!(s.contains("failed to probe media stream"));
    }

    struct PanicRead;

    impl Read for PanicRead {
        fn read(&mut self, _buf: &mut [u8]) -> std::io::Result<usize> {
            panic!("boom");
        }
    }

    #[test]
    fn transcribe_with_encoder_reports_decoder_thread_panic() {
        let scribble = Scribble::with_backend(DummyBackend);
        let opts = default_opts(OutputType::Json);
        let mut encoder = NoopEncoder;

        let err = scribble
            .transcribe_with_encoder(PanicRead, &opts, &mut encoder)
            .unwrap_err();
        assert!(err.to_string().contains("decoder thread panicked"));
    }

    struct FailingWriter;

    impl std::io::Write for FailingWriter {
        fn write(&mut self, _buf: &[u8]) -> std::io::Result<usize> {
            Err(std::io::Error::other("write failed"))
        }

        fn flush(&mut self) -> std::io::Result<()> {
            Err(std::io::Error::other("flush failed"))
        }
    }

    #[test]
    fn transcribe_surfaces_json_close_error_when_run_ok() -> anyhow::Result<()> {
        let scribble = Scribble::with_backend(DummyBackend);
        let opts = default_opts(OutputType::Json);
        let input = std::fs::File::open("tests/fixtures/jfk.wav")?;

        let err = scribble
            .transcribe(input, FailingWriter, &opts)
            .unwrap_err();
        assert!(err.to_string().contains("write failed"));
        Ok(())
    }

    #[test]
    fn transcribe_surfaces_vtt_close_error_when_run_ok() -> anyhow::Result<()> {
        let scribble = Scribble::with_backend(DummyBackend);
        let opts = default_opts(OutputType::Vtt);
        let input = std::fs::File::open("tests/fixtures/jfk.wav")?;

        let err = scribble
            .transcribe(input, FailingWriter, &opts)
            .unwrap_err();
        assert!(
            err.to_string().contains("flush failed") || err.to_string().contains("write failed")
        );
        Ok(())
    }
}
