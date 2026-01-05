use std::collections::HashMap;
use std::path::Path;

use anyhow::{Result, anyhow, ensure};
use whisper_rs::WhisperContext;

use crate::backend::{Backend, BackendStream};
use crate::decoder::SamplesSink;
use crate::opts::Opts;
use crate::segment_encoder::SegmentEncoder;

mod ctx;
mod incremental;
mod logging;
mod segments;
mod token;

use incremental::BufferedSegmentTranscriber;
use segments::emit_segments;

/// Built-in backend powered by `whisper-rs` / `whisper.cpp`.
pub struct WhisperBackend {
    first_model_key: String,
    first_model: WhisperContext,
    models: HashMap<String, WhisperContext>,
    vad_model_path: String,
}

/// Streaming state for [`WhisperBackend`].
pub struct WhisperStream<'a> {
    inner: BufferedSegmentTranscriber<'a>,
}

impl BackendStream for WhisperStream<'_> {
    fn on_samples(&mut self, samples_16k_mono: &[f32]) -> Result<bool> {
        self.inner.on_samples(samples_16k_mono)
    }

    fn finish(&mut self) -> Result<()> {
        self.inner.finish()
    }
}

impl WhisperBackend {
    /// Load whisper.cpp model(s) and initialize a backend.
    ///
    /// Model keys are derived from the model filename (not the full path).
    pub fn new<I, P>(model_paths: I, vad_model_path: &str) -> Result<Self>
    where
        I: IntoIterator<Item = P>,
        P: AsRef<str>,
    {
        ensure!(
            !vad_model_path.trim().is_empty(),
            "VAD model path must be provided"
        );

        let vad_path = Path::new(vad_model_path);
        ensure!(
            vad_path.exists(),
            "VAD model not found at '{}'",
            vad_model_path
        );
        ensure!(
            vad_path.is_file(),
            "VAD model path is not a file: '{}'",
            vad_model_path
        );

        let mut first_model_key: Option<String> = None;
        let mut first_model: Option<WhisperContext> = None;
        let mut models = HashMap::new();

        for model_path in model_paths {
            let model_path = model_path.as_ref();
            ensure!(!model_path.trim().is_empty(), "model path must be provided");

            let model_key = Self::model_key_from_path(model_path)?;
            ensure!(
                first_model_key.as_deref() != Some(&model_key) && !models.contains_key(&model_key),
                "duplicate model key '{model_key}' derived from path '{model_path}'"
            );

            let ctx = ctx::get_context(model_path)?;
            if first_model_key.is_none() {
                first_model_key = Some(model_key);
                first_model = Some(ctx);
            } else {
                models.insert(model_key, ctx);
            }
        }

        let first_model_key = first_model_key
            .ok_or_else(|| anyhow!("at least one whisper model must be provided"))?;
        let first_model =
            first_model.ok_or_else(|| anyhow!("missing default whisper model context"))?;

        Ok(Self {
            first_model_key,
            first_model,
            models,
            vad_model_path: vad_model_path.to_owned(),
        })
    }

    /// Access the default Whisper context (the first loaded model).
    pub fn context(&self) -> &WhisperContext {
        &self.first_model
    }

    /// Access the configured VAD model path.
    pub fn vad_model_path(&self) -> &str {
        &self.vad_model_path
    }

    /// The model key used when `Opts::model_key` is `None`.
    pub fn default_model_key(&self) -> &str {
        self.first_model_key.as_str()
    }

    /// List available model keys (sorted).
    pub fn model_keys(&self) -> Vec<String> {
        let mut keys = Vec::with_capacity(self.models.len() + 1);
        keys.push(self.first_model_key.clone());
        keys.extend(self.models.keys().cloned());
        keys.sort_unstable();
        keys
    }

    fn model_key_from_path(model_path: &str) -> Result<String> {
        let path = Path::new(model_path);
        let Some(file_name) = path.file_name() else {
            return Err(anyhow!(
                "model path '{model_path}' does not have a filename"
            ));
        };
        let Some(file_name) = file_name.to_str() else {
            return Err(anyhow!(
                "model filename for path '{model_path}' is not valid UTF-8"
            ));
        };
        ensure!(
            !file_name.trim().is_empty(),
            "model filename for path '{model_path}' is empty"
        );
        Ok(file_name.to_owned())
    }

    fn selected_model_key<'a>(&'a self, opts: &'a Opts) -> Result<&'a str> {
        if let Some(key) = opts.model_key.as_deref() {
            if key == self.first_model_key || self.models.contains_key(key) {
                return Ok(key);
            }
            return Err(anyhow!(
                "unknown model key '{key}' (available: {})",
                self.available_model_keys()
            ));
        }

        Ok(self.first_model_key.as_str())
    }

    fn selected_context<'a>(&'a self, opts: &'a Opts) -> Result<&'a WhisperContext> {
        let key = self.selected_model_key(opts)?;
        if key == self.first_model_key {
            return Ok(&self.first_model);
        }
        self.models
            .get(key)
            .ok_or_else(|| anyhow!("selected model '{key}' was not loaded"))
    }

    fn available_model_keys(&self) -> String {
        let mut keys: Vec<&str> = self.models.keys().map(|k| k.as_str()).collect();
        keys.push(self.first_model_key.as_str());
        keys.sort_unstable();
        keys.join(", ")
    }
}

impl Backend for WhisperBackend {
    type Stream<'a>
        = WhisperStream<'a>
    where
        Self: 'a;

    fn transcribe_full(
        &self,
        opts: &Opts,
        encoder: &mut dyn SegmentEncoder,
        samples: &[f32],
    ) -> Result<()> {
        if samples.is_empty() {
            return Ok(());
        }

        let ctx = self.selected_context(opts)?;

        // VAD workflow is temporarily disabled while the streaming-focused version is reworked.
        let _ = opts.enable_voice_activity_detection;
        emit_segments(ctx, opts, samples, &mut |seg| encoder.write_segment(seg))
    }

    fn create_stream<'a>(
        &'a self,
        opts: &'a Opts,
        encoder: &'a mut dyn SegmentEncoder,
    ) -> Result<Self::Stream<'a>> {
        let ctx = self.selected_context(opts)?;

        // VAD workflow is temporarily disabled while the streaming-focused version is reworked.
        let _ = opts.enable_voice_activity_detection;
        Ok(WhisperStream {
            inner: BufferedSegmentTranscriber::new(ctx, opts, encoder),
        })
    }
}
