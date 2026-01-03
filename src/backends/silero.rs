use std::collections::HashMap;
use std::fs;
use std::path::Path;

use anyhow::{Context, Result, anyhow, ensure};
use ort::session::Session;
use ort::tensor::TensorElementType;
use ort::value::Tensor;
use ort::value::ValueType;

use crate::audio_pipeline::WHISPER_SAMPLE_RATE;
use crate::backend::{Backend, BackendStream};
use crate::decoder::SamplesSink;
use crate::opts::Opts;
use crate::segment_encoder::SegmentEncoder;
use crate::segments::Segment;

/// Default alphabet for CTC-style Silero English models.
///
/// Notes:
/// - Index 0 is reserved for the CTC blank token.
/// - Some models use `|` for “space”; we normalize that to `" "` during decoding.
const DEFAULT_EN_ALPHABET: &[&str] = &[
    "_", "a", "b", "c", "d", "e", "f", "g", "h", "i", "j", "k", "l", "m", "n", "o", "p", "q", "r",
    "s", "t", "u", "v", "w", "x", "y", "z", "'", " ", "|",
];

/// Configuration for [`SileroBackend`].
#[derive(Debug, Clone)]
pub struct SileroConfig {
    /// Output tensor index to read logits/token IDs from.
    ///
    /// Most ONNX Silero models have exactly one output, so `0` is a sensible default.
    pub output_index: usize,

    /// Decoder strategy used to turn frame-level logits into tokens.
    pub decoder: SileroDecoder,

    /// Beam width used when `decoder` is [`SileroDecoder::BeamSearch`].
    pub beam_width: usize,

    /// Per-frame token shortlist when `decoder` is [`SileroDecoder::BeamSearch`].
    pub frame_top_k: usize,

    /// CTC blank token ID (almost always `0` for Silero exports).
    pub blank_id: usize,

    /// Optional word-boundary token (often `" "` or `"|"`) to bias beam search toward better spacing.
    ///
    /// If `None`, we auto-detect a boundary token by trying `" "` and then `"|"`.
    pub word_boundary_token: Option<String>,

    /// Log-space bonus added when the word-boundary token is emitted during beam search.
    ///
    /// Set to `0.0` to disable.
    pub word_boundary_bonus: f32,

    /// Minimum number of emitted tokens between word-boundary tokens for the bonus to apply.
    ///
    /// This helps avoid over-segmentation (spaces between subword pieces).
    pub word_boundary_min_gap: usize,

    /// Token strings used to decode model outputs.
    ///
    /// For CTC-style models:
    /// - `alphabet[0]` is the blank token.
    /// - other entries map argmax indices to text.
    pub alphabet: Vec<String>,
}

#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum SileroDecoder {
    Greedy,
    BeamSearch,
}

impl Default for SileroConfig {
    fn default() -> Self {
        Self {
            output_index: 0,
            decoder: SileroDecoder::Greedy,
            beam_width: 16,
            frame_top_k: 32,
            blank_id: 0,
            word_boundary_token: None,
            word_boundary_bonus: 0.0,
            word_boundary_min_gap: 4,
            alphabet: DEFAULT_EN_ALPHABET
                .iter()
                .map(|s| (*s).to_owned())
                .collect(),
        }
    }
}

impl SileroConfig {
    /// Replace the tokenizer/alphabet with entries loaded from a newline-delimited vocab file.
    ///
    /// Supported formats:
    /// - Newline-delimited tokens (one token per line)
    /// - JSON array of strings (e.g. `["<blank>", "a", ...]`)
    /// - JSON object with a `labels` or `tokens` array
    pub fn with_vocab_file(mut self, path: impl AsRef<Path>) -> Result<Self> {
        let bytes = fs::read(path.as_ref()).with_context(|| {
            format!(
                "failed to read Silero vocab file '{}'",
                path.as_ref().display()
            )
        })?;

        let s = String::from_utf8(bytes).context("Silero vocab file was not valid UTF-8")?;
        let tokens = parse_vocab_text(&s).with_context(|| {
            format!(
                "failed to parse Silero vocab file '{}'",
                path.as_ref().display()
            )
        })?;

        ensure!(
            !tokens.is_empty(),
            "Silero vocab file '{}' contained no tokens",
            path.as_ref().display()
        );

        self.alphabet = tokens;
        Ok(self)
    }
}

fn parse_vocab_text(s: &str) -> Result<Vec<String>> {
    let trimmed = s.trim_start();

    // Try JSON first (Silero publishes `*_labels.json` for STT models).
    if (trimmed.starts_with('[') || trimmed.starts_with('{'))
        && let Ok(v) = serde_json::from_str::<serde_json::Value>(trimmed)
    {
        if let Some(arr) = v.as_array() {
            return Ok(arr
                .iter()
                .filter_map(|x| x.as_str().map(|t| t.to_owned()))
                .collect());
        }

        if let Some(obj) = v.as_object() {
            for key in ["labels", "tokens"] {
                if let Some(arr) = obj.get(key).and_then(|x| x.as_array()) {
                    return Ok(arr
                        .iter()
                        .filter_map(|x| x.as_str().map(|t| t.to_owned()))
                        .collect());
                }
            }
        }
    }

    // Fallback: newline-delimited.
    Ok(s.lines()
        .map(|l| l.trim())
        .filter(|l| !l.is_empty())
        .map(|l| l.to_owned())
        .collect())
}

/// Experimental backend that runs a Silero ASR ONNX model via ONNX Runtime (`ort`).
///
/// This backend currently emits a single [`Segment`] for the full audio buffer.
pub struct SileroBackend {
    session: Session,
    cfg: SileroConfig,
}

/// Streaming state for [`SileroBackend`].
pub struct SileroStream<'a> {
    backend: &'a mut SileroBackend,
    opts: &'a Opts,
    encoder: &'a mut dyn SegmentEncoder,
    buffered: Vec<f32>,
}

impl SamplesSink for SileroStream<'_> {
    fn on_samples(&mut self, samples_16k_mono: &[f32]) -> Result<bool> {
        self.buffered.extend_from_slice(samples_16k_mono);
        Ok(true)
    }
}

impl BackendStream for SileroStream<'_> {
    fn finish(&mut self) -> Result<()> {
        if self.buffered.is_empty() {
            return Ok(());
        }
        self.backend
            .transcribe_full(self.opts, self.encoder, &self.buffered)
    }
}

impl SileroBackend {
    /// Load an ONNX model from disk and initialize a backend.
    pub fn new(model_path: impl AsRef<Path>) -> Result<Self> {
        Self::with_config(model_path, SileroConfig::default())
    }

    /// Load an ONNX model from disk and initialize a backend with custom configuration.
    pub fn with_config(model_path: impl AsRef<Path>, cfg: SileroConfig) -> Result<Self> {
        ensure!(
            WHISPER_SAMPLE_RATE == 16_000,
            "SileroBackend assumes 16kHz input; got WHISPER_SAMPLE_RATE={WHISPER_SAMPLE_RATE}"
        );
        ensure!(!cfg.alphabet.is_empty(), "alphabet must not be empty");

        let session = Session::builder()
            .context("failed to create ONNX Runtime session builder")?
            .commit_from_file(model_path.as_ref())
            .with_context(|| {
                format!(
                    "failed to load Silero ONNX model from '{}'",
                    model_path.as_ref().display()
                )
            })?;

        if looks_like_silero_vad(&session) {
            return Err(anyhow!(
                "the provided ONNX model looks like Silero VAD (inputs include `state`+`sr`, output is `[-1, 1]`); SileroBackend expects an ASR/STT model"
            ));
        }

        if std::env::var_os("SCRIBBLE_DEBUG_SILERO").is_some() {
            eprintln!("Silero ONNX inputs:");
            for i in &session.inputs {
                eprintln!("  - {}: {:?}", i.name, i.input_type);
            }
            eprintln!("Silero ONNX outputs:");
            for o in &session.outputs {
                eprintln!("  - {}: {:?}", o.name, o.output_type);
            }
        }

        Ok(Self { session, cfg })
    }

    fn infer_text(&mut self, samples_16k_mono: &[f32]) -> Result<String> {
        // Most Silero ASR exports accept a rank-2 tensor.
        //
        // For `v5_en`, the input dim symbols are typically ["batch", "samples"].
        let samples_len = samples_16k_mono.len();
        let samples = samples_16k_mono.to_vec();

        let input_rank = silero_input_rank(&self.session)?;
        ensure!(
            input_rank == 2,
            "SileroBackend currently supports rank-2 inputs; model expects rank {input_rank}"
        );

        // [batch=1, samples]
        if std::env::var_os("SCRIBBLE_DEBUG_SILERO").is_some() {
            eprintln!("Silero attempt: input shape [1, {samples_len}]");
        }
        let input = Tensor::from_array(([1usize, samples_len], samples.clone().into_boxed_slice()))
            .context("failed to build ONNX input tensor ([1, samples])")?;
        let first_err = match self.run_and_decode(input) {
            Ok(text) => return Ok(text),
            Err(err) => {
                if std::env::var_os("SCRIBBLE_DEBUG_SILERO").is_some() {
                    eprintln!("Silero attempt failed ([1, {samples_len}]): {err:#}");
                }
                err
            }
        };

        // If the model ran but we can't decode due to vocab mismatch, don't bother trying alternative shapes.
        if first_err.to_string().contains("vocab size") {
            return Err(first_err);
        }

        // [samples, batch=1]
        if std::env::var_os("SCRIBBLE_DEBUG_SILERO").is_some() {
            eprintln!("Silero attempt: input shape [{samples_len}, 1]");
        }
        let input = Tensor::from_array(([samples_len, 1usize], samples.into_boxed_slice()))
            .context("failed to build ONNX input tensor ([samples, 1])")?;
        match self.run_and_decode(input) {
            Ok(text) => Ok(text),
            Err(err) => Err(err.context(first_err)),
        }
    }

    fn run_and_decode(&mut self, audio: Tensor<f32>) -> Result<String> {
        let inputs = build_session_inputs(&self.session, audio)?;

        // `SessionOutputs` borrows the session; keep it scoped so we can run again if needed.
        let text = {
            let outputs = self
                .session
                .run(inputs)
                .context("failed to run Silero ONNX model")?;

            ensure!(
                self.cfg.output_index < outputs.len(),
                "invalid output_index {} (model has {} outputs)",
                self.cfg.output_index,
                outputs.len()
            );

            let v = &outputs[self.cfg.output_index];
            decode_silero_output(v, &self.cfg)?
        };

        Ok(text)
    }
}

fn build_session_inputs(
    session: &Session,
    audio: Tensor<f32>,
) -> Result<Vec<(String, ort::session::SessionInputValue<'static>)>> {
    // We prefer named inputs so models with extra scalars (e.g. `sr`) work without relying on ordering.
    // The most common Silero signatures are:
    // - inputs: ["input"] (audio)
    // - inputs: ["input", "sr"] (audio + sample rate)
    let mut audio_input_name: Option<String> = None;
    let mut state_input: Option<(String, ValueType)> = None;
    let mut sr_input: Option<(String, TensorElementType)> = None;

    for input in &session.inputs {
        if input.name == "sr" {
            let ValueType::Tensor { ty, .. } = input.input_type else {
                return Err(anyhow!(
                    "Silero input 'sr' is not a tensor: {}",
                    input.input_type
                ));
            };
            sr_input = Some((input.name.clone(), ty));
            continue;
        }

        if input.name == "state" {
            state_input = Some((input.name.clone(), input.input_type.clone()));
            continue;
        }

        if audio_input_name.is_some() {
            return Err(anyhow!(
                "Silero model has multiple non-'sr' inputs; unsupported: {:?}",
                session
                    .inputs
                    .iter()
                    .map(|i| i.name.as_str())
                    .collect::<Vec<_>>()
            ));
        }
        audio_input_name = Some(input.name.clone());
    }

    let audio_name = audio_input_name.unwrap_or_else(|| "input".to_owned());

    // We need `SessionInputValue<'static>` for the Vec conversion; use owned values.
    let mut inputs: Vec<(String, ort::session::SessionInputValue<'static>)> = Vec::new();
    inputs.push((
        audio_name,
        ort::session::SessionInputValue::Owned(audio.into_dyn()),
    ));

    if let Some((name, ty)) = state_input {
        let ValueType::Tensor { ty, shape, .. } = ty else {
            return Err(anyhow!("Silero input 'state' is not a tensor: {ty}"));
        };

        let dims: Vec<usize> = shape
            .iter()
            .map(|d| {
                if *d < 0 {
                    Ok(1usize)
                } else {
                    (*d).try_into()
                        .map_err(|_| anyhow!("state shape dimension did not fit in usize: {d}"))
                }
            })
            .collect::<Result<_>>()?;

        let numel = dims.iter().copied().product::<usize>();
        let state_value = match ty {
            TensorElementType::Float32 => {
                Tensor::from_array((dims, vec![0.0f32; numel].into_boxed_slice()))
                    .context("failed to build 'state' input tensor (f32)")?
                    .into_dyn()
            }
            TensorElementType::Float64 => {
                Tensor::from_array((dims, vec![0.0f64; numel].into_boxed_slice()))
                    .context("failed to build 'state' input tensor (f64)")?
                    .into_dyn()
            }
            other => {
                return Err(anyhow!(
                    "unsupported 'state' input tensor element type: {other}"
                ));
            }
        };

        inputs.push((name, ort::session::SessionInputValue::Owned(state_value)));
    }

    if let Some((name, ty)) = sr_input {
        let sr = WHISPER_SAMPLE_RATE as i64;
        let sr_value = match ty {
            TensorElementType::Int64 => Tensor::from_array(((), vec![sr].into_boxed_slice()))
                .context("failed to build 'sr' input tensor (i64)")?
                .into_dyn(),
            TensorElementType::Int32 => {
                Tensor::from_array(((), vec![sr as i32].into_boxed_slice()))
                    .context("failed to build 'sr' input tensor (i32)")?
                    .into_dyn()
            }
            other => {
                return Err(anyhow!(
                    "unsupported 'sr' input tensor element type: {other}"
                ));
            }
        };
        inputs.push((name, ort::session::SessionInputValue::Owned(sr_value)));
    }

    Ok(inputs)
}

fn looks_like_silero_vad(session: &Session) -> bool {
    let has_state_input = session.inputs.iter().any(|i| i.name == "state");
    let has_sr_input = session.inputs.iter().any(|i| i.name == "sr");
    let has_state_output = session
        .outputs
        .iter()
        .any(|o| o.name == "stateN" || o.name == "state");

    let output_is_time_by_one = session.outputs.iter().any(|o| {
        matches!(
            o.output_type,
            ValueType::Tensor { ty: TensorElementType::Float32, ref shape, .. }
                if shape.len() == 2 && shape[0] == -1 && shape[1] == 1
        )
    });

    has_state_input && has_sr_input && has_state_output && output_is_time_by_one
}

impl Backend for SileroBackend {
    type Stream<'a>
        = SileroStream<'a>
    where
        Self: 'a;

    fn transcribe_full(
        &mut self,
        opts: &Opts,
        encoder: &mut dyn SegmentEncoder,
        samples_16k_mono: &[f32],
    ) -> Result<()> {
        if samples_16k_mono.is_empty() {
            return Ok(());
        }

        let text = self.infer_text(samples_16k_mono)?;
        let duration_seconds = samples_16k_mono.len() as f32 / WHISPER_SAMPLE_RATE as f32;

        let seg = Segment {
            start_seconds: 0.0,
            end_seconds: duration_seconds,
            text,
            tokens: vec![],
            language_code: opts.language.clone().unwrap_or_else(|| "und".to_owned()),
            next_speaker_turn: false,
        };

        encoder.write_segment(&seg)
    }

    fn create_stream<'a>(
        &'a mut self,
        opts: &'a Opts,
        encoder: &'a mut dyn SegmentEncoder,
    ) -> Result<Self::Stream<'a>> {
        Ok(SileroStream {
            backend: self,
            opts,
            encoder,
            buffered: Vec::new(),
        })
    }
}

fn decode_silero_output(output: &ort::value::DynValue, cfg: &SileroConfig) -> Result<String> {
    if let Ok((shape, ids)) = output.try_extract_tensor::<i64>() {
        return decode_token_ids(shape, ids, &cfg.alphabet);
    }
    if let Ok((shape, ids)) = output.try_extract_tensor::<i32>() {
        return decode_token_ids(shape, ids, &cfg.alphabet);
    }
    if let Ok((shape, logits)) = output.try_extract_tensor::<f32>() {
        return decode_f32_output(shape, logits, cfg);
    }

    Err(anyhow!(
        "unsupported Silero output value type: {}",
        output.dtype()
    ))
}

fn decode_f32_output(
    shape: &ort::tensor::Shape,
    values: &[f32],
    cfg: &SileroConfig,
) -> Result<String> {
    // Some Silero exports produce token IDs as `f32` with shape [time, 1] or [time].
    // Others produce logits with shape [time, vocab] (or [1, time, vocab]).
    let dims: Vec<usize> = shape
        .iter()
        .map(|d| {
            (*d).try_into()
                .map_err(|_| anyhow!("output shape contained a negative dimension: {d}"))
        })
        .collect::<Result<Vec<_>>>()?;

    match dims.as_slice() {
        [time] => decode_token_ids_f32(values, *time, &cfg.alphabet),
        [time, 1] => decode_token_ids_f32(values, *time, &cfg.alphabet),
        _ => decode_logits(shape, values, cfg),
    }
}

fn decode_token_ids_f32(values: &[f32], time: usize, alphabet: &[String]) -> Result<String> {
    ensure!(
        values.len() >= time,
        "token-id buffer too small for shape [{time}] (len={})",
        values.len()
    );

    let mut out = String::new();
    let mut prev: Option<usize> = None;

    for &v in values.iter().take(time) {
        let idx = v.round() as isize;
        if idx < 0 {
            continue;
        }
        let idx = idx as usize;

        // Greedy CTC collapse:
        // - remove repeats
        // - drop blanks (index 0)
        if prev == Some(idx) {
            continue;
        }
        prev = Some(idx);
        if idx == 0 {
            continue;
        }

        let token = alphabet.get(idx).ok_or_else(|| {
            anyhow!(
                "token id {idx} out of range (alphabet size {})",
                alphabet.len()
            )
        })?;
        push_token(&mut out, token);
    }

    Ok(normalize_text(out))
}

fn decode_token_ids<T>(shape: &ort::tensor::Shape, ids: &[T], alphabet: &[String]) -> Result<String>
where
    T: Copy + TryInto<usize>,
{
    let _ = shape; // shape is currently unused for token-id outputs
    let mut out = String::new();
    let mut prev: Option<usize> = None;
    for v in ids.iter().copied() {
        let idx: usize = v
            .try_into()
            .map_err(|_| anyhow!("token id did not fit into usize"))?;

        if prev == Some(idx) {
            continue;
        }
        prev = Some(idx);

        if idx == 0 {
            continue;
        }
        let token = alphabet.get(idx).ok_or_else(|| {
            anyhow!(
                "token id {idx} out of range (alphabet size {})",
                alphabet.len()
            )
        })?;
        push_token(&mut out, token);
    }
    Ok(normalize_text(out))
}

fn decode_logits(shape: &ort::tensor::Shape, logits: &[f32], cfg: &SileroConfig) -> Result<String> {
    let dims: Vec<usize> = shape
        .iter()
        .map(|d| {
            (*d).try_into()
                .map_err(|_| anyhow!("logits shape contained a negative dimension: {d}"))
        })
        .collect::<Result<Vec<_>>>()?;

    match dims.as_slice() {
        [time, vocab] => decode_logits_2d_with_cfg(*time, *vocab, logits, cfg),
        [batch, time, vocab] if *batch == 1 => {
            decode_logits_2d_with_cfg(*time, *vocab, logits, cfg)
        }
        [time, batch, vocab] if *batch == 1 => {
            decode_logits_2d_with_cfg(*time, *vocab, logits, cfg)
        }
        [time, one, vocab] if *one == 1 => decode_logits_2d_with_cfg(*time, *vocab, logits, cfg),
        [batch, time, one, vocab] if *batch == 1 && *one == 1 => {
            decode_logits_2d_with_cfg(*time, *vocab, logits, cfg)
        }
        _ => Err(anyhow!(
            "unsupported Silero logits shape {:?} (expected [time, vocab] or batch=1 variants)",
            dims
        )),
    }
}

fn decode_logits_2d_with_cfg(
    time: usize,
    vocab: usize,
    logits: &[f32],
    cfg: &SileroConfig,
) -> Result<String> {
    match cfg.decoder {
        SileroDecoder::Greedy => decode_logits_2d_greedy(time, vocab, logits, &cfg.alphabet),
        SileroDecoder::BeamSearch => decode_logits_2d_beam(time, vocab, logits, cfg),
    }
}

fn decode_logits_2d_greedy(
    time: usize,
    vocab: usize,
    logits: &[f32],
    alphabet: &[String],
) -> Result<String> {
    let mut out = String::new();
    let mut prev: Option<usize> = None;

    for t in 0..time {
        let start = t * vocab;
        let row = &logits[start..start + vocab];

        let mut best_i = 0usize;
        let mut best_v = f32::NEG_INFINITY;
        for (i, &v) in row.iter().enumerate() {
            if v > best_v {
                best_v = v;
                best_i = i;
            }
        }

        // Greedy CTC collapse:
        // - remove repeats
        // - drop blanks (index 0)
        if prev == Some(best_i) {
            continue;
        }
        prev = Some(best_i);
        if best_i == 0 {
            continue;
        }
        let token = &alphabet[best_i];
        push_token(&mut out, token);
    }

    Ok(normalize_text(out))
}

fn decode_logits_2d_beam(
    time: usize,
    vocab: usize,
    logits: &[f32],
    cfg: &SileroConfig,
) -> Result<String> {
    ensure!(cfg.beam_width > 0, "beam_width must be > 0");
    ensure!(cfg.frame_top_k > 0, "frame_top_k must be > 0");
    ensure!(
        cfg.blank_id < cfg.alphabet.len(),
        "blank_id {} out of range (alphabet size {})",
        cfg.blank_id,
        cfg.alphabet.len()
    );
    ensure!(
        vocab == cfg.alphabet.len(),
        "logits vocab size ({vocab}) does not match vocab/alphabet size ({})",
        cfg.alphabet.len()
    );

    let word_boundary_id = word_boundary_id(cfg);

    // Prefix beam state: p_b (ends with blank), p_nb (ends with non-blank).
    let mut beam: HashMap<Vec<usize>, (f32, f32)> = HashMap::new();
    beam.insert(Vec::new(), (0.0, f32::NEG_INFINITY)); // log(1), log(0)

    for t in 0..time {
        let start = t * vocab;
        let row = &logits[start..start + vocab];

        let log_z = log_sum_exp(row);
        let blank_logp = row[cfg.blank_id] - log_z;

        let candidates = top_k_indices(row, cfg.frame_top_k, cfg.blank_id);

        let mut next: HashMap<Vec<usize>, (f32, f32)> = HashMap::new();

        for (prefix, (pb, pnb)) in beam.iter() {
            let p_total = log_add_exp(*pb, *pnb);

            // Emit blank: prefix unchanged.
            let entry = next
                .entry(prefix.clone())
                .or_insert((f32::NEG_INFINITY, f32::NEG_INFINITY));
            entry.0 = log_add_exp(entry.0, p_total + blank_logp);

            for &c in &candidates {
                if c == cfg.blank_id {
                    continue;
                }
                let mut logp_c = row[c] - log_z;
                if cfg.word_boundary_bonus != 0.0 && word_boundary_id == Some(c) {
                    let ok_gap = match last_boundary_age(prefix, c) {
                        Some(age) => age >= cfg.word_boundary_min_gap,
                        None => true,
                    };
                    if ok_gap && !prefix.is_empty() {
                        logp_c += cfg.word_boundary_bonus;
                    }
                }

                let last = prefix.last().copied();
                if last == Some(c) {
                    // Repeating char without going through blank stays on the same prefix.
                    let entry_same = next
                        .entry(prefix.clone())
                        .or_insert((f32::NEG_INFINITY, f32::NEG_INFINITY));
                    entry_same.1 = log_add_exp(entry_same.1, *pnb + logp_c);

                    // Repeating char after a blank extends.
                    let mut extended = prefix.clone();
                    extended.push(c);
                    let entry_ext = next
                        .entry(extended)
                        .or_insert((f32::NEG_INFINITY, f32::NEG_INFINITY));
                    entry_ext.1 = log_add_exp(entry_ext.1, *pb + logp_c);
                } else {
                    let mut extended = prefix.clone();
                    extended.push(c);
                    let entry_ext = next
                        .entry(extended)
                        .or_insert((f32::NEG_INFINITY, f32::NEG_INFINITY));
                    entry_ext.1 = log_add_exp(entry_ext.1, p_total + logp_c);
                }
            }
        }

        // Prune to top `beam_width`.
        let mut scored: Vec<(Vec<usize>, (f32, f32), f32)> = next
            .into_iter()
            .map(|(p, (pb, pnb))| {
                let score = log_add_exp(pb, pnb);
                (p, (pb, pnb), score)
            })
            .collect();

        scored.sort_by(|a, b| b.2.total_cmp(&a.2));
        scored.truncate(cfg.beam_width);

        beam = scored.into_iter().map(|(p, st, _)| (p, st)).collect();
    }

    let (best_prefix, _) = beam
        .into_iter()
        .max_by(|(_, (a_pb, a_pnb)), (_, (b_pb, b_pnb))| {
            log_add_exp(*a_pb, *a_pnb).total_cmp(&log_add_exp(*b_pb, *b_pnb))
        })
        .unwrap_or((Vec::new(), (f32::NEG_INFINITY, f32::NEG_INFINITY)));

    // Final CTC collapse (remove repeats) + drop blanks.
    let mut collapsed: Vec<usize> = Vec::new();
    for &id in &best_prefix {
        if id == cfg.blank_id {
            continue;
        }
        if collapsed.last().copied() == Some(id) {
            continue;
        }
        collapsed.push(id);
    }

    let mut out = String::new();
    for id in collapsed {
        if id == cfg.blank_id {
            continue;
        }
        let token = cfg.alphabet.get(id).ok_or_else(|| {
            anyhow!(
                "token id {id} out of range (alphabet size {})",
                cfg.alphabet.len()
            )
        })?;
        push_token(&mut out, token);
    }

    Ok(normalize_text(out))
}

fn word_boundary_id(cfg: &SileroConfig) -> Option<usize> {
    let token = cfg
        .word_boundary_token
        .as_deref()
        .or_else(|| cfg.alphabet.iter().any(|t| t == " ").then_some(" "))
        .or_else(|| cfg.alphabet.iter().any(|t| t == "|").then_some("|"))?;

    cfg.alphabet.iter().position(|t| t == token)
}

fn last_boundary_age(prefix: &[usize], boundary_id: usize) -> Option<usize> {
    // Returns the number of tokens since the last boundary token.
    for (i, &id) in prefix.iter().enumerate().rev() {
        if id == boundary_id {
            return Some(prefix.len().saturating_sub(i + 1));
        }
    }
    None
}

fn push_token(out: &mut String, token: &str) {
    // Some Silero alphabets represent space as `|`.
    if token == "|" {
        out.push(' ');
    } else {
        out.push_str(token);
    }
}

fn normalize_text(s: String) -> String {
    s.split_whitespace().collect::<Vec<_>>().join(" ")
}

fn top_k_indices(row: &[f32], k: usize, blank_id: usize) -> Vec<usize> {
    let mut pairs: Vec<(usize, f32)> = row.iter().copied().enumerate().collect();
    pairs.sort_by(|a, b| b.1.total_cmp(&a.1));
    let mut out: Vec<usize> = pairs.into_iter().take(k).map(|(i, _)| i).collect();
    if !out.contains(&blank_id) {
        out.push(blank_id);
    }
    out
}

fn log_sum_exp(xs: &[f32]) -> f32 {
    let mut max_v = f32::NEG_INFINITY;
    for &v in xs {
        if v > max_v {
            max_v = v;
        }
    }
    if !max_v.is_finite() {
        return max_v;
    }
    let mut sum = 0.0f32;
    for &v in xs {
        sum += (v - max_v).exp();
    }
    max_v + sum.ln()
}

fn log_add_exp(a: f32, b: f32) -> f32 {
    if a == f32::NEG_INFINITY {
        return b;
    }
    if b == f32::NEG_INFINITY {
        return a;
    }
    let m = a.max(b);
    m + ((a - m).exp() + (b - m).exp()).ln()
}

fn silero_input_rank(session: &Session) -> Result<usize> {
    let input = session
        .inputs
        .first()
        .ok_or_else(|| anyhow!("Silero model has no inputs"))?;
    let ValueType::Tensor { shape, .. } = &input.input_type else {
        return Err(anyhow!(
            "Silero input is not a tensor: {}",
            input.input_type
        ));
    };
    Ok(shape.len())
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn top_k_indices_always_includes_blank() {
        let row = [0.0_f32, 10.0, 9.0];
        let got = top_k_indices(&row, 1, 0);
        assert!(got.contains(&0));
        assert!(got.contains(&1));
        assert_eq!(got.len(), 2);
    }

    #[test]
    fn beam_decoder_runs_on_tiny_logits() -> Result<()> {
        let cfg = SileroConfig {
            output_index: 0,
            decoder: SileroDecoder::BeamSearch,
            beam_width: 4,
            frame_top_k: 1,
            blank_id: 0,
            word_boundary_token: Some("|".into()),
            word_boundary_bonus: 0.5,
            word_boundary_min_gap: 2,
            alphabet: vec!["_".into(), "a".into(), "b".into()],
        };

        // time=2, vocab=3
        // t0: a
        // t1: b
        let logits = [
            -10.0, 10.0, 0.0, // t0
            -10.0, -10.0, 10.0, // t1
        ];

        let text = decode_logits_2d_with_cfg(2, 3, &logits, &cfg)?;
        assert_eq!(text, "ab");
        Ok(())
    }
}
