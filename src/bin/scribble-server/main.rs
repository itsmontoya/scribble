use std::io::Cursor;
use std::net::SocketAddr;
use std::sync::Arc;

use anyhow::{Context, Result, anyhow};
use axum::body::{Body, Bytes};
use axum::extract::{DefaultBodyLimit, Query, State};
use axum::http::{HeaderValue, StatusCode, header};
use axum::middleware::from_fn;
use axum::response::{IntoResponse, Response};
use axum::routing::{get, post};
use axum::{Json, Router};
use clap::Parser;
use futures_util::stream::BoxStream;
use futures_util::{StreamExt, TryStreamExt};
use serde::{Deserialize, Serialize};
use symphonia::core::io::ReadOnlySource;
use tokio::net::TcpListener;
use tokio::sync::oneshot;
use tokio_util::io::{ReaderStream, SyncIoBridge};
use tower_http::trace::{DefaultMakeSpan, DefaultOnFailure, DefaultOnResponse, TraceLayer};
use tracing::{Level, error, info, warn};

mod metrics;

use scribble::{Opts, OutputType, Scribble, WhisperBackend};

type BodyDataStream = BoxStream<'static, std::result::Result<Bytes, axum::Error>>;

#[derive(Parser, Debug)]
#[command(name = "scribble-server")]
#[command(about = "HTTP server for audio/video transcription")]
struct Params {
    /// Path(s) to whisper.cpp model file(s) (e.g. `ggml-large-v3.bin`).
    #[arg(short = 'm', long = "model", required = true, num_args = 1..)]
    model_paths: Vec<String>,

    /// Path to a Whisper-VAD model file.
    #[arg(short = 'v', long = "vad-model", required = true)]
    vad_model_path: String,

    /// Host interface to bind to.
    #[arg(long = "host", default_value = "127.0.0.1")]
    host: String,

    /// TCP port to listen on.
    #[arg(long = "port", default_value_t = 8080)]
    port: u16,

    /// Maximum request body size (bytes).
    #[arg(long = "max-bytes", default_value_t = 100 * 1024 * 1024)]
    max_bytes: usize,
}

#[derive(Clone)]
struct AppState {
    scribble: Arc<Scribble<WhisperBackend>>,
}

#[derive(Debug, Deserialize)]
struct TranscribeQuery {
    #[serde(default, alias = "output_type")]
    output: Option<String>,
    #[serde(default)]
    model_key: Option<String>,
    #[serde(default)]
    enable_vad: Option<bool>,
    #[serde(default)]
    translate_to_english: Option<bool>,
    #[serde(default)]
    language: Option<String>,
}

#[derive(Debug, Serialize)]
struct ModelsResponse {
    default_model_key: String,
    model_keys: Vec<String>,
    vad_model_path: String,
}

#[derive(Debug, Serialize)]
struct ErrorBody {
    error: String,
}

struct AppError {
    status: StatusCode,
    message: String,
}

impl AppError {
    fn bad_request(message: impl Into<String>) -> Self {
        Self {
            status: StatusCode::BAD_REQUEST,
            message: message.into(),
        }
    }

    fn unsupported_media(message: impl Into<String>) -> Self {
        Self {
            status: StatusCode::UNSUPPORTED_MEDIA_TYPE,
            message: message.into(),
        }
    }
}

impl IntoResponse for AppError {
    fn into_response(self) -> Response {
        let body = Json(ErrorBody {
            error: self.message,
        });
        (self.status, body).into_response()
    }
}

#[tokio::main]
async fn main() {
    scribble::init_logging();

    if let Err(err) = run().await {
        error!(error = ?err, "scribble-server failed");
        std::process::exit(1);
    }
}

async fn run() -> Result<()> {
    let params = Params::parse();

    if let Err(err) = metrics::init() {
        warn!(error = ?err, "metrics disabled (init failed)");
    }

    let addr: SocketAddr = format!("{}:{}", params.host, params.port)
        .parse()
        .context("invalid host/port bind address")?;

    let scribble = Scribble::new(params.model_paths, &params.vad_model_path)
        .context("failed to initialize Scribble backend")?;

    let state = AppState {
        scribble: Arc::new(scribble),
    };

    let app = Router::new()
        .route("/", get(root))
        .route("/healthz", get(healthz))
        .route("/metrics", get(metrics::prometheus_metrics))
        .route("/v1/models", get(models))
        .route("/v1/transcribe", post(transcribe))
        .route_layer(from_fn(metrics::track_http_metrics))
        .with_state(state)
        .layer(DefaultBodyLimit::max(params.max_bytes))
        .layer(
            TraceLayer::new_for_http()
                .make_span_with(
                    DefaultMakeSpan::new()
                        .level(Level::INFO)
                        .include_headers(false),
                )
                .on_response(DefaultOnResponse::new().level(Level::INFO))
                .on_failure(DefaultOnFailure::new().level(Level::ERROR)),
        );

    let listener = TcpListener::bind(addr).await.context("bind failed")?;
    info!(%addr, "listening");
    axum::serve(listener, app).await.context("server error")?;

    Ok(())
}

async fn root() -> &'static str {
    "scribble-server: POST /v1/transcribe (multipart field: file)"
}

async fn healthz() -> &'static str {
    "ok"
}

async fn models(
    State(state): State<AppState>,
) -> std::result::Result<Json<ModelsResponse>, AppError> {
    let backend = state.scribble.backend();

    Ok(Json(ModelsResponse {
        default_model_key: backend.default_model_key().to_owned(),
        model_keys: backend.model_keys(),
        vad_model_path: backend.vad_model_path().to_owned(),
    }))
}

async fn transcribe(
    State(state): State<AppState>,
    Query(query): Query<TranscribeQuery>,
    body: Body,
) -> std::result::Result<Response, AppError> {
    // We want request bodies to be streaming (for very long/live uploads), but we still want to
    // fail fast for obviously unsupported inputs. We do a small, bounded probe against the
    // initial prefix and then replay that prefix into the decoder so transcription starts at
    // byte 0 without buffering the whole upload.
    const MAX_PROBE_BYTES: usize = 512 * 1024;
    let body_stream: BodyDataStream = body.into_data_stream().boxed();
    let (prefix_bytes, prefix_chunks, body_stream) =
        get_prefix_bytes(body_stream, MAX_PROBE_BYTES).await?;

    validate_media_prefix(&prefix_bytes)?;

    let output_type = parse_output_type(query.output.as_deref())
        .map_err(|err| AppError::bad_request(err.to_string()))?;

    let opts = Opts {
        model_key: query.model_key,
        enable_translate_to_english: query.translate_to_english.unwrap_or(false),
        enable_voice_activity_detection: query.enable_vad.unwrap_or(false),
        language: query.language,
        output_type,
        incremental_min_window_seconds: 1,
    };

    let content_type = match opts.output_type {
        OutputType::Json => HeaderValue::from_static("application/json; charset=utf-8"),
        OutputType::Vtt => HeaderValue::from_static("text/vtt; charset=utf-8"),
    };

    let scribble = state.scribble.clone();
    let prefix_stream =
        futures_util::stream::iter(prefix_chunks.into_iter().map(Ok::<Bytes, axum::Error>));
    let input_stream = prefix_stream.chain(body_stream);
    let input_reader =
        tokio_util::io::StreamReader::new(input_stream.map_err(std::io::Error::other));

    let (out_tx, out_rx) = tokio::io::duplex(64 * 1024);
    let (done_tx, done_rx) = oneshot::channel::<std::result::Result<(), String>>();

    tokio::task::spawn_blocking(move || {
        let mut writer = SyncIoBridge::new(out_tx);
        let input = SyncIoBridge::new(input_reader);
        let res = scribble
            .transcribe(input, &mut writer, &opts)
            .map_err(|err| err.to_string());
        let _ = done_tx.send(res);
    });

    tokio::spawn(async move {
        if let Ok(Err(msg)) = done_rx.await {
            error!(%msg, "transcription failed");
        }
    });

    let body = Body::from_stream(ReaderStream::new(out_rx));
    Ok(([(header::CONTENT_TYPE, content_type)], body).into_response())
}

async fn get_prefix_bytes(
    mut body_stream: BodyDataStream,
    max_probe_bytes: usize,
) -> std::result::Result<(Vec<u8>, Vec<Bytes>, BodyDataStream), AppError> {
    let mut prefix_bytes = Vec::<u8>::new();
    let mut prefix_chunks = Vec::<Bytes>::new();

    while prefix_bytes.len() < max_probe_bytes {
        let Some(chunk) = body_stream.next().await else {
            break;
        };
        let chunk = chunk.map_err(|err| AppError::bad_request(err.to_string()))?;
        if chunk.is_empty() {
            continue;
        }

        let remaining = max_probe_bytes - prefix_bytes.len();
        if chunk.len() <= remaining {
            prefix_bytes.extend_from_slice(&chunk);
            prefix_chunks.push(chunk);
            continue;
        }

        // Split the chunk so we only buffer up to the probe limit.
        prefix_bytes.extend_from_slice(&chunk[..remaining]);
        prefix_chunks.push(chunk.slice(..remaining));

        // Put the tail back into the stream for transcription.
        let tail = chunk.slice(remaining..);
        let tail_stream: BodyDataStream =
            futures_util::stream::once(async move { Ok::<Bytes, axum::Error>(tail) }).boxed();
        body_stream = tail_stream.chain(body_stream).boxed();
        break;
    }

    if prefix_bytes.is_empty() {
        return Err(AppError::bad_request("request body was empty"));
    }

    Ok((prefix_bytes, prefix_chunks, body_stream))
}

fn validate_media_prefix(prefix: &[u8]) -> std::result::Result<(), AppError> {
    let source = ReadOnlySource::new(Cursor::new(prefix.to_vec()));
    if let Err(err) = probe_source_and_pick_default_track(Box::new(source), None) {
        return Err(AppError::unsupported_media(format!(
            "unsupported or unrecognized media container: {err}"
        )));
    }
    Ok(())
}

fn probe_source_and_pick_default_track(
    source: Box<dyn symphonia::core::io::MediaSource>,
    hint_extension: Option<&str>,
) -> Result<(
    Box<dyn symphonia::core::formats::FormatReader>,
    symphonia::core::formats::Track,
)> {
    use symphonia::core::codecs::CODEC_TYPE_NULL;
    use symphonia::core::io::{MediaSourceStream, MediaSourceStreamOptions};
    use symphonia::core::meta::MetadataOptions;
    use symphonia::core::probe::Hint;

    let mss_opts = MediaSourceStreamOptions {
        buffer_len: 256 * 1024,
    };

    let mss = MediaSourceStream::new(source, mss_opts);

    let mut hint = Hint::new();
    if let Some(ext) = hint_extension {
        hint.with_extension(ext);
    }

    let format_opts: symphonia::core::formats::FormatOptions = Default::default();
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

fn parse_output_type(output: Option<&str>) -> Result<OutputType> {
    match output {
        None => Ok(OutputType::Vtt),
        Some(raw) => match raw.trim().to_ascii_lowercase().as_str() {
            "json" => Ok(OutputType::Json),
            "vtt" => Ok(OutputType::Vtt),
            other => Err(anyhow!(
                "unknown output type '{other}' (expected 'json' or 'vtt')"
            )),
        },
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use futures_util::{StreamExt, TryStreamExt};

    fn stream_from_chunks(chunks: Vec<&'static [u8]>) -> BodyDataStream {
        futures_util::stream::iter(
            chunks
                .into_iter()
                .map(|c| Ok::<Bytes, axum::Error>(Bytes::from_static(c))),
        )
        .boxed()
    }

    #[test]
    fn parse_output_type_defaults_to_vtt() -> anyhow::Result<()> {
        assert!(matches!(parse_output_type(None)?, OutputType::Vtt));
        Ok(())
    }

    #[test]
    fn parse_output_type_accepts_known_values_case_insensitively() -> anyhow::Result<()> {
        assert!(matches!(
            parse_output_type(Some(" json "))?,
            OutputType::Json
        ));
        assert!(matches!(parse_output_type(Some("VTT"))?, OutputType::Vtt));
        Ok(())
    }

    #[test]
    fn parse_output_type_rejects_unknown_value() {
        let err = parse_output_type(Some("nope")).unwrap_err();
        assert!(err.to_string().contains("unknown output type"));
    }

    #[tokio::test]
    async fn get_prefix_bytes_errors_on_empty_body() {
        let res = get_prefix_bytes(stream_from_chunks(vec![]), 16).await;
        assert!(res.is_err());
        let err = res.err().expect("expected AppError");
        assert!(err.message.contains("request body was empty"));
    }

    #[tokio::test]
    async fn get_prefix_bytes_skips_empty_chunks() {
        let (prefix_bytes, prefix_chunks, _tail) =
            match get_prefix_bytes(stream_from_chunks(vec![b"", b"abc"]), 16).await {
                Ok(v) => v,
                Err(err) => panic!("unexpected error: {}", err.message),
            };
        assert_eq!(prefix_bytes, b"abc");
        assert_eq!(prefix_chunks.len(), 1);
        assert_eq!(prefix_chunks[0].as_ref(), b"abc");
    }

    #[tokio::test]
    async fn get_prefix_bytes_splits_large_chunk_and_replays_tail() {
        let (prefix_bytes, prefix_chunks, tail) =
            match get_prefix_bytes(stream_from_chunks(vec![b"helloWORLD"]), 5).await {
                Ok(v) => v,
                Err(err) => panic!("unexpected error: {}", err.message),
            };

        assert_eq!(prefix_bytes, b"hello");
        assert_eq!(prefix_chunks.len(), 1);
        assert_eq!(prefix_chunks[0].as_ref(), b"hello");

        let tail_chunks: Vec<Bytes> = match tail.try_collect().await {
            Ok(v) => v,
            Err(err) => panic!("unexpected tail stream error: {err}"),
        };
        assert_eq!(tail_chunks.len(), 1);
        assert_eq!(tail_chunks[0].as_ref(), b"WORLD");
    }

    #[test]
    fn validate_media_prefix_accepts_wav_fixture() {
        let bytes = std::fs::read("tests/fixtures/jfk.wav").expect("read wav fixture");
        if let Err(err) = validate_media_prefix(&bytes) {
            panic!(
                "expected WAV fixture to probe successfully: {}",
                err.message
            );
        }
    }
}
