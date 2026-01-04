use std::io::Cursor;
use std::net::SocketAddr;
use std::sync::Arc;

use anyhow::{Context, Result, anyhow};
use axum::extract::{DefaultBodyLimit, Multipart, Query, State};
use axum::http::{HeaderValue, StatusCode, header};
use axum::response::{IntoResponse, Response};
use axum::routing::{get, post};
use axum::{Json, Router};
use clap::Parser;
use serde::{Deserialize, Serialize};
use tokio::net::TcpListener;
use tokio::sync::Mutex;

use scribble::opts::Opts;
use scribble::output_type::OutputType;
use scribble::scribble::Scribble;

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
    scribble: Arc<Mutex<Scribble<scribble::backends::whisper::WhisperBackend>>>,
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
    #[serde(default)]
    incremental_min_window_seconds: Option<usize>,
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

    fn internal(message: impl Into<String>) -> Self {
        Self {
            status: StatusCode::INTERNAL_SERVER_ERROR,
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
async fn main() -> Result<()> {
    let params = Params::parse();

    let addr: SocketAddr = format!("{}:{}", params.host, params.port)
        .parse()
        .context("invalid host/port bind address")?;

    let scribble = Scribble::new(params.model_paths, &params.vad_model_path)
        .context("failed to initialize Scribble backend")?;

    let state = AppState {
        scribble: Arc::new(Mutex::new(scribble)),
    };

    let app = Router::new()
        .route("/", get(root))
        .route("/healthz", get(healthz))
        .route("/v1/models", get(models))
        .route("/v1/transcribe", post(transcribe))
        .with_state(state)
        .layer(DefaultBodyLimit::max(params.max_bytes));

    let listener = TcpListener::bind(addr).await.context("bind failed")?;
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
    let scribble = state.scribble.lock().await;
    let backend = scribble.backend();

    Ok(Json(ModelsResponse {
        default_model_key: backend.default_model_key().to_owned(),
        model_keys: backend.model_keys(),
        vad_model_path: backend.vad_model_path().to_owned(),
    }))
}

async fn transcribe(
    State(state): State<AppState>,
    Query(query): Query<TranscribeQuery>,
    mut multipart: Multipart,
) -> std::result::Result<Response, AppError> {
    let file_bytes = read_file_field(&mut multipart)
        .await
        .map_err(|err| AppError::bad_request(err.to_string()))?;

    let output_type = parse_output_type(query.output.as_deref())
        .map_err(|err| AppError::bad_request(err.to_string()))?;

    let opts = Opts {
        model_key: query.model_key,
        enable_translate_to_english: query.translate_to_english.unwrap_or(false),
        enable_voice_activity_detection: query.enable_vad.unwrap_or(false),
        language: query.language,
        output_type,
        incremental_min_window_seconds: query.incremental_min_window_seconds.unwrap_or(1),
    };

    let mut scribble = state.scribble.lock().await;
    let mut output = Vec::new();
    scribble
        .transcribe(Cursor::new(file_bytes), &mut output, &opts)
        .map_err(|err| AppError::internal(err.to_string()))?;

    let content_type = match opts.output_type {
        OutputType::Json => HeaderValue::from_static("application/json; charset=utf-8"),
        OutputType::Vtt => HeaderValue::from_static("text/vtt; charset=utf-8"),
    };

    Ok(([(header::CONTENT_TYPE, content_type)], output).into_response())
}

async fn read_file_field(multipart: &mut Multipart) -> Result<Vec<u8>> {
    while let Some(field) = multipart.next_field().await? {
        let name = field.name().unwrap_or_default().to_owned();
        if name == "file" || name == "media" {
            let bytes = field.bytes().await?;
            if bytes.is_empty() {
                return Err(anyhow!("multipart field '{name}' was empty"));
            }
            return Ok(bytes.to_vec());
        }
    }

    Err(anyhow!(
        "missing multipart field 'file' (or 'media') with the input container bytes"
    ))
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
