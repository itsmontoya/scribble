use std::sync::OnceLock;
use std::time::Instant;

use axum::body::Body;
use axum::extract::MatchedPath;
use axum::http::Request;
use axum::http::{HeaderValue, StatusCode, header};
use axum::middleware::Next;
use axum::response::{IntoResponse, Response};
use prometheus::{
    Encoder, HistogramOpts, HistogramVec, IntCounterVec, IntGauge, Opts as PromOpts, Registry,
    TextEncoder,
};

struct Metrics {
    registry: Registry,
    http_requests_total: IntCounterVec,
    http_request_duration_seconds: HistogramVec,
    http_in_flight_requests: IntGauge,
}

static METRICS: OnceLock<Metrics> = OnceLock::new();

fn metrics() -> &'static Metrics {
    METRICS.get_or_init(|| {
        let registry = Registry::new();

        let http_requests_total = IntCounterVec::new(
            PromOpts::new(
                "scribble_http_requests_total",
                "Total HTTP requests served by scribble-server.",
            ),
            &["status"],
        )
        .expect("metrics definition must be valid");

        let http_request_duration_seconds = HistogramVec::new(
            HistogramOpts::new(
                "scribble_http_request_duration_seconds",
                "HTTP request latency in seconds.",
            ),
            &["status"],
        )
        .expect("metrics definition must be valid");

        let http_in_flight_requests = IntGauge::new(
            "scribble_http_in_flight_requests",
            "Current number of in-flight HTTP requests.",
        )
        .expect("metrics definition must be valid");

        registry
            .register(Box::new(http_requests_total.clone()))
            .expect("metrics must register");
        registry
            .register(Box::new(http_request_duration_seconds.clone()))
            .expect("metrics must register");
        registry
            .register(Box::new(http_in_flight_requests.clone()))
            .expect("metrics must register");

        Metrics {
            registry,
            http_requests_total,
            http_request_duration_seconds,
            http_in_flight_requests,
        }
    })
}

pub fn init() {
    let _ = metrics();
}

pub async fn prometheus_metrics() -> Response {
    let families = metrics().registry.gather();
    let mut buf = Vec::new();
    if TextEncoder::new().encode(&families, &mut buf).is_err() {
        return (
            StatusCode::INTERNAL_SERVER_ERROR,
            "failed to encode metrics",
        )
            .into_response();
    }

    (
        [(
            header::CONTENT_TYPE,
            HeaderValue::from_static("text/plain; version=0.0.4; charset=utf-8"),
        )],
        buf,
    )
        .into_response()
}

pub async fn track_http_metrics(req: Request<Body>, next: Next) -> Response {
    let route = req
        .extensions()
        .get::<MatchedPath>()
        .map(|path| path.as_str())
        .unwrap_or_else(|| req.uri().path())
        .to_owned();

    if route == "/metrics" || route == "/healthz" {
        return next.run(req).await;
    }

    let start = Instant::now();

    metrics().http_in_flight_requests.inc();
    let response = next.run(req).await;
    metrics().http_in_flight_requests.dec();

    let status = response.status().as_u16().to_string();
    metrics()
        .http_requests_total
        .with_label_values(&[&status])
        .inc();
    metrics()
        .http_request_duration_seconds
        .with_label_values(&[&status])
        .observe(start.elapsed().as_secs_f64());

    response
}
