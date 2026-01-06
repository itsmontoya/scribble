use std::sync::OnceLock;
use std::time::Instant;

use anyhow::{Context, Result};
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

fn build_metrics() -> Result<Metrics> {
    let registry = Registry::new();

    let http_requests_total = IntCounterVec::new(
        PromOpts::new(
            "scribble_http_requests_total",
            "Total HTTP requests served by scribble-server.",
        ),
        &["status"],
    )
    .context("invalid definition for scribble_http_requests_total")?;

    let http_request_duration_seconds = HistogramVec::new(
        HistogramOpts::new(
            "scribble_http_request_duration_seconds",
            "HTTP request latency in seconds.",
        ),
        &["status"],
    )
    .context("invalid definition for scribble_http_request_duration_seconds")?;

    let http_in_flight_requests = IntGauge::new(
        "scribble_http_in_flight_requests",
        "Current number of in-flight HTTP requests.",
    )
    .context("invalid definition for scribble_http_in_flight_requests")?;

    registry
        .register(Box::new(http_requests_total.clone()))
        .context("failed to register scribble_http_requests_total")?;
    registry
        .register(Box::new(http_request_duration_seconds.clone()))
        .context("failed to register scribble_http_request_duration_seconds")?;
    registry
        .register(Box::new(http_in_flight_requests.clone()))
        .context("failed to register scribble_http_in_flight_requests")?;

    Ok(Metrics {
        registry,
        http_requests_total,
        http_request_duration_seconds,
        http_in_flight_requests,
    })
}

fn metrics() -> Option<&'static Metrics> {
    METRICS.get()
}

pub fn init() -> Result<()> {
    if metrics().is_some() {
        return Ok(());
    }

    let built = build_metrics()?;
    let _ = METRICS.set(built);
    Ok(())
}

pub async fn prometheus_metrics() -> Response {
    if metrics().is_none()
        && let Err(err) = init()
    {
        return (
            StatusCode::INTERNAL_SERVER_ERROR,
            format!("failed to initialize metrics: {err:#}"),
        )
            .into_response();
    }

    let Some(metrics) = metrics() else {
        return (StatusCode::INTERNAL_SERVER_ERROR, "metrics not initialized").into_response();
    };

    let families = metrics.registry.gather();
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

    if route == "/metrics" || route == "/health" {
        return next.run(req).await;
    }

    let Some(metrics) = metrics() else {
        return next.run(req).await;
    };

    let start = Instant::now();

    metrics.http_in_flight_requests.inc();
    let response = next.run(req).await;
    metrics.http_in_flight_requests.dec();

    let status = response.status().as_u16().to_string();
    metrics
        .http_requests_total
        .with_label_values(&[&status])
        .inc();
    metrics
        .http_request_duration_seconds
        .with_label_values(&[&status])
        .observe(start.elapsed().as_secs_f64());

    response
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn init_is_idempotent_and_registers_metrics() {
        init().unwrap();
        init().unwrap();

        metrics()
            .unwrap()
            .http_requests_total
            .with_label_values(&["200"])
            .inc();
        metrics()
            .unwrap()
            .http_request_duration_seconds
            .with_label_values(&["200"])
            .observe(0.001);
        metrics().unwrap().http_in_flight_requests.inc();

        let families = metrics().unwrap().registry.gather();
        let names: Vec<&str> = families.iter().map(|f| f.name()).collect();
        assert!(names.contains(&"scribble_http_requests_total"));
        assert!(names.contains(&"scribble_http_request_duration_seconds"));
        assert!(names.contains(&"scribble_http_in_flight_requests"));
    }

    #[tokio::test]
    async fn prometheus_metrics_returns_text_format() -> anyhow::Result<()> {
        init().unwrap();
        metrics()
            .unwrap()
            .http_requests_total
            .with_label_values(&["200"])
            .inc();
        metrics()
            .unwrap()
            .http_request_duration_seconds
            .with_label_values(&["200"])
            .observe(0.001);

        let resp = prometheus_metrics().await;

        assert_eq!(resp.status(), StatusCode::OK);
        assert_eq!(
            resp.headers()
                .get(header::CONTENT_TYPE)
                .expect("content-type header")
                .to_str()?,
            "text/plain; version=0.0.4; charset=utf-8"
        );

        let bytes = axum::body::to_bytes(resp.into_body(), usize::MAX).await?;
        let text = std::str::from_utf8(&bytes)?;
        assert!(text.contains("scribble_http_requests_total"));
        Ok(())
    }
}
