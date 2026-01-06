/// Initialize structured JSON logging.
///
/// Defaults to `error` level unless overridden by `SCRIBBLE_LOG`.
#[cfg(feature = "logging")]
pub fn init() {
    use tracing_subscriber::EnvFilter;
    use tracing_subscriber::layer::SubscriberExt;
    use tracing_subscriber::util::SubscriberInitExt;

    let filter = EnvFilter::builder()
        .with_env_var("SCRIBBLE_LOG")
        .with_default_directive(tracing::level_filters::LevelFilter::ERROR.into())
        .from_env_lossy();

    let _ = tracing_subscriber::registry()
        .with(filter)
        .with(
            tracing_subscriber::fmt::layer()
                .json()
                .with_current_span(true)
                .with_span_list(true),
        )
        .try_init();
}

/// Initialize logging when the `logging` feature is not enabled.
///
/// We keep this as a no-op so library consumers can call `scribble::init_logging()` without
/// needing to pull in `tracing-subscriber`.
#[cfg(not(feature = "logging"))]
pub fn init() {}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn init_is_idempotent() {
        init();
        init();
    }
}
