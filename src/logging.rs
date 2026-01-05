use tracing_subscriber::EnvFilter;
use tracing_subscriber::layer::SubscriberExt;
use tracing_subscriber::util::SubscriberInitExt;

/// Initialize structured JSON logging.
///
/// Defaults to `error` level unless overridden by `SCRIBBLE_LOG`.
pub fn init() {
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

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn init_is_idempotent() {
        init();
        init();
    }
}
