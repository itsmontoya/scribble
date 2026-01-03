use anyhow::{Context, Result};
use whisper_rs::{WhisperContext, WhisperContextParameters};

use super::logging::init_whisper_logging;

/// Load a Whisper model and return an initialized `WhisperContext`.
pub fn get_context(model_path: &str) -> Result<WhisperContext> {
    init_whisper_logging();

    let ctx_params = WhisperContextParameters::default();
    let ctx = WhisperContext::new_with_params(model_path, ctx_params)
        .with_context(|| format!("failed to load model from path: {model_path}"))?;

    Ok(ctx)
}

