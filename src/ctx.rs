use anyhow::{Context, Result};
use whisper_rs::{WhisperContext, WhisperContextParameters};

use crate::logging::init_whisper_logging;

pub fn get_context(model_path: &String) -> Result<WhisperContext> {
    init_whisper_logging();
    let ctx_params = WhisperContextParameters::default();
    let ctx: WhisperContext = WhisperContext::new_with_params(&model_path, ctx_params)
        .with_context(|| format!("failed to load model from {}", model_path))?;
    Ok(ctx)
}
