use anyhow::{Context, Result};
use whisper_rs::{WhisperContext, WhisperContextParameters};

use crate::logging::init_whisper_logging;

/// Load a Whisper model and return an initialized `WhisperContext`.
///
/// Why this exists:
/// - We centralize model loading in one place so error handling and defaults stay consistent.
///
/// Design notes:
/// - We accept `&str` instead of `&String` to keep the API flexible for callers.
/// - We intentionally control whisper.rs / whisper.cpp logging to keep CLI output clean.
pub fn get_context(model_path: &str) -> Result<WhisperContext> {
    // We currently silence logs emitted by whisper.rs / whisper.cpp because they can be very noisy
    // (and because our binaries may want to fully control what gets printed).
    //
    // Later, we can evolve this into a real logging configuration hook (e.g. level/targets/env),
    // but for now it is a deliberate "quiet by default" choice.
    init_whisper_logging();

    // We start with default Whisper context parameters.
    // If we need to tune performance or enable optional features later, we can do it here.
    let ctx_params = WhisperContextParameters::default();

    // Load the model and attach useful context to errors for easier debugging.
    let ctx = WhisperContext::new_with_params(model_path, ctx_params)
        .with_context(|| format!("failed to load model from path: {model_path}"))?;

    Ok(ctx)
}
