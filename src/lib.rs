//! `scribble` — a small, focused transcription library with a pluggable ASR backend.
//!
//! # Overview
//!
//! Scribble provides a clean, idiomatic Rust API for audio transcription,
//! designed to work equally well in CLI tools and long-running services.
//!
//! At a high level, Scribble wires together:
//! - Media demuxing and audio decoding (via Symphonia)
//! - Audio normalization and resampling (mono, 16 kHz)
//! - Backend inference (built-in Whisper backend available)
//! - Pluggable output encoders (JSON, VTT, etc.)
//!
//! The library emphasizes:
//! - Explicit control flow
//! - Streaming-friendly design
//! - Clear separation of concerns
//! - Minimal surprises for callers
//!
//! Most consumers should start with [`crate::Scribble`].

// ─────────────────────────────────────────────────────────────────────────────
// High-level API
// ─────────────────────────────────────────────────────────────────────────────

mod backend;
mod backends;
mod error;
mod opts;
mod scribble;
mod vad;

// ─────────────────────────────────────────────────────────────────────────────
// Transcription core
// ─────────────────────────────────────────────────────────────────────────────

mod segments;
mod token;

// ─────────────────────────────────────────────────────────────────────────────
// Audio preprocessing pipeline
// ─────────────────────────────────────────────────────────────────────────────

mod audio_pipeline;
mod decode;
mod decoder;
mod demux;
#[cfg(test)]
mod wav;

// ─────────────────────────────────────────────────────────────────────────────
// Output encoding
// ─────────────────────────────────────────────────────────────────────────────

mod json_array_encoder;
mod output_type;
mod segment_encoder;
mod vtt_encoder;

// ─────────────────────────────────────────────────────────────────────────────
// Infrastructure
// ─────────────────────────────────────────────────────────────────────────────

mod logging;

/// Internal adapters used to keep the high-level transcription loop linear and explicit.
pub(crate) mod samples_rx;

pub use crate::backend::{Backend, BackendStream};
pub use crate::backends::whisper::WhisperBackend;
pub use crate::error::{Error, Result};
pub use crate::logging::init as init_logging;
pub use crate::opts::Opts;
pub use crate::output_type::OutputType;
pub use crate::scribble::Scribble;
pub use crate::segment_encoder::SegmentEncoder;
pub use crate::segments::Segment;
pub use crate::vad::{VadProcessor, VadStream, VadStreamReceiver};
