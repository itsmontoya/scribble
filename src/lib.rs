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
//! Most consumers should start with [`scribble::Scribble`].

// ─────────────────────────────────────────────────────────────────────────────
// High-level API
// ─────────────────────────────────────────────────────────────────────────────

/// User-facing transcription entry point and orchestration logic.
pub mod scribble;

/// Pluggable ASR backend trait.
pub mod backend;

/// Built-in backend implementations.
pub mod backends;

/// User-configurable transcription options.
pub mod opts;

/// Voice activity detection (VAD) helpers.
pub mod vad;

// ─────────────────────────────────────────────────────────────────────────────
// Transcription core
// ─────────────────────────────────────────────────────────────────────────────

/// Segment data structures and transcription helpers.
pub mod segments;
/// Token data structures.
pub mod token;

// ─────────────────────────────────────────────────────────────────────────────
// Audio preprocessing pipeline
// ─────────────────────────────────────────────────────────────────────────────

/// Streaming-friendly audio decoding and normalization helpers.
pub mod decoder;

/// Low-level demux helpers (container probing, packet iteration).
pub mod demux;

/// Codec-level decode helpers.
pub mod decode;

/// Audio resampling, downmixing, and chunk emission pipeline.
pub mod audio_pipeline;

/// WAV helpers (primarily for tests and simple inputs).
pub mod wav;

// ─────────────────────────────────────────────────────────────────────────────
// Output encoding
// ─────────────────────────────────────────────────────────────────────────────

/// Output format selection.
pub mod output_type;

/// Shared encoder trait definitions.
pub mod segment_encoder;

/// JSON array encoder.
pub mod json_array_encoder;

/// WebVTT encoder.
pub mod vtt_encoder;

// ─────────────────────────────────────────────────────────────────────────────
// Infrastructure
// ─────────────────────────────────────────────────────────────────────────────
