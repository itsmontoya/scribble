//! `scribble` — a small, focused transcription library built on top of Whisper.
//!
//! This crate provides:
//! - Model loading and context management
//! - Audio preprocessing (VAD, WAV decoding)
//! - Segment extraction
//! - Pluggable output encoders (JSON, VTT, etc.)
//!
//! The library is designed to be used by both CLI tools and long-running services,
//! with an emphasis on clarity, streaming output, and minimal surprises.

// High-level API (most consumers should start here).
pub mod opts;
pub mod scribble;

// Core Whisper context management.
pub mod ctx;

// Segment data structures and transcription helpers.
pub mod segments;

// Audio preprocessing and decoding.
pub mod vad;
pub mod wav;

// Output selection and encoder interfaces.
pub mod output_type;
pub mod segment_encoder;

// Output encoders that serialize segments into various formats.
pub mod json_array_encoder;
pub mod vtt_encoder;

// Logging configuration and control.
pub mod logging;

pub mod decoder;
