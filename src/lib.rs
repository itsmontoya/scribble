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

// Core Whisper context management.
pub mod ctx;

// Output encoders that serialize segments into various formats.
pub mod json_array_encoder;
pub mod vtt_encoder;

// Shared encoder traits and output selection.
pub mod output_type;
pub mod segment_encoder;

// Segment data structures and helpers.
pub mod segments;

// Audio preprocessing and decoding.
pub mod vad;
pub mod wav;

// Logging configuration and control.
pub mod logging;
