//! A small adapter for consuming decoded audio samples.
//!
//! We want the transcription loop to be simple and explicit:
//! - pull a chunk of samples,
//! - hand it to the backend stream,
//! - repeat until the input ends.
//!
//! When VAD is enabled, we still want the same control flow, just with a different
//! source of samples. `SamplesRx` gives us that “receiver-like” shape without
//! introducing trait objects or implicit behavior.

use std::sync::mpsc;

use anyhow::{Result, anyhow};

use crate::vad::VadStreamReceiver;

/// A receiver-like source of sample chunks (`Vec<f32>`).
///
/// We intentionally keep this enum small and concrete:
/// - `Plain` is the raw decoder output channel.
/// - `Vad` wraps that channel and yields VAD-filtered chunks.
pub enum SamplesRx<'a> {
    Plain(mpsc::Receiver<Vec<f32>>),
    Vad(VadStreamReceiver<'a>),
}

impl<'a> SamplesRx<'a> {
    /// Receive the next chunk of samples.
    ///
    /// We mirror the blocking behavior of `std::sync::mpsc::Receiver::recv`.
    /// When the underlying channel disconnects, we return an error rather than
    /// inventing a sentinel value. This keeps end-of-stream handling explicit
    /// in callers (`while let Ok(chunk) = rx.recv()`).
    pub fn recv(&mut self) -> Result<Vec<f32>> {
        match self {
            SamplesRx::Plain(rx) => rx
                .recv()
                .map_err(|_| anyhow!("decoder output channel disconnected")),
            SamplesRx::Vad(rx) => rx.recv(),
        }
    }
}
