//! A small adapter for consuming decoded audio samples.
//!
//! Keeps the transcription loop simple and explicit:
//! - pull a chunk of samples,
//! - hand it to the backend stream,
//! - repeat until the input ends.
//!
//! When VAD is enabled, keep the same control flow, just with a different source of samples.
//! `SamplesRx` provides a receiver-like shape without introducing trait objects or implicit
//! behavior.

use std::sync::mpsc;

use anyhow::{Result, anyhow};

use crate::vad::VadStreamReceiver;

/// A receiver-like source of sample chunks (`Vec<f32>`).
///
/// Keeps this enum small and concrete:
/// - `Plain` is the raw decoder output channel.
/// - `Vad` wraps that channel and yields VAD-filtered chunks.
pub enum SamplesRx {
    Plain(mpsc::Receiver<Vec<f32>>),
    Vad(VadStreamReceiver),
}

impl SamplesRx {
    /// Receive the next chunk of samples.
    ///
    /// Mirrors the blocking behavior of `std::sync::mpsc::Receiver::recv`.
    /// When the underlying channel disconnects, returns an error rather than inventing a sentinel
    /// value. This keeps end-of-stream handling explicit
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

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn samples_rx_plain_reports_disconnect_as_error() {
        let (tx, rx) = mpsc::channel::<Vec<f32>>();
        drop(tx);
        let mut rx = SamplesRx::Plain(rx);
        let err = rx.recv().unwrap_err();
        assert!(
            err.to_string()
                .contains("decoder output channel disconnected")
        );
    }
}
