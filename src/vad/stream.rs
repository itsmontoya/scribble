use std::sync::mpsc;

use anyhow::{Result, anyhow};

use super::VadProcessor;

/// Streaming VAD adapter.
///
/// Buffers audio into analysis windows (with a small holdback tail) so VAD decisions have enough
/// context to detect speech and apply padding, while still allowing Scribble to feed the backend
/// incrementally.
///
/// This type is an internal building block. The public, receiver-like surface area is
/// [`VadStreamReceiver`], kept intentionally simple and predictable.
pub struct VadStream {
    vad: VadProcessor,
    window_frames: usize,
    holdback_frames: usize,
    pending_tail: Vec<f32>,
    in_buf: Vec<f32>,
    out_buf: Vec<f32>,
    out_cursor: usize,
}

impl VadStream {
    pub(crate) fn new(vad: VadProcessor) -> Self {
        const SAMPLE_RATE_HZ: f32 = 16_000.0;

        let policy = vad.policy();
        let holdback_ms = policy
            .pre_pad_ms
            .max(policy.post_pad_ms)
            .max(policy.gap_merge_ms);

        Self {
            vad,
            window_frames: (SAMPLE_RATE_HZ as usize) * 2,
            holdback_frames: ms_to_samples(holdback_ms, SAMPLE_RATE_HZ),
            pending_tail: Vec::new(),
            in_buf: Vec::new(),
            out_buf: Vec::new(),
            out_cursor: 0,
        }
    }

    pub(crate) fn push(&mut self, chunk: &[f32]) -> Result<()> {
        self.in_buf.extend_from_slice(chunk);
        self.process_ready_windows()
    }

    pub(crate) fn flush(&mut self) -> Result<()> {
        self.process_ready_windows()?;

        let mut window = Vec::with_capacity(self.pending_tail.len() + self.in_buf.len());
        window.extend_from_slice(&self.pending_tail);
        window.extend_from_slice(&self.in_buf);

        self.pending_tail.clear();
        self.in_buf.clear();

        if window.is_empty() {
            return Ok(());
        }

        let _ = self.vad.apply(&mut window)?;
        self.out_buf.extend_from_slice(&window);
        Ok(())
    }

    pub(crate) fn peek_chunk(&self, frames: usize) -> Option<&[f32]> {
        let available = self.out_buf.len().saturating_sub(self.out_cursor);
        if available >= frames {
            Some(&self.out_buf[self.out_cursor..self.out_cursor + frames])
        } else {
            None
        }
    }

    pub(crate) fn consume_chunk(&mut self, frames: usize) {
        self.out_cursor = (self.out_cursor + frames).min(self.out_buf.len());

        // Periodically compact to avoid unbounded growth if the caller consumes in small steps.
        // This is conservative (no clever ring buffers) and keeps behavior easy to reason about.
        if self.out_cursor >= 65_536 && self.out_cursor * 2 >= self.out_buf.len() {
            self.out_buf.drain(..self.out_cursor);
            self.out_cursor = 0;
        }
    }

    pub(crate) fn peek_remainder(&self) -> Option<&[f32]> {
        if self.out_cursor < self.out_buf.len() {
            Some(&self.out_buf[self.out_cursor..])
        } else {
            None
        }
    }

    pub(crate) fn consume_remainder(&mut self) {
        self.out_cursor = self.out_buf.len();
        self.out_buf.clear();
        self.out_cursor = 0;
    }

    fn process_ready_windows(&mut self) -> Result<()> {
        while self.in_buf.len() >= self.window_frames {
            let segment: Vec<f32> = self.in_buf.drain(..self.window_frames).collect();

            let mut window = Vec::with_capacity(self.pending_tail.len() + segment.len());
            window.extend_from_slice(&self.pending_tail);
            window.extend_from_slice(&segment);

            let _ = self.vad.apply(&mut window)?;

            if self.holdback_frames > 0 && window.len() > self.holdback_frames {
                let split = window.len() - self.holdback_frames;
                self.out_buf.extend_from_slice(&window[..split]);
                self.pending_tail.clear();
                self.pending_tail.extend_from_slice(&window[split..]);
            } else {
                self.out_buf.extend_from_slice(&window);
                self.pending_tail.clear();
            }
        }

        Ok(())
    }
}

/// Adapter that turns a `Receiver<Vec<f32>>` into a VAD-processed receiver-like stream.
///
/// Shaped like `std::sync::mpsc::Receiver` so the transcription loop stays explicit:
/// receive a chunk, hand it to the backend, and repeat.
///
/// Call `recv()` repeatedly to get VAD-filtered chunks. When the input channel disconnects and all
/// buffered audio is flushed/drained, `recv()` returns an error.
pub struct VadStreamReceiver {
    inner: mpsc::Receiver<Vec<f32>>,
    vad: VadStream,
    emit_frames: usize,
    flushed: bool,
}

impl VadStreamReceiver {
    /// Create a new VAD receiver wrapper.
    ///
    /// `emit_frames` controls the chunk size yielded downstream. It is clamped to at least `1` so
    /// callers don't have to handle a degenerate “0-sized chunk” mode.
    pub fn new(inner: mpsc::Receiver<Vec<f32>>, vad: VadProcessor, emit_frames: usize) -> Self {
        Self {
            inner,
            vad: VadStream::new(vad),
            emit_frames: emit_frames.max(1),
            flushed: false,
        }
    }

    /// Receive the next VAD-processed chunk.
    ///
    /// Uses a `loop` (not recursion) so the control flow is obvious:
    /// - if buffered output exists, return it;
    /// - otherwise, pull more input;
    /// - once input ends, flush exactly once and drain any remaining output.
    pub fn recv(&mut self) -> Result<Vec<f32>> {
        loop {
            if let Some(chunk) = self.vad.peek_chunk(self.emit_frames) {
                let out = chunk.to_vec();
                self.vad.consume_chunk(self.emit_frames);
                return Ok(out);
            }

            if self.flushed {
                if let Some(rem) = self.vad.peek_remainder() {
                    let out = rem.to_vec();
                    self.vad.consume_remainder();
                    return Ok(out);
                }
                return Err(anyhow!(
                    "vad input channel disconnected (all buffered audio drained)"
                ));
            }

            match self.inner.recv() {
                Ok(chunk) => self.vad.push(&chunk)?,
                Err(_) => {
                    // Flush once when the input channel closes so VAD can emit buffered
                    // context/padding. After this, only drain output until exhausted.
                    self.vad.flush()?;
                    self.flushed = true;
                }
            }
        }
    }
}

fn ms_to_samples(ms: u32, sample_rate: f32) -> usize {
    ((ms as f32 / 1000.0) * sample_rate).round() as usize
}
