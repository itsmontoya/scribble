use anyhow::Result;

use super::VadProcessor;

/// Streaming VAD adapter.
///
/// This buffers audio into analysis windows (with a small holdback tail) so that VAD decisions
/// have enough context to detect speech and apply padding, while still allowing Scribble to
/// feed the backend incrementally.
pub struct VadStream<'a> {
    vad: &'a mut VadProcessor,
    window_frames: usize,
    holdback_frames: usize,
    pending_tail: Vec<f32>,
    in_buf: Vec<f32>,
    out_buf: Vec<f32>,
    out_cursor: usize,
}

impl<'a> VadStream<'a> {
    pub fn new(vad: &'a mut VadProcessor) -> Self {
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

    pub fn push(&mut self, chunk: &[f32]) -> Result<()> {
        self.in_buf.extend_from_slice(chunk);
        self.process_ready_windows()
    }

    pub fn flush(&mut self) -> Result<()> {
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

    pub fn peek_chunk(&self, frames: usize) -> Option<&[f32]> {
        let available = self.out_buf.len().saturating_sub(self.out_cursor);
        if available >= frames {
            Some(&self.out_buf[self.out_cursor..self.out_cursor + frames])
        } else {
            None
        }
    }

    pub fn consume_chunk(&mut self, frames: usize) {
        self.out_cursor = (self.out_cursor + frames).min(self.out_buf.len());

        // Periodically compact to avoid unbounded growth.
        if self.out_cursor >= 65_536 && self.out_cursor * 2 >= self.out_buf.len() {
            self.out_buf.drain(..self.out_cursor);
            self.out_cursor = 0;
        }
    }

    pub fn peek_remainder(&self) -> Option<&[f32]> {
        if self.out_cursor < self.out_buf.len() {
            Some(&self.out_buf[self.out_cursor..])
        } else {
            None
        }
    }

    pub fn consume_remainder(&mut self) {
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

fn ms_to_samples(ms: u32, sample_rate: f32) -> usize {
    ((ms as f32 / 1000.0) * sample_rate).round() as usize
}
