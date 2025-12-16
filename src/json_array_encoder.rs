use anyhow::Result;
use std::io::Write;

use crate::segment_encoder::SegmentEncoder;
use crate::segments::Segment;

pub struct JsonArrayEncoder<W: Write> {
    w: W,
    started: bool,
    first: bool,
    closed: bool,
}

impl<W: Write> JsonArrayEncoder<W> {
    pub fn new(w: W) -> Self {
        Self {
            w,
            started: false,
            first: true,
            closed: false,
        }
    }

    fn start_if_needed(&mut self) -> Result<()> {
        if !self.started {
            self.w.write_all(b"[")?;
            self.started = true;
        }
        Ok(())
    }
}

impl<W: Write> SegmentEncoder for JsonArrayEncoder<W> {
    fn write_segment(&mut self, seg: &Segment) -> Result<()> {
        if self.closed {
            anyhow::bail!("encoder is closed");
        }

        self.start_if_needed()?;

        if !self.first {
            self.w.write_all(b",")?;
        }
        self.first = false;

        serde_json::to_writer(&mut self.w, seg)?;
        Ok(())
    }

    fn close(&mut self) -> Result<()> {
        if self.closed {
            return Ok(());
        }

        // Ensure we still output a valid JSON array even if no segments were written.
        self.start_if_needed()?;
        self.w.write_all(b"]")?;
        self.w.flush()?;
        self.closed = true;
        Ok(())
    }
}
