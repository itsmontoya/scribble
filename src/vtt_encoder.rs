use anyhow::Result;
use std::io::Write;

use crate::segment_encoder::SegmentEncoder;
use crate::segments::Segment;

pub struct VttEncoder<W: Write> {
    w: W,
    started: bool,
    closed: bool,
    idx: u32,
}

impl<W: Write> VttEncoder<W> {
    pub fn new(w: W) -> Self {
        Self {
            w,
            started: false,
            closed: false,
            idx: 1,
        }
    }

    fn start_if_needed(&mut self) -> Result<()> {
        if !self.started {
            self.w.write_all(b"WEBVTT\n\n")?;
            self.started = true;
        }
        Ok(())
    }
}

impl<W: Write> SegmentEncoder for VttEncoder<W> {
    fn write_segment(&mut self, seg: &Segment) -> Result<()> {
        if self.closed {
            anyhow::bail!("encoder is closed");
        }

        self.start_if_needed()?;

        let start = get_formatted_timestamp(seg.start_seconds);
        let end = get_formatted_timestamp(seg.end_seconds);

        writeln!(&mut self.w, "{}", self.idx)?;
        writeln!(&mut self.w, "{} --> {}", start, end)?;
        writeln!(&mut self.w, "{}", seg.text)?;
        writeln!(&mut self.w)?;
        self.idx += 1;

        Ok(())
    }

    fn close(&mut self) -> Result<()> {
        if self.closed {
            return Ok(());
        }
        self.w.flush()?;
        self.closed = true;
        Ok(())
    }
}

fn get_formatted_timestamp(t: f32) -> String {
    let total_ms = (t * 1000.0).round() as u64;
    let ms = total_ms % 1000;
    let total_s = total_ms / 1000;
    let s = total_s % 60;
    let total_m = total_s / 60;
    let m = total_m % 60;
    let h = total_m / 60;
    format!("{h:02}:{m:02}:{s:02}.{ms:03}")
}
