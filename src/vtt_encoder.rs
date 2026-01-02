use anyhow::Result;
use std::io::Write;

use crate::segment_encoder::SegmentEncoder;
use crate::segments::Segment;

/// A `SegmentEncoder` that writes segments in WebVTT format.
///
/// Design:
/// - We stream output directly to a `Write` implementation.
/// - We write the WebVTT header lazily on the first segment so that:
///   - callers can construct the encoder without immediately writing output
///   - even "no segments" runs still behave predictably (close just flushes)
pub struct VttEncoder<W: Write> {
    /// The underlying writer we stream VTT into.
    w: W,

    /// Whether we've written the `WEBVTT` header.
    started: bool,

    /// Whether the encoder has been closed.
    closed: bool,
}

impl<W: Write> VttEncoder<W> {
    /// Create a new VTT encoder that writes to the provided writer.
    pub fn new(w: W) -> Self {
        Self {
            w,
            started: false,
            closed: false,
        }
    }

    /// Write the WebVTT header if we haven't written it yet.
    fn start_if_needed(&mut self) -> Result<()> {
        if !self.started {
            // WebVTT files begin with a mandatory header line followed by a blank line.
            self.w.write_all(b"WEBVTT\n\n")?;
            self.started = true;
        }
        Ok(())
    }
}

impl<W: Write> SegmentEncoder for VttEncoder<W> {
    /// Write a single cue in WebVTT format.
    fn write_segment(&mut self, seg: &Segment) -> Result<()> {
        if self.closed {
            anyhow::bail!("cannot write segment: encoder is already closed");
        }

        self.start_if_needed()?;

        // WebVTT timestamps use `HH:MM:SS.mmm`.
        let start = format_timestamp_vtt(seg.start_seconds);
        let end = format_timestamp_vtt(seg.end_seconds);

        // Cue timing line.
        writeln!(&mut self.w, "{start} --> {end}")?;

        // Cue text. (We write it verbatim; if we later want to sanitize/escape,
        // this is where we'd do it.)
        writeln!(&mut self.w, "{}", seg.text)?;

        // Blank line separates cues.
        writeln!(&mut self.w)?;

        // Flush so streaming consumers (stdout, pipes, sockets) see output promptly.
        self.w.flush()?;

        Ok(())
    }

    /// Flush the underlying writer. This is idempotent.
    fn close(&mut self) -> Result<()> {
        if self.closed {
            return Ok(());
        }

        // We flush so callers get output immediately (especially important for streaming to stdout).
        self.w.flush()?;
        self.closed = true;

        Ok(())
    }
}

/// Format seconds into a WebVTT timestamp (`HH:MM:SS.mmm`).
///
/// Rounding policy:
/// - We round to the nearest millisecond to reduce drift when converting from `f32`.
fn format_timestamp_vtt(seconds: f32) -> String {
    let total_ms = (seconds * 1000.0).round() as u64;

    let ms = total_ms % 1000;
    let total_s = total_ms / 1000;

    let s = total_s % 60;
    let total_m = total_s / 60;

    let m = total_m % 60;
    let h = total_m / 60;

    format!("{h:02}:{m:02}:{s:02}.{ms:03}")
}
