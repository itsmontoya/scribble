use anyhow::Result;
use std::io::Write;

use crate::segment_encoder::SegmentEncoder;
use crate::segments::Segment;

/// A `SegmentEncoder` that writes segments as a single JSON array.
///
/// Design:
/// - We stream output directly to a `Write` implementation to avoid buffering
///   all segments in memory.
/// - The encoder is stateful so we can emit a well-formed JSON array incrementally.
///
/// Example output:
/// ```json
/// [
///   { "start": 0.0, "end": 1.2, "text": "hello" },
///   { "start": 1.2, "end": 2.5, "text": "world" }
/// ]
/// ```
pub struct JsonArrayEncoder<W: Write> {
    /// The underlying writer we stream JSON into.
    w: W,

    /// Whether we have written the opening `[` of the JSON array.
    started: bool,

    /// Whether the next element will be the first element in the array.
    /// This lets us correctly place commas between elements.
    first: bool,

    /// Whether the encoder has been closed.
    /// Once closed, no further writes are allowed.
    closed: bool,
}

impl<W: Write> JsonArrayEncoder<W> {
    /// Create a new JSON array encoder that writes to the given writer.
    ///
    /// At creation time:
    /// - We have not written anything yet.
    /// - The JSON array is opened lazily on the first write or on close.
    pub fn new(w: W) -> Self {
        Self {
            w,
            started: false,
            first: true,
            closed: false,
        }
    }

    /// Write the opening `[` of the JSON array if we have not already done so.
    ///
    /// We defer writing the opening bracket so that:
    /// - Empty output still results in valid JSON (`[]`)
    /// - We do not emit partial output unless a segment is actually written
    fn start_if_needed(&mut self) -> Result<()> {
        if !self.started {
            self.w.write_all(b"[")?;
            self.started = true;
        }
        Ok(())
    }
}

impl<W: Write> SegmentEncoder for JsonArrayEncoder<W> {
    /// Serialize a single segment and append it to the JSON array.
    fn write_segment(&mut self, seg: &Segment) -> Result<()> {
        if self.closed {
            anyhow::bail!("cannot write segment: encoder is already closed");
        }

        // Ensure the JSON array has been started.
        self.start_if_needed()?;

        // Write a comma before every element except the first.
        if !self.first {
            self.w.write_all(b",")?;
        }
        self.first = false;

        // Stream the segment directly into the writer as JSON.
        serde_json::to_writer(&mut self.w, seg)?;

        Ok(())
    }

    /// Finalize the JSON array and flush the underlying writer.
    ///
    /// This method is idempotent:
    /// - Calling `close()` multiple times is safe.
    /// - After closing, no further segments may be written.
    fn close(&mut self) -> Result<()> {
        if self.closed {
            return Ok(());
        }

        // Ensure we still output a valid JSON array even if no segments were written.
        self.start_if_needed()?;

        // Close the JSON array.
        self.w.write_all(b"]")?;
        self.w.flush()?;

        self.closed = true;
        Ok(())
    }
}
