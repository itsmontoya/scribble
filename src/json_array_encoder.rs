use std::io::Write;

use crate::Result;
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
            return Err(crate::Error::msg(
                "cannot write segment: encoder is already closed",
            ));
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

        // Flush so streaming consumers (stdout, pipes, sockets) see output promptly.
        self.w.flush()?;

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

#[cfg(test)]
mod tests {
    use super::*;

    fn seg(start: f32, end: f32, text: &str) -> Segment {
        Segment {
            start_seconds: start,
            end_seconds: end,
            text: text.to_string(),
            tokens: Vec::new(),
            language_code: "en".to_string(),
            next_speaker_turn: false,
        }
    }

    #[test]
    fn json_array_close_without_segments_emits_empty_array() -> anyhow::Result<()> {
        let mut out = Vec::new();
        let mut enc = JsonArrayEncoder::new(&mut out);
        enc.close()?;
        assert_eq!(std::str::from_utf8(&out)?, "[]");
        Ok(())
    }

    #[test]
    fn json_array_writes_valid_json_incrementally() -> anyhow::Result<()> {
        let mut out = Vec::new();
        let mut enc = JsonArrayEncoder::new(&mut out);

        enc.write_segment(&seg(0.0, 1.0, "hello"))?;
        enc.write_segment(&seg(1.0, 2.5, "world"))?;
        enc.close()?;

        let s = std::str::from_utf8(&out)?;
        let parsed: serde_json::Value = serde_json::from_str(s)?;
        let arr = parsed.as_array().expect("expected JSON array");
        assert_eq!(arr.len(), 2);
        assert_eq!(arr[0]["text"], "hello");
        assert_eq!(arr[1]["text"], "world");
        Ok(())
    }

    #[test]
    fn json_array_close_is_idempotent() -> anyhow::Result<()> {
        let mut out = Vec::new();
        let mut enc = JsonArrayEncoder::new(&mut out);
        enc.close()?;
        enc.close()?;
        assert_eq!(std::str::from_utf8(&out)?, "[]");
        Ok(())
    }

    #[test]
    fn json_array_write_after_close_errors() -> anyhow::Result<()> {
        let mut out = Vec::new();
        let mut enc = JsonArrayEncoder::new(&mut out);
        enc.close()?;
        let err = enc.write_segment(&seg(0.0, 1.0, "nope")).unwrap_err();
        assert!(err.to_string().contains("already closed"));
        Ok(())
    }
}
