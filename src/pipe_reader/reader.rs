use std::io::{self, Read};

/// A `Read` wrapper that:
/// - reads from an underlying reader until it has at least `min_bytes_read` (or EOF),
/// - calls `process_bytes` on that chunk,
/// - then serves the processed bytes to the caller.
///
/// `process_bytes` may change the size of the data.
pub struct PipeReader<R, F>
where
    R: Read,
    F: FnMut(&[u8]) -> io::Result<Vec<u8>>,
{
    inner: R,
    process_bytes: F,
    min_bytes_read: usize,

    // Raw input bytes waiting to be processed
    in_buf: Vec<u8>,

    // Processed output bytes waiting to be consumed by `read()`
    out_buf: Vec<u8>,
    out_pos: usize,

    // Whether we've seen EOF from the inner reader
    eof: bool,
}

impl<R, F> PipeReader<R, F>
where
    R: Read,
    F: FnMut(&[u8]) -> io::Result<Vec<u8>>,
{
    pub fn new(inner: R, min_bytes_read: usize, process_bytes: F) -> Self {
        Self {
            inner,
            process_bytes,
            min_bytes_read,
            in_buf: Vec::new(),
            out_buf: Vec::new(),
            out_pos: 0,
            eof: false,
        }
    }

    fn out_remaining(&self) -> usize {
        self.out_buf.len().saturating_sub(self.out_pos)
    }

    fn drain_out_into(&mut self, dst: &mut [u8]) -> usize {
        let n = dst.len().min(self.out_remaining());
        if n == 0 {
            return 0;
        }
        dst[..n].copy_from_slice(&self.out_buf[self.out_pos..self.out_pos + n]);
        self.out_pos += n;

        // If fully drained, reset buffers to keep things tidy
        if self.out_pos == self.out_buf.len() {
            self.out_buf.clear();
            self.out_pos = 0;
        }

        n
    }

    /// Ensure `out_buf` has something to serve, if possible.
    /// Returns:
    /// - Ok(true) if `out_buf` now has bytes (or already had bytes)
    /// - Ok(false) if there's nothing left to produce (true EOF)
    fn refill_out_buf(&mut self) -> io::Result<bool> {
        if self.out_remaining() > 0 {
            return Ok(true);
        }

        // Read until we hit min_bytes_read or EOF
        while !self.eof && self.in_buf.len() < self.min_bytes_read {
            // Read in chunks to amortize syscalls; size can be tuned.
            let mut tmp = [0u8; 8192];
            let n = self.inner.read(&mut tmp)?;
            if n == 0 {
                self.eof = true;
                break;
            }
            self.in_buf.extend_from_slice(&tmp[..n]);
        }

        // If we have any input bytes (enough OR partial at EOF), process them.
        if !self.in_buf.is_empty() && (self.in_buf.len() >= self.min_bytes_read || self.eof) {
            let input = std::mem::take(&mut self.in_buf);
            let processed = (self.process_bytes)(&input)?;
            self.out_buf = processed;
            self.out_pos = 0;

            // It is legal for processing to output 0 bytes; if so, try again
            // (but avoid infinite loops if we're truly done).
            if self.out_remaining() == 0 {
                if self.eof {
                    // No more input will arrive and processing produced nothing.
                    return Ok(false);
                }
                // Otherwise, keep going: read more input and try processing again.
                return self.refill_out_buf();
            }

            return Ok(true);
        }

        // No input, no output, and EOF reached => done.
        Ok(false)
    }
}

impl<R, F> Read for PipeReader<R, F>
where
    R: Read,
    F: FnMut(&[u8]) -> io::Result<Vec<u8>>,
{
    fn read(&mut self, dst: &mut [u8]) -> io::Result<usize> {
        if dst.is_empty() {
            return Ok(0);
        }

        // If we can serve from existing processed output, do that first.
        let n = self.drain_out_into(dst);
        if n > 0 {
            return Ok(n);
        }

        // Otherwise, produce more processed bytes (if possible), then drain.
        if !self.refill_out_buf()? {
            return Ok(0); // true EOF
        }

        Ok(self.drain_out_into(dst))
    }
}
