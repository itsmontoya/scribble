use std::io;
use std::sync::mpsc;

/// An iterator adapter that:
/// - pulls `Vec<T>` chunks from a channel until it has at least `min_items` (or the channel closes),
/// - calls `process_items` on that buffered input,
/// - yields the processed `Vec<T>`.
///
/// `process_items` may change the size of the data.
pub struct PipeReceiver<T, F>
where
    F: FnMut(&[T]) -> io::Result<Vec<T>>,
{
    rx: mpsc::Receiver<Vec<T>>,
    process_items: F,
    min_items: usize,

    in_buf: Vec<T>,
    eof: bool,
}

impl<T, F> PipeReceiver<T, F>
where
    F: FnMut(&[T]) -> io::Result<Vec<T>>,
{
    pub fn new(rx: mpsc::Receiver<Vec<T>>, min_items: usize, process_items: F) -> Self {
        Self {
            rx,
            process_items,
            min_items,
            in_buf: Vec::new(),
            eof: false,
        }
    }

    fn fill_in_buf(&mut self) {
        while !self.eof && self.in_buf.len() < self.min_items {
            match self.rx.recv() {
                Ok(chunk) => self.in_buf.extend(chunk),
                Err(_) => self.eof = true,
            }
        }
    }

/// Receive and process the next chunk, if any.
///
/// This matches the blocking behavior of `std::sync::mpsc::Receiver::recv`:
/// it waits for enough input to be available to run `process_items`, and returns
/// `io::ErrorKind::BrokenPipe` once the channel is disconnected and no buffered
/// input remains.
    pub fn recv(&mut self) -> io::Result<Vec<T>> {
        loop {
            self.fill_in_buf();

            if self.in_buf.is_empty() && self.eof {
                return Err(io::Error::new(
                    io::ErrorKind::BrokenPipe,
                    "pipe receiver disconnected",
                ));
            }

            if self.in_buf.len() >= self.min_items || self.eof {
                let input = std::mem::take(&mut self.in_buf);
                let processed = (self.process_items)(&input)?;

                if processed.is_empty() {
                    if self.eof {
                        return Err(io::Error::new(
                            io::ErrorKind::BrokenPipe,
                            "pipe receiver disconnected",
                        ));
                    }
                    continue;
                }

                return Ok(processed);
            }
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn yields_processed_once_min_items_reached() {
        let (tx, rx) = mpsc::sync_channel::<Vec<u32>>(8);
        tx.send(vec![1, 2]).unwrap();
        tx.send(vec![3]).unwrap();
        drop(tx);

        let mut pr =
            PipeReceiver::new(rx, 3, |items: &[u32]| Ok(items.iter().map(|v| v * 2).collect()));
        let out = pr.recv().unwrap();
        assert_eq!(out, vec![2, 4, 6]);
        let err = pr.recv().unwrap_err();
        assert_eq!(err.kind(), io::ErrorKind::BrokenPipe);
    }

    #[test]
    fn flushes_remainder_on_close() {
        let (tx, rx) = mpsc::sync_channel::<Vec<u32>>(8);
        tx.send(vec![1, 2]).unwrap();
        drop(tx);

        let mut pr = PipeReceiver::new(rx, 10, |items: &[u32]| Ok(items.to_vec()));
        let out = pr.recv().unwrap();
        assert_eq!(out, vec![1, 2]);
        let err = pr.recv().unwrap_err();
        assert_eq!(err.kind(), io::ErrorKind::BrokenPipe);
    }
}
