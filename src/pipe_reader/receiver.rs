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
    /// Returns:
    /// - `Ok(Some(vec))` when a processed chunk is produced,
    /// - `Ok(None)` when the channel is closed and no buffered input remains.
    pub fn recv_processed(&mut self) -> io::Result<Option<Vec<T>>> {
        self.fill_in_buf();

        if self.in_buf.is_empty() && self.eof {
            return Ok(None);
        }

        if self.in_buf.len() >= self.min_items || self.eof {
            let input = std::mem::take(&mut self.in_buf);
            let processed = (self.process_items)(&input)?;

            if processed.is_empty() {
                if self.eof {
                    return Ok(None);
                }
                return self.recv_processed();
            }

            return Ok(Some(processed));
        }

        Ok(None)
    }
}

impl<T, F> Iterator for PipeReceiver<T, F>
where
    F: FnMut(&[T]) -> io::Result<Vec<T>>,
{
    type Item = io::Result<Vec<T>>;

    fn next(&mut self) -> Option<Self::Item> {
        match self.recv_processed() {
            Ok(Some(v)) => Some(Ok(v)),
            Ok(None) => None,
            Err(e) => Some(Err(e)),
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
        let out = pr.next().unwrap().unwrap();
        assert_eq!(out, vec![2, 4, 6]);
        assert!(pr.next().is_none());
    }

    #[test]
    fn flushes_remainder_on_close() {
        let (tx, rx) = mpsc::sync_channel::<Vec<u32>>(8);
        tx.send(vec![1, 2]).unwrap();
        drop(tx);

        let mut pr = PipeReceiver::new(rx, 10, |items: &[u32]| Ok(items.to_vec()));
        let out = pr.next().unwrap().unwrap();
        assert_eq!(out, vec![1, 2]);
        assert!(pr.next().is_none());
    }
}
