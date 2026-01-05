use std::cell::Cell;
use std::io::{Read, Result as IoResult};

use scribble::decoder::{SamplesSink, StreamDecodeOpts, decode_to_stream_from_read};

/// Ensures the streaming decoder accepts `Read + Send` inputs that are not `Sync`.
///
/// This matters for truly streaming sources (like HTTP request bodies) that are moved into a
/// dedicated decode thread and are never accessed concurrently.
#[test]
fn decoder_accepts_send_non_sync_readers() {
    struct NotSyncReader {
        inner: std::io::Cursor<Vec<u8>>,
        _marker: Cell<u8>,
    }

    impl Read for NotSyncReader {
        fn read(&mut self, buf: &mut [u8]) -> IoResult<usize> {
            self.inner.read(buf)
        }
    }

    struct NoopSink;
    impl SamplesSink for NoopSink {
        fn on_samples(&mut self, _samples_16k_mono: &[f32]) -> anyhow::Result<bool> {
            Ok(true)
        }
    }

    let reader = NotSyncReader {
        inner: std::io::Cursor::new(Vec::new()),
        _marker: Cell::new(0),
    };

    // We expect probing to fail on empty input; the point of this test is that it compiles and
    // runs without requiring `R: Sync`.
    let res = decode_to_stream_from_read(reader, StreamDecodeOpts::default(), &mut NoopSink);
    assert!(res.is_err());
}
