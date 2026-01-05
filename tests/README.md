# Tests

This directory contains integration tests that exercise Scribble end-to-end (decode → transcription → encoding).

## Running

```bash
cargo test --all-features
```

## Local assets

Some tests require local model files and fixtures that are not downloaded automatically.

- Whisper model: `./models/ggml-tiny.bin`
- VAD model: `./models/ggml-silero-v6.2.0.bin`
- Fixture audio: `./tests/fixtures/jfk.wav`

## Skip behavior

- If required model files are missing, the integration test(s) skip (not fail) so `cargo test` remains friendly for contributors.
- If the fixture audio file is missing, the integration test(s) fail (this indicates a broken repo checkout).

## Fixture provenance

See `tests/fixtures/README.md`.

