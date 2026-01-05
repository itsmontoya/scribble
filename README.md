# Scribble &emsp; [![Build Status]][actions] [![Latest Version]][crates.io]

[Build Status]: https://img.shields.io/github/actions/workflow/status/itsmontoya/scribble/ci.yaml?branch=main
[actions]: https://github.com/itsmontoya/scribble/actions?query=branch%3Amain
[Latest Version]: https://img.shields.io/crates/v/scribble.svg
[crates.io]: https://crates.io/crates/scribble

Scribble is a fast, lightweight transcription engine written in Rust, with a built-in Whisper backend and a backend trait for custom implementations.

![banner](https://github.com/itsmontoya/scribble/blob/main/banner.png?raw=true "Scribble banner")

Scribble will demux/decode **audio *or* video containers** (MP4, MP3, WAV, FLAC, OGG, WebM, MKV, etc.), downmix to mono, and resample to 16 kHz — no preprocessing required.

## Demo
<img src="https://github.com/itsmontoya/scribble/blob/main/demo/demo.gif?raw=true" />

## Project goals

- Provide a clean, idiomatic Rust API for audio transcription
- Support multiple output formats (JSON, VTT, plain text, etc.)
- Work equally well as a CLI tool or embedded library
- **Be streaming-first:** designed to support incremental, chunk-based transcription pipelines (live audio, long-running streams, and low-latency workflows)
- **Enable composable pipelines:** VAD → transcription → encoding, with clear extension points for streaming and real-time use cases
- Keep the core simple, explicit, and easy to extend

> Scribble is built with **streaming and real-time transcription** in mind, even when operating on static files today.

## Installation

Clone the repository and build the binaries:

```bash
cargo build --release
```

This will produce the following binaries:

- `scribble-cli` — transcribe audio/video (decodes + normalizes to mono 16 kHz)
- `scribble-server` — HTTP server for transcription
- `model-downloader` — download Whisper and VAD models

## model-downloader

`model-downloader` is a small helper CLI for downloading **known-good Whisper and Whisper-VAD models** into a local directory.

### List available models

```bash
cargo run --bin model-downloader -- --list
```

Example output:

```text
Whisper models:
  - tiny
  - base.en
  - large-v3-turbo
  - large-v3-turbo-q8_0
  ...

VAD models:
  - silero-v5.1.2
  - silero-v6.2.0
```

### Download a model

```bash
cargo run --bin model-downloader -- --name large-v3-turbo
```

By default, models are downloaded into `./models`.

### Download into a custom directory

```bash
cargo run --bin model-downloader -- \
  --name silero-v6.2.0 \
  --dir /opt/scribble/models
```

Downloads are performed safely:

- written to `*.part`
- fsynced
- atomically renamed into place

## scribble-cli

`scribble-cli` is the main transcription CLI.

It accepts audio or video containers and normalizes them to Whisper’s required mono 16 kHz internally. Provide:

- an input media path (e.g. MP4, MP3, WAV, FLAC, OGG, WebM, MKV) or `-` to stream from stdin
- a Whisper model
- a Whisper-VAD model (used when `--enable-vad` is set)

### Basic transcription (VTT output)

```bash
cargo run --bin scribble-cli -- \
  --model ./models/ggml-large-v3-turbo.bin \
  --vad-model ./models/ggml-silero-v6.2.0.bin \
  --input ./input.mp4
```

Output is written to `stdout` in WebVTT format by default.

## scribble-server

`scribble-server` is a long-running HTTP server that loads models once and accepts transcription requests over HTTP.

### Start the server

```bash
cargo run --bin scribble-server -- \
  --model ./models/ggml-large-v3-turbo.bin \
  --vad-model ./models/ggml-silero-v6.2.0.bin \
  --host 127.0.0.1 \
  --port 8080
```

### Transcribe via HTTP (multipart upload)

```bash
curl -sS --data-binary @./input.mp4 \
  "http://127.0.0.1:8080/v1/transcribe?output=vtt" \
  > transcript.vtt
```

For JSON output:

```bash
curl -sS --data-binary @./input.wav \
  "http://127.0.0.1:8080/v1/transcribe?output=json" \
  > transcript.json
```

Example using all query params:

```bash
curl -sS --data-binary @./input.mp4 \
  "http://127.0.0.1:8080/v1/transcribe?output=json&output_type=json&model_key=ggml-large-v3-turbo.bin&enable_vad=true&translate_to_english=true&language=en" \
  > transcript.json
```

### Prometheus metrics

`scribble-server` exposes Prometheus metrics at `GET /metrics`.

```bash
curl -sS "http://127.0.0.1:8080/metrics"
```

Key metrics:

- `scribble_http_requests_total` (labels: `status`)
- `scribble_http_request_duration_seconds` (labels: `status`)
- `scribble_http_in_flight_requests`

### Logging

All binaries emit structured JSON logs to `stderr`.

- Default level: `error`
- Override with `SCRIBBLE_LOG` (e.g. `SCRIBBLE_LOG=info`)

### JSON output

```bash
cargo run --bin scribble-cli -- \
  --model ./models/ggml-large-v3-turbo.bin \
  --vad-model ./models/ggml-silero-v6.2.0.bin \
  --input ./input.wav \
  --output-type json
```

### Enable voice activity detection (VAD)

```bash
cargo run --bin scribble-cli -- \
  --model ./models/ggml-large-v3-turbo.bin \
  --vad-model ./models/ggml-silero-v6.2.0.bin \
  --enable-vad \
  --input ./input.wav
```

When VAD is enabled:

- non-speech regions are suppressed
- if no speech is detected, no output is produced

### Specify language explicitly

```bash
cargo run --bin scribble-cli -- \
  --model ./models/ggml-large-v3-turbo.bin \
  --vad-model ./models/ggml-silero-v6.2.0.bin \
  --input ./input.wav \
  --language en
```

If `--language` is omitted, Whisper will auto-detect.

### Write output to a file

```bash
cargo run --bin scribble-cli -- \
  --model ./models/ggml-large-v3-turbo.bin \
  --vad-model ./models/ggml-silero-v6.2.0.bin \
  --input ./input.wav \
  --output-type vtt \
  > transcript.vtt
```

## Library usage

Scribble is also designed to be embedded as a library.

High-level usage looks like:

```rust
use scribble::{opts::Opts, output_type::OutputType, scribble::Scribble};
use std::fs::File;

let mut scribble = Scribble::new(
    ["./models/ggml-large-v3-turbo.bin"],
    "./models/ggml-silero-v6.2.0.bin",
)?;

let mut input = File::open("audio.wav")?;
let mut output = Vec::new();

let opts = Opts {
    model_key: None,
    enable_translate_to_english: false,
    enable_voice_activity_detection: true,
    language: None,
    output_type: OutputType::Json,
    incremental_min_window_seconds: 1,
};

scribble.transcribe(&mut input, &mut output, &opts)?;

let json = String::from_utf8(output)?;
println!("{json}");
```

## Goals

- [X] Make VAD streaming-capable
- [X] Support streaming and incremental transcription
- [X] Select the primary audio track in multi-track video containers
- [X] Implement a web server
- [X] Add Prometheus metrics endpoint
- [X] Add structured logs (tracing)
- [ ] Expand test coverage to 80%+

## Coverage

This project uses [`cargo-llvm-cov`](https://github.com/taiki-e/cargo-llvm-cov) for coverage locally and in CI.

One-time setup:

```bash
rustup component add llvm-tools-preview
cargo install cargo-llvm-cov
```

Run coverage locally:

```bash
# Print a summary to stdout
cargo llvm-cov --all-features --all-targets

# Generate an HTML report (writes to ./target/llvm-cov/html)
cargo llvm-cov --all-features --all-targets --html
```

## Status

Scribble is under active development. The API is not yet stable, but the foundations are in place and evolving quickly.

## Contributing

See [`STYLEGUIDE.md`](STYLEGUIDE.md) for code style and repository conventions.

## License

MIT

![footer](https://github.com/itsmontoya/scribble/blob/main/footer.png?raw=true "Scribble footer")
