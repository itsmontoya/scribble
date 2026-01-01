# Scribble &emsp; [![Build Status]][actions] [![Latest Version]][crates.io]

[Build Status]: https://img.shields.io/github/actions/workflow/status/itsmontoya/scribble/ci.yml?branch=master
[actions]: https://github.com/itsmontoya/scribble/actions?query=branch%3Amaster
[Latest Version]: https://img.shields.io/crates/v/scribble.svg
[crates.io]: https://crates.io/crates/scribble

Scribble is a fast, lightweight transcription engine written in Rust, built on top of Whisper and designed for both CLI and webserver use.

![billboard](https://github.com/itsmontoya/scribble/blob/main/banner.png?raw=true "Scribble billboard")

## Goals

- Provide a clean, idiomatic Rust API for audio transcription
- Support multiple output formats (JSON, VTT, plain text, etc.)
- Work equally well as a CLI tool or embedded service
- Keep the core simple, explicit, and easy to extend

## Installation

Clone the repository and build the binaries:

```bash
cargo build --release
```

This will produce the following binaries:

- `scribble-cli` — transcribe WAV files
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

It expects:

- a **mono, 16 kHz WAV file**
- a Whisper model
- (optionally) a Whisper-VAD model

### Basic transcription (VTT output)

```bash
cargo run --bin scribble-cli -- \
  --model ./models/ggml-large-v3-turbo.bin \
  --audio ./audio.wav
```

Output is written to `stdout` in WebVTT format by default.

### JSON output

```bash
cargo run --bin scribble-cli -- \
  --model ./models/ggml-large-v3-turbo.bin \
  --audio ./audio.wav \
  --output-type json
```

### Enable voice activity detection (VAD)

```bash
cargo run --bin scribble-cli -- \
  --model ./models/ggml-large-v3-turbo.bin \
  --vad-model ./models/ggml-silero-v6.2.0.bin \
  --enable-vad \
  --audio ./audio.wav
```

When VAD is enabled:

- non-speech regions are suppressed
- if no speech is detected, no output is produced

### Specify language explicitly

```bash
cargo run --bin scribble-cli -- \
  --model ./models/ggml-large-v3-turbo.bin \
  --audio ./audio.wav \
  --language en
```

If `--language` is omitted, Whisper will auto-detect.

### Write output to a file

```bash
cargo run --bin scribble-cli -- \
  --model ./models/ggml-large-v3-turbo.bin \
  --audio ./audio.wav \
  --output-type vtt \
  > transcript.vtt
```

## Library usage

Scribble is also designed to be embedded as a library.

High-level usage looks like:

```rust
use scribble::{opts::Opts, output_type::OutputType, scribble::Scribble};
use std::fs::File;

let scribble = Scribble::new(
    "./models/ggml-large-v3-turbo.bin",
    "./models/ggml-silero-v6.2.0.bin",
)?;

let mut input = File::open("audio.wav")?;
let mut output = Vec::new();

let opts = Opts {
    enable_translate_to_english: false,
    enable_voice_activity_detection: true,
    language: None,
    output_type: OutputType::Json,
};

scribble.transcribe(&mut input, &mut output, &opts)?;

let json = String::from_utf8(output)?;
println!("{json}");
```

## TODOs

- Expand testing (goal of 80%+ test coverage)
- Update VAD to utilize streaming approach
- Implement the webserver
- Streaming / incremental transcription support

## Status

Scribble is under active development. The API is not yet stable, but the foundations are in place and evolving quickly.

## License

MIT
