# Scribble

Scribble is a fast, lightweight transcription engine written in Rust, built on top of Whisper and designed for both CLI and webserver use.

![billboard](https://github.com/itsmontoya/scribble/blob/main/banner.png?raw=true "Scribble billboard")

## Goals

- Provide a clean, idiomatic Rust API for audio transcription
- Support multiple output formats (JSON, VTT, plain text, etc.)
- Work equally well as a CLI tool or embedded service
- Keep the core simple, explicit, and easy to extend

## TODOs

- Expand testing (goal of 80%+ test coverage)
- Implement the webserver
- Streaming / incremental transcription support

## Status

Scribble is under active development. The API is not yet stable, but the foundations are in place and evolving quickly.

## License

MIT
