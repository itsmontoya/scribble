# Contributing

Thanks for your interest in contributing to Scribble.

## Quick start

- Install Rust (this repo pins the toolchain via `rust-toolchain.toml`).
- Run checks before opening a PR:
  - `cargo fmt --all -- --check`
  - `cargo clippy --all-targets --all-features -- -D warnings`
  - `cargo test --all-features`

## Style and conventions

See `STYLEGUIDE.md` for:
- code style and review expectations
- error handling and documentation conventions
- the standard `fmt`/`clippy` flags we use

## Pull requests

- Keep PRs focused and easy to review.
- Include a clear description of the behavior change and any tradeoffs.
- Use the PR template in `.github/pull_request_template.md`.

## Questions

If you’re unsure about an API or design direction, open an issue first so we can align before you implement a larger change.
