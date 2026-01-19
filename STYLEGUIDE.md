# Style Guide

This document describes how we write, review, and change code in this repository.
It applies to everyone making changes here, whether human or automated.

Our goal is simple: boring, explicit, well-documented Rust with predictable behavior.

## Documentation & comments

Our goals:
- Write purposeful comments: capture *why* a choice exists, the tradeoff, and the invariant.
- Keep public APIs well-documented and unsurprising: clear names, clear inputs/outputs, and clear failure modes.
- Prefer concise, neutral voice. Avoid chatty or “collective” phrasing (e.g., “we …”) unless it adds necessary context.

Guidelines:
- Prefer module/type/function documentation for “why” and “how to use”; prefer inline comments only when the code’s intent is not obvious.
- If a decision has a non-obvious tradeoff, we document it near the decision.
- Avoid narrating code (“increment i”) unless it clarifies an invariant or edge case.
- Prefer short sentences. If a comment needs multiple points, use a small bullet list.

Voice examples:
- ✅ “Buffers at least N bytes so downstream processing is amortized and predictable.”
- ✅ “Returns EOF early to preserve streaming semantics.”
- ❌ “Calls read() and then process_bytes().”
- ❌ “We do X here to make things nicer.”

Public API documentation should answer:
- What it does (one sentence)
- Inputs, outputs, and important invariants
- Error cases and when they occur
- Performance characteristics only when relevant and stable
- A short example when it helps prevent misuse

## Code style

Our preferences:
- We value clarity and correctness over cleverness.
- We prefer explicit control flow and avoid hidden or “magic” behavior.
- We use conservative, readable error handling with context.
- We prefer stable, predictable behavior over micro-optimizations.

Guidelines:
- Prefer simple, explicit state machines over hidden behavior.
- Be explicit about buffering, ownership, lifetimes, and boundaries.
- Avoid surprising implicit conversions, fallthrough, or overly generic abstractions unless they clearly reduce complexity.
- Handle edge cases intentionally (EOF, partial reads, empty inputs, cancellation where applicable).

Error handling:
- Add context at boundaries (I/O, parsing, decoding, external calls).
- Preserve root causes; do not discard errors.
- Prefer meaningful error messages over clever formatting.

Testing expectations:
- We add or adjust tests when behavior changes.
- We test edge cases and invariants, especially around streaming or buffering.

## Tooling & verification

When making code changes, we run the following before considering a change complete:

- `cargo fmt --all -- --check`
- `cargo clippy --all-targets --features bin-scribble-cli,bin-model-downloader,bin-scribble-server -- -D warnings`
- `cargo check --features bin-scribble-cli,bin-model-downloader,bin-scribble-server`
- `./scripts/test-all.sh`

We treat formatting and clippy warnings as part of the API contract, not optional hygiene.

For PR expectations and repo-level checklists, see `AGENTS.md`.
