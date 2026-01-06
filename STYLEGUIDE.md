# Style Guide

This document describes how we write, review, and change code in this repository.
It applies to everyone making changes here, whether human or automated.

Our goal is simple: boring, explicit, well-documented Rust with predictable behavior.

## Documentation & comments

Our goals:
- We write generous but purposeful comments: explain *why* a choice exists, what tradeoff we accepted, and what invariant we rely on.
- We keep public APIs well-documented and unsurprising: clear names, clear inputs/outputs, and clear failure modes.
- We use “we” in comments and documentation to describe design intent and decisions.

Guidelines:
- Prefer module/type/function documentation for “why” and “how to use”; prefer inline comments only when the code’s intent is not obvious.
- If a decision has a non-obvious tradeoff, we document it near the decision.
- We avoid narrating code (“we increment i”) unless it clarifies an invariant or edge case.

Voice examples:
- ✅ “We buffer at least N bytes so downstream processing is amortized and predictable.”
- ✅ “We return EOF early here to preserve streaming semantics.”
- ❌ “We call read() and then process_bytes().”

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
- `cargo clippy --all-targets --all-features -- -D warnings`
- `cargo check --all-features`
- `cargo test --all-features`

We treat formatting and clippy warnings as part of the API contract, not optional hygiene.

## Pull requests

When creating pull requests:
- We write the PR description inside a fenced code block using Markdown.
- The description focuses on intent and behavior, not a line-by-line diff.
- We explicitly call out behavior changes, edge cases, and follow-up work.

## Before finishing a change, we quickly check:

- Did we preserve behavior (or clearly document the behavior change)?
- Are errors contextual at the boundary?
- Are “we” statements used for intent in docs and comments?
- Is the control flow understandable to a new contributor in ~60 seconds?
- Did we run `cargo fmt` and `cargo clippy` with the project’s standard flags?
