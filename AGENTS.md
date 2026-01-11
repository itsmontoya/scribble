# Agent Guidelines

This file contains repo-specific instructions for automated coding agents and contributors.

For writing, code style, and verification conventions, see `STYLEGUIDE.md`.

## Pull requests

When creating pull requests:

- Write the PR description inside a fenced code block using Markdown.
- Focus on intent and behavior, not a line-by-line diff.
- Explicitly call out behavior changes, edge cases, and follow-up work.

## Before finishing a change

- Preserve behavior (or clearly document the behavior change).
- Add error context at boundaries (I/O, parsing, decoding, external calls).
- Keep control flow understandable to a new contributor in ~60 seconds.
- Run the standard verification commands in `STYLEGUIDE.md`.
