use clap::ValueEnum;

/// The supported output formats for encoded transcription segments.
///
/// Why this exists:
/// - We want a single, strongly-typed representation of output formats
///   across the CLI and library code.
/// - Using an enum avoids stringly-typed conditionals and keeps format
///   selection explicit and discoverable.
///
/// Integration notes:
/// - `ValueEnum` allows this enum to be used directly as a CLI flag with `clap`.
/// - Each variant maps to a concrete `SegmentEncoder` implementation.
#[derive(Debug, Clone, ValueEnum)]
pub enum OutputType {
    /// Output segments as a JSON array.
    Json,

    /// Output segments in WebVTT subtitle format.
    Vtt,
}
