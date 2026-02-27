use crate::output_type::OutputType;

/// Options that control how a transcription is performed.
///
/// This struct represents *library-level configuration*, not CLI flags directly.
/// The CLI is responsible for mapping user input into this type so that:
/// - the library remains reusable outside of a CLI context
/// - other frontends (APIs, tests, batch jobs) can construct options programmatically
#[derive(Debug, Clone)]
pub struct Opts {
    /// Optional key selecting which loaded model to use for transcription.
    ///
    /// For built-in Whisper backends that support loading multiple models, the key is derived
    /// from the model filename (not the full path).
    ///
    /// When `None`, the backend uses its default model (typically the first loaded).
    pub model_key: Option<String>,

    /// Whether to translate speech to English instead of transcribing verbatim.
    ///
    /// Note:
    /// - This flag is parsed and propagated today, but translation is not yet
    ///   fully wired into the Whisper parameter configuration.
    pub enable_translate_to_english: bool,

    /// Whether to apply voice activity detection (VAD) before transcription.
    ///
    /// When enabled:
    /// - Non-speech regions are zeroed out in the audio buffer.
    /// - If no speech is detected at all, transcription exits early with no output.
    pub enable_voice_activity_detection: bool,

    /// Optional language hint (e.g. `"en"`, `"es"`).
    ///
    /// When `None`, Whisper auto-detects the spoken language.
    /// This field exists to support future CLI flags or API usage.
    pub language: Option<String>,

    /// The desired output format for transcription segments.
    pub output_type: OutputType,

    /// Minimum buffered audio duration (seconds) before running Whisper in incremental mode.
    ///
    /// This only affects the streaming/incremental path (when VAD is disabled). Larger windows
    /// increase latency but can improve segmentation stability.
    pub incremental_min_window_seconds: usize,

    /// Whether to emit single segments immediately in incremental mode.
    ///
    /// By default (`false`), the incremental transcriber waits for 2+ segments before emitting,
    /// treating the last segment as potentially incomplete. This provides better accuracy for
    /// long-form transcription but adds latency for short utterances.
    ///
    /// When `true`, single segments are emitted as soon as detected. This is useful for
    /// voice assistants and real-time applications where low latency is more important than
    /// waiting for sentence boundaries.
    pub emit_single_segments: bool,
}
