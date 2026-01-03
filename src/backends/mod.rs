/// Built-in ASR backends.
pub mod whisper;

/// Experimental Silero ASR backend (ONNX).
#[cfg(feature = "silero-onnx")]
pub mod silero;
