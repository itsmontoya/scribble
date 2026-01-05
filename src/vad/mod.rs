//! Voice Activity Detection (VAD) utilities.
//!
//! Scribble applies VAD (when enabled) in the high-level pipeline before passing audio into a
//! backend. This keeps backends focused on ASR and makes preprocessing behavior consistent.
//!
//! We expose a receiver-like adapter (`VadStreamReceiver`) rather than the lower-level buffering
//! machinery. This keeps public APIs unsurprising and preserves explicit control flow at call sites.

mod processor;
mod stream;
mod to_speech;

pub use processor::VadProcessor;
pub use stream::VadStreamReceiver;
