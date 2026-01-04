//! Voice Activity Detection (VAD) utilities.
//!
//! Scribble applies VAD (when enabled) in the high-level pipeline before passing audio into a
//! backend. This keeps backends focused on ASR and makes preprocessing behavior consistent.

mod processor;
mod stream;
mod to_speech;

pub use processor::VadProcessor;
pub use stream::VadStream;
pub use stream::VadStreamReceiver;
pub use to_speech::{DEFAULT_VAD_POLICY, VadPolicy};
