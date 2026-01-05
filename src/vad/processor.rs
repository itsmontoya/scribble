use anyhow::Result;
use whisper_rs::{WhisperVadContext, WhisperVadContextParams};

use super::to_speech::{DEFAULT_VAD_POLICY, VadPolicy, to_speech_only_with_policy};

/// Voice Activity Detection (VAD) processor.
///
/// Current implementation uses whisper.cpp's built-in VAD via `whisper-rs`.
pub struct VadProcessor {
    ctx: WhisperVadContext,
    policy: VadPolicy,
}

impl VadProcessor {
    pub fn new(model_path: &str) -> Result<Self> {
        let params = WhisperVadContextParams::default();
        let ctx = WhisperVadContext::new(model_path, params)?;
        Ok(Self {
            ctx,
            policy: DEFAULT_VAD_POLICY,
        })
    }

    pub(crate) fn policy(&self) -> VadPolicy {
        self.policy
    }

    /// Apply VAD in-place, attenuating non-speech regions.
    ///
    /// Returns `Ok(true)` when any speech is detected, `Ok(false)` otherwise.
    pub fn apply(&mut self, samples_16k_mono: &mut [f32]) -> Result<bool> {
        to_speech_only_with_policy(&mut self.ctx, 16_000, samples_16k_mono, self.policy)
    }
}
