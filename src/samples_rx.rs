use std::sync::mpsc;

use anyhow::{Result, anyhow};

use crate::vad::VadStreamReceiver;

pub enum SamplesRx<'a> {
    Plain(mpsc::Receiver<Vec<f32>>),
    Vad(VadStreamReceiver<'a>),
}

impl<'a> SamplesRx<'a> {
    pub fn recv(&mut self) -> Result<Vec<f32>> {
        match self {
            SamplesRx::Plain(rx) => rx
                .recv()
                .map_err(|_| anyhow!("decoder output channel disconnected")),
            SamplesRx::Vad(rx) => rx.recv(),
        }
    }
}

