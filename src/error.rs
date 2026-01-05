use std::error::Error as StdError;

use thiserror::Error;

/// Scribble's crate-wide result type.
pub type Result<T> = std::result::Result<T, Error>;

/// Scribble's crate-wide error type.
///
/// This is intentionally decoupled from `anyhow` so downstream libraries aren't forced to
/// adopt `anyhow` in their own public APIs.
#[derive(Debug, Error)]
pub enum Error {
    #[error("{0}")]
    Message(String),

    #[error(transparent)]
    Other(#[from] Box<dyn StdError + Send + Sync>),
}

impl Error {
    pub(crate) fn msg(message: impl Into<String>) -> Self {
        Self::Message(message.into())
    }
}

impl From<anyhow::Error> for Error {
    fn from(err: anyhow::Error) -> Self {
        Self::Message(format!("{err:#}"))
    }
}

impl From<std::io::Error> for Error {
    fn from(err: std::io::Error) -> Self {
        Self::Other(Box::new(err))
    }
}

impl From<serde_json::Error> for Error {
    fn from(err: serde_json::Error) -> Self {
        Self::Other(Box::new(err))
    }
}

impl From<std::str::Utf8Error> for Error {
    fn from(err: std::str::Utf8Error) -> Self {
        Self::Other(Box::new(err))
    }
}
