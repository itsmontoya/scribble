use std::error::Error as StdError;
use std::fmt;

use thiserror::Error;

/// Scribble's crate-wide result type.
pub type Result<T> = std::result::Result<T, Error>;

/// Scribble's crate-wide error type.
///
/// This is intentionally decoupled from `anyhow` so downstream libraries aren't forced to
/// adopt `anyhow` in their own public APIs.
#[derive(Debug, Error)]
pub enum Error {
    #[error("invalid input: {message}")]
    InvalidInput {
        message: String,
        #[source]
        source: Option<Box<dyn StdError + Send + Sync>>,
    },

    #[error("I/O error: {message}")]
    Io {
        message: String,
        #[source]
        source: std::io::Error,
    },

    #[error("decode error: {message}")]
    Decode {
        message: String,
        #[source]
        source: Option<Box<dyn StdError + Send + Sync>>,
    },

    #[error("backend error: {message}")]
    Backend {
        message: String,
        #[source]
        source: Option<Box<dyn StdError + Send + Sync>>,
    },

    #[error("{message}")]
    Other {
        message: String,
        #[source]
        source: Option<Box<dyn StdError + Send + Sync>>,
    },
}

impl Error {
    pub(crate) fn invalid_input(message: impl Into<String>) -> Self {
        Self::InvalidInput {
            message: message.into(),
            source: None,
        }
    }
}

#[derive(Debug)]
struct AnyhowChainError {
    rendered: String,
}

impl fmt::Display for AnyhowChainError {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        f.write_str(&self.rendered)
    }
}

impl StdError for AnyhowChainError {}

impl From<anyhow::Error> for Error {
    fn from(err: anyhow::Error) -> Self {
        Error::Other {
            message: err.to_string(),
            source: Some(Box::new(AnyhowChainError {
                rendered: format!("{err:#}"),
            })),
        }
    }
}

impl From<std::io::Error> for Error {
    fn from(err: std::io::Error) -> Self {
        Self::Io {
            message: err.to_string(),
            source: err,
        }
    }
}

impl From<serde_json::Error> for Error {
    fn from(err: serde_json::Error) -> Self {
        Self::Other {
            message: err.to_string(),
            source: Some(Box::new(err)),
        }
    }
}

impl From<std::str::Utf8Error> for Error {
    fn from(err: std::str::Utf8Error) -> Self {
        Self::Other {
            message: err.to_string(),
            source: Some(Box::new(err)),
        }
    }
}
