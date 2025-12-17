use std::os::raw::{c_char, c_void};
use std::sync::Once;

/// A no-op log callback used to silence logs emitted by whisper.cpp.
///
/// Why this exists:
/// - whisper.cpp can emit a large volume of logs directly from native code
/// - for CLIs and services, we want full control over what gets printed
///
/// Today:
/// - We intentionally discard all whisper logs.
///
/// Future:
/// - This callback can be expanded to forward logs into `tracing`, `log`,
///   or another structured logging system with configurable levels.
unsafe extern "C" fn whisper_log_callback(
    // Log level provided by whisper.cpp (currently unused).
    _level: u32,

    // Null-terminated C string containing the log message (currently ignored).
    _c_msg: *const c_char,

    // Optional user data pointer (unused for now).
    _user_data: *mut c_void,
) {
    // Intentionally left empty.
    //
    // We discard all whisper logs to keep output quiet and predictable.
}

/// Ensure whisper logging is configured exactly once for the lifetime of the process.
///
/// Why this exists:
/// - whisper.rs exposes a global log callback
/// - calling `set_log_callback` multiple times is unnecessary and potentially confusing
///
/// Today:
/// - We register a no-op callback to silence all logs.
///
/// Future:
/// - This can become a configuration point for log levels or integrations.
pub fn init_whisper_logging() {
    static INIT: Once = Once::new();

    INIT.call_once(|| unsafe {
        // Register our log callback and ignore user data.
        whisper_rs::set_log_callback(Some(whisper_log_callback), std::ptr::null_mut());
    });
}
