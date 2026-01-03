use std::os::raw::{c_char, c_void};
use std::sync::Once;

/// A no-op log callback used to silence logs emitted by whisper.cpp.
unsafe extern "C" fn whisper_log_callback(
    _level: u32,
    _c_msg: *const c_char,
    _user_data: *mut c_void,
) {
    // Intentionally left empty.
}

/// Ensure whisper logging is configured exactly once for the lifetime of the process.
pub fn init_whisper_logging() {
    static INIT: Once = Once::new();

    INIT.call_once(|| unsafe {
        whisper_rs::set_log_callback(Some(whisper_log_callback), std::ptr::null_mut());
    });
}
