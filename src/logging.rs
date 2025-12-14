use std::os::raw::{c_char, c_void};

unsafe extern "C" fn whisper_log_callback(
    // level
    _: u32,
    // c_msg
    _: *const c_char,
    _user_data: *mut c_void,
) {
}

pub fn init_whisper_logging() {
    unsafe {
        // callback + user_data (we don't use user_data here)
        whisper_rs::set_log_callback(Some(whisper_log_callback), std::ptr::null_mut());
    }
}
