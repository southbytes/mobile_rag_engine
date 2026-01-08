pub mod api;
mod frb_generated;

// Logger is automatically initialized by flutter_rust_bridge's setup_default_user_utils()
// iOS: uses oslog (output to Xcode console)
// Android: uses android_logger (output to logcat)
