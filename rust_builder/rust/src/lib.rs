pub mod api;
mod frb_generated;

// 로거는 flutter_rust_bridge의 setup_default_user_utils()가 자동 초기화함
// iOS: oslog 사용 (Xcode 콘솔에 출력)
// Android: android_logger 사용 (logcat에 출력)
