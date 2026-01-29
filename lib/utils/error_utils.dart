import '../src/rust/api/error.dart';

/// Extension to provide user-friendly error messages from RagError.
extension RagErrorUi on RagError {
  /// A user-friendly message suitable for UI display (Snackbars, Dialogs).
  String get userFriendlyMessage {
    return when(
      databaseError: (_) => '데이터베이스 오류가 발생했습니다. 잠시 후 다시 시도해주세요.',
      ioError: (_) => '파일을 읽거나 쓸 수 없습니다. 저장소 권한을 확인해주세요.',
      modelLoadError: (_) => 'AI 모델을 불러오는데 실패했습니다. 앱을 재시작해주세요.',
      invalidInput: (msg) => '입력값이 올바르지 않습니다: $msg',
      internalError: (_) => '일시적인 내부 오류가 발생했습니다.',
      unknown: (_) => '알 수 없는 오류가 발생했습니다.',
    );
  }

  /// The technical details for debugging (same as original message).
  String get technicalMessage {
    return when(
      databaseError: (msg) => msg,
      ioError: (msg) => msg,
      modelLoadError: (msg) => msg,
      invalidInput: (msg) => msg,
      internalError: (msg) => msg,
      unknown: (msg) => msg,
    );
  }
}
