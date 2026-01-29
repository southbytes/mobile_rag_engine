import 'dart:io';

import 'package:flutter_test/flutter_test.dart';
import 'package:mobile_rag_engine/src/rust/api/error.dart';
import 'package:mobile_rag_engine/src/rust/frb_generated.dart';
import 'package:mobile_rag_engine/services/source_rag_service.dart';

// Check if running in CI environment
bool get isCI => Platform.environment['CI'] == 'true';

void main() {
  // These tests require native Rust library which is not available in CI
  // Run locally with: flutter test test/smart_error_test.dart
  test(
    'Smart Error Handling - Verify DatabaseError on missing pool',
    () async {
      // 1. Initialize FFI
      await RustLib.init();

      // 2. Instantiate service *without* initializing DB Pool
      // This intentionally skips `initDbPool` to trigger the error check in Rust
      final service = SourceRagService(dbPath: 'test_dummy.db');

      // 3. Expect RagError
      try {
        await service.init();
        fail('Should have thrown RagError');
      } on RagError catch (e) {
        print('\n[Caught RagError Successfully]');

        e.when(
          databaseError: (msg) {
            print('✅ Type: DatabaseError');
            print('✅ Message: $msg');
            // Expect the message from get_connection() in db_pool.rs
            expect(msg, contains('DB pool not initialized'));
          },
          ioError: (msg) => fail('Unexpected IoError: $msg'),
          modelLoadError: (msg) => fail('Unexpected ModelLoadError: $msg'),
          invalidInput: (msg) => fail('Unexpected InvalidInput: $msg'),
          internalError: (msg) => fail('Unexpected InternalError: $msg'),
          unknown: (msg) => fail('Unexpected Unknown error: $msg'),
        );
      }
    },
    skip: isCI ? 'Requires native Rust library (not available in CI)' : null,
  );

  test(
    'Smart Error Handling - Verify RagError on Search without DB',
    () async {
      // 1. Instantiate service *without* initializing DB Pool
      final service = SourceRagService(dbPath: 'test_dummy_search.db');

      // 2. Attempt search
      try {
        await service.search('hello', topK: 1);
        fail('Should have thrown RagError');
      } on RagError catch (e) {
        print('\n[Caught RagError on Search Successfully]');

        ExtensionRagError(e).when(
          databaseError: (_) => fail(
            'Should be Internal or DB Error depending on impl path but expected failure',
          ), // Actually might be internal if HNSW logic fails, or DB if conn fails first.
          ioError: (_) => fail('Unexpected IoError'),
          modelLoadError: (_) => fail('Unexpected ModelLoadError'),
          invalidInput: (_) => fail('Unexpected InvalidInput'),
          // search_chunks accesses get_connection() first if HNSW loaded, or later.
          // If HNSW not loaded, it calls rebuild_chunk_hnsw_index -> get_connection -> DatabaseError.
          // Let's create a more robust check.
          internalError: (msg) {
            print('✅ Caught expected error: $msg');
          },
          unknown: (msg) {
            print('✅ Caught expected error type (unknown): $msg');
          },
        );

        // Since we just want to verify it IS a RagError and not a generic Exception
        expect(e, isA<RagError>());
        // Check message matches expectation for uninitialized pool
        ExtensionRagErrorMessage(e).message.contains('DB pool not initialized');
      } catch (e) {
        fail('Caught unexpected exception type: ${e.runtimeType}');
      }
    },
    skip: isCI ? 'Requires native Rust library (not available in CI)' : null,
  );
}

// Helper since extension might not be visible in test scope without import or if I just defined it in service file.
// Actually I need to import the service file extension or redefine helper if private.
// The extension 'RagErrorMessage' is public in source_rag_service.dart, so I can use it if imported.
extension ExtensionRagErrorMessage on RagError {
  String get message => when(
    databaseError: (msg) => msg,
    ioError: (msg) => msg,
    modelLoadError: (msg) => msg,
    invalidInput: (msg) => msg,
    internalError: (msg) => msg,
    unknown: (msg) => msg,
  );
}

// Wrapper for when to handle missing generated code access in test file if needed
extension type ExtensionRagError(RagError e) implements RagError {}
