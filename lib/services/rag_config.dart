/// Configuration for RagEngine initialization.
///
/// Use [RagConfig.fromAssets] for convenient asset-based configuration:
///
/// ```dart
/// final config = RagConfig.fromAssets(
///   tokenizerAsset: 'assets/tokenizer.json',
///   modelAsset: 'assets/model.onnx',
/// );
/// ```
library;

/// Configuration options for RagEngine initialization.
class RagConfig {
  /// Asset path for tokenizer JSON file.
  ///
  /// Example: `'assets/tokenizer.json'`
  final String tokenizerAsset;

  /// Asset path for ONNX embedding model.
  ///
  /// Example: `'assets/model.onnx'`
  final String modelAsset;

  /// Name of the SQLite database file.
  ///
  /// If null, defaults to `'rag.sqlite'`.
  /// The file will be created in the app's documents directory.
  ///
  /// Both `.sqlite` and `.db` extensions work (e.g., `'rag.sqlite'` or `'rag.db'`).
  final String? databaseName;

  /// Maximum characters per chunk (default: 500).
  final int maxChunkChars;

  /// Overlap characters between chunks for context continuity (default: 50).
  final int overlapChars;

  /// Creates a RagConfig with all options.
  const RagConfig({
    required this.tokenizerAsset,
    required this.modelAsset,
    this.databaseName,
    this.maxChunkChars = 500,
    this.overlapChars = 50,
  });

  /// Convenience factory for asset-based initialization.
  ///
  /// ```dart
  /// final config = RagConfig.fromAssets(
  ///   tokenizerAsset: 'assets/tokenizer.json',
  ///   modelAsset: 'assets/model.onnx',
  ///   databaseName: 'my_rag.sqlite', // optional
  /// );
  /// ```
  factory RagConfig.fromAssets({
    required String tokenizerAsset,
    required String modelAsset,
    String? databaseName,
    int maxChunkChars = 500,
    int overlapChars = 50,
  }) => RagConfig(
    tokenizerAsset: tokenizerAsset,
    modelAsset: modelAsset,
    databaseName: databaseName,
    maxChunkChars: maxChunkChars,
    overlapChars: overlapChars,
  );
}
