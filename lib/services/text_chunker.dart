/// Utility for splitting text into semantic chunks.
library;

import '../src/rust/api/semantic_chunker.dart' as chunker;

/// Utility class for splitting text into smaller chunks for RAG.
///
/// Wraps the low-level Rust semantic chunker functions.
class TextChunker {
  TextChunker._();

  /// Chunk text using Markdown structure-aware splitter.
  ///
  /// Preserves headers, code blocks, and tables.
  ///
  /// [text] - The markdown text to split.
  /// [maxChars] - Maximum characters per chunk (default: 500).
  static Future<List<chunker.StructuredChunk>> markdown(
    String text, {
    int maxChars = 500,
  }) async => chunker.markdownChunk(text: text, maxChars: maxChars);

  /// Chunk text using standard recursive character splitter.
  ///
  /// Splits by paragraphs, then sentences, then words to fit within [maxChars].
  ///
  /// [text] - The text to split.
  /// [maxChars] - Maximum characters per chunk (default: 500).
  /// [overlapChars] - Overlap between chunks for context continuity (default: 50).
  static Future<List<chunker.SemanticChunk>> recursive(
    String text, {
    int maxChars = 500,
    int overlapChars = 50,
  }) async => chunker.semanticChunkWithOverlap(
    text: text,
    maxChars: maxChars,
    overlapChars: overlapChars,
  );

  /// Classify a text chunk by type (e.g., definition, list, code).
  ///
  /// Uses rule-based pattern matching to determine the likely content type.
  static Future<chunker.ChunkType> classify(String text) async =>
      chunker.classifyChunk(text: text);
}
