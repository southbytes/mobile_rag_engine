/// Utility for extracting text from documents.
library;

import '../src/rust/api/document_parser.dart' as raw;

/// Utility class for parsing documents (PDF, DOCX, etc.).
///
/// Wraps the low-level Rust document parser functions.
class DocumentParser {
  DocumentParser._();

  /// Extract text from PDF bytes.
  static Future<String> parsePdf(List<int> bytes) =>
      raw.extractTextFromPdf(fileBytes: bytes);

  /// Extract text from DOCX bytes.
  static Future<String> parseDocx(List<int> bytes) =>
      raw.extractTextFromDocx(fileBytes: bytes);

  /// Auto-detect document type and extract text.
  static Future<String> parse(List<int> bytes) =>
      raw.extractTextFromDocument(fileBytes: bytes);
}
