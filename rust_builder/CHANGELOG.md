# Changelog

## 0.6.0
- **DB Connection Pool**: Implemented connection pooling with `r2d2` for 50-90% search performance improvement
- **Resource Optimization**: Eliminated redundant SQLite connections to reduce file descriptor usage
- **Refactoring**: Updated API to use pooled connections instead of direct file opens

## 0.5.1
- **Unit Tests**: Added tests for `hnsw_index` and `document_parser` modules
- **BM25 Korean Support**: Improved Korean tokenization using `unicode-segmentation` crate for better word boundary detection
- **Code Quality**: Enhanced test coverage for core Rust modules

## 0.5.0
- **PDF/DOCX Text Extraction**: New text extraction with smart dehyphenation
- **Markdown Chunking**: Structure-aware chunking with header path inheritance
- **PDF Fix**: Enhanced text normalization to preserve paragraph structure
- **Safety**: Added 50MB file size processing limit

## 0.4.0
- **Fix binary mismatch**: Rebuilt native binaries to resolve hash mismatch with Dart bindings.
## 0.3.0

- **Fix platform directories missing**: Include ios/, android/, macos/, linux/, windows/ in package
- Add .pubignore to prevent parent ignore rules from excluding platform configs

## 0.2.0

- **Fix package structure**: Include rust/ directory in package for correct pub.dev distribution
- Update platform build configs (iOS, macOS, Android, Linux, Windows) to reference internal rust/ path

## 0.1.0

- Initial release
- High-performance tokenization with HuggingFace tokenizers
- HNSW vector indexing for O(log n) similarity search  
- SQLite integration for persistent vector storage
- Semantic text chunking with Unicode boundary detection
- Prebuilt binaries for iOS, macOS, and Android
