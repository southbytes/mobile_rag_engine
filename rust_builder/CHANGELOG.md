# Changelog

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
