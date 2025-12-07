# Changelog

All notable changes to this project will be documented in this file.

The format is based on [Keep a Changelog](https://keepachangelog.com/en/1.0.0/),
and this project adheres to [Semantic Versioning](https://semver.org/spec/v2.0.0.html).

## [0.1.0] - 2024-12-08

### Added
- Initial release
- On-device semantic search with HNSW vector indexing
- Rust-powered tokenization via HuggingFace tokenizers
- ONNX Runtime integration for embedding generation
- SQLite-based vector storage with content deduplication
- Batch embedding support with progress callback
- Cross-platform support (iOS and Android)

### Features
- `initDb()` - Initialize SQLite database
- `addDocument()` - Add documents with SHA256 deduplication
- `searchSimilar()` - HNSW-based semantic search
- `rebuildHnswIndex()` - Manual index rebuild
- `EmbeddingService.embed()` - Generate embeddings
- `EmbeddingService.embedBatch()` - Batch embedding

### Performance
- HNSW search: O(log n) complexity
- Tokenization: ~0.8ms for short text
- Embedding: ~4ms for short text, ~36ms for long text
- Search (100 docs): ~1ms
