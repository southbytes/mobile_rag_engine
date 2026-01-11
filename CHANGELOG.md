# Changelog

All notable changes to this project will be documented in this file.

The format is based on [Keep a Changelog](https://keepachangelog.com/en/1.0.0/),
and this project adheres to [Semantic Versioning](https://semver.org/spec/v2.0.0.html).

## 0.4.1

### Added
- **PDF/DOCX Text Extraction**: New `extractTextFromPdf()`, `extractTextFromDocx()`, and `extractTextFromDocument()` functions
- **Smart Dehyphenation**: Automatically rejoins words split by line breaks and page boundaries
- **Page Number Removal**: Strips standalone page numbers from PDF text extraction
- **macOS Entitlements**: Added file read permissions for macOS file picker support

### Changed
- **Rust Core**: Added `pdf-extract`, `docx-lite`, and `regex` crates for document processing

## 0.4.0

### Changed
- **README Cleanup**: Removed all emojis and unnecessary sections for cleaner documentation

## 0.3.9

### Fixed
- **README Images**: Updated image paths to use GitHub raw URLs for pub.dev compatibility

## 0.3.8

### Changed
- **ONNX Runtime**: Reverted to `onnxruntime ^1.4.1` for CocoaPods compatibility (1.23.2 not yet available)
- **README**: Added benchmark result screenshots (iOS/Android) and architecture diagram
- **Platform Support**: Removed Linux/Windows from publish (no pre-compiled binaries available)

### Removed
- **ChunkingTestScreen**: Removed unnecessary test screen from example app

### Added
- **Android Platform**: Added Android support to example app

## 0.3.7

### Changed
- **ONNX Runtime Upgrade**: Migrated from `onnxruntime` to `onnxruntime_v2` (v1.23.2) with optional GPU acceleration support
- **README Remake**: Completely redesigned README with "No Rust Installation Required" emphasis, accurate benchmarks, and Mermaid architecture diagram
- **Benchmark UI Overhaul**: Visual separation of Rust-powered (fast) vs ONNX (standard) operations with category headers and icons

### Added
- **GPU Acceleration Option**: `EmbeddingService.init()` now accepts `useGpuAcceleration` parameter (CoreML/NNAPI support, disabled by default)
- **macOS Support for Example App**: Example app now supports macOS platform
- **Benchmark Categories**: Results now grouped by `BenchmarkCategory.rust` and `BenchmarkCategory.onnx`

### Fixed
- **Pub Point Warning**: Removed non-existent `assets/` directory reference from pubspec.yaml
- **Static Analysis**: Fixed all lint issues (unnecessary imports, avoid_print, curly braces)

## 0.3.5
- Globalization: Removed all Korean text and logic, replaced with English.
- Updated prompt builder and semantic chunker for better international support.
- Updated default language settings to English.

## 0.3.4

- Fix model download URLs in README (use correct Teradata/bge-m3 and BAAI/bge-m3 sources)
- Add production model deployment strategies guide

## 0.3.3

- Improve README with Quick Start guide and model download instructions
- Update to pub.dev dependency instead of git

## 0.3.2

- Update `rag_engine_flutter` dependency to `^0.3.0` (fixes platform directory issue)

## [0.3.1] - 2026-01-08

### Fixed
- **Package structure fix**: Update `rag_engine_flutter` dependency to v0.2.0 which includes rust/ source

## [0.3.0] - 2026-01-08

### Changed
- **Package Rename**: Rust crate renamed to `rag_engine_flutter` for pub.dev distribution.
- **iOS Podspec Fix**: Resolved linker path issues for iOS builds.
- **Asset Handling**: Force-overwrite asset files to prevent stale cache issues.

### Removed
- Deprecated `test_app` and `local-gemma-macos` directories.

## [0.2.0] - 2024-12-08


### Added
- **LLM-Optimized Chunking**: Introduced `ChunkingService` with Recursive Character Splitting and Overlap support.
- **Improved Data Model**: Separated storage into `Source` (original document) and `Chunk` (searchable parts).
- **Context Assembly**: Added `ContextBuilder` to intelligently assemble LLM context within a token budget.
- **High-Level API**: New `SourceRagService` for automated chunking, embedding, and indexing pipeline.
- **Context Strategies**: Support for `relevanceFirst`, `diverseSources`, and `chronological` context assembly strategies.

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
