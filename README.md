# Mobile RAG Engine

A Flutter package for fully local RAG (Retrieval-Augmented Generation) on mobile devices.

## Why I Built This

Implementing AI-powered search on mobile typically requires a server. Embedding generation, vector storage, similarity search—all handled server-side, with the app just making API calls.

But this approach has problems:
- No internet, no functionality
- User data gets sent to servers
- Ongoing server costs

So I found a way to do **everything on-device**.

## Technical Challenges

I first tried pure Dart. Loading ONNX models, tokenizing, generating embeddings—it was too slow. Vector search became noticeably laggy with just 1,000 documents.

So I brought in Rust.

### Rust + Flutter Architecture

> 📖 **[Architecture Guide](docs/guides/architecture_guide.md)** - Detailed system architecture documentation

```
Flutter (Dart)
    │
    ├── EmbeddingService (ONNX Runtime)
    │       └── text → 384-dim vector
    │
    └── flutter_rust_bridge (FFI)
            │
            ▼
        Rust
            ├── Tokenizer (HuggingFace tokenizers)
            ├── SQLite (vector storage)
            └── HNSW Index (O(log n) search)
```

Rust's `tokenizers` crate is 10x+ faster than Dart for tokenization. Vector search improved from O(n) to O(log n) using the `instant-distance` HNSW implementation.

## How It Differs

### vs. Server-based RAG
- Works completely offline
- Data never leaves the device
- Zero network latency

### vs. Pure Dart Implementation
- Native Rust performance
- HNSW enables fast search even with large document sets
- Memory-efficient vector storage

### vs. Existing Flutter Vector DBs
- Direct ONNX model loading (no external APIs needed)
- Swappable models for Korean/multilingual support
- Integrated pipeline from embedding to search

## Performance

Tested on iOS Simulator (Apple Silicon Mac):

| Operation | Time |
|-----------|------|
| Tokenization (short text) | 0.8ms |
| Embedding generation (short text) | 4ms |
| Embedding generation (long text) | 36ms |
| HNSW search (100 docs) | 1ms |

With 1ms search on 100 documents, real-time search is feasible up to 10,000+ documents.

## Problems Solved During Development

### 1. iOS Cross-Compilation
Initially, the `onig` regex library blocked iOS builds. `___chkstk_darwin` symbol missing error. Switched to pure Rust `fancy-regex` to fix it.

### 2. HNSW Index Timing
Rebuilding HNSW on every document insert results in O(n²) complexity. Changed to rebuild once after bulk inserts.

### 3. Duplicate Document Handling
Identical documents caused duplicates in search results. Added SHA256 content hashing to skip already-stored documents.

### 4. ONNX Runtime Thread Safety
Tried parallel batch embedding, but `onnxruntime`'s `OrtSession` isn't thread-safe. Switched to sequential processing—still fast enough for real-world use since individual embeddings are quick.

## Usage

### Installation

```yaml
dependencies:
  mobile_rag_engine:
    git:
      url: https://github.com/dev07060/mobile_rag_engine.git
  
  # Required for iOS static library loading
  flutter_rust_bridge: ^2.0.0
```

### iOS Setup

For iOS, the Rust library is statically linked. Add this import and use `ExternalLibrary.process()`:

```dart
import 'dart:io' show Platform;
import 'package:flutter_rust_bridge/flutter_rust_bridge_for_generated.dart';
```

### Initialization

```dart
import 'package:mobile_rag_engine/mobile_rag_engine.dart';
import 'package:flutter_rust_bridge/flutter_rust_bridge_for_generated.dart';

Future<void> initializeRag() async {
  // Initialize Rust library (platform-specific)
  if (Platform.isIOS || Platform.isMacOS) {
    // iOS/macOS: static library loaded via DynamicLibrary.process()
    await RustLib.init(
      externalLibrary: ExternalLibrary.process(iKnowHowToUseIt: true),
    );
  } else {
    // Android/Linux/Windows: dynamic library
    await RustLib.init();
  }

  // Load tokenizer
  await initTokenizer(tokenizerPath: 'path/to/tokenizer.json');

  // Load ONNX model
  final modelBytes = await rootBundle.load('assets/model.onnx');
  await EmbeddingService.init(modelBytes.buffer.asUint8List());

  // Initialize DB
  await initDb(dbPath: 'path/to/rag.db');
}
```

> **Note**: On iOS, if you encounter `symbol not found` errors, ensure your Podfile has been updated with `pod install` after adding the dependency.


### Adding Documents

```dart
final text = "Flutter is a cross-platform UI framework.";
final embedding = await EmbeddingService.embed(text);

final result = await addDocument(
  dbPath: dbPath,
  content: text,
  embedding: embedding,
);

if (result.isDuplicate) {
  print("Document already exists");
}

// Rebuild index after bulk inserts
await rebuildHnswIndex(dbPath: dbPath);
```

### Searching

```dart
final query = "cross-platform development";
final queryEmbedding = await EmbeddingService.embed(query);

final results = await searchSimilar(
  dbPath: dbPath,
  queryEmbedding: queryEmbedding,
  topK: 5,
);

for (final doc in results) {
  print(doc);
}
```

## Required Models

You need a Sentence Transformer model in ONNX format.

```bash
pip install optimum[exporters]
optimum-cli export onnx \
  --model sentence-transformers/all-MiniLM-L6-v2 \
  ./model_output
```

Add `model_output/model.onnx` and `tokenizer.json` to your app's assets.

## Releases

- **[v0.3.0 - Rust Semantic Chunking](docs/guides/v0.3.0_semantic_chunking_update.md)** - Migrated to Unicode-based semantic chunking
- **[v0.2.0 - LLM-Optimized Chunking](docs/guides/v0.2.0_chunking_update.md)** - Added chunking and context assembly

## Future Plans

- INT8 quantization to reduce model size
- Korean-specific models (KoSimCSE, KR-SBERT)
- ~~Chunking strategies for long documents~~ ✅ (v0.2.0, v0.3.0)
- Hybrid search (keyword + semantic)

## License

MIT

## Contributing

Bug reports, feature requests, and PRs are all welcome.
