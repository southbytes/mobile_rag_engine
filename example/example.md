# Mobile RAG Engine - Example

A complete on-device RAG (Retrieval-Augmented Generation) implementation.

## Quick Start

```dart
import 'package:flutter/services.dart';
import 'package:path_provider/path_provider.dart';
import 'package:mobile_rag_engine/mobile_rag_engine.dart';

void main() async {
  WidgetsFlutterBinding.ensureInitialized();
  
  // 1. Initialize Rust library
  await RustLib.init();
  
  // 2. Load tokenizer
  final dir = await getApplicationDocumentsDirectory();
  await initTokenizer(tokenizerPath: '${dir.path}/tokenizer.json');
  
  // 3. Load ONNX embedding model
  final modelBytes = await rootBundle.load('assets/model.onnx');
  await EmbeddingService.init(modelBytes.buffer.asUint8List());
  
  // 4. Initialize RAG service
  final ragService = SourceRagService(dbPath: '${dir.path}/rag.db');
  await ragService.init();
  
  runApp(MyApp(ragService: ragService));
}
```

## Adding Documents

```dart
// Add a document with automatic chunking and embedding
final result = await ragService.addSourceWithChunking(
  'Flutter is Google\'s UI toolkit for building beautiful apps...',
  onProgress: (done, total) => print('Embedding: $done/$total'),
);

print('Created ${result.chunkCount} chunks');

// Rebuild HNSW index after adding documents
await ragService.rebuildIndex();
```

## Semantic Search

```dart
// Search for relevant chunks
final searchResult = await ragService.search(
  'How to build mobile apps?',
  topK: 5,
  tokenBudget: 2000,
);

// Get assembled context for LLM
print('Found ${searchResult.chunks.length} chunks');
print('Context tokens: ${searchResult.context.estimatedTokens}');

// Format prompt for LLM
final prompt = ragService.formatPrompt(
  'How to build mobile apps?',
  searchResult,
);
```

## Batch Embedding

```dart
// Embed multiple texts efficiently
final embeddings = await EmbeddingService.embedBatch(
  ['Text 1', 'Text 2', 'Text 3'],
  onProgress: (done, total) => print('Progress: $done/$total'),
);
```

## Performance

| Operation | Time | Engine |
|:----------|-----:|:-------|
| Tokenization | 0.04ms | Rust |
| HNSW Search | 0.3ms | Rust |
| Embedding | 25-100ms | ONNX |

See the full example app in the [GitHub repository](https://github.com/dev07060/mobile_rag_engine/tree/main/example).
