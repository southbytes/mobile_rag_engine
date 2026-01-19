# Quick Start Guide

Get started with `mobile_rag_engine` in 5 minutes.

---

## Prerequisites

- Flutter 3.9+
- iOS 13.0+ / Android API 21+ / macOS 10.15+

---

## Step 1: Add Dependency

```yaml
# pubspec.yaml
dependencies:
  mobile_rag_engine: ^0.4.3
```

```bash
flutter pub get
```

---

## Step 2: Download Model

Run from your project root:

```bash
mkdir -p assets && cd assets

# BGE-m3 (multilingual, Korean support)
curl -L -o model.onnx "https://huggingface.co/Teradata/bge-m3/resolve/main/onnx/model_int8.onnx"
curl -L -o tokenizer.json "https://huggingface.co/BAAI/bge-m3/resolve/main/tokenizer.json"
```

Register assets in `pubspec.yaml`:

```yaml
flutter:
  assets:
    - assets/model.onnx
    - assets/tokenizer.json
```

---

## Step 3: Initialize

```dart
import 'package:flutter/services.dart';
import 'package:path_provider/path_provider.dart';
import 'package:mobile_rag_engine/mobile_rag_engine.dart';

Future<void> initializeRAG() async {
  // 1. Initialize Rust library
  await RustLib.init();
  
  // 2. Copy assets to documents directory (first run only)
  final dir = await getApplicationDocumentsDirectory();
  await _copyAssetIfNeeded('assets/tokenizer.json', '${dir.path}/tokenizer.json');
  await _copyAssetIfNeeded('assets/model.onnx', '${dir.path}/model.onnx');
  
  // 3. Initialize tokenizer
  await initTokenizer(tokenizerPath: '${dir.path}/tokenizer.json');
  
  // 4. Initialize embedding service
  final modelBytes = await File('${dir.path}/model.onnx').readAsBytes();
  await EmbeddingService.init(modelBytes);
  
  // 5. Initialize RAG service
  final ragService = SourceRagService(dbPath: '${dir.path}/rag.db');
  await ragService.init();
}

Future<void> _copyAssetIfNeeded(String asset, String targetPath) async {
  final file = File(targetPath);
  if (!await file.exists()) {
    final data = await rootBundle.load(asset);
    await file.writeAsBytes(data.buffer.asUint8List());
  }
}
```

---

## Step 4: Add Documents

```dart
// Add text
await ragService.addSourceWithChunking(
  'Flutter is Google\'s UI toolkit for building beautiful apps.',
);

// Add PDF/DOCX
final bytes = await File('document.pdf').readAsBytes();
final text = await extractTextFromDocument(fileBytes: bytes);
await ragService.addSourceWithChunking(text);

// Rebuild index (important!)
await ragService.rebuildIndex();
```

---

## Step 5: Search

```dart
final result = await ragService.search(
  'What is Flutter?',
  topK: 5,
);

for (final chunk in result.chunks) {
  print('Score: ${chunk.score}');
  print('Content: ${chunk.content}');
}
```

---

## Complete Example

```dart
import 'dart:io';
import 'package:flutter/material.dart';
import 'package:flutter/services.dart';
import 'package:path_provider/path_provider.dart';
import 'package:mobile_rag_engine/mobile_rag_engine.dart';

void main() async {
  WidgetsFlutterBinding.ensureInitialized();
  
  // Initialize
  await RustLib.init();
  final dir = await getApplicationDocumentsDirectory();
  
  // Setup tokenizer & model
  await initTokenizer(tokenizerPath: '${dir.path}/tokenizer.json');
  final modelBytes = await File('${dir.path}/model.onnx').readAsBytes();
  await EmbeddingService.init(modelBytes);
  
  // Create RAG service
  final ragService = SourceRagService(dbPath: '${dir.path}/rag.db');
  await ragService.init();
  
  // Add a document
  await ragService.addSourceWithChunking(
    'Flutter is an open-source UI framework by Google.',
  );
  await ragService.rebuildIndex();
  
  // Search
  final result = await ragService.search('What is Flutter?', topK: 3);
  print('Found ${result.chunks.length} results');
  
  runApp(MyApp(ragService: ragService));
}
```

---

## Next Steps

- [Model Setup Guide](model_setup.md) - Model selection and deployment strategies
- [FAQ](faq.md) - Frequently asked questions
- [Troubleshooting](troubleshooting.md) - Problem solving guide
