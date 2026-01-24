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
  mobile_rag_engine:
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
import 'package:mobile_rag_engine/mobile_rag_engine.dart';

Future<void> initializeRAG() async {
  // Initialize in just 1 line!
  await MobileRag.initialize(
    tokenizerAsset: 'assets/tokenizer.json',
    modelAsset: 'assets/model.onnx',
  );
}
```

---

## Step 4: Add Documents

```dart
// Add text
await MobileRag.instance.addDocument(
  'Flutter is Google\'s UI toolkit for building beautiful apps.',
);

// Add PDF/DOCX
final bytes = await File('document.pdf').readAsBytes();
final text = await extractTextFromDocument(fileBytes: bytes.toList());
await MobileRag.instance.addDocument(text, filePath: 'document.pdf');

// Rebuild index (important!)
await MobileRag.instance.rebuildIndex();
```

---

## Step 5: Search

```dart
final result = await MobileRag.instance.search(
  'What is Flutter?',
  topK: 5,
  tokenBudget: 2000,
);

// LLM-ready context
print(result.context.text);

// Or iterate chunks
for (final chunk in result.chunks) {
  print('Score: ${chunk.similarity}');
  print('Content: ${chunk.content}');
}
```

---

## Complete Example

```dart
import 'package:flutter/material.dart';
import 'package:mobile_rag_engine/mobile_rag_engine.dart';

void main() async {
  WidgetsFlutterBinding.ensureInitialized();
  
  // Initialize
  await MobileRag.initialize(
    tokenizerAsset: 'assets/tokenizer.json',
    modelAsset: 'assets/model.onnx',
  );
  
  // Add a document
  await MobileRag.instance.addDocument(
    'Flutter is an open-source UI framework by Google.',
  );
  await MobileRag.instance.rebuildIndex();
  
  // Search
  final result = await MobileRag.instance.search('What is Flutter?', topK: 3);
  print('Found ${result.chunks.length} results');
  print('Context: ${result.context.text}');
  
  runApp(const MyApp());
}
```

---

## Advanced Usage

For fine-grained control, you can still use the low-level APIs:

```dart
// Manual initialization
await initTokenizer(tokenizerPath: 'path/to/tokenizer.json');
await EmbeddingService.init(modelBytes);
final ragService = SourceRagService(dbPath: 'path/to/rag.db');
await ragService.init();
```

---

## Next Steps

- [Model Setup Guide](model_setup.md) - Model selection and deployment strategies
- [FAQ](faq.md) - Frequently asked questions
- [Troubleshooting](troubleshooting.md) - Problem solving guide
