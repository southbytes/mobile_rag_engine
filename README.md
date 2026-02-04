# Mobile RAG Engine

![pub package](https://img.shields.io/pub/v/mobile_rag_engine)
![flutter](https://img.shields.io/badge/Flutter-3.9%2B-blue)
![rust](https://img.shields.io/badge/Core-Rust-orange)
![platform](https://img.shields.io/badge/Platform-iOS%20|%20Android%20|%20macOS-lightgrey)
[![License: MIT](https://img.shields.io/badge/License-MIT-yellow.svg)](https://opensource.org/licenses/MIT)

**Production-ready, fully local RAG (Retrieval-Augmented Generation) engine for Flutter.**

Powered by a **Rust core**, it delivers lightning-fast vector search and embedding generation directly on the device. No servers, no API costs, no latency.

---

## Why this package?

### No Rust Installation Required

**You do NOT need to install Rust, Cargo, or Android NDK.**

This package includes **pre-compiled binaries** for iOS, Android, and macOS. Just `pub add` and run.

### Performance

| Feature | Pure Dart | **Mobile RAG Engine (Rust)** |
|:---|:---:|:---:|
| **Tokenization** | Slow | **10x Faster** (HuggingFace tokenizers) |
| **Vector Search** | O(n) | **O(log n)** (HNSW Index) |
| **Memory Usage** | High | **Optimized** (Zero-copy FFI) |

### 100% Offline & Private

Data never leaves the user's device. Perfect for privacy-focused apps (journals, secure chats, enterprise tools).

---

## Features

- **Cross-Platform:** Works seamlessly on **iOS, Android, and macOS**
- **HNSW Vector Index:** Fast approximate nearest neighbor search (proven scale up to 10k+ docs)
- **Hybrid Search Ready:** Supports semantic search combined with exact matching
- **Markdown Structure-Aware Chunking:** Preserves headers, code blocks, and tables during chunking with header path inheritance
- **PDF/DOCX Text Extraction:** Built-in document parsing with smart dehyphenation and page number removal
- **Auto-Chunking:** Intelligent text splitting strategies included (Unicode-based semantic chunking)
- **Model Flexibility:** Use standard ONNX models (e.g., `bge-m3`, `all-MiniLM-L6-v2`)
- **Independent Source Search:** Filter by specific documents (Exact Scan guarantees recall)
- **Data Management:** Delete individual sources or clear all data easily

---

## Benchmark Results

<p align="center">
  <img src="https://raw.githubusercontent.com/dev07060/mobile_rag_engine/main/assets/readme-sources/ios_benchmark_result.png" width="280" alt="iOS Benchmark">
  <img src="https://raw.githubusercontent.com/dev07060/mobile_rag_engine/main/assets/readme-sources/android_benchmark_result.png" width="280" alt="Android Benchmark">
</p>

---

## Requirements

| Platform | Minimum Version |
|:---------|:----------------|
| **iOS** | 13.0+ |
| **Android** | API 21+ (Android 5.0 Lollipop) |
| **macOS** | 10.15+ (Catalina) |

> **ONNX Runtime** is bundled automatically via the [`onnxruntime`](https://pub.dev/packages/onnxruntime) plugin. No additional native setup required.

---

## Installation

### 1. Add the dependency

```yaml
dependencies:
  mobile_rag_engine:
```

### 2. Download Model Files

```bash
# Create assets folder
mkdir -p assets && cd assets

# Download BGE-m3 model (INT8 quantized, multilingual)
curl -L -o model.onnx "https://huggingface.co/Teradata/bge-m3/resolve/main/onnx/model_int8.onnx"
curl -L -o tokenizer.json "https://huggingface.co/BAAI/bge-m3/resolve/main/tokenizer.json"
```

> See [Model Setup Guide](https://github.com/dev07060/mobile_rag_engine/blob/main/docs/guides/model_setup.md) for alternative models and production deployment strategies.

---

## Quick Start

Initialize the engine once in your `main()` function:

```dart
import 'package:mobile_rag_engine/mobile_rag_engine.dart';

void main() async {
  WidgetsFlutterBinding.ensureInitialized();
  
  // 1. Initialize (Just 1 step!)
  await MobileRag.initialize(
    tokenizerAsset: 'assets/tokenizer.json',
    modelAsset: 'assets/model.onnx',
  );

  runApp(const MyApp());
}
```

Then use it anywhere in your app:

```dart
class MySearchScreen extends StatelessWidget {
  Future<void> _search() async {
    // 2. Add Documents (auto-chunked & embedded)
    await MobileRag.instance.addDocument(
      'Flutter is a UI toolkit for building apps.',
    );
    await MobileRag.instance.rebuildIndex();
  
    // 3. Search with LLM-ready context
    final result = await MobileRag.instance.search(
      'What is Flutter?', 
      tokenBudget: 2000,
    );
    
    print(result.context.text); // Ready to send to LLM
  }
}
```

> **Advanced Usage:** For fine-grained control, you can still use the low-level APIs (`initTokenizer`, `EmbeddingService`, `SourceRagService`) directly. See the [API Reference](https://pub.dev/documentation/mobile_rag_engine/latest/).

---

## PDF/DOCX Import

Extract text from documents and add to RAG:

```dart
import 'dart:io';
import 'package:file_picker/file_picker.dart';
import 'package:mobile_rag_engine/mobile_rag_engine.dart';

Future<void> importDocument() async {
  // Pick file
  final result = await FilePicker.platform.pickFiles(
    type: FileType.custom,
    allowedExtensions: ['pdf', 'docx'],
  );
  if (result == null) return;

  // Extract text (handles hyphenation, page numbers automatically)
  final bytes = await File(result.files.single.path!).readAsBytes();
  final text = await extractTextFromDocument(fileBytes: bytes.toList());

  // Add to RAG with auto-chunking
  await MobileRag.instance.addDocument(text, filePath: result.files.single.path);
  await MobileRag.instance.rebuildIndex();
}
```

> **Note:** `file_picker` is optional. You can obtain file bytes from any source (network, camera, etc.) and pass to `extractTextFromDocument()`.

---

## Benchmarks

**Rust-powered components** (M3 Pro macOS):

| Operation | Time | Notes |
|:---|:---:|:---|
| **Tokenization** (234 chars) | **0.04ms** | HuggingFace `tokenizers` crate |
| **HNSW Search** (100 docs) | **0.3ms** | `instant-distance` (O(log n)) |

> These are the components where Rust provides **10-100x speedup** over pure Dart implementations.
>
> Embedding generation uses ONNX Runtime (platform-dependent, typically 25-100ms per text).

---

## Architecture

This package bridges the best of two worlds: **Flutter for UI** and **Rust for heavy lifting**.

<p align="center">
  <img src="https://raw.githubusercontent.com/dev07060/mobile_rag_engine/main/assets/readme-sources/rag_engine_architecture.png" width="600" alt="Architecture">
</p>

| Component | Technology |
|:---|:---|
| **Embedding** | ONNX Runtime with quantized models (INT8) |
| **Storage** | SQLite for metadata + memory-mapped vector index |
| **Search** | `instant-distance` (HNSW) for low-latency retrieval |
| **Tokenization** | HuggingFace `tokenizers` crate |

---

## Model Options

| Model | Dimensions | Size | Max Tokens | Languages |
|:------|:----------:|:----:|:----------:|:----------|
| [Teradata/bge-m3](https://huggingface.co/Teradata/bge-m3) (INT8) | **1024** | ~542 MB | 8,194 | 100+ (multilingual) |
| [all-MiniLM-L6-v2](https://huggingface.co/sentence-transformers/all-MiniLM-L6-v2) | **384** | ~25 MB | 256 | English only |

> **Important:** The embedding dimension must be consistent across all documents. Switching models requires re-embedding your entire corpus.

**Custom Models:** Export any Sentence Transformer to ONNX:
```bash
pip install optimum[exporters]
optimum-cli export onnx --model sentence-transformers/YOUR_MODEL ./output
```

See [Model Setup Guide](https://github.com/dev07060/mobile_rag_engine/blob/main/docs/guides/model_setup.md) for deployment strategies and troubleshooting.

---

## Documentation

| Guide | Description |
|:------|:------------|
| [Quick Start](https://github.com/dev07060/mobile_rag_engine/blob/main/docs/guides/quick_start.md) | Get started in 5 minutes |
| [Model Setup](https://github.com/dev07060/mobile_rag_engine/blob/main/docs/guides/model_setup.md) | Model selection, download, deployment strategies |
| [FAQ](https://github.com/dev07060/mobile_rag_engine/blob/main/docs/guides/faq.md) | Frequently asked questions |
| [Troubleshooting](https://github.com/dev07060/mobile_rag_engine/blob/main/docs/guides/troubleshooting.md) | Problem solving guide |

---

## Sample App

Check out the example application using this package. This desktop app demonstrates **full RAG pipeline integration with an LLM (Gemma 2B)** running locally on-device.

[mobile-ondevice-rag-desktop](https://github.com/dev07060/mobile-ondevice-rag-desktop)

<p align="center">
  <img src="https://raw.githubusercontent.com/dev07060/mobile_rag_engine/main/assets/readme-sources/sample_app.png" width="600" alt="Sample App Screenshot">
</p>

---

## Contributing

Bug reports, feature requests, and PRs are all welcome!

## License

This project is licensed under the [MIT License](LICENSE).
