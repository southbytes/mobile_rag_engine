/// Mobile RAG Engine
///
/// A high-performance, on-device RAG (Retrieval-Augmented Generation) engine
/// for Flutter. Run semantic search completely offline on iOS and Android.
///
/// ## Quick Start (Recommended)
///
/// ```dart
/// import 'package:mobile_rag_engine/mobile_rag_engine.dart';
///
/// // Initialize (just 3 lines!)
/// await RustLib.init();
/// final rag = await RagEngine.initialize(
///   config: RagConfig.fromAssets(
///     tokenizerAsset: 'assets/tokenizer.json',
///     modelAsset: 'assets/model.onnx',
///   ),
/// );
///
/// // Add documents
/// await rag.addDocument('Your long document text here');
/// await rag.rebuildIndex();
///
/// // Search with LLM context assembly
/// final result = await rag.search('query', tokenBudget: 2000);
/// final prompt = rag.formatPrompt('query', result);
/// // Send prompt to LLM
/// ```
///
/// ## Advanced Usage (Low-Level API)
///
/// For fine-grained control, use the individual services directly:
///
/// ```dart
/// await initTokenizer(tokenizerPath: 'path/to/tokenizer.json');
/// await EmbeddingService.init(modelBytes);
/// final rag = SourceRagService(dbPath: 'path/to/rag.db');
/// await rag.init();
/// ```
library;

// High-level unified API (recommended)
export 'services/rag_config.dart';
export 'services/rag_engine.dart';

// Core RAG functions (simple API)
export 'src/rust/api/simple_rag.dart';

// Source-based RAG functions (chunked API)
export 'src/rust/api/source_rag.dart';

// Tokenizer functions
export 'src/rust/api/tokenizer.dart';

// Services
export 'services/embedding_service.dart';
export 'src/rust/api/semantic_chunker.dart'; // Rust-based semantic chunking
export 'src/rust/api/hybrid_search.dart'; // Hybrid search (Vector + BM25)
export 'src/rust/api/bm25_search.dart'; // BM25 keyword search
export 'services/context_builder.dart';
export 'services/source_rag_service.dart';
export 'services/prompt_compressor.dart'; // REFRAG-style prompt compression
export 'src/rust/api/compression_utils.dart'; // Low-level compression utilities
export 'src/rust/api/document_parser.dart'; // PDF/DOCX text extraction

// Rust library initialization
export 'src/rust/frb_generated.dart' show RustLib;
