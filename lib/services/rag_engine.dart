/// Unified RAG engine with simplified initialization.
///
/// This class combines tokenizer, embedding model, and RAG service
/// initialization into a single `initialize()` call.
///
/// ## Quick Start
///
/// ```dart
/// import 'package:mobile_rag_engine/mobile_rag_engine.dart';
///
/// // NOTE: For most apps, use [MobileRag] singleton instead.
/// // It wraps this class and handles global access.
///
/// // If you need a standalone engine instance:
/// final rag = await RagEngine.initialize(
///   config: RagConfig.fromAssets(
///     tokenizerAsset: 'assets/tokenizer.json',
///     modelAsset: 'assets/model.onnx',
///   ),
/// );
///
/// // Use the engine
/// await rag.addDocument('Your document text here');
/// await rag.rebuildIndex();
/// final result = await rag.search('query', tokenBudget: 2000);
/// ```
library;

import 'dart:io';

import 'package:flutter/services.dart';
import 'package:flutter/foundation.dart';
import 'package:onnxruntime/onnxruntime.dart';
import 'package:path_provider/path_provider.dart';

import '../src/rust/api/tokenizer.dart';
import '../src/rust/api/source_rag.dart' show SourceStats, SourceEntry;
import '../src/rust/api/db_pool.dart';
import 'embedding_service.dart';
import 'rag_config.dart';
import 'source_rag_service.dart';
import 'context_builder.dart';
import '../src/rust/api/hybrid_search.dart' as hybrid;
import 'package:flutter_rust_bridge/flutter_rust_bridge_for_generated.dart';
import '../src/rust/frb_generated.dart';

/// Unified RAG engine with simplified initialization.
///
/// Wraps [SourceRagService] with automatic dependency initialization.
class RagEngine {
  static bool _isRustInitialized = false;

  /// Ensures RustLib is initialized (safe to call multiple times).
  static Future<void> _ensureRustInitialized() async {
    if (_isRustInitialized) return;

    if (Platform.isMacOS) {
      await RustLib.init(
        externalLibrary: ExternalLibrary.process(iKnowHowToUseIt: true),
      );
    } else {
      await RustLib.init();
    }
    _isRustInitialized = true;
  }

  final SourceRagService _ragService;

  /// Path to the SQLite database.
  final String dbPath;

  /// Vocabulary size of the loaded tokenizer.
  final int vocabSize;

  RagEngine._({
    required SourceRagService ragService,
    required this.dbPath,
    required this.vocabSize,
  }) : _ragService = ragService;

  /// Initialize RagEngine with all dependencies.
  ///
  /// This method handles:
  /// 1. Copying tokenizer asset to documents directory
  /// 2. Initializing the tokenizer
  /// 3. Loading the ONNX embedding model
  /// 4. Initializing the RAG database
  ///
  /// [config] - Configuration containing asset paths and options.
  /// [onProgress] - Optional callback for initialization status updates.
  ///
  /// Example:
  /// ```dart
  /// final rag = await RagEngine.initialize(
  ///   config: RagConfig.fromAssets(
  ///     tokenizerAsset: 'assets/tokenizer.json',
  ///     modelAsset: 'assets/model.onnx',
  ///   ),
  ///   onProgress: (status) => setState(() => _status = status),
  /// );
  /// ```
  static Future<RagEngine> initialize({
    required RagConfig config,
    void Function(String status)? onProgress,
  }) async {
    // 0. Auto-initialize Rust library (safe to call multiple times)
    await _ensureRustInitialized();

    // 1. Get app documents directory
    final dir = await getApplicationDocumentsDirectory();
    final dbPath = "${dir.path}/${config.databaseName ?? 'rag.sqlite'}";
    final tokenizerPath = "${dir.path}/tokenizer.json";
    final modelPath = "${dir.path}/${config.modelAsset.split('/').last}";

    // 2. Copy and initialize tokenizer
    onProgress?.call('Initializing tokenizer...');
    await _copyAssetToFile(config.tokenizerAsset, tokenizerPath);
    await initTokenizer(tokenizerPath: tokenizerPath);
    final vocabSize = getVocabSize();

    // 3. Prepare ONNX embedding model (Copy logic)
    onProgress?.call('Preparing embedding model...');
    // Copy model asset to file (optimized for memory)
    await _copyAssetToFile(config.modelAsset, modelPath);

    // Configure session options if thread limit is requested
    OrtSessionOptions? sessionOptions;

    // Default to half the cores if not specified to prevent full CPU usage
    // Calculate threads based on configuration
    int threads;
    final totalCores = Platform.numberOfProcessors;

    if (config.threadLevel != null) {
      // 1. Thread Level (Percentage based)
      switch (config.threadLevel!) {
        case ThreadUseLevel.low:
          threads = (totalCores * 0.2).ceil();
          break;
        case ThreadUseLevel.medium:
          threads = (totalCores * 0.4).ceil();
          break;
        case ThreadUseLevel.high:
          threads = (totalCores * 0.8).ceil();
          break;
      }
    } else if (config.embeddingIntraOpNumThreads != null) {
      // 2. Manual Count
      threads = config.embeddingIntraOpNumThreads!;
    } else {
      // 3. Priority: Default (50% safe fallback)
      threads = (totalCores > 1 ? (totalCores / 2).ceil() : 1);
    }

    // Ensure at least 1 thread
    if (threads < 1) threads = 1;

    // Apply thread configuration
    sessionOptions = OrtSessionOptions();
    try {
      sessionOptions.setIntraOpNumThreads(threads);
      debugPrint(
        '[RagEngine] Configured ONNX embedding threads: $threads (Total Cores: $totalCores)',
      );
    } catch (e) {
      debugPrint('[RagEngine] Warning: Failed to set intra-op num threads: $e');
    }

    // Init EmbeddingService with file path
    onProgress?.call('Loading embedding model...');
    await EmbeddingService.init(modelPath: modelPath, options: sessionOptions);

    // 4. Initialize database connection pool
    onProgress?.call('Initializing connection pool...');
    await initDbPool(dbPath: dbPath, maxSize: 4);

    // 5. Initialize RAG service
    onProgress?.call('Initializing database...');
    final ragService = SourceRagService(
      dbPath: dbPath,
      maxChunkChars: config.maxChunkChars,
      overlapChars: config.overlapChars,
    );
    await ragService.init();

    onProgress?.call('Ready!');
    return RagEngine._(
      ragService: ragService,
      dbPath: dbPath,
      vocabSize: vocabSize,
    );
  }

  /// Copy asset file to filesystem if it doesn't exist.
  static Future<void> _copyAssetToFile(
    String assetPath,
    String filePath,
  ) async {
    final file = File(filePath);
    if (!await file.exists()) {
      final data = await rootBundle.load(assetPath);
      await file.writeAsBytes(data.buffer.asUint8List());
    }
  }

  // ─────────────────────────────────────────────────────────────────────────
  // Delegated methods from SourceRagService
  // ─────────────────────────────────────────────────────────────────────────

  /// Add a document with automatic chunking and embedding.
  ///
  /// The document is:
  /// 1. Split into chunks based on file type (auto-detected from [filePath])
  /// 2. Each chunk is embedded using the loaded model
  /// 3. Source and chunks are stored in the database
  ///
  /// Remember to call [rebuildIndex] after adding documents for optimal
  /// search performance.
  Future<SourceAddResult> addDocument(
    String content, {
    String? metadata,
    String? name,
    String? filePath,
    ChunkingStrategy? strategy,
    Duration? chunkDelay,
    void Function(int done, int total)? onProgress,
  }) => _ragService.addSourceWithChunking(
    content,
    metadata: metadata,
    name: name,
    filePath: filePath,
    strategy: strategy,
    chunkDelay: chunkDelay,
    onProgress: onProgress,
  );

  /// Search for relevant chunks and assemble context for LLM.
  ///
  /// [query] - The search query text.
  /// [topK] - Number of top results to return (default: 10).
  /// [tokenBudget] - Maximum tokens for assembled context (default: 2000).
  /// [strategy] - Context assembly strategy (default: relevanceFirst).
  /// [adjacentChunks] - Include N chunks before/after matches (default: 0).
  /// [singleSourceMode] - Only include chunks from most relevant source.
  Future<RagSearchResult> search(
    String query, {
    int topK = 10,
    int tokenBudget = 2000,
    ContextStrategy strategy = ContextStrategy.relevanceFirst,
    int adjacentChunks = 0,
    bool singleSourceMode = false,
    List<int>? sourceIds,
  }) => _ragService.search(
    query,
    topK: topK,
    tokenBudget: tokenBudget,
    strategy: strategy,
    adjacentChunks: adjacentChunks,
    singleSourceMode: singleSourceMode,
    sourceIds: sourceIds,
  );

  /// Hybrid search combining vector and keyword (BM25) search.
  ///
  /// Uses Reciprocal Rank Fusion (RRF) to combine semantic and keyword results.
  Future<List<hybrid.HybridSearchResult>> searchHybrid(
    String query, {
    int topK = 10,
    double vectorWeight = 0.5,
    double bm25Weight = 0.5,
    List<int>? sourceIds,
  }) => _ragService.searchHybrid(
    query,
    topK: topK,
    vectorWeight: vectorWeight,
    bm25Weight: bm25Weight,
    sourceIds: sourceIds,
  );

  /// Hybrid search with context assembly for LLM.
  ///
  /// [adjacentChunks] - Include N chunks before/after matches (default: 0).
  /// [singleSourceMode] - Only include chunks from most relevant source.
  Future<RagSearchResult> searchHybridWithContext(
    String query, {
    int topK = 10,
    int tokenBudget = 2000,
    ContextStrategy strategy = ContextStrategy.relevanceFirst,
    double vectorWeight = 0.5,
    double bm25Weight = 0.5,
    List<int>? sourceIds,
    int adjacentChunks = 0,
    bool singleSourceMode = false,
  }) => _ragService.searchHybridWithContext(
    query,
    topK: topK,
    tokenBudget: tokenBudget,
    strategy: strategy,
    vectorWeight: vectorWeight,
    bm25Weight: bm25Weight,
    sourceIds: sourceIds,
    adjacentChunks: adjacentChunks,
    singleSourceMode: singleSourceMode,
  );

  /// Rebuild the HNSW index after adding documents.
  ///
  /// Call this after adding one or more documents for optimal search
  /// performance. The index enables fast approximate nearest neighbor search.
  Future<void> rebuildIndex() => _ragService.rebuildIndex();

  /// Try to load a cached HNSW index from disk.
  ///
  /// Returns true if a previously built index exists.
  Future<bool> tryLoadCachedIndex() => _ragService.tryLoadCachedIndex();

  /// Save the HNSW index marker to disk.
  Future<void> saveIndex() => _ragService.saveIndex();

  /// Get statistics about stored sources and chunks.
  Future<SourceStats> getStats() => _ragService.getStats();

  /// Remove a source and all its chunks from the database.
  Future<void> removeSource(int sourceId) => _ragService.removeSource(sourceId);

  /// Get a list of all stored sources.
  Future<List<SourceEntry>> listSources() => _ragService.listSources();

  /// Format search results as an LLM prompt.
  String formatPrompt(String query, RagSearchResult result) =>
      _ragService.formatPrompt(query, result);

  /// Regenerate embeddings for all existing chunks.
  ///
  /// Use this when the embedding model has been updated.
  Future<void> regenerateAllEmbeddings({
    void Function(int done, int total)? onProgress,
  }) => _ragService.regenerateAllEmbeddings(onProgress: onProgress);

  /// Clear all data (database and index files) and reset the engine.
  ///
  /// This is a destructive operation that:
  /// 1. Closes the database connection
  /// 2. Deletes the SQLite database file
  /// 3. Deletes the HNSW index file
  /// 4. Re-initializes the database and service
  Future<void> clearAllData() async {
    debugPrint('[RagEngine] clearAllData: Starting...');
    // 1. Close DB pool
    debugPrint('[RagEngine] clearAllData: Closing DB pool...');
    await closeDbPool();
    debugPrint('[RagEngine] clearAllData: DB pool closed.');

    // 2. Delete DB file
    final dbFile = File(dbPath);
    if (await dbFile.exists()) {
      debugPrint('[RagEngine] clearAllData: Deleting DB file at $dbPath...');
      await dbFile.delete();
      debugPrint('[RagEngine] clearAllData: DB file deleted.');
    } else {
      debugPrint('[RagEngine] clearAllData: DB file not found.');
    }

    // 3. Delete HNSW index file (and lock file if exists)
    final indexFile = File('${dbPath.replaceAll('.db', '')}_hnsw');
    if (await indexFile.exists()) {
      debugPrint('[RagEngine] clearAllData: Deleting index file...');
      await indexFile.delete();
    }
    // Also try checking for .pbin if naming convention varies
    final indexFileAlt = File('${dbPath.replaceAll('.db', '')}_hnsw.pbin');
    if (await indexFileAlt.exists()) {
      debugPrint('[RagEngine] clearAllData: Deleting alt index file...');
      await indexFileAlt.delete();
    }

    // 4. Re-initialize DB pool
    debugPrint('[RagEngine] clearAllData: Re-initializing DB pool...');
    await initDbPool(dbPath: dbPath, maxSize: 4);
    debugPrint('[RagEngine] clearAllData: DB pool initialized.');

    // 5. Re-initialize service
    debugPrint('[RagEngine] clearAllData: Re-initializing service...');
    await _ragService.init();
    debugPrint('[RagEngine] clearAllData: Service initialized. Done.');
  }

  /// Access to the underlying [SourceRagService] for advanced operations.
  SourceRagService get service => _ragService;

  /// Dispose of resources.
  ///
  /// Call this when done using the engine to release resources.
  static Future<void> dispose() async {
    EmbeddingService.dispose();
    await closeDbPool();
  }
}
