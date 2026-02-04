/// High-level RAG service for managing sources and chunks.
///
/// This service provides a convenient API that combines:
/// - Rust semantic chunking for document splitting (Unicode sentence/word boundaries)
/// - EmbeddingService for vector generation
/// - Rust source_rag APIs for storage and search
/// - ContextBuilder for LLM context assembly
/// - Hybrid search combining vector and BM25 keyword search
library;

import 'dart:typed_data';
import '../src/rust/api/error.dart';
import '../src/rust/api/source_rag.dart' as rust_rag;
import '../src/rust/api/source_rag.dart'
    show
        SourceStats,
        ChunkSearchResult,
        // AddSourceResult, // Unused
        ChunkData,
        SourceEntry,
        initSourceDb,
        addSource,
        addChunks,
        // listSources, // Removed to avoid collision, using rust_rag.listSources
        searchChunks, // Added back
        deleteSource,
        getAdjacentChunks,
        getSource,
        getSourceStats,
        getAllChunkIdsAndContents,
        updateChunkEmbedding,
        rebuildChunkHnswIndex,
        rebuildChunkBm25Index;
import '../src/rust/api/semantic_chunker.dart';
import '../src/rust/api/hybrid_search.dart' as hybrid;
import '../src/rust/api/hnsw_index.dart' as hnsw;
import 'context_builder.dart';
import 'embedding_service.dart';
import '../utils/error_utils.dart';
import '../src/rust/api/logger.dart';
import 'package:flutter_rust_bridge/flutter_rust_bridge_for_generated.dart'
    as frb;
import 'dart:async';

extension RagErrorMessage on RagError {
  String get message => when(
    databaseError: (msg) => msg,
    ioError: (msg) => msg,
    modelLoadError: (msg) => msg,
    invalidInput: (msg) => msg,
    internalError: (msg) => msg,
    unknown: (msg) => msg,
  );
}

/// Chunking strategy for document processing.
enum ChunkingStrategy {
  /// Default paragraph-based chunking (recursive character splitting)
  recursive,

  /// Markdown-aware chunking (preserves headers, code blocks, tables)
  markdown,
}

/// Result of adding a source document with automatic chunking.
class SourceAddResult {
  final int sourceId;
  final bool isDuplicate;
  final int chunkCount;
  final String message;

  SourceAddResult({
    required this.sourceId,
    required this.isDuplicate,
    required this.chunkCount,
    required this.message,
  });
}

/// Search result with assembled context.
class RagSearchResult {
  /// Individual chunk results.
  final List<ChunkSearchResult> chunks;

  /// Assembled context for LLM.
  final AssembledContext context;

  RagSearchResult({required this.chunks, required this.context});
}

/// High-level service for source-based RAG operations.
class SourceRagService {
  final String dbPath;

  /// Maximum characters per chunk (default: 500)
  final int maxChunkChars;

  /// Overlap characters between chunks for context continuity
  final int overlapChars;

  SourceRagService({
    required this.dbPath,
    this.maxChunkChars = 500,
    this.overlapChars = 50,
  });

  /// Detect the appropriate chunking strategy based on file extension.
  static ChunkingStrategy detectChunkingStrategy(String? filePath) {
    if (filePath == null) return ChunkingStrategy.recursive;

    final ext = filePath.split('.').lastOrNull?.toLowerCase() ?? '';
    return switch (ext) {
      'md' || 'markdown' => ChunkingStrategy.markdown,
      _ => ChunkingStrategy.recursive,
    };
  }

  StreamSubscription<String>? _logSubscription;

  /// Helper to convert List<int> to the specific Int64List type required by FRB.
  /// We do this manually because frb.Int64List.fromList() might return a native
  /// dart:typed_data Int64List which can cause type mismatch errors if FRB
  /// expects its own wrapped type.
  frb.Int64List _toInt64List(List<int> list) {
    // Try using the constructor directly
    final result = frb.Int64List(list.length);
    for (var i = 0; i < list.length; i++) {
      result[i] = list[i];
    }
    return result;
  }

  /// Get the HNSW index path (derived from dbPath)
  String get _indexPath => dbPath.replaceAll('.db', '_hnsw');

  /// Initialize the source database.
  Future<void> init() async {
    try {
      // 1. Initialize Rust logger (idempotent)
      await initLogger();

      // 2. Close any existing log stream (both Dart and Rust side)
      await _cleanupLogStream();

      // 3. Start fresh log stream (logs already have [LEVEL] prefix)
      _logSubscription = initLogStream().listen((log) {
        print(log);
      });

      // 4. Initialize DB
      await initSourceDb();

      // 5. Rebuild indexes for existing chunks (both are in-memory only)
      // This is critical because neither HNSW nor BM25 indexes are persisted to disk
      await rebuildChunkHnswIndex(); // Vector search
      await rebuildChunkBm25Index(); // Keyword search
    } on RagError catch (e) {
      // Smart Error Handling integration
      print(
        '[SmartError] ${e.userFriendlyMessage} (Tech: ${e.technicalMessage})',
      );
      rethrow;
    }
  }

  /// Clean up log stream resources (both Dart subscription and Rust sink).
  Future<void> _cleanupLogStream() async {
    await _logSubscription?.cancel();
    _logSubscription = null;
    try {
      closeLogStream(); // Close Rust-side sink
    } catch (_) {
      // Ignore errors during cleanup
    }
  }

  /// Dispose resources held by this service.
  /// Call this when the service is no longer needed.
  Future<void> dispose() async {
    await _cleanupLogStream();
  }

  /// Try to load cached HNSW index.
  ///
  /// Returns true if a previously built index exists (marker found).
  /// The actual index data is rebuilt from DB when [rebuildIndex] is called.
  ///
  /// Usage pattern:
  /// ```dart
  /// await service.init();
  /// final hasCached = await service.tryLoadCachedIndex();
  /// if (!hasCached || forceRebuild) {
  ///   await service.rebuildIndex();
  /// }
  /// ```
  Future<bool> tryLoadCachedIndex() async {
    final exists = await hnsw.loadHnswIndex(basePath: _indexPath);
    return exists;
  }

  /// Save HNSW index marker to disk.
  ///
  /// Call this after [rebuildIndex] to mark that an index was built.
  /// This allows faster startup detection on next app launch.
  Future<void> saveIndex() async {
    await hnsw.saveHnswIndex(basePath: _indexPath);
  }

  /// Add a source document with automatic chunking and embedding.
  ///
  /// The document is:
  /// 1. Split into chunks based on file type (auto-detected from [filePath])
  /// 2. Each chunk is embedded
  /// 3. Source and chunks are stored in DB
  ///
  /// If [filePath] is provided, chunking strategy is auto-detected:
  /// - `.md`, `.markdown` → Markdown-aware chunking (preserves headers, code blocks)
  /// - Other files → Default recursive chunking
  Future<SourceAddResult> addSourceWithChunking(
    String content, {
    String? metadata,
    String? name,
    String? filePath,
    ChunkingStrategy? strategy,
    void Function(int done, int total)? onProgress,
  }) async {
    // 1. Add source document
    late SourceAddResult sourceResult;
    try {
      final res = await addSource(
        content: content,
        metadata: metadata,
        name: name,
      );
      sourceResult = SourceAddResult(
        sourceId: res.sourceId.toInt(),
        isDuplicate: res.isDuplicate,
        chunkCount: res.chunkCount,
        message: res.message,
      );
    } on RagError catch (e) {
      e.when(
        databaseError: (msg) =>
            print('[SmartError] Storage write failed: $msg'),
        ioError: (msg) => print('[SmartError] IO error adding source: $msg'),
        modelLoadError: (_) {},
        invalidInput: (msg) =>
            print('[SmartError] Invalid source content: $msg'),
        internalError: (msg) =>
            print('[SmartError] Internal error adding source: $msg'),
        unknown: (msg) =>
            print('[SmartError] Unknown error adding source: $msg'),
      );
      rethrow;
    }

    if (sourceResult.isDuplicate) {
      return SourceAddResult(
        sourceId: sourceResult.sourceId.toInt(),
        isDuplicate: true,
        chunkCount: 0,
        message: sourceResult.message,
      );
    }

    // 2. Determine chunking strategy
    final effectiveStrategy = strategy ?? detectChunkingStrategy(filePath);

    // 3. Split content based on strategy
    final chunkDataList = <ChunkData>[];

    if (effectiveStrategy == ChunkingStrategy.markdown) {
      // Markdown-aware chunking
      final chunks = markdownChunk(text: content, maxChars: maxChunkChars);

      for (var i = 0; i < chunks.length; i++) {
        onProgress?.call(i, chunks.length);
        final chunk = chunks[i];
        final embedding = await EmbeddingService.embed(chunk.content);

        // Include header path in chunk type for context
        final enrichedType = chunk.headerPath.isNotEmpty
            ? '${chunk.chunkType}|${chunk.headerPath}'
            : chunk.chunkType;

        chunkDataList.add(
          ChunkData(
            content: chunk.content,
            chunkIndex: chunk.index,
            startPos: chunk.startPos,
            endPos: chunk.endPos,
            chunkType: enrichedType,
            embedding: Float32List.fromList(embedding),
          ),
        );
      }
      onProgress?.call(chunks.length, chunks.length);
    } else {
      // Default recursive chunking
      final chunks = semanticChunkWithOverlap(
        text: content,
        maxChars: maxChunkChars,
        overlapChars: overlapChars,
      );

      for (var i = 0; i < chunks.length; i++) {
        onProgress?.call(i, chunks.length);
        final chunk = chunks[i];
        final embedding = await EmbeddingService.embed(chunk.content);

        chunkDataList.add(
          ChunkData(
            content: chunk.content,
            chunkIndex: chunk.index,
            startPos: chunk.startPos,
            endPos: chunk.endPos,
            chunkType: chunk.chunkType,
            embedding: Float32List.fromList(embedding),
          ),
        );
      }
      onProgress?.call(chunks.length, chunks.length);
    }

    if (chunkDataList.isEmpty) {
      return SourceAddResult(
        sourceId: sourceResult.sourceId.toInt(),
        isDuplicate: false,
        chunkCount: 0,
        message: 'No chunks created',
      );
    }

    // 4. Store chunks
    await addChunks(sourceId: sourceResult.sourceId, chunks: chunkDataList);

    return SourceAddResult(
      sourceId: sourceResult.sourceId.toInt(),
      isDuplicate: false,
      chunkCount: chunkDataList.length,
      message:
          'Added ${chunkDataList.length} chunks (${effectiveStrategy.name})',
    );
  }

  /// Rebuild the HNSW and BM25 indexes after adding sources.
  Future<void> rebuildIndex() async {
    try {
      // Rebuild HNSW for vector search
      await rebuildChunkHnswIndex();
      // Rebuild BM25 for keyword search (critical for hybrid search!)
      await rebuildChunkBm25Index();
    } on RagError catch (e) {
      e.when(
        databaseError: (msg) =>
            print('[SmartError] DB error rebuilding index: $msg'),
        ioError: (msg) => print('[SmartError] IO error rebuilding index: $msg'),
        modelLoadError: (_) {},
        invalidInput: (_) {},
        internalError: (msg) =>
            print('[SmartError] Internal error rebuilding indexes: $msg'),
        unknown: (_) {},
      );
      rethrow;
    }
  }

  /// Regenerate embeddings for all existing chunks using the current model.
  /// This is needed when the embedding model or tokenizer has been updated.
  Future<void> regenerateAllEmbeddings({
    void Function(int done, int total)? onProgress,
  }) async {
    // 1. Get all chunk IDs and contents
    final chunks = await getAllChunkIdsAndContents();
    print(
      '[regenerateAllEmbeddings] Found ${chunks.length} chunks to re-embed',
    );

    // 2. Re-embed each chunk
    for (var i = 0; i < chunks.length; i++) {
      final chunk = chunks[i];
      final embedding = await EmbeddingService.embed(chunk.content);

      // 3. Update in DB
      await updateChunkEmbedding(
        chunkId: chunk.chunkId,
        embedding: Float32List.fromList(embedding),
      );

      onProgress?.call(i + 1, chunks.length);

      // Log progress every 50 chunks
      if ((i + 1) % 50 == 0) {
        print('[regenerateAllEmbeddings] Progress: ${i + 1}/${chunks.length}');
      }
    }

    print('[regenerateAllEmbeddings] Completed. Rebuilding HNSW index...');

    // 4. Rebuild HNSW index
    await rebuildIndex();

    print('[regenerateAllEmbeddings] Done!');
  }

  /// Search for relevant chunks and assemble context for LLM.
  ///
  /// [adjacentChunks] - Number of adjacent chunks to include before/after each
  /// matched chunk (default: 0). Setting this to 1 will include the chunk
  /// before and after each matched chunk, helping with long articles.
  /// [singleSourceMode] - If true, only include chunks from the most relevant source.
  Future<RagSearchResult> search(
    String query, {
    int topK = 10,
    int tokenBudget = 2000,
    ContextStrategy strategy = ContextStrategy.relevanceFirst,
    int adjacentChunks = 0,
    bool singleSourceMode = false,
    List<int>? sourceIds,
  }) async {
    // 1. Generate query embedding
    final queryEmbedding = await EmbeddingService.embed(query);

    // DEBUG: Log embedding stats
    final embNorm = queryEmbedding.fold<double>(0, (sum, v) => sum + v * v);
    print('[DEBUG] Query: "$query"');
    print(
      '[DEBUG] Embedding norm: ${embNorm.toStringAsFixed(4)}, dims: ${queryEmbedding.length}',
    );
    print(
      '[DEBUG] First 5 values: ${queryEmbedding.take(5).map((v) => v.toStringAsFixed(4)).toList()}',
    );

    // 2. Search chunks
    late List<ChunkSearchResult> chunks;
    try {
      if (sourceIds != null && sourceIds.isNotEmpty) {
        // Use hybrid search with filter for vector search capability is limited in filters
        // For strictly vector search with filters, we'd need HNSW filtering which is more complex.
        // For now, we'll use hybrid search with 1.0 vector weight if sourceIds are provided.
        final hybridResults = await searchHybrid(
          query,
          topK: topK,
          vectorWeight: 1.0,
          bm25Weight: 0.0,
          sourceIds: sourceIds,
        );
        chunks = hybridResults
            .map(
              (r) => ChunkSearchResult(
                chunkId: r.docId,
                sourceId: r.sourceId,
                content: r.content,
                chunkIndex: 0, // Lost in hybrid search
                chunkType: 'general',
                similarity: r.score,
                metadata: r.metadata,
              ),
            )
            .toList();
      } else {
        chunks = await searchChunks(queryEmbedding: queryEmbedding, topK: topK);
      }
    } on RagError catch (e) {
      e.when(
        databaseError: (msg) =>
            print('[SmartError] Search failed (database): $msg'),
        ioError: (msg) => print('[SmartError] Search IO error: $msg'),
        modelLoadError: (_) {},
        invalidInput: (msg) =>
            print('[SmartError] Invalid search parameters: $msg'),
        internalError: (msg) =>
            print('[SmartError] Search engine failure: $msg'),
        unknown: (msg) => print('[SmartError] Unknown search error: $msg'),
      );
      rethrow;
    }

    // 3. Filter to single source FIRST (before adjacent expansion)
    // Pass the original query for text matching
    if (singleSourceMode && chunks.isNotEmpty) {
      chunks = _filterToMostRelevantSource(chunks, query);
    }

    // 4. Expand with adjacent chunks (only for the selected source)
    if (adjacentChunks > 0 && chunks.isNotEmpty) {
      chunks = await _expandWithAdjacentChunks(chunks, adjacentChunks);
    }

    // 5. Assemble context (pass singleSourceMode to skip headers when single source)
    final context = ContextBuilder.build(
      searchResults: chunks,
      tokenBudget: tokenBudget,
      strategy: strategy,
      singleSourceMode: singleSourceMode, // Pass through to skip headers
    );

    return RagSearchResult(chunks: chunks, context: context);
  }

  /// Filter to only chunks from the most relevant source.
  /// Prioritizes sources that contain exact text from the query.
  List<ChunkSearchResult> _filterToMostRelevantSource(
    List<ChunkSearchResult> results,
    String query,
  ) {
    if (results.isEmpty) return results;

    // Extract key phrases from query
    final queryLower = query.toLowerCase();

    // First: Try to find source that contains the exact query text
    final sourceTextMatches = <int, int>{}; // sourceId -> match count

    for (final chunk in results) {
      final sourceId = chunk.sourceId.toInt();
      final contentLower = chunk.content.toLowerCase();

      // Check if chunk contains significant part of query
      // Split query into meaningful segments and check for matches
      final queryWords = query
          .split(RegExp(r'[\s\(\)]+'))
          .where((w) => w.length > 2)
          .toList();
      int matchCount = 0;
      for (final word in queryWords) {
        if (contentLower.contains(word.toLowerCase())) {
          matchCount++;
        }
      }

      // Bonus for exact phrase match
      if (contentLower.contains(queryLower) || chunk.content.contains(query)) {
        matchCount += 100; // Strong bonus for exact match
      }

      sourceTextMatches[sourceId] =
          (sourceTextMatches[sourceId] ?? 0) + matchCount;
    }

    // Find source with highest text match count
    int? bestSourceByText;
    int bestTextMatchCount = 0;
    for (final entry in sourceTextMatches.entries) {
      if (entry.value > bestTextMatchCount) {
        bestTextMatchCount = entry.value;
        bestSourceByText = entry.key;
      }
    }

    // If we have a source with good text matches, use it
    if (bestSourceByText != null && bestTextMatchCount > 0) {
      return results
          .where((c) => c.sourceId.toInt() == bestSourceByText)
          .toList();
    }

    // Fallback: Sum similarity scores by source
    final sourceScores = <int, double>{};
    for (final chunk in results) {
      final sourceId = chunk.sourceId.toInt();
      sourceScores[sourceId] = (sourceScores[sourceId] ?? 0) + chunk.similarity;
    }

    // Find source with highest total score
    int? bestSourceId;
    double bestScore = -1;
    for (final entry in sourceScores.entries) {
      if (entry.value > bestScore) {
        bestScore = entry.value;
        bestSourceId = entry.key;
      }
    }

    if (bestSourceId == null) return results;

    // Filter to only that source
    return results.where((c) => c.sourceId.toInt() == bestSourceId).toList();
  }

  /// Expand search results with adjacent chunks from the same source.
  Future<List<ChunkSearchResult>> _expandWithAdjacentChunks(
    List<ChunkSearchResult> chunks,
    int adjacentCount,
  ) async {
    final expandedMap = <String, ChunkSearchResult>{};

    // Add original chunks first (keyed by sourceId:chunkIndex)
    for (final chunk in chunks) {
      final key = '${chunk.sourceId}:${chunk.chunkIndex}';
      expandedMap[key] = chunk;
    }

    // Fetch adjacent chunks for each matched chunk
    for (final chunk in chunks) {
      final minIndex = (chunk.chunkIndex - adjacentCount).clamp(0, 999999);
      final maxIndex = chunk.chunkIndex + adjacentCount;

      List<ChunkSearchResult> adjacent = [];
      try {
        adjacent = await getAdjacentChunks(
          sourceId: chunk.sourceId,
          minIndex: minIndex,
          maxIndex: maxIndex,
        );
      } on RagError catch (e) {
        print('[SmartError] Failed to fetch adjacent chunks: ${e.message}');
        // Non-critical, just continue without expansion
        continue;
      }

      for (final adj in adjacent) {
        final key = '${adj.sourceId}:${adj.chunkIndex}';
        if (!expandedMap.containsKey(key)) {
          expandedMap[key] = adj;
        }
      }
    }

    // Sort by sourceId then chunkIndex for coherent reading order
    final result = expandedMap.values.toList();
    result.sort((a, b) {
      final sourceCompare = a.sourceId.compareTo(b.sourceId);
      if (sourceCompare != 0) return sourceCompare;
      return a.chunkIndex.compareTo(b.chunkIndex);
    });

    return result;
  }

  /// Get the original source document for a chunk.
  Future<String?> getSourceForChunk(ChunkSearchResult chunk) async {
    return await getSource(sourceId: chunk.sourceId);
  }

  /// Get statistics about stored sources and chunks.
  Future<SourceStats> getStats() async {
    return await getSourceStats();
  }

  /// Format search results as an LLM prompt.
  String formatPrompt(String query, RagSearchResult result) {
    return ContextBuilder.formatForPrompt(
      query: query,
      context: result.context,
    );
  }

  /// Get a list of all stored sources.
  Future<List<SourceEntry>> listSources() async {
    return await rust_rag.listSources();
  }

  /// Remove a source and all its chunks from the database.
  Future<void> removeSource(int sourceId) async {
    try {
      await deleteSource(sourceId: sourceId);
    } on RagError catch (e) {
      print('[SmartError] Failed to remove source $sourceId: ${e.message}');
      rethrow;
    }
    // Note: HNSW index is not automatically updated.
    // It's recommended to call rebuildIndex() if many sources are deleted.
  }

  /// Hybrid search combining vector and keyword (BM25) search.
  ///
  /// This method uses Reciprocal Rank Fusion (RRF) to combine:
  /// - Vector search: semantic similarity using embeddings
  /// - BM25 search: keyword matching for exact terms
  ///
  /// Use this for better results when searching for:
  /// - Proper nouns (names, product models)
  /// - Technical terms or code snippets
  /// - Exact keyword matches
  ///
  /// Parameters:
  /// - [query]: The search query text
  /// - [topK]: Number of results to return (default: 10)
  /// - [vectorWeight]: Weight for vector search (0.0-1.0, default: 0.5)
  /// - [bm25Weight]: Weight for BM25 search (0.0-1.0, default: 0.5)
  Future<List<hybrid.HybridSearchResult>> searchHybrid(
    String query, {
    int topK = 10,
    double vectorWeight = 0.5,
    double bm25Weight = 0.5,
    List<int>? sourceIds,
  }) async {
    // 1. Generate query embedding
    final queryEmbedding = await EmbeddingService.embed(query);

    // 2. Perform hybrid search with RRF fusion
    late List<hybrid.HybridSearchResult> results;
    try {
      // Improve post-filtering recall:
      // If filtering by source, fetch more candidates internally to avoid
      // the dominant source occupying all top-k slots before filtering.
      final effectiveTopK = sourceIds != null
          ? (topK * 10).clamp(50, 200)
          : topK;

      results = await hybrid.searchHybrid(
        queryText: query,
        queryEmbedding: queryEmbedding,
        topK: effectiveTopK,
        config: hybrid.RrfConfig(
          k: 60,
          vectorWeight: vectorWeight,
          bm25Weight: bm25Weight,
        ),
        filter: sourceIds != null
            ? hybrid.SearchFilter(sourceIds: _toInt64List(sourceIds))
            : null,
      );
    } on RagError catch (e) {
      e.when(
        databaseError: (msg) =>
            print('[SmartError] Hybrid search DB error: $msg'),
        ioError: (_) {},
        modelLoadError: (_) {},
        invalidInput: (_) {},
        internalError: (msg) =>
            print('[SmartError] Hybrid search engine error: $msg'),
        unknown: (msg) =>
            print('[SmartError] Hybrid search unknown error: $msg'),
      );
      rethrow;
    }

    return results.take(topK).toList();
  }

  /// Hybrid search with context assembly for LLM.
  ///
  /// Similar to [search] but uses hybrid (vector + BM25) search.
  Future<RagSearchResult> searchHybridWithContext(
    String query, {
    int topK = 10,
    int tokenBudget = 2000,
    ContextStrategy strategy = ContextStrategy.relevanceFirst,
    double vectorWeight = 0.5,
    double bm25Weight = 0.5,
    List<int>? sourceIds,
  }) async {
    // 1. Get hybrid search results
    final hybridResults = await searchHybrid(
      query,
      topK: topK,
      vectorWeight: vectorWeight,
      bm25Weight: bm25Weight,
      sourceIds: sourceIds,
    );

    // 2. Convert to ChunkSearchResult format for context building
    // Note: Hybrid search returns content directly, so we create minimal chunks
    final chunks = hybridResults
        .map(
          (r) => ChunkSearchResult(
            chunkId: r.docId,
            sourceId: r.sourceId, // Correctly map from HybridSearchResult
            content: r.content,
            chunkIndex: 0,
            chunkType: 'general', // Hybrid search doesn't return chunk type
            similarity: r.score, // RRF score as similarity
            metadata: r.metadata,
          ),
        )
        .toList();

    // 3. Assemble context
    final context = ContextBuilder.build(
      searchResults: chunks,
      tokenBudget: tokenBudget,
      strategy: strategy,
    );

    return RagSearchResult(chunks: chunks, context: context);
  }
}
