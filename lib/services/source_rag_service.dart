/// High-level RAG service for managing sources and chunks.
///
/// This service provides a convenient API that combines:
/// - Rust semantic chunking for document splitting (Unicode sentence/word boundaries)
/// - EmbeddingService for vector generation
/// - Rust source_rag APIs for storage and search
/// - ContextBuilder for LLM context assembly
/// - Hybrid search combining vector and BM25 keyword search
library;

import 'dart:developer';
import 'dart:io';

import 'package:flutter/foundation.dart';
import '../src/rust/api/error.dart';
import '../src/rust/api/source_rag.dart' as rust_rag;
import '../src/rust/api/source_rag.dart'
    show SourceStats, ChunkSearchResult, ChunkData, SourceEntry;
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

  /// Helper to convert `List<int>` to the specific `Int64List` type required by FRB.
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

  /// File marker to indicate if the index is dirty (needs rebuild).
  /// This persists across app restarts for crash recovery.
  File get _dirtyMarkerFile => File('${dbPath.replaceAll('.db', '')}.dirty');

  /// Memory flag to track dirty state alongside the file marker.
  bool _needsRebuild = false;

  /// Mark the index as dirty (needs rebuild).
  /// Creates a persistent marker file.
  Future<void> _markDirty() async {
    _needsRebuild = true;
    try {
      if (!await _dirtyMarkerFile.exists()) {
        await _dirtyMarkerFile.create();
      }
    } catch (e) {
      debugPrint('[SourceRagService] Failed to create dirty marker: $e');
    }
  }

  /// Mark the index as clean (rebuild complete).
  /// Deletes the persistent marker file.
  Future<void> _markClean() async {
    _needsRebuild = false;
    try {
      if (await _dirtyMarkerFile.exists()) {
        await _dirtyMarkerFile.delete();
      }
    } catch (e) {
      debugPrint('[SourceRagService] Failed to delete dirty marker: $e');
    }
  }

  /// Initialize the source database.
  Future<void> init() async {
    try {
      debugPrint('[SourceRagService] init: Starting...');

      // 1-3. Initialize Logger (only if not already active)
      // Skipping re-init avoids potential deadlocks with closeLogStream/logging threads
      if (_logSubscription == null) {
        debugPrint('[SourceRagService] init: Initializing logger system...');
        await initLogger(); // Idempotent

        // Close any existing (though _logSubscription is null, be safe)
        await _cleanupLogStream();

        // Start fresh log stream
        debugPrint('[SourceRagService] init: Starting log stream...');
        _logSubscription = initLogStream().listen((log) {
          debugPrint(log);
        });
      } else {
        debugPrint(
          '[SourceRagService] init: Logger stream already active, skipping re-init.',
        );
      }

      // 4. Initialize DB
      debugPrint('[SourceRagService] init: Initializing Source DB...');
      await rust_rag.initSourceDb();

      // 4.1 Check for dirty marker (Crash Recovery)
      if (await _dirtyMarkerFile.exists()) {
        debugPrint(
          '[SourceRagService] Found dirty marker. Previous session might have crashed.',
        );
        debugPrint('[SourceRagService] Index will be rebuilt automatically.');
        _needsRebuild = true;
      }

      // 5. Load or Rebuild indexes
      // BM25 is currently in-memory only and fast to rebuild from SQLite
      debugPrint('[SourceRagService] init: Rebuilding BM25 index...');
      await rust_rag.rebuildChunkBm25Index();

      // HNSW can be slow to rebuild for large datasets, so we try loading from disk first
      debugPrint('[SourceRagService] init: Attempting to load HNSW index from disk...');
      final loaded = await tryLoadCachedIndex();
      
      if (loaded && !_needsRebuild) {
        debugPrint('[SourceRagService] init: HNSW index loaded successfully from cache.');
      } else {
        if (_needsRebuild) {
          debugPrint('[SourceRagService] init: Dirty marker found, forcing HNSW rebuild.');
        } else {
          debugPrint('[SourceRagService] init: No cached HNSW index found, rebuilding from DB.');
        }
        await rust_rag.rebuildChunkHnswIndex();
        // Save the newly built index for next time
        await saveIndex();
      }

      debugPrint('[SourceRagService] init: Done!');
    } on RagError catch (e) {
      // Smart Error Handling integration
      debugPrint(
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
    Duration? chunkDelay,
    void Function(int done, int total)? onProgress,
  }) async {
    // 1. Add source document
    late SourceAddResult sourceResult;
    try {
      final res = await rust_rag.addSource(
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

      // Mark index as dirty (Persistent)
      await _markDirty();
    } on RagError catch (e) {
      e.when(
        databaseError: (msg) =>
            debugPrint('[SmartError] Storage write failed: $msg'),
        ioError: (msg) =>
            debugPrint('[SmartError] IO error adding source: $msg'),
        modelLoadError: (_) {},
        invalidInput: (msg) =>
            debugPrint('[SmartError] Invalid source content: $msg'),
        internalError: (msg) =>
            debugPrint('[SmartError] Internal error adding source: $msg'),
        unknown: (msg) =>
            debugPrint('[SmartError] Unknown error adding source: $msg'),
      );
      rethrow;
    }

    // 2. Determine chunking strategy
    final effectiveStrategy = strategy ?? detectChunkingStrategy(filePath);

    // 3. Split content based on strategy
    List<dynamic> allRawChunks; // Either MarkdownChunk or String/Chunk
    if (effectiveStrategy == ChunkingStrategy.markdown) {
      allRawChunks = markdownChunk(text: content, maxChars: maxChunkChars);
    } else {
      allRawChunks = semanticChunkWithOverlap(
        text: content,
        maxChars: maxChunkChars,
        overlapChars: overlapChars,
      );
    }

    // 4. Check for existing progress (Resume capability)
    int startIndex = 0;
    if (sourceResult.isDuplicate) {
      try {
        final existingCount = await rust_rag.getSourceChunkCount(
          sourceId: sourceResult.sourceId.toInt(),
        );
        if (existingCount >= allRawChunks.length) {
          // Ensure it's marked as completed if it wasn't already
          await rust_rag.updateSourceStatus(
            sourceId: sourceResult.sourceId,
            status: 'completed',
          );

          return SourceAddResult(
            sourceId: sourceResult.sourceId.toInt(),
            isDuplicate: true,
            chunkCount: 0, // No new chunks added
            message: 'Already processed (${allRawChunks.length} chunks)',
          );
        }
        startIndex = existingCount;
        debugPrint(
          '[MobileRag] Resuming source ${sourceResult.sourceId} from chunk $startIndex/${allRawChunks.length}',
        );
      } catch (e) {
        debugPrint(
          '[MobileRag] Failed to check existing chunks, restarting: $e',
        );
      }
    }

    // 5. Process chunks with batching
    final chunkDataBatch = <ChunkData>[];
    int chunksAdded = 0;
    const batchSize = 50;

    try {
      for (var i = startIndex; i < allRawChunks.length; i++) {
        onProgress?.call(i, allRawChunks.length);
        final rawChunk = allRawChunks[i];

        // Throttling: Wait if delay is configured
        if (chunkDelay != null) {
          await Future.delayed(chunkDelay);
        }

        String contentStr;
        String chunkType;
        int chunkIdx;
        int startPos;
        int endPos;

        if (effectiveStrategy == ChunkingStrategy.markdown) {
          final c = rawChunk as StructuredChunk;
          contentStr = c.content;
          chunkType = c.headerPath.isNotEmpty
              ? '${c.chunkType}|${c.headerPath}'
              : c.chunkType;
          chunkIdx = c.index;
          startPos = c.startPos;
          endPos = c.endPos;
        } else {
          final c = rawChunk as SemanticChunk;
          contentStr = c.content;
          chunkType = c.chunkType;
          chunkIdx = c.index;
          startPos = c.startPos;
          endPos = c.endPos;
        }

        final embedding = await EmbeddingService.embed(contentStr);

        chunkDataBatch.add(
          ChunkData(
            content: contentStr,
            chunkIndex: chunkIdx,
            startPos: startPos,
            endPos: endPos,
            chunkType: chunkType,
            embedding: Float32List.fromList(embedding),
          ),
        );
        chunksAdded++;

        // Incremental save
        if (chunkDataBatch.length >= batchSize) {
          await rust_rag.addChunks(
            sourceId: sourceResult.sourceId,
            chunks: chunkDataBatch,
          );
          chunkDataBatch.clear();
          // Optional: Update status to 'processing' or 'partial' if needed, but 'pending' serves as "not done".
          // We could update to 'processing' here to indicate active work.
          await rust_rag.updateSourceStatus(
            sourceId: sourceResult.sourceId,
            status: 'processing',
          );
        }
      }

      // Save remaining chunks
      if (chunkDataBatch.isNotEmpty) {
        await rust_rag.addChunks(
          sourceId: sourceResult.sourceId,
          chunks: chunkDataBatch,
        );
      }

      onProgress?.call(allRawChunks.length, allRawChunks.length);

      // Mark as completed
      await rust_rag.updateSourceStatus(
        sourceId: sourceResult.sourceId,
        status: 'completed',
      );

      return SourceAddResult(
        sourceId: sourceResult.sourceId.toInt(),
        isDuplicate: sourceResult.isDuplicate,
        chunkCount: chunksAdded,
        message:
            'Added $chunksAdded chunks (Resumed from $startIndex, Total ${allRawChunks.length})',
      );
    } catch (e) {
      // Mark as failed if error occurs during chunking/embedding loop
      debugPrint(
        '[MobileRag] Error adding source ${sourceResult.sourceId}: $e',
      );
      await rust_rag.updateSourceStatus(
        sourceId: sourceResult.sourceId,
        status: 'failed',
      );
      rethrow;
    }
  }

  /// Rebuild the HNSW and BM25 indexes after adding sources.
  /// [force] - If true, rebuilds even if no changes were detected (default: false).
  Future<void> rebuildIndex({bool force = false}) async {
    if (!force && !_needsRebuild) {
      debugPrint(
        '[SourceRagService] Index is already up to date. Skipping rebuild.',
      );
      return;
    }
    try {
      // Rebuild HNSW for vector search
      await rust_rag.rebuildChunkHnswIndex();
      // Rebuild BM25 for keyword search (critical for hybrid search!)
      await rust_rag.rebuildChunkBm25Index();

      await _markClean(); // Mark index as clean (Persistent)
    } on RagError catch (e) {
      e.when(
        databaseError: (msg) =>
            debugPrint('[SmartError] DB error rebuilding index: $msg'),
        ioError: (msg) =>
            debugPrint('[SmartError] IO error rebuilding index: $msg'),
        modelLoadError: (_) {},
        invalidInput: (_) {},
        internalError: (msg) =>
            debugPrint('[SmartError] Internal error rebuilding indexes: $msg'),
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
    // 1. Get total stats for progress tracking
    final stats = await rust_rag.getSourceStats();
    final totalChunks = stats.chunkCount.toInt();

    debugPrint(
      '[regenerateAllEmbeddings] Found $totalChunks chunks to re-embed (Safe Batch Mode)',
    );

    // 2. Iterate by Source to ensure memory safety
    // Instead of loading all chunks (which could be huge), we process source by source
    final sources = await rust_rag.listSources();
    int processedCount = 0;

    for (final source in sources) {
      final sourceId = source.id.toInt();
      final chunkCount = await rust_rag.getSourceChunkCount(
        sourceId: source.id,
      );

      // Process chunks for this source in batches
      const batchSize = 50;
      for (var offset = 0; offset < chunkCount; offset += batchSize) {
        // Fetch batch of chunks
        // We use getAdjacentChunks to fetch specific ranges safely
        // minIndex = offset, maxIndex = offset + batchSize - 1
        List<ChunkSearchResult> batch;
        try {
          batch = await rust_rag.getAdjacentChunks(
            sourceId: source.id,
            minIndex: offset,
            maxIndex: offset + batchSize - 1, // Inclusive
          );
        } catch (e) {
          debugPrint(
            '[regenerateAllEmbeddings] Failed to fetch batch for source $sourceId: $e',
          );
          continue;
        }

        // Re-embed and update each chunk
        for (final chunk in batch) {
          try {
            final embedding = await EmbeddingService.embed(chunk.content);
            await rust_rag.updateChunkEmbedding(
              chunkId: chunk.chunkId,
              embedding: Float32List.fromList(embedding),
            );
          } catch (e) {
            debugPrint(
              '[regenerateAllEmbeddings] Failed to update chunk ${chunk.chunkId}: $e',
            );
          }
        }

        processedCount += batch.length;
        onProgress?.call(processedCount, totalChunks);

        // Yield to event loop to prevent UI jank
        await Future.delayed(Duration.zero);
      }
    }

    debugPrint('[regenerateAllEmbeddings] Completed. Rebuilding HNSW index...');

    // 3. Rebuild HNSW index
    await rebuildIndex(force: true);

    debugPrint('[regenerateAllEmbeddings] Done!');
  }

  /// Fetch adjacent chunks for a given source and index range.
  ///
  /// Useful for "Show More" or "Read Previous/Next" features.
  Future<List<ChunkSearchResult>> getAdjacentChunks({
    required int sourceId,
    required int minIndex,
    required int maxIndex,
  }) {
    return rust_rag.getAdjacentChunks(
      sourceId: sourceId,
      minIndex: minIndex,
      maxIndex: maxIndex,
    );
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
    debugPrint('[DEBUG] Query: "$query"');
    debugPrint(
      '[DEBUG] Embedding norm: ${embNorm.toStringAsFixed(4)}, dims: ${queryEmbedding.length}',
    );
    debugPrint(
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
        chunks = await rust_rag.searchChunks(
          queryEmbedding: queryEmbedding,
          topK: topK,
        );
      }
    } on RagError catch (e) {
      e.when(
        databaseError: (msg) =>
            debugPrint('[SmartError] Search failed (database): $msg'),
        ioError: (msg) => debugPrint('[SmartError] Search IO error: $msg'),
        modelLoadError: (_) {},
        invalidInput: (msg) =>
            debugPrint('[SmartError] Invalid search parameters: $msg'),
        internalError: (msg) =>
            debugPrint('[SmartError] Search engine failure: $msg'),
        unknown: (msg) => debugPrint('[SmartError] Unknown search error: $msg'),
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
        adjacent = await rust_rag.getAdjacentChunks(
          sourceId: chunk.sourceId,
          minIndex: minIndex,
          maxIndex: maxIndex,
        );
      } on RagError catch (e) {
        debugPrint(
          '[SmartError] Failed to fetch adjacent chunks: ${e.message}',
        );
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
    return await rust_rag.getSource(sourceId: chunk.sourceId);
  }

  /// Get statistics about stored sources and chunks.
  Future<SourceStats> getStats() async {
    return await rust_rag.getSourceStats();
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
      await rust_rag.deleteSource(sourceId: sourceId);
      await _markDirty(); // Mark index as dirty (Persistent)
    } on RagError catch (e) {
      debugPrint(
        '[SmartError] Failed to remove source $sourceId: ${e.message}',
      );
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
            debugPrint('[SmartError] Hybrid search DB error: $msg'),
        ioError: (_) {},
        modelLoadError: (_) {},
        invalidInput: (_) {},
        internalError: (msg) =>
            log('[SmartError] Hybrid search engine error: $msg'),
        unknown: (msg) => log('[SmartError] Hybrid search unknown error: $msg'),
      );
      rethrow;
    }

    return results.take(topK).toList();
  }

  /// Hybrid search with context assembly for LLM.
  ///
  /// Similar to [search] but uses hybrid (vector + BM25) search.
  ///
  /// [adjacentChunks] - Number of adjacent chunks to include before/after each
  /// matched chunk (default: 0). Setting this to 1 will include the chunk
  /// before and after each matched chunk, helping with long articles.
  /// [singleSourceMode] - If true, only include chunks from the most relevant source.
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
    var chunks = hybridResults
        .map(
          (r) => ChunkSearchResult(
            chunkId: r.docId,
            sourceId: r.sourceId,
            content: r.content,
            chunkIndex: r.chunkIndex, // Now available from Rust!
            chunkType: 'general', // Hybrid search doesn't return chunk type
            similarity: r.score, // RRF score as similarity
            metadata: r.metadata,
          ),
        )
        .toList();

    // 3. Filter to single source FIRST (before adjacent expansion)
    if (singleSourceMode && chunks.isNotEmpty) {
      chunks = _filterToMostRelevantSource(chunks, query);
    }

    // 4. Expand with adjacent chunks (only for the selected source)
    if (adjacentChunks > 0 && chunks.isNotEmpty) {
      chunks = await _expandWithAdjacentChunks(chunks, adjacentChunks);
    }

    // 5. Assemble context
    final context = ContextBuilder.build(
      searchResults: chunks,
      tokenBudget: tokenBudget,
      strategy: strategy,
      singleSourceMode: singleSourceMode,
    );

    return RagSearchResult(chunks: chunks, context: context);
  }
}
