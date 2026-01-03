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
import '../src/rust/api/source_rag.dart';
import '../src/rust/api/semantic_chunker.dart';
import '../src/rust/api/hybrid_search.dart' as hybrid;
import '../src/rust/api/hnsw_index.dart' as hnsw;
import 'context_builder.dart';
import 'embedding_service.dart';

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

  /// Get the HNSW index path (derived from dbPath)
  String get _indexPath => dbPath.replaceAll('.db', '_hnsw');

  /// Initialize the source database.
  Future<void> init() async {
    await initSourceDb(dbPath: dbPath);
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
  /// 1. Split into chunks based on [chunkConfig]
  /// 2. Each chunk is embedded
  /// 3. Source and chunks are stored in DB
  Future<SourceAddResult> addSourceWithChunking(
    String content, {
    String? metadata,
    void Function(int done, int total)? onProgress,
  }) async {
    // 1. Add source document
    final sourceResult = await addSource(
      dbPath: dbPath,
      content: content,
      metadata: metadata,
    );

    if (sourceResult.isDuplicate) {
      return SourceAddResult(
        sourceId: sourceResult.sourceId.toInt(),
        isDuplicate: true,
        chunkCount: 0,
        message: sourceResult.message,
      );
    }

    // 2. Split into semantic chunks using Rust (Unicode sentence/word boundaries)
    final chunks = semanticChunkWithOverlap(
      text: content,
      maxChars: maxChunkChars,
      overlapChars: overlapChars,
    );

    if (chunks.isEmpty) {
      return SourceAddResult(
        sourceId: sourceResult.sourceId.toInt(),
        isDuplicate: false,
        chunkCount: 0,
        message: 'No chunks created',
      );
    }

    // 3. Generate embeddings for each chunk
    final chunkDataList = <ChunkData>[];

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
          embedding: Float32List.fromList(embedding),
        ),
      );
    }

    onProgress?.call(chunks.length, chunks.length);

    // 4. Store chunks
    await addChunks(
      dbPath: dbPath,
      sourceId: sourceResult.sourceId,
      chunks: chunkDataList,
    );

    return SourceAddResult(
      sourceId: sourceResult.sourceId.toInt(),
      isDuplicate: false,
      chunkCount: chunks.length,
      message: 'Added ${chunks.length} chunks',
    );
  }

  /// Rebuild the HNSW index after adding sources.
  Future<void> rebuildIndex() async {
    await rebuildChunkHnswIndex(dbPath: dbPath);
  }

  /// Regenerate embeddings for all existing chunks using the current model.
  /// This is needed when the embedding model or tokenizer has been updated.
  Future<void> regenerateAllEmbeddings({
    void Function(int done, int total)? onProgress,
  }) async {
    // 1. Get all chunk IDs and contents
    final chunks = await getAllChunkIdsAndContents(dbPath: dbPath);
    print(
      '[regenerateAllEmbeddings] Found ${chunks.length} chunks to re-embed',
    );

    // 2. Re-embed each chunk
    for (var i = 0; i < chunks.length; i++) {
      final chunk = chunks[i];
      final embedding = await EmbeddingService.embed(chunk.content);

      // 3. Update in DB
      await updateChunkEmbedding(
        dbPath: dbPath,
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
    var chunks = await searchChunks(
      dbPath: dbPath,
      queryEmbedding: queryEmbedding,
      topK: topK,
    );

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

    // Extract key phrases from query (look for patterns like "제 X 조 (내용)")
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

      final adjacent = await getAdjacentChunks(
        dbPath: dbPath,
        sourceId: chunk.sourceId,
        minIndex: minIndex,
        maxIndex: maxIndex,
      );

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
    return await getSource(dbPath: dbPath, sourceId: chunk.sourceId);
  }

  /// Get statistics about stored sources and chunks.
  Future<SourceStats> getStats() async {
    return await getSourceStats(dbPath: dbPath);
  }

  /// Format search results as an LLM prompt.
  String formatPrompt(String query, RagSearchResult result) {
    return ContextBuilder.formatForPrompt(
      query: query,
      context: result.context,
    );
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
  }) async {
    // 1. Generate query embedding
    final queryEmbedding = await EmbeddingService.embed(query);

    // 2. Perform hybrid search with RRF fusion
    final results = await hybrid.searchHybridWeighted(
      dbPath: dbPath,
      queryText: query,
      queryEmbedding: queryEmbedding,
      topK: topK,
      vectorWeight: vectorWeight,
      bm25Weight: bm25Weight,
    );

    return results;
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
  }) async {
    // 1. Get hybrid search results
    final hybridResults = await searchHybrid(
      query,
      topK: topK,
      vectorWeight: vectorWeight,
      bm25Weight: bm25Weight,
    );

    // 2. Convert to ChunkSearchResult format for context building
    // Note: Hybrid search returns content directly, so we create minimal chunks
    final chunks = hybridResults
        .map(
          (r) => ChunkSearchResult(
            chunkId: r.docId,
            sourceId: r.docId, // Same as chunk ID for simple docs
            content: r.content,
            chunkIndex: 0,
            similarity: r.score, // RRF score as similarity
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
