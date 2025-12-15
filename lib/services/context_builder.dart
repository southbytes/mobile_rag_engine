/// Context assembly for LLM prompts.
///
/// ContextBuilder takes search results and assembles them into
/// an optimized context string within a token budget.

import '../src/rust/api/source_rag.dart';

/// Assembled context ready for LLM consumption.
class AssembledContext {
  /// The combined text of all selected chunks.
  final String text;
  
  /// Chunks that were included in the context.
  final List<ChunkSearchResult> includedChunks;
  
  /// Approximate token count used.
  final int estimatedTokens;
  
  /// Tokens remaining from budget.
  final int remainingBudget;

  const AssembledContext({
    required this.text,
    required this.includedChunks,
    required this.estimatedTokens,
    required this.remainingBudget,
  });

  @override
  String toString() => 'AssembledContext(${includedChunks.length} chunks, ~$estimatedTokens tokens)';
}

/// Strategy for selecting and ordering chunks.
enum ContextStrategy {
  /// Include highest similarity chunks first until budget exhausted.
  relevanceFirst,
  
  /// Ensure diversity by not including chunks from same source consecutively.
  diverseSources,
  
  /// Order by chunk index within each source (maintains document order).
  chronological,
}

/// Builds optimized context for LLM prompts.
class ContextBuilder {
  ContextBuilder._();

  /// Assemble context from search results within a token budget.
  ///
  /// [searchResults] - Chunks ranked by relevance (highest first).
  /// [tokenBudget] - Maximum tokens to use.
  /// [strategy] - How to select and order chunks.
  /// [separator] - Text between chunks.
  static AssembledContext build({
    required List<ChunkSearchResult> searchResults,
    int tokenBudget = 2000,
    ContextStrategy strategy = ContextStrategy.relevanceFirst,
    String separator = '\n\n---\n\n',
  }) {
    if (searchResults.isEmpty) {
      return const AssembledContext(
        text: '',
        includedChunks: [],
        estimatedTokens: 0,
        remainingBudget: 0,
      );
    }

    // Apply strategy
    final orderedResults = switch (strategy) {
      ContextStrategy.relevanceFirst => searchResults,
      ContextStrategy.diverseSources => _diversifySources(searchResults),
      ContextStrategy.chronological => _orderChronologically(searchResults),
    };

    // Select chunks within budget
    final selected = <ChunkSearchResult>[];
    var usedTokens = 0;
    final separatorTokens = (separator.length / 4).ceil();

    for (final chunk in orderedResults) {
      final chunkTokens = (chunk.content.length / 4).ceil();
      final totalIfAdded = usedTokens + chunkTokens + (selected.isEmpty ? 0 : separatorTokens);

      if (totalIfAdded <= tokenBudget) {
        selected.add(chunk);
        usedTokens = totalIfAdded;
      } else {
        break; // Budget exhausted
      }
    }

    // Build final text
    final text = selected.map((c) => c.content).join(separator);

    return AssembledContext(
      text: text,
      includedChunks: selected,
      estimatedTokens: usedTokens,
      remainingBudget: tokenBudget - usedTokens,
    );
  }

  /// Diversify by avoiding consecutive chunks from same source.
  static List<ChunkSearchResult> _diversifySources(List<ChunkSearchResult> results) {
    final diverse = <ChunkSearchResult>[];
    final remaining = List<ChunkSearchResult>.from(results);
    int? lastSourceId;

    while (remaining.isNotEmpty) {
      // Find next chunk not from last source
      final idx = remaining.indexWhere((r) => r.sourceId.toInt() != lastSourceId);
      
      if (idx >= 0) {
        diverse.add(remaining.removeAt(idx));
        lastSourceId = diverse.last.sourceId.toInt();
      } else {
        // All remaining are from same source
        diverse.addAll(remaining);
        break;
      }
    }

    return diverse;
  }

  /// Order by source ID then chunk index (maintains document order).
  static List<ChunkSearchResult> _orderChronologically(List<ChunkSearchResult> results) {
    final sorted = List<ChunkSearchResult>.from(results);
    sorted.sort((a, b) {
      final sourceCompare = a.sourceId.compareTo(b.sourceId);
      if (sourceCompare != 0) return sourceCompare;
      return a.chunkIndex.compareTo(b.chunkIndex);
    });
    return sorted;
  }

  /// Format context for LLM prompt.
  static String formatForPrompt({
    required String query,
    required AssembledContext context,
    String systemInstruction = 'Answer based on the following documents:',
  }) {
    if (context.text.isEmpty) {
      return query;
    }

    return '''$systemInstruction

${context.text}

Question: $query''';
  }
}
