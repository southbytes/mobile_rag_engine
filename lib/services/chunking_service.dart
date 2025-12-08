/// Chunking strategies for splitting long documents into LLM-friendly pieces.
///
/// The chunking service splits documents while:
/// - Respecting semantic boundaries (sentences, paragraphs)
/// - Maintaining overlap for context continuity
/// - Staying within token limits for LLM consumption

/// Represents a single chunk of text from a larger document.
class Chunk {
  /// The text content of this chunk.
  final String content;
  
  /// Index of this chunk within the source document (0-based).
  final int index;
  
  /// Character position where this chunk starts in the original document.
  final int startPosition;
  
  /// Character position where this chunk ends in the original document.
  final int endPosition;

  const Chunk({
    required this.content,
    required this.index,
    required this.startPosition,
    required this.endPosition,
  });

  /// Approximate token count (rough estimate: 1 token ≈ 4 chars for English).
  int get estimatedTokens => (content.length / 4).ceil();

  @override
  String toString() => 'Chunk($index, ${content.length} chars, ~$estimatedTokens tokens)';
}

/// Configuration for chunking strategy.
class ChunkConfig {
  /// Maximum characters per chunk.
  final int maxChars;
  
  /// Number of overlapping characters between consecutive chunks.
  final int overlap;
  
  /// Separators to try, in order of preference (paragraph, line, sentence, word).
  final List<String> separators;

  const ChunkConfig({
    this.maxChars = 500,
    this.overlap = 50,
    this.separators = const ['\n\n', '\n', '. ', ', ', ' '],
  });

  /// Preset for short chunks (good for precise retrieval).
  static const short = ChunkConfig(maxChars: 300, overlap: 30);
  
  /// Preset for medium chunks (balanced).
  static const medium = ChunkConfig(maxChars: 500, overlap: 50);
  
  /// Preset for long chunks (more context per chunk).
  static const long = ChunkConfig(maxChars: 1000, overlap: 100);
}

/// Service for splitting documents into chunks.
class ChunkingService {
  ChunkingService._();

  /// Split text into chunks using recursive character splitting.
  /// 
  /// This strategy tries to split on natural boundaries (paragraphs, sentences)
  /// before falling back to character-level splitting.
  static List<Chunk> chunk(String text, {ChunkConfig config = ChunkConfig.medium}) {
    if (text.isEmpty) return [];
    
    // If text is short enough, return as single chunk
    if (text.length <= config.maxChars) {
      return [
        Chunk(
          content: text,
          index: 0,
          startPosition: 0,
          endPosition: text.length,
        ),
      ];
    }

    final chunks = <Chunk>[];
    var currentPosition = 0;
    var chunkIndex = 0;

    while (currentPosition < text.length) {
      // Calculate end position for this chunk
      var endPosition = currentPosition + config.maxChars;
      
      if (endPosition >= text.length) {
        // Last chunk - take the rest
        endPosition = text.length;
      } else {
        // Find the best split point
        endPosition = _findBestSplitPoint(
          text,
          currentPosition,
          endPosition,
          config.separators,
        );
      }

      // Extract chunk content
      final content = text.substring(currentPosition, endPosition).trim();
      
      if (content.isNotEmpty) {
        chunks.add(Chunk(
          content: content,
          index: chunkIndex,
          startPosition: currentPosition,
          endPosition: endPosition,
        ));
        chunkIndex++;
      }

      // Move to next position with overlap
      currentPosition = endPosition - config.overlap;
      if (currentPosition <= chunks.last.startPosition) {
        currentPosition = endPosition; // Avoid infinite loop
      }
    }

    return chunks;
  }

  /// Find the best position to split text, preferring natural boundaries.
  static int _findBestSplitPoint(
    String text,
    int start,
    int idealEnd,
    List<String> separators,
  ) {
    // Search backwards from ideal end to find a separator
    for (final separator in separators) {
      // Look for separator in the last 30% of the chunk
      final searchStart = start + ((idealEnd - start) * 0.7).toInt();
      final searchRegion = text.substring(searchStart, idealEnd);
      
      final lastIndex = searchRegion.lastIndexOf(separator);
      if (lastIndex != -1) {
        return searchStart + lastIndex + separator.length;
      }
    }
    
    // No good separator found, split at ideal position
    return idealEnd;
  }

  /// Estimate total tokens for a list of chunks.
  static int estimateTotalTokens(List<Chunk> chunks) {
    return chunks.fold(0, (sum, chunk) => sum + chunk.estimatedTokens);
  }

  /// Get chunks that fit within a token budget, ordered by relevance.
  /// 
  /// [rankedChunks] should be pre-sorted by relevance (highest first).
  static List<Chunk> selectWithinBudget(
    List<Chunk> rankedChunks,
    int tokenBudget,
  ) {
    final selected = <Chunk>[];
    var usedTokens = 0;

    for (final chunk in rankedChunks) {
      if (usedTokens + chunk.estimatedTokens <= tokenBudget) {
        selected.add(chunk);
        usedTokens += chunk.estimatedTokens;
      }
    }

    return selected;
  }
}
