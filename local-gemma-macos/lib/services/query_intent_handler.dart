// local-gemma-macos/lib/services/query_intent_handler.dart
//
// Handles slash command intents for RAG queries

import 'package:mobile_rag_engine/src/rust/api/user_intent.dart';

/// Intent-specific configuration for RAG and LLM
class IntentConfig {
  /// System prompt template
  final String systemPrompt;

  /// Preferred chunk types for boosting
  final List<String> preferredChunkTypes;

  /// Token budget for RAG context
  final int tokenBudget;

  /// Number of adjacent chunks to fetch
  final int adjacentChunks;

  /// TopK for RAG search
  final int topK;

  /// Whether to use LLM general knowledge
  final bool useLlmKnowledge;

  const IntentConfig({
    required this.systemPrompt,
    this.preferredChunkTypes = const [],
    this.tokenBudget = 2000,
    this.adjacentChunks = 1,
    this.topK = 10,
    this.useLlmKnowledge = false,
  });
}

/// Service to handle intent-based RAG configuration
class QueryIntentHandler {
  static const _summaryPrompt = '''
You are a summarization assistant. Analyze the provided context and create a clear, concise summary.

Rules:
1. Focus on key points and main ideas
2. Use bullet points for clarity
3. Keep the summary under 200 words
4. Only use information from the provided context

Context:
{context}

Question: {query}

Provide a concise summary:''';

  static const _definePrompt = '''
You are a technical definition assistant. Provide a clear, precise definition based on the context.

Rules:
1. Start with a formal definition
2. Include key characteristics
3. Provide an example if available
4. Use academic/professional tone

Context:
{context}

Define: {term}

Definition:''';

  static const _morePrompt = '''
You are a knowledge expansion assistant. Combine the provided RAG context with your general knowledge.

IMPORTANT FORMATTING RULES:
1. Start with "📚 문서 기반 정보:" section for RAG context-based answers
2. Then add "💡 추가 지식:" section for your general knowledge
3. Each section should be clearly separated
4. If no general knowledge is needed, skip the 💡 section

Format Example:
📚 문서 기반 정보:
- [Information from RAG context]

💡 추가 지식:
- [Your additional knowledge not in the documents]

RAG Context (Primary Source):
{context}

Question: {query}

Answer using the format above:''';

  static const _generalPrompt = '''
You are a helpful assistant. Answer the question based on the provided context.

Context:
{context}

Question: {query}

Answer:''';

  /// Get configuration for a parsed intent
  static IntentConfig getConfig(ParsedIntent intent) {
    switch (intent.intentType) {
      case 'summary':
        return IntentConfig(
          systemPrompt: _summaryPrompt,
          preferredChunkTypes: ['list', 'definition', 'procedure'],
          tokenBudget: 1500,
          adjacentChunks: 1,
          topK: 8,
          useLlmKnowledge: false,
        );

      case 'define':
        return IntentConfig(
          systemPrompt: _definePrompt,
          preferredChunkTypes: ['definition', 'example'],
          tokenBudget: 1000,
          adjacentChunks: 0,
          topK: 5,
          useLlmKnowledge: false,
        );

      case 'more':
        return IntentConfig(
          systemPrompt: _morePrompt,
          preferredChunkTypes: [],
          tokenBudget: 2500,
          adjacentChunks: 2,
          topK: 12,
          useLlmKnowledge: true, // Use LLM general knowledge
        );

      case 'general':
      default:
        return IntentConfig(
          systemPrompt: _generalPrompt,
          preferredChunkTypes: [],
          tokenBudget: 2000,
          adjacentChunks: 1,
          topK: 10,
          useLlmKnowledge: false,
        );
    }
  }

  /// Format the system prompt with context and query
  static String formatPrompt({
    required IntentConfig config,
    required String context,
    required String query,
  }) {
    return config.systemPrompt
        .replaceAll('{context}', context)
        .replaceAll('{query}', query)
        .replaceAll('{term}', query);
  }

  /// Apply chunk type boosting - returns boosted similarity scores
  /// Chunks with preferred types get a boost factor
  static double applyBoost({
    required double similarity,
    required String chunkType,
    required List<String> preferredTypes,
    double boostFactor = 0.1,
  }) {
    if (preferredTypes.isEmpty) return similarity;

    if (preferredTypes.contains(chunkType)) {
      return (similarity + boostFactor).clamp(0.0, 1.0);
    }
    return similarity;
  }
}
