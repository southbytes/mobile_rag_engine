// local-gemma-macos/lib/models/chat_models.dart
//
// Data models for RAG chat functionality

import 'package:mobile_rag_engine/mobile_rag_engine.dart';

/// Query intent types for RAG parameter optimization
enum QueryIntent {
  summary, // 요약, 정리, 핵심 → 적은 청크, 낮은 토큰
  definition, // ~란, 뜻, 의미 → 정확한 정의
  broad, // 전체, 모든, 목록 → 많은 청크
  detail, // 자세히, 왜, 어떻게 → 중간 청크
  general, // 기본 질문
}

/// Analysis result from LLM intent classification
class QueryAnalysis {
  final QueryIntent intent;
  final int adjacentChunks;
  final int tokenBudget;
  final int topK;
  final String refinedQuery; // LLM이 정제한 검색 키워드

  const QueryAnalysis({
    required this.intent,
    required this.adjacentChunks,
    required this.tokenBudget,
    required this.topK,
    required this.refinedQuery,
  });

  /// Default fallback analysis
  factory QueryAnalysis.defaultFor(String query) {
    return QueryAnalysis(
      intent: QueryIntent.general,
      adjacentChunks: 2,
      tokenBudget: 2000,
      topK: 10,
      refinedQuery: query,
    );
  }

  @override
  String toString() =>
      'QueryAnalysis(intent: $intent, adjacent: $adjacentChunks, budget: $tokenBudget, topK: $topK, query: "$refinedQuery")';
}

/// Message model for chat
class ChatMessage {
  final String content;
  final bool isUser;
  final DateTime timestamp;
  final List<ChunkSearchResult>? retrievedChunks;
  final int? tokensUsed;
  final double? compressionRatio; // 0.0-1.0, lower = more compressed
  final int? originalTokens; // Before compression

  // Processing metadata for UI display
  final String? queryType; // explanation, definition, factual, etc.

  // Timing metrics for debug
  final Duration? ragSearchTime;
  final Duration? llmGenerationTime;
  final Duration? totalTime;

  ChatMessage({
    required this.content,
    required this.isUser,
    DateTime? timestamp,
    this.retrievedChunks,
    this.tokensUsed,
    this.compressionRatio,
    this.originalTokens,
    this.queryType,
    this.ragSearchTime,
    this.llmGenerationTime,
    this.totalTime,
  }) : timestamp = timestamp ?? DateTime.now();
}
