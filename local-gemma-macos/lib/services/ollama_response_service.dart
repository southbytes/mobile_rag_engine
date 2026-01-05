// local-gemma-macos/lib/services/ollama_response_service.dart
//
// Service for generating LLM responses with RAG context
// Handles response mode selection (STRICT/HYBRID/FALLBACK) and prompt construction

import 'package:flutter/foundation.dart';
import 'package:mobile_rag_engine/mobile_rag_engine.dart';
import 'package:ollama_dart/ollama_dart.dart';

/// Response mode based on RAG similarity scores
enum ResponseMode {
  strict, // High similarity (>= 0.7): use only document context
  hybrid, // Medium similarity (>= 0.5): combine document + general knowledge
  fallback, // Low/no similarity: use general knowledge
}

/// Result of response generation
class OllamaResponseResult {
  final String response;
  final ResponseMode mode;
  final double bestSimilarity;

  const OllamaResponseResult({
    required this.response,
    required this.mode,
    required this.bestSimilarity,
  });
}

/// Service for generating Ollama LLM responses with RAG context
class OllamaResponseService {
  final OllamaClient ollamaClient;
  final String modelName;

  // Thresholds for response mode selection
  static const double hybridThreshold = 0.5;
  static const double strictThreshold = 0.7;

  OllamaResponseService({
    required this.ollamaClient,
    this.modelName = 'gemma3:4b',
  });

  /// Generate a response using RAG context
  /// Returns the response text and metadata
  Future<OllamaResponseResult> generateResponse({
    required String query,
    required String contextText,
    required RagSearchResult ragResult,
    required bool hasRelevantContext,
    required List<Message> chatHistory,
    void Function(Message)? onHistoryUpdate,
  }) async {
    // Calculate best similarity score for mode decision
    final bestSimilarity = _calculateBestSimilarity(ragResult);

    // Determine response mode
    final mode = _determineResponseMode(hasRelevantContext, bestSimilarity);

    debugPrint(
      '🎯 Response Mode: ${mode.name.toUpperCase()} '
      '(bestSim: ${bestSimilarity.toStringAsFixed(3)})',
    );

    try {
      // Build messages
      final messages = <Message>[];

      // 1. System Prompt - varies by mode
      messages.add(_buildSystemPrompt(mode));

      // 2. Chat History (last 6 messages)
      final historyStart = chatHistory.length > 6 ? chatHistory.length - 6 : 0;
      messages.addAll(chatHistory.sublist(historyStart));

      // 3. Current User Message (WITH RAG CONTEXT)
      final userMessage = _buildUserMessage(query, contextText, mode);
      messages.add(Message(role: MessageRole.user, content: userMessage));

      // Save raw query to history (not the huge context prompt)
      onHistoryUpdate?.call(Message(role: MessageRole.user, content: query));

      // Debug: Log prompt structure
      debugPrint('📨 === Prompt to LLM ===');
      debugPrint('📨 System: ${messages[0].content}');
      debugPrint('📨 History: ${chatHistory.length} messages');
      debugPrint('📨 User Query: $query');
      debugPrint('📨 Context Length: ${contextText.length} chars');
      debugPrint('📨 Mode: ${mode.name}');

      // Stream response from Ollama
      final responseBuffer = StringBuffer();
      final thinkingBuffer = StringBuffer();
      bool isInThinking = false;
      int chunkCount = 0;

      debugPrint('📝 === LLM Streaming Start ===');

      final stream = ollamaClient.generateChatCompletionStream(
        request: GenerateChatCompletionRequest(
          model: modelName,
          messages: messages,
        ),
      );

      await for (final chunk in stream) {
        final content = chunk.message.content;
        responseBuffer.write(content);
        chunkCount++;

        // Detect thinking/reasoning sections (some models use <think> tags)
        if (content.contains('<think>')) {
          isInThinking = true;
          debugPrint('🧠 [THINKING START]');
        }
        if (content.contains('</think>')) {
          isInThinking = false;
          debugPrint('🧠 [THINKING END]');
        }

        // Log chunk content
        if (isInThinking) {
          thinkingBuffer.write(content);
          // Print thinking chunks with special prefix
          final cleanContent = content.replaceAll('\n', '↵');
          debugPrint('🧠 $cleanContent');
        } else {
          // Print response chunks
          final cleanContent = content.replaceAll('\n', '↵');
          if (cleanContent.isNotEmpty) {
            // debugPrint('💬 $cleanContent');
          }
        }
      }

      debugPrint('📝 === LLM Streaming End ($chunkCount chunks) ===');

      // Log thinking summary if any
      if (thinkingBuffer.isNotEmpty) {
        debugPrint('🧠 === Thinking Summary ===');
        debugPrint(thinkingBuffer.toString());
        debugPrint('🧠 === End Thinking ===');
      }

      final response = responseBuffer.toString().trim();

      // Save assistant response to history
      onHistoryUpdate?.call(
        Message(role: MessageRole.assistant, content: response),
      );

      if (response.isEmpty) {
        return OllamaResponseResult(
          response:
              '⚠️ The model returned an empty response. Please try again.',
          mode: mode,
          bestSimilarity: bestSimilarity,
        );
      }

      return OllamaResponseResult(
        response: response,
        mode: mode,
        bestSimilarity: bestSimilarity,
      );
    } catch (e, stackTrace) {
      debugPrint('🔴 Ollama Error: $e');
      debugPrint('🔴 Stack Trace: $stackTrace');

      return OllamaResponseResult(
        response:
            '⚠️ Ollama Error: $e\n\n'
            'Make sure Ollama is running (ollama serve) and the model is installed.',
        mode: mode,
        bestSimilarity: bestSimilarity,
      );
    }
  }

  /// Calculate best similarity score from RAG results
  double _calculateBestSimilarity(RagSearchResult ragResult) {
    if (ragResult.chunks.isEmpty) return 0.0;

    return ragResult.chunks
        .map((c) => c.similarity)
        .where((s) => s > 0) // Exclude adjacent chunks with 0.0
        .fold(0.0, (a, b) => a > b ? a : b);
  }

  /// Determine response mode based on context and similarity
  ResponseMode _determineResponseMode(
    bool hasRelevantContext,
    double bestSimilarity,
  ) {
    if (!hasRelevantContext) return ResponseMode.fallback;
    if (bestSimilarity >= strictThreshold) return ResponseMode.strict;
    if (bestSimilarity >= hybridThreshold) return ResponseMode.hybrid;
    return ResponseMode.fallback;
  }

  /// Build system prompt based on response mode
  Message _buildSystemPrompt(ResponseMode mode) {
    switch (mode) {
      case ResponseMode.strict:
        return const Message(
          role: MessageRole.system,
          content:
              '당신은 제공된 문맥을 기반으로 정확하게 답변하는 AI 비서입니다. '
              '문맥에 있는 정보를 우선하여 답변하세요.',
        );
      case ResponseMode.hybrid:
        return const Message(
          role: MessageRole.system,
          content:
              '당신은 제공된 문맥과 일반 지식을 결합하여 답변하는 AI 비서입니다. '
              '문맥의 정보를 우선하되, 필요시 일반 지식으로 보완하세요. '
              '단, 문맥에서 온 정보와 일반 지식을 구분하여 설명하세요.',
        );
      case ResponseMode.fallback:
        return const Message(
          role: MessageRole.system,
          content: '당신은 도움이 되는 AI 비서입니다.',
        );
    }
  }

  /// Build user message with context based on response mode
  String _buildUserMessage(
    String query,
    String contextText,
    ResponseMode mode,
  ) {
    switch (mode) {
      case ResponseMode.strict:
        return '''
[참고 문서]
$contextText
[참고 문서 종료]

위 문서의 내용을 바탕으로 다음 질문에 답변하세요.

질문: $query''';

      case ResponseMode.hybrid:
        return '''
[관련 문서]
$contextText
[관련 문서 종료]

위 문서에 관련 내용이 있습니다. 문서 내용을 참고하여 답변하되, 
필요한 경우 일반적인 지식으로 보완해도 됩니다.
문서에서 직접 확인된 내용과 일반 지식을 구분해서 설명해 주세요.

질문: $query''';

      case ResponseMode.fallback:
        return '''
질문: $query

참고: 업로드된 문서에서 직접적으로 관련된 정보를 찾지 못했습니다.
일반적인 지식으로 답변하되, 더 정확한 정보가 필요하면 관련 문서를 추가해달라고 안내하세요.''';
    }
  }
}
