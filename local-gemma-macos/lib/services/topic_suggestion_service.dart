// local-gemma-macos/lib/services/topic_suggestion_service.dart
//
// Service for generating topic-based question suggestions from knowledge base
// Uses LLM to analyze sampled chunks and generate relevant questions
// Includes RAG-based validation to filter out unanswerable questions

import 'dart:convert';
import 'dart:math';

import 'package:flutter/foundation.dart';
import 'package:mobile_rag_engine/mobile_rag_engine.dart';
import 'package:ollama_dart/ollama_dart.dart';

/// A suggested question with its source topic and validation score
class SuggestedQuestion {
  final String question;
  final String topic;
  final double? validationScore; // Highest similarity from RAG validation

  const SuggestedQuestion({
    required this.question,
    required this.topic,
    this.validationScore,
  });

  @override
  String toString() =>
      'SuggestedQuestion(topic: $topic, question: $question, score: $validationScore)';
}

/// Service for generating topic-based question suggestions
class TopicSuggestionService {
  // Cached suggestions
  List<SuggestedQuestion>? _cachedSuggestions;
  DateTime? _cacheTime;

  // Cache duration (regenerate if older than 1 hour)
  static const Duration _cacheDuration = Duration(hours: 1);

  // Validation threshold: minimum similarity score for a valid question
  static const double _validationThreshold = 0.5;

  // Minimum number of high-quality chunks needed
  static const int _minValidChunks = 2;

  /// Get cached suggestions if available and not expired
  List<SuggestedQuestion>? getCachedSuggestions() {
    if (_cachedSuggestions == null || _cacheTime == null) {
      return null;
    }

    final now = DateTime.now();
    if (now.difference(_cacheTime!) > _cacheDuration) {
      return null; // Cache expired
    }

    return _cachedSuggestions;
  }

  /// Invalidate the cache (call when new documents are added)
  void invalidateCache() {
    _cachedSuggestions = null;
    _cacheTime = null;
    debugPrint('🔄 Topic suggestions cache invalidated');
  }

  /// Generate topic-based question suggestions from the knowledge base
  ///
  /// This method:
  /// 1. Fetches all chunks from the database
  /// 2. Samples a subset of chunks for analysis
  /// 3. Uses LLM to extract topics and generate relevant questions
  /// 4. Validates each question with RAG search to ensure answerability
  Future<List<SuggestedQuestion>> generateSuggestions({
    required SourceRagService ragService,
    required OllamaClient ollamaClient,
    String? modelName,
    int maxSuggestions = 3,
    int sampleSize = 15,
  }) async {
    // Return cached if available
    final cached = getCachedSuggestions();
    if (cached != null && cached.isNotEmpty) {
      debugPrint('📋 Returning ${cached.length} cached suggestions');
      return cached;
    }

    debugPrint('🔍 Generating topic suggestions...');
    final stopwatch = Stopwatch()..start();

    try {
      // 1. Get all chunks from database
      final stats = await ragService.getStats();
      if (stats.chunkCount == 0) {
        debugPrint('📭 No chunks in database, cannot generate suggestions');
        return [];
      }

      // 2. Get chunk contents for sampling
      final allChunks = await getAllChunkIdsAndContents(
        dbPath: ragService.dbPath,
      );

      if (allChunks.isEmpty) {
        debugPrint('📭 No chunk contents found');
        return [];
      }

      // 3. Sample chunks for topic analysis
      final sampledChunks = _sampleChunks(allChunks, sampleSize);
      final sampleText = sampledChunks
          .map((c) => c.content)
          .join('\n\n---\n\n');

      debugPrint('📊 Sampled ${sampledChunks.length} chunks for analysis');

      // 4. Generate MORE suggestions than needed (to have room after filtering)
      final candidateCount = maxSuggestions * 2; // Generate 2x to filter
      final candidates = await _generateWithLLM(
        ollamaClient: ollamaClient,
        modelName: modelName,
        sampleText: sampleText,
        maxSuggestions: candidateCount,
      );

      debugPrint('📝 Generated ${candidates.length} candidate questions');

      // 5. Validate each question with RAG search
      final validatedSuggestions = await _validateQuestions(
        candidates: candidates,
        ragService: ragService,
        maxValid: maxSuggestions,
      );

      stopwatch.stop();
      debugPrint(
        '✅ Validated ${validatedSuggestions.length}/${candidates.length} suggestions in ${stopwatch.elapsedMilliseconds}ms',
      );

      // 6. Cache results
      _cachedSuggestions = validatedSuggestions;
      _cacheTime = DateTime.now();

      return validatedSuggestions;
    } catch (e, stackTrace) {
      debugPrint('❌ Error generating suggestions: $e');
      debugPrint('Stack trace: $stackTrace');
      return [];
    }
  }

  /// Validate questions using RAG search to ensure they can be answered
  Future<List<SuggestedQuestion>> _validateQuestions({
    required List<SuggestedQuestion> candidates,
    required SourceRagService ragService,
    required int maxValid,
  }) async {
    final validatedList = <SuggestedQuestion>[];

    for (final candidate in candidates) {
      if (validatedList.length >= maxValid) break;

      try {
        // Run RAG search for the question
        final result = await ragService.search(
          candidate.question,
          topK: 5,
          tokenBudget: 500, // Small budget, we just need similarity scores
          adjacentChunks: 0, // No adjacent chunks for validation
        );

        // Count chunks with high similarity
        final highQualityChunks = result.chunks
            .where((c) => c.similarity >= _validationThreshold)
            .length;

        // Get best similarity score
        final bestScore = result.chunks.isNotEmpty
            ? result.chunks
                  .map((c) => c.similarity)
                  .reduce((a, b) => a > b ? a : b)
            : 0.0;

        debugPrint(
          '🔍 Validating: "${candidate.question.substring(0, min(50, candidate.question.length))}..." → '
          'bestScore: ${bestScore.toStringAsFixed(3)}, highQuality: $highQualityChunks',
        );

        // Question is valid if it has enough high-quality matches
        if (highQualityChunks >= _minValidChunks &&
            bestScore >= _validationThreshold) {
          validatedList.add(
            SuggestedQuestion(
              question: candidate.question,
              topic: candidate.topic,
              validationScore: bestScore,
            ),
          );
          debugPrint('  ✅ PASSED validation');
        } else {
          debugPrint(
            '  ❌ FAILED validation (need $highQualityChunks >= $_minValidChunks chunks with sim >= $_validationThreshold)',
          );
        }
      } catch (e) {
        debugPrint('  ⚠️ Validation error: $e');
      }
    }

    // Sort by validation score (highest first)
    validatedList.sort(
      (a, b) => (b.validationScore ?? 0).compareTo(a.validationScore ?? 0),
    );

    return validatedList;
  }

  /// Sample chunks randomly, preferring diverse content
  List<ChunkForReembedding> _sampleChunks(
    List<ChunkForReembedding> chunks,
    int sampleSize,
  ) {
    if (chunks.length <= sampleSize) {
      return chunks;
    }

    // Shuffle and take sample
    final shuffled = List<ChunkForReembedding>.from(chunks);
    shuffled.shuffle(Random());

    // Take sample, preferring longer chunks (more content)
    shuffled.sort((a, b) => b.content.length.compareTo(a.content.length));

    // Take top half by length, then shuffle again for diversity
    final topHalf = shuffled.take((shuffled.length / 2).ceil()).toList();
    topHalf.shuffle(Random());

    return topHalf.take(sampleSize).toList();
  }

  /// Use LLM to analyze sample text and generate questions
  Future<List<SuggestedQuestion>> _generateWithLLM({
    required OllamaClient ollamaClient,
    String? modelName,
    required String sampleText,
    required int maxSuggestions,
  }) async {
    // Truncate sample text if too long (keep within context window)
    final truncatedText = sampleText.length > 4000
        ? '${sampleText.substring(0, 4000)}...'
        : sampleText;

    // 🔍 DEBUG: Log sample text for analysis
    debugPrint('=' * 70);
    debugPrint(
      '📜 [DEBUG] SAMPLE TEXT SENT TO LLM (${truncatedText.length} chars):',
    );
    debugPrint('=' * 70);
    debugPrint(truncatedText);
    debugPrint('=' * 70);

    final prompt =
        '''아래는 지식 베이스에 저장된 문서들의 샘플입니다. 
이 내용을 분석하여 사용자가 물어볼 만한 유용한 질문 $maxSuggestions개를 생성해주세요.

규칙:
1. 질문은 반드시 아래 문서에서 직접적으로 답할 수 있는 내용이어야 합니다.
2. 문서에 명시적으로 언급된 사실, 데이터, 개념에 대해 질문하세요.
3. 문서에 없는 외부 정보를 필요로 하는 질문은 피하세요.
4. 질문은 간결하게 작성하세요 (1-2문장).
5. 질문은 한국어로 작성해주세요 (문서가 한국어인 경우).

문서 샘플:
$truncatedText

JSON 형식으로만 응답하세요:
[
  {"topic": "주제1", "question": "질문1?"},
  {"topic": "주제2", "question": "질문2?"}
]''';

    try {
      final response = await ollamaClient.generateCompletion(
        request: GenerateCompletionRequest(
          model: modelName ?? 'gemma3:4b',
          prompt: prompt,
          options: const RequestOptions(
            temperature: 0.5, // Lower temperature for more grounded questions
            numPredict: 800, // More tokens for more questions
          ),
        ),
      );

      final responseText = response.response?.trim() ?? '';
      debugPrint('🤖 LLM response: $responseText');

      final questions = _parseResponse(responseText);

      // 🔍 DEBUG: Analyze keyword matching between questions and sample text
      debugPrint('=' * 70);
      debugPrint('🔬 [DEBUG] KEYWORD MATCHING ANALYSIS:');
      debugPrint('=' * 70);
      final sampleLower = truncatedText.toLowerCase();
      for (final q in questions) {
        final questionWords = q.question
            .replaceAll(RegExp(r'[^\w가-힣]'), ' ')
            .split(RegExp(r'\s+'))
            .where((w) => w.length > 2)
            .toSet();

        final matches = <String>[];
        final misses = <String>[];
        for (final word in questionWords) {
          if (sampleLower.contains(word.toLowerCase())) {
            matches.add(word);
          } else {
            misses.add(word);
          }
        }

        debugPrint('📌 Q: "${q.question}"');
        debugPrint('   Topic: ${q.topic}');
        debugPrint('   ✅ In sample: ${matches.join(", ")}');
        debugPrint(
          '   ❌ NOT in sample: ${misses.isEmpty ? "(all matched)" : misses.join(", ")}',
        );
        if (misses.isNotEmpty) {
          debugPrint(
            '   ⚠️ HALLUCINATION DETECTED: ${misses.length}/${questionWords.length} keywords missing',
          );
        }
        debugPrint('');
      }
      debugPrint('=' * 70);

      return questions;
    } catch (e) {
      debugPrint('❌ LLM call failed: $e');
      return [];
    }
  }

  /// Parse LLM response JSON into SuggestedQuestion list
  List<SuggestedQuestion> _parseResponse(String responseText) {
    try {
      // Find JSON array in response
      final jsonMatch = RegExp(r'\[[\s\S]*\]').firstMatch(responseText);
      if (jsonMatch == null) {
        debugPrint('⚠️ No JSON array found in response');
        return [];
      }

      String jsonStr = jsonMatch.group(0)!;

      // Sanitize JSON string: remove control characters and fix smart quotes
      jsonStr = jsonStr
          .replaceAll(''', "'")  // Left single quote
          .replaceAll(''', "'") // Right single quote
          .replaceAll('"', '"') // Left double quote
          .replaceAll('"', '"') // Right double quote
          .replaceAll('–', '-') // En dash
          .replaceAll('—', '-') // Em dash
          .replaceAll(RegExp(r'[\x00-\x1F\x7F]'), ''); // Remove control chars

      final List<dynamic> jsonList = jsonDecode(jsonStr) as List<dynamic>;

      return jsonList
          .map((item) {
            final map = item as Map<String, dynamic>;
            return SuggestedQuestion(
              topic: (map['topic'] as String?) ?? '',
              question: (map['question'] as String?) ?? '',
            );
          })
          .where((q) => q.question.isNotEmpty)
          .toList();
    } catch (e) {
      debugPrint('⚠️ Failed to parse response: $e');
      return [];
    }
  }
}
