// local-gemma-macos/lib/services/query_understanding_service.dart
//
// Comprehensive query understanding service for RAG
// Analyzes user intent, validates queries, and normalizes for consistent results

import 'dart:convert';
import 'package:flutter/foundation.dart';
import 'package:ollama_dart/ollama_dart.dart';

/// Query type classification
enum QueryType {
  definition, // X란? X가 뭐야? or just "X" = asking for definition
  explanation, // 왜? 어떻게? = asking for explanation
  factual, // 언제? 어디서? 누가? 얼마? = factual query
  comparison, // A vs B, 차이점 = comparison
  listing, // 목록, 종류, 전체 = listing request
  summary, // 요약, 정리 = summary request
  opinion, // 생각? 의견? = opinion (may reject)
  greeting, // 인사, 안녕 = greeting (reject)
  unclear, // 모호한 입력 (reject)
  unknown, // Cannot determine
}

/// Result of query understanding analysis
class QueryUnderstanding {
  final bool isValid;
  final QueryType type;
  final String originalQuery;
  final String normalizedQuery;
  final List<String> keywords;
  final String? implicitIntent;
  final double confidence;
  final String? rejectionReason;

  const QueryUnderstanding({
    required this.isValid,
    required this.type,
    required this.originalQuery,
    required this.normalizedQuery,
    required this.keywords,
    this.implicitIntent,
    required this.confidence,
    this.rejectionReason,
  });

  /// Create an invalid/rejected query understanding
  factory QueryUnderstanding.invalid(String originalQuery, String reason) {
    return QueryUnderstanding(
      isValid: false,
      type: QueryType.unknown,
      originalQuery: originalQuery,
      normalizedQuery: '',
      keywords: [],
      confidence: 0.0,
      rejectionReason: reason,
    );
  }

  @override
  String toString() =>
      'QueryUnderstanding(valid: $isValid, type: ${type.name}, '
      'normalized: "$normalizedQuery", confidence: ${confidence.toStringAsFixed(2)}, '
      'keywords: $keywords)';
}

/// Service for understanding and analyzing user queries
class QueryUnderstandingService {
  final OllamaClient ollamaClient;
  final String? modelName;

  QueryUnderstandingService({required this.ollamaClient, this.modelName});

  /// Analyze a user query to understand intent and validate
  Future<QueryUnderstanding> analyze(String query) async {
    final trimmedQuery = query.trim();

    // === Stage 1: Basic validity check ===
    final basicCheck = _basicValidityCheck(trimmedQuery);
    if (basicCheck != null) {
      debugPrint(
        '🚫 Query rejected (basic check): ${basicCheck.rejectionReason}',
      );
      return basicCheck;
    }

    // === Stage 2: LLM-based deep analysis ===
    final stopwatch = Stopwatch()..start();
    final llmResult = await _analyzeWithLLM(trimmedQuery);
    stopwatch.stop();

    debugPrint('🧠 Query Understanding (${stopwatch.elapsedMilliseconds}ms):');
    debugPrint('   Original: "$trimmedQuery"');
    debugPrint('   Type: ${llmResult.type.name}');
    debugPrint('   Normalized: "${llmResult.normalizedQuery}"');
    debugPrint('   Implicit Intent: ${llmResult.implicitIntent}');
    debugPrint('   Keywords: ${llmResult.keywords}');
    debugPrint('   Confidence: ${llmResult.confidence.toStringAsFixed(2)}');

    // === Stage 3: Confidence threshold check ===
    if (llmResult.confidence < 0.4) {
      return QueryUnderstanding.invalid(
        trimmedQuery,
        '질문을 이해하지 못했습니다. 더 구체적으로 질문해주세요.',
      );
    }

    return llmResult;
  }

  /// Basic validity check before LLM analysis
  QueryUnderstanding? _basicValidityCheck(String query) {
    // Empty or too short
    if (query.isEmpty) {
      return QueryUnderstanding.invalid(query, '질문을 입력해주세요.');
    }

    if (query.length < 2) {
      return QueryUnderstanding.invalid(query, '너무 짧은 입력입니다.');
    }

    // Only special characters or punctuation
    final onlySpecialChars = RegExp(r'^[\s\p{P}\p{S}]+$', unicode: true);
    if (onlySpecialChars.hasMatch(query)) {
      return QueryUnderstanding.invalid(query, '의미 있는 질문을 입력해주세요.');
    }

    // Only numbers
    final onlyNumbers = RegExp(r'^[\d\s.,]+$');
    if (onlyNumbers.hasMatch(query)) {
      return QueryUnderstanding.invalid(query, '숫자만으로는 질문을 이해할 수 없습니다.');
    }

    return null; // Passed basic check
  }

  /// Use LLM to deeply analyze the query
  Future<QueryUnderstanding> _analyzeWithLLM(String query) async {
    final prompt =
        '''사용자 입력을 분석하세요. JSON으로만 응답하세요.

입력: "$query"

분석 규칙:
1. normalized_query와 keywords는 반드시 입력에 포함된 단어만 사용하세요.
2. 입력에 없는 새로운 단어를 만들어내지 마세요.
3. 의미 없거나 모호한 입력은 is_valid: false로 처리하세요.

분석 항목:
1. is_valid: 의미 있는 질문/요청인가? (true/false)
   - "모르겠네", "뭐지", "음..." 같은 모호한 표현 → false
   - 명확한 주제나 키워드가 있는 경우 → true
2. query_type: 질문 유형
   - "definition": 정의/의미 요청 (예: "X란?", "X가 뭐야?", 또는 용어 단독 입력)
   - "explanation": 이유/방법 설명 (예: "왜?", "어떻게?")
   - "factual": 사실 확인 (예: "언제?", "얼마?")
   - "comparison": 비교 (예: "A vs B")
   - "listing": 목록/나열 (예: "종류", "목록")
   - "summary": 요약 요청 (예: "요약해줘")
   - "greeting": 인사 (예: "안녕")
   - "unclear": 의도 파악 불가
3. implicit_intent: 추론된 의도 (입력 단어 기반으로만 작성)
4. normalized_query: 검색용 쿼리 (입력에 있는 단어만 사용!)
5. keywords: 입력에서 추출한 핵심 키워드 (배열, 입력에 있는 단어만!)
6. confidence: 분석 신뢰도 (0.0-1.0)
   - 모호한 입력: 0.3 이하
   - 명확한 입력: 0.7 이상

JSON:
{
  "is_valid": true/false,
  "query_type": "...",
  "implicit_intent": "...",
  "normalized_query": "입력에 있는 단어만",
  "keywords": ["입력에", "있는", "단어만"],
  "confidence": 0.0-1.0
}''';

    try {
      final response = await ollamaClient.generateCompletion(
        request: GenerateCompletionRequest(
          model: modelName ?? 'gemma3:4b',
          prompt: prompt,
          options: const RequestOptions(temperature: 0.0, numPredict: 200),
        ),
      );

      final responseText = response.response?.trim() ?? '';
      debugPrint('🤖 LLM Analysis Response: $responseText');

      return _parseLLMResponse(query, responseText);
    } catch (e) {
      debugPrint('❌ LLM analysis failed: $e');
      // Fallback: treat as simple query
      return QueryUnderstanding(
        isValid: true,
        type: QueryType.definition,
        originalQuery: query,
        normalizedQuery: query,
        keywords: [query],
        confidence: 0.5,
      );
    }
  }

  /// Parse LLM response into QueryUnderstanding
  QueryUnderstanding _parseLLMResponse(
    String originalQuery,
    String responseText,
  ) {
    try {
      // Extract JSON from response
      final jsonMatch = RegExp(r'\{[\s\S]*\}').firstMatch(responseText);
      if (jsonMatch == null) {
        debugPrint('⚠️ No JSON found in LLM response');
        return _fallbackUnderstanding(originalQuery);
      }

      // Sanitize and parse JSON
      String jsonStr = jsonMatch.group(0)!;
      jsonStr = jsonStr
          .replaceAll(''', "'")
          .replaceAll(''', "'")
          .replaceAll('"', '"')
          .replaceAll('"', '"')
          .replaceAll(RegExp(r'[\x00-\x1F\x7F]'), '');

      final json = jsonDecode(jsonStr) as Map<String, dynamic>;

      final isValid = json['is_valid'] as bool? ?? true;
      final queryTypeStr =
          (json['query_type'] as String?)?.toLowerCase() ?? 'unknown';
      final implicitIntent = json['implicit_intent'] as String?;
      final normalizedQuery =
          (json['normalized_query'] as String?) ?? originalQuery;
      final keywordsList = json['keywords'] as List<dynamic>? ?? [];
      final confidence = (json['confidence'] as num?)?.toDouble() ?? 0.5;

      // Map query type string to enum
      final queryType = switch (queryTypeStr) {
        'definition' => QueryType.definition,
        'explanation' => QueryType.explanation,
        'factual' => QueryType.factual,
        'comparison' => QueryType.comparison,
        'listing' => QueryType.listing,
        'summary' => QueryType.summary,
        'greeting' => QueryType.greeting,
        'opinion' => QueryType.opinion,
        'unclear' => QueryType.unclear,
        _ => QueryType.unknown,
      };

      // Reject greetings and unclear queries
      if (queryType == QueryType.greeting) {
        return QueryUnderstanding.invalid(
          originalQuery,
          '안녕하세요! 문서에 대해 궁금한 점을 질문해주세요.',
        );
      }

      if (queryType == QueryType.unclear || queryType == QueryType.unknown) {
        return QueryUnderstanding.invalid(
          originalQuery,
          '질문을 이해하지 못했습니다. 더 구체적으로 질문해주세요.',
        );
      }

      if (!isValid) {
        return QueryUnderstanding.invalid(
          originalQuery,
          '질문을 이해하지 못했습니다. 다시 표현해주세요.',
        );
      }

      return QueryUnderstanding(
        isValid: true,
        type: queryType,
        originalQuery: originalQuery,
        normalizedQuery: normalizedQuery,
        keywords: keywordsList.map((k) => k.toString()).toList(),
        implicitIntent: implicitIntent,
        confidence: confidence,
      );
    } catch (e) {
      debugPrint('⚠️ Failed to parse LLM response: $e');
      return _fallbackUnderstanding(originalQuery);
    }
  }

  /// Fallback when LLM parsing fails
  QueryUnderstanding _fallbackUnderstanding(String query) {
    return QueryUnderstanding(
      isValid: true,
      type: QueryType.definition,
      originalQuery: query,
      normalizedQuery: query,
      keywords: _extractSimpleKeywords(query),
      confidence: 0.5,
    );
  }

  /// Simple keyword extraction fallback
  List<String> _extractSimpleKeywords(String query) {
    // Remove common Korean particles and question words
    final cleaned = query
        .replaceAll(RegExp(r'[은는이가을를에서로의와과란]'), ' ')
        .replaceAll(RegExp(r'[?？!！.,。，]'), ' ')
        .replaceAll(RegExp(r'\s+'), ' ')
        .trim();

    return cleaned.split(' ').where((w) => w.length > 1).take(5).toList();
  }
}
