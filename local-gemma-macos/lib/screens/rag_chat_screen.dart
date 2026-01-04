// local-gemma-macos/lib/screens/rag_chat_screen.dart
//
// RAG + Ollama Chat Screen
// Uses mobile_rag_engine for retrieval and ollama_dart for generation

import 'dart:io';
import 'package:flutter/material.dart';
import 'package:flutter/services.dart';
import 'package:path_provider/path_provider.dart';

import 'package:mobile_rag_engine/mobile_rag_engine.dart';
import 'package:ollama_dart/ollama_dart.dart';
import 'dart:convert';

import '../services/topic_suggestion_service.dart';
import '../models/chat_models.dart';
import '../widgets/knowledge_graph_panel.dart';
import '../widgets/chunk_detail_sidebar.dart';
import '../widgets/suggestion_chips.dart';
import '../widgets/chat_input_area.dart';
import '../widgets/document_style_response.dart';

// Models are now in models/chat_models.dart

class RagChatScreen extends StatefulWidget {
  final bool mockLlm;
  final String? modelName;

  const RagChatScreen({super.key, this.mockLlm = false, this.modelName});

  @override
  State<RagChatScreen> createState() => _RagChatScreenState();
}

class _RagChatScreenState extends State<RagChatScreen> {
  final TextEditingController _messageController = TextEditingController();
  final ScrollController _scrollController = ScrollController();
  final FocusNode _focusNode = FocusNode();

  final List<ChatMessage> _messages = [];

  SourceRagService? _ragService;
  bool _isInitialized = false;
  bool _isLoading = false;
  bool _isGenerating = false;
  String _status = 'Initializing...';

  // Ollama client and chat history
  final OllamaClient _ollamaClient = OllamaClient();
  final List<Message> _chatHistory = [];

  // Debug info
  bool _showDebugInfo = true;
  int _totalChunks = 0;
  int _totalSources = 0;

  // Compression settings (Phase 1)
  int _compressionLevel = 1; // 0=minimal, 1=balanced, 2=aggressive

  // Similarity threshold for RAG
  final double _minSimilarityThreshold = 0.35;

  // Topic suggestion service
  final TopicSuggestionService _topicService = TopicSuggestionService();
  List<SuggestedQuestion> _suggestedQuestions = [];
  bool _isLoadingSuggestions = false;

  // UI state for collapsible sections
  bool _isSuggestionsExpanded = true; // Collapsible suggestions panel
  final Set<int> _expandedSourceMessages =
      {}; // Track which messages have sources expanded

  // Knowledge Graph panel state
  bool _showGraphPanel = true; // Toggle graph panel visibility
  ChunkSearchResult? _selectedChunk; // Currently selected chunk in graph
  String? _lastQuery; // Last query for graph display
  List<ChunkSearchResult> _lastChunks = []; // Last chunks for graph display
  int? _activeGraphMessageIndex; // Track which message's chunks are in graph

  /// Calculate optimal adjacent chunks based on query characteristics.
  ///
  /// Returns a higher value for queries that need more context (summaries,
  /// definitions) and lower for specific short queries.
  int _calculateAdjacentChunks(String query) {
    final queryLength = query.length;
    final queryLower = query.toLowerCase();

    // Keywords indicating need for broader context
    final broadContextKeywords = ['전체', '요약', '정의', '설명', '개요', '모든', '전반'];
    final narrowContextKeywords = ['뭐야', '뭔가요', '무엇', '어디', '누가', '언제'];

    // Check for broad context keywords → more adjacent chunks
    for (final keyword in broadContextKeywords) {
      if (queryLower.contains(keyword)) {
        return 4; // 넓은 문맥 필요
      }
    }

    // Short specific queries → narrower context
    if (queryLength < 15) {
      // Check if it's a simple lookup question
      for (final keyword in narrowContextKeywords) {
        if (queryLower.contains(keyword)) {
          return 1; // 짧고 직접적인 질문
        }
      }
      return 2; // 짧지만 일반적인 질문
    }

    // Long queries with specific terms → medium context
    if (queryLength > 50) {
      return 2; // 긴 질문은 이미 충분한 맥락 포함
    }

    // Default: balanced context
    return 2;
  }

  /// Analyze query intent using LLM to get optimal RAG parameters.
  /// Returns a QueryAnalysis with intent type and tuned parameters.
  Future<QueryAnalysis> _analyzeQueryIntent(String query) async {
    try {
      final intentStopwatch = Stopwatch()..start();

      // Quick LLM call to classify intent and refine query
      final response = await _ollamaClient.generateCompletion(
        request: GenerateCompletionRequest(
          model: widget.modelName ?? 'gemma3:4b',
          prompt:
              '''사용자 질문을 분석하여 JSON으로만 응답하세요. 다른 텍스트 없이 JSON만 출력하세요.

질문: "$query"

분석 기준:
- intent: "summary" (요약, 정리, 핵심), "definition" (정의, ~란, 뜻), "broad" (전체, 모든, 목록), "detail" (자세히, 왜, 어떻게), "general" (기타)
- search_query: 검색에 사용할 핵심 키워드 (조사, 질문 형식 제거)

JSON 형식:
{"intent": "...", "search_query": "..."}''',
          options: RequestOptions(
            temperature: 0.0, // Deterministic for classification
            numPredict: 100, // Short response expected
          ),
        ),
      );

      intentStopwatch.stop();
      final responseText = response.response?.trim() ?? '';
      debugPrint(
        '🧠 Intent analysis (${intentStopwatch.elapsedMilliseconds}ms): $responseText',
      );

      // Parse JSON response
      final jsonMatch = RegExp(r'\{[^}]+\}').firstMatch(responseText);
      if (jsonMatch != null) {
        final json = jsonDecode(jsonMatch.group(0)!) as Map<String, dynamic>;
        final intentStr =
            (json['intent'] as String?)?.toLowerCase() ?? 'general';
        final searchQuery = (json['search_query'] as String?) ?? query;

        // Map intent string to enum and parameters
        final (intent, adjacent, budget, topK) = switch (intentStr) {
          'summary' => (QueryIntent.summary, 1, 1500, 5),
          'definition' => (QueryIntent.definition, 1, 1000, 5),
          'broad' => (QueryIntent.broad, 3, 4000, 15),
          'detail' => (QueryIntent.detail, 2, 2500, 10),
          _ => (QueryIntent.general, 2, 2000, 10),
        };

        final analysis = QueryAnalysis(
          intent: intent,
          adjacentChunks: adjacent,
          tokenBudget: budget,
          topK: topK,
          refinedQuery: searchQuery,
        );

        debugPrint('📊 Query Analysis: $analysis');
        return analysis;
      }
    } catch (e) {
      debugPrint('⚠️ Intent analysis failed: $e');
    }

    // Fallback to default analysis
    return QueryAnalysis.defaultFor(query);
  }

  @override
  void initState() {
    super.initState();
    _initialize();
  }

  Future<void> _initialize() async {
    setState(() {
      _isLoading = true;
      _status = 'Initializing RAG engine...';
    });

    try {
      final dir = await getApplicationDocumentsDirectory();
      final dbPath = '${dir.path}/local_gemma_rag.db';
      final tokenizerPath = '${dir.path}/tokenizer.json';

      // 1. Copy BGE-m3 tokenizer
      await _copyAsset('assets/bge-m3-tokenizer.json', tokenizerPath);
      await initTokenizer(tokenizerPath: tokenizerPath);

      // 2. Load BGE-m3 ONNX model (int8 quantized, 1024 dim output)
      setState(() => _status = 'Loading BGE-m3 embedding model...');
      final modelBytes = await rootBundle.load('assets/bge-m3-int8.onnx');
      await EmbeddingService.init(modelBytes.buffer.asUint8List());

      // 3. Initialize RAG service
      _ragService = SourceRagService(
        dbPath: dbPath,
        maxChunkChars: 500,
        overlapChars: 50,
      );
      await _ragService!.init();

      // Get stats
      final stats = await _ragService!.getStats();
      _totalSources = stats.sourceCount.toInt();
      _totalChunks = stats.chunkCount.toInt();

      setState(() {
        _isInitialized = true;
        _isLoading = false;
        _status = 'Ready! Sources: $_totalSources, Chunks: $_totalChunks';
      });

      // Add welcome message
      _addSystemMessage(
        'Welcome! I can answer questions based on the documents you add.\n\n'
        '• Use the 📎 button to add documents\n'
        '• Ask me questions about the documents\n'
        '• ${widget.mockLlm ? "(Mock mode - no LLM)" : "Using Ollama: ${widget.modelName ?? 'default'}"}',
      );

      // Generate topic suggestions if we have documents
      if (_totalChunks > 0 && !widget.mockLlm) {
        _generateTopicSuggestions();
      }
    } catch (e) {
      setState(() {
        _isLoading = false;
        _status = 'Error: $e';
      });
    }
  }

  Future<void> _copyAsset(String assetPath, String targetPath) async {
    final file = File(targetPath);
    if (!await file.exists()) {
      final data = await rootBundle.load(assetPath);
      await file.writeAsBytes(data.buffer.asUint8List());
    }
  }

  void _addSystemMessage(String content) {
    setState(() {
      _messages.insert(0, ChatMessage(content: content, isUser: false));
    });
  }

  /// Generate topic suggestions from knowledge base
  Future<void> _generateTopicSuggestions() async {
    if (_ragService == null || widget.mockLlm) return;

    setState(() => _isLoadingSuggestions = true);

    try {
      final suggestions = await _topicService.generateSuggestions(
        ragService: _ragService!,
        ollamaClient: _ollamaClient,
        modelName: widget.modelName,
        maxSuggestions: 3,
      );

      if (mounted) {
        setState(() {
          _suggestedQuestions = suggestions;
          _isLoadingSuggestions = false;
        });
      }
    } catch (e) {
      debugPrint('❌ Topic suggestion error: $e');
      if (mounted) {
        setState(() => _isLoadingSuggestions = false);
      }
    }
  }

  /// Send a suggested question as user message
  void _sendSuggestedQuestion(SuggestedQuestion question) {
    // Remove clicked question from list
    setState(() {
      _suggestedQuestions.remove(question);
      _isSuggestionsExpanded = false; // Collapse panel after clicking
    });
    _messageController.text = question.question;
    _sendMessage();

    // Auto-regenerate suggestions when all are used up
    if (_suggestedQuestions.isEmpty && !widget.mockLlm) {
      _topicService.invalidateCache();
      _generateTopicSuggestions();
    }
  }

  Future<void> _sendMessage() async {
    final text = _messageController.text.trim();
    if (text.isEmpty || !_isInitialized || _isGenerating) return;

    _messageController.clear();
    _focusNode.unfocus();

    // Add user message
    setState(() {
      _messages.insert(0, ChatMessage(content: text, isUser: true));
      _isGenerating = true;
    });

    try {
      final totalStopwatch = Stopwatch()..start();

      // 1. Intent Analysis (LLM 1차 호출)
      final intentStopwatch = Stopwatch()..start();
      final queryAnalysis = await _analyzeQueryIntent(text);
      intentStopwatch.stop();
      debugPrint(
        '⏱️ Intent analysis: ${intentStopwatch.elapsedMilliseconds}ms',
      );

      // 2. RAG Search with analyzed parameters
      final ragStopwatch = Stopwatch()..start();
      debugPrint(
        '📐 Using: intent=${queryAnalysis.intent.name}, adjacent=${queryAnalysis.adjacentChunks}, budget=${queryAnalysis.tokenBudget}, topK=${queryAnalysis.topK}',
      );
      debugPrint('🔎 Refined query: "${queryAnalysis.refinedQuery}"');

      final ragResult = await _ragService!.search(
        queryAnalysis.refinedQuery, // LLM이 정제한 검색어 사용
        topK: queryAnalysis.topK,
        tokenBudget: queryAnalysis.tokenBudget,
        strategy: ContextStrategy.relevanceFirst,
        adjacentChunks: queryAnalysis.adjacentChunks,
        singleSourceMode: false, // Changed: search across all sources
      );
      ragStopwatch.stop();
      final ragSearchTime = ragStopwatch.elapsed;

      // DEBUG: Log BGE-m3 search results
      debugPrint('🔍 BGE-m3 search for: "$text"');
      debugPrint('   Found ${ragResult.chunks.length} chunks');
      for (var i = 0; i < ragResult.chunks.length && i < 5; i++) {
        final c = ragResult.chunks[i];
        final preview = c.content.length > 50
            ? '${c.content.substring(0, 50)}...'
            : c.content;
        debugPrint('   [$i] sim=${c.similarity.toStringAsFixed(3)}: $preview');
      }

      // Filter low similarity chunks
      // We allow similarity of 0.0 because those are "adjacent chunks" (neighbors)
      // added for context, which don't have a computed similarity score.
      final relevantChunks = ragResult.chunks
          .where(
            (c) =>
                c.similarity >= _minSimilarityThreshold || c.similarity == 0.0,
          )
          .toList();

      if (relevantChunks.length < ragResult.chunks.length) {
        debugPrint(
          '   🧹 Filtered ${ragResult.chunks.length - relevantChunks.length} low similarity chunks (<$_minSimilarityThreshold)',
        );
      }

      // Use RAG context directly (no compression)
      // Check if we have ANY relevant chunks after filtering
      final bool hasRelevantContext = relevantChunks.isNotEmpty;

      final contextText = ragResult
          .context
          .text; // Still using full context for now, but prompt will handle "no info"
      final estimatedTokens = ragResult.context.estimatedTokens;
      final chunkCount = ragResult.chunks.length;

      debugPrint(
        '📊 RAG Context: $estimatedTokens tokens, $chunkCount chunks (Relevant: ${relevantChunks.length})',
      );

      // 3. LLM Generation with timing
      final llmStopwatch = Stopwatch()..start();
      String response;
      if (widget.mockLlm) {
        // Mock mode
        response = _generateMockResponse(text, ragResult, '0');
      } else {
        // Real LLM generation with Ollama - use RAG context directly
        response = await _generateOllamaResponse(
          text,
          hasRelevantContext ? contextText : '',
          ragResult,
          hasRelevantContext,
        );
      }
      llmStopwatch.stop();
      final llmGenerationTime = llmStopwatch.elapsed;

      totalStopwatch.stop();

      // Add AI response with stats
      setState(() {
        // Update graph panel data
        _lastQuery = text;
        _lastChunks = ragResult.chunks;

        _messages.insert(
          0,
          ChatMessage(
            content: response,
            isUser: false,
            retrievedChunks: ragResult.chunks,
            tokensUsed: estimatedTokens,
            ragSearchTime: ragSearchTime,
            llmGenerationTime: llmGenerationTime,
            totalTime: totalStopwatch.elapsed,
          ),
        );
      });
    } catch (e) {
      setState(() {
        _messages.insert(0, ChatMessage(content: '❌ Error: $e', isUser: false));
      });
    } finally {
      setState(() => _isGenerating = false);
    }

    _scrollToBottom();
  }

  String _generateMockResponse(
    String query,
    RagSearchResult ragResult,
    String savedPercent,
  ) {
    if (ragResult.chunks.isEmpty) {
      return '📭 No relevant documents found.\n\nPlease add some documents using the menu.';
    }

    final buffer = StringBuffer();
    buffer.writeln('📚 Found ${ragResult.chunks.length} relevant chunks:');
    buffer.writeln('📊 Using ~${ragResult.context.estimatedTokens} tokens');
    buffer.writeln('🗜️ Reduced by $savedPercent%\n');

    for (var i = 0; i < ragResult.chunks.length && i < 3; i++) {
      final chunk = ragResult.chunks[i];
      final preview = chunk.content.length > 100
          ? '${chunk.content.substring(0, 100)}...'
          : chunk.content;
      buffer.writeln('${i + 1}. $preview\n');
    }

    buffer.writeln('---');
    buffer.writeln(
      '💡 This is a mock response. Install an LLM model for real answers.',
    );

    return buffer.toString();
  }

  /// Generate response using Ollama with pre-compressed context text
  Future<String> _generateOllamaResponseWithCompressedText(
    String query,
    String compressedContext,
    RagSearchResult ragResult,
  ) async {
    try {
      // Build messages with compressed context
      final messages = <Message>[];

      // System message with compressed RAG context
      if (compressedContext.isNotEmpty) {
        messages.add(
          Message(
            role: MessageRole.system,
            content: '''당신은 주어진 문맥을 기반으로 질문에 답변하는 도우미입니다.

규칙:
1. 아래 문맥의 정보만을 기반으로 답변하세요.
2. 문맥에 관련 정보가 없으면 "문서에서 해당 정보를 찾을 수 없습니다"라고 답변하세요.
3. 질문과 동일한 언어로 답변하세요.
4. 조항 번호(제X조)는 문맥에 있는 그대로 인용하세요.

문맥:
$compressedContext''',
          ),
        );
      } else {
        messages.add(
          const Message(
            role: MessageRole.system,
            content: 'You are a helpful assistant.',
          ),
        );
      }

      // Add current user message
      messages.add(Message(role: MessageRole.user, content: query));

      // Save to history
      _chatHistory.add(Message(role: MessageRole.user, content: query));

      // Stream response from Ollama
      final responseBuffer = StringBuffer();

      final stream = _ollamaClient.generateChatCompletionStream(
        request: GenerateChatCompletionRequest(
          model: widget.modelName ?? 'gemma3:4b',
          messages: messages,
        ),
      );

      await for (final chunk in stream) {
        responseBuffer.write(chunk.message.content);
      }

      final response = responseBuffer.toString().trim();

      // Save assistant response to history
      _chatHistory.add(Message(role: MessageRole.assistant, content: response));

      if (response.isEmpty) {
        return '⚠️ The model returned an empty response. Please try again.';
      }

      return response;
    } catch (e, stackTrace) {
      debugPrint('🔴 Ollama Error: $e');
      debugPrint('🔴 Stack Trace: $stackTrace');

      return '⚠️ Ollama Error: $e\n\n'
          'Make sure Ollama is running (ollama serve) and the model is installed.';
    }
  }

  Future<String> _generateOllamaResponse(
    String query,
    String contextText,
    RagSearchResult ragResult,
    bool hasRelevantContext,
  ) async {
    try {
      // Build messages
      final messages = <Message>[];

      // Calculate best similarity score for hybrid mode decision
      final bestSimilarity = ragResult.chunks.isNotEmpty
          ? ragResult.chunks
                .map((c) => c.similarity)
                .where((s) => s > 0) // Exclude adjacent chunks with 0.0
                .fold(0.0, (a, b) => a > b ? a : b)
          : 0.0;

      // Hybrid Mode Thresholds
      const double hybridThreshold = 0.5; // Use hybrid mode if sim >= 0.5
      const double strictThreshold = 0.7; // Use strict mode if sim >= 0.7

      final bool useHybridMode =
          hasRelevantContext && bestSimilarity >= hybridThreshold;
      final bool useStrictMode =
          hasRelevantContext && bestSimilarity >= strictThreshold;

      debugPrint(
        '🎯 Response Mode: ${useStrictMode
            ? "STRICT"
            : useHybridMode
            ? "HYBRID"
            : "FALLBACK"} '
        '(bestSim: ${bestSimilarity.toStringAsFixed(3)})',
      );

      // 1. System Prompt - varies by mode
      if (useStrictMode) {
        // High similarity: strict document-only mode
        messages.add(
          const Message(
            role: MessageRole.system,
            content:
                '당신은 제공된 문맥을 기반으로 정확하게 답변하는 AI 비서입니다. '
                '문맥에 있는 정보를 우선하여 답변하세요.',
          ),
        );
      } else if (useHybridMode) {
        // Medium similarity: hybrid mode (document + Gemma knowledge)
        messages.add(
          const Message(
            role: MessageRole.system,
            content:
                '당신은 제공된 문맥과 일반 지식을 결합하여 답변하는 AI 비서입니다. '
                '문맥의 정보를 우선하되, 필요시 일반 지식으로 보완하세요. '
                '단, 문맥에서 온 정보와 일반 지식을 구분하여 설명하세요.',
          ),
        );
      } else {
        // No relevant context
        messages.add(
          const Message(
            role: MessageRole.system,
            content: '당신은 도움이 되는 AI 비서입니다.',
          ),
        );
      }

      // 2. Chat History (last 6 messages)
      final historyStart = _chatHistory.length > 6
          ? _chatHistory.length - 6
          : 0;
      messages.addAll(_chatHistory.sublist(historyStart));

      // 3. Current User Message (WITH RAG CONTEXT)
      String finalUserContent;

      if (useStrictMode) {
        // High similarity: strict mode
        finalUserContent =
            '''
[참고 문서]
$contextText
[참고 문서 종료]

위 문서의 내용을 바탕으로 다음 질문에 답변하세요.

질문: $query''';
      } else if (useHybridMode) {
        // Medium similarity: hybrid mode (NotebookLM style)
        finalUserContent =
            '''
[관련 문서]
$contextText
[관련 문서 종료]

위 문서에 관련 내용이 있습니다. 문서 내용을 참고하여 답변하되, 
필요한 경우 일반적인 지식으로 보완해도 됩니다.
문서에서 직접 확인된 내용과 일반 지식을 구분해서 설명해 주세요.

질문: $query''';
      } else {
        // No relevant context: general knowledge mode
        finalUserContent =
            '''
질문: $query

참고: 업로드된 문서에서 직접적으로 관련된 정보를 찾지 못했습니다.
일반적인 지식으로 답변하되, 더 정확한 정보가 필요하면 관련 문서를 추가해달라고 안내하세요.''';
      }

      messages.add(Message(role: MessageRole.user, content: finalUserContent));

      // Save raw query to history (not the huge context prompt)
      _chatHistory.add(Message(role: MessageRole.user, content: query));

      // Stream response from Ollama
      final responseBuffer = StringBuffer();

      final stream = _ollamaClient.generateChatCompletionStream(
        request: GenerateChatCompletionRequest(
          model: widget.modelName ?? 'gemma3:4b',
          messages: messages,
        ),
      );

      await for (final chunk in stream) {
        responseBuffer.write(chunk.message.content);
      }

      final response = responseBuffer.toString().trim();

      // Save assistant response to history
      _chatHistory.add(Message(role: MessageRole.assistant, content: response));

      if (response.isEmpty) {
        return '⚠️ The model returned an empty response. Please try again.';
      }

      return response;
    } catch (e, stackTrace) {
      debugPrint('🔴 Ollama Error: $e');
      debugPrint('🔴 Stack Trace: $stackTrace');

      return '⚠️ Ollama Error: $e\n\n'
          'Make sure Ollama is running (ollama serve) and the model is installed.';
    }
  }

  /// Start a new chat session - clears messages and chat history
  Future<void> _startNewChat() async {
    setState(() {
      _isLoading = true;
      _status = 'Starting new chat...';
    });

    _chatHistory.clear();

    setState(() {
      _messages.clear();
      _isLoading = false;
      _status = 'Ready! Sources: $_totalSources, Chunks: $_totalChunks';
    });

    _addSystemMessage(
      '🔄 New chat started! Chat history has been cleared.\n\n'
      '• Ask me questions about your documents',
    );
  }

  /// Parse simple markdown (bold **text**) into TextSpan
  TextSpan _parseMarkdown(String text) {
    final spans = <TextSpan>[];
    final regex = RegExp(r'\*\*(.+?)\*\*');
    int lastEnd = 0;

    for (final match in regex.allMatches(text)) {
      // Add text before the match
      if (match.start > lastEnd) {
        spans.add(TextSpan(text: text.substring(lastEnd, match.start)));
      }
      // Add bold text
      spans.add(
        TextSpan(
          text: match.group(1),
          style: const TextStyle(fontWeight: FontWeight.bold),
        ),
      );
      lastEnd = match.end;
    }

    // Add remaining text
    if (lastEnd < text.length) {
      spans.add(TextSpan(text: text.substring(lastEnd)));
    }

    // If no matches, return plain text
    if (spans.isEmpty) {
      spans.add(TextSpan(text: text));
    }

    return TextSpan(children: spans);
  }

  void _scrollToBottom() {
    WidgetsBinding.instance.addPostFrameCallback((_) {
      if (_scrollController.hasClients) {
        _scrollController.animateTo(
          0,
          duration: const Duration(milliseconds: 300),
          curve: Curves.easeOut,
        );
      }
    });
  }

  Future<void> _addSampleDocuments() async {
    if (!_isInitialized) return;

    setState(() {
      _isLoading = true;
      _status = 'Adding sample documents...';
    });

    try {
      final samples = [
        '''Flutter is an open source framework by Google for building beautiful, 
natively compiled, multi-platform applications from a single codebase.
Flutter uses the Dart programming language and provides a rich set of 
pre-designed widgets for creating modern user interfaces.''',

        '''RAG (Retrieval-Augmented Generation) is a technique that combines 
information retrieval with text generation. It first retrieves relevant 
documents from a knowledge base, then uses that context to generate 
more accurate and informed responses.''',

        '''Mobile RAG Engine is a Flutter package that provides on-device 
semantic search capabilities. It uses HNSW (Hierarchical Navigable Small World) 
graphs for efficient vector similarity search, and supports automatic 
document chunking for optimal LLM context assembly.''',

        '''Ollama is an open-source tool that allows you to run large language 
models locally on your machine. It supports various models including Llama, 
Gemma, Mistral, and more. Ollama provides a simple API for generating 
text completions and chat responses.''',
      ];

      for (var i = 0; i < samples.length; i++) {
        setState(
          () => _status = 'Adding document ${i + 1}/${samples.length}...',
        );
        await _ragService!.addSourceWithChunking(samples[i]);
      }

      await _ragService!.rebuildIndex();

      final stats = await _ragService!.getStats();
      _totalSources = stats.sourceCount.toInt();
      _totalChunks = stats.chunkCount.toInt();

      setState(() {
        _isLoading = false;
        _status =
            'Added ${samples.length} documents! Total chunks: $_totalChunks';
      });

      _addSystemMessage(
        '✅ Added ${samples.length} sample documents with $_totalChunks chunks.',
      );

      // Regenerate topic suggestions with new content
      if (!widget.mockLlm) {
        _topicService.invalidateCache();
        _generateTopicSuggestions();
      }
    } catch (e) {
      setState(() {
        _isLoading = false;
        _status = 'Error: $e';
      });
    }
  }

  /// Manually rebuild HNSW index to sync with all chunks in DB.
  Future<void> _rebuildHnswIndex() async {
    if (!_isInitialized) return;

    setState(() {
      _isLoading = true;
      _status = 'Rebuilding HNSW index...';
    });

    try {
      final stopwatch = Stopwatch()..start();
      await _ragService!.rebuildIndex();
      stopwatch.stop();

      final stats = await _ragService!.getStats();
      _totalChunks = stats.chunkCount.toInt();

      setState(() {
        _isLoading = false;
        _status =
            'Index rebuilt in ${stopwatch.elapsedMilliseconds}ms! Chunks: $_totalChunks';
      });

      _addSystemMessage(
        '🔄 HNSW 인덱스 재구축 완료!\n'
        '• 소요 시간: ${stopwatch.elapsedMilliseconds}ms\n'
        '• 인덱싱된 청크: $_totalChunks개',
      );
    } catch (e) {
      setState(() {
        _isLoading = false;
        _status = 'Error rebuilding index: $e';
      });
    }
  }

  /// Regenerate all chunk embeddings using current model.
  /// This fixes the issue when stored embeddings don't match the current model.
  Future<void> _regenerateAllEmbeddings() async {
    if (!_isInitialized) return;

    setState(() {
      _isLoading = true;
      _status = 'Regenerating embeddings...';
    });

    try {
      final stopwatch = Stopwatch()..start();

      await _ragService!.regenerateAllEmbeddings(
        onProgress: (done, total) {
          setState(() {
            _status = 'Re-embedding: $done/$total';
          });
        },
      );

      stopwatch.stop();

      final stats = await _ragService!.getStats();
      _totalChunks = stats.chunkCount.toInt();

      setState(() {
        _isLoading = false;
        _status =
            'Embeddings regenerated in ${(stopwatch.elapsedMilliseconds / 1000).toStringAsFixed(1)}s!';
      });

      _addSystemMessage(
        '🔄 임베딩 재생성 완료!\n'
        '• 소요 시간: ${(stopwatch.elapsedMilliseconds / 1000).toStringAsFixed(1)}초\n'
        '• 처리된 청크: $_totalChunks개\n'
        '• 이제 검색이 정상 작동합니다.',
      );
    } catch (e) {
      setState(() {
        _isLoading = false;
        _status = 'Error: $e';
      });
    }
  }

  @override
  Widget build(BuildContext context) {
    return Scaffold(
      backgroundColor: const Color(0xFF121212),
      appBar: AppBar(
        backgroundColor: const Color(0xFF1A1A1A),
        title: const Text('RAG Chat', style: TextStyle(color: Colors.white)),
        centerTitle: true,
        iconTheme: const IconThemeData(color: Colors.white),
        actions: [
          // Graph panel toggle
          IconButton(
            icon: Icon(
              _showGraphPanel ? Icons.hub : Icons.hub_outlined,
              color: _showGraphPanel ? Colors.purple : Colors.grey,
            ),
            tooltip: 'Toggle Knowledge Graph',
            onPressed: () => setState(() => _showGraphPanel = !_showGraphPanel),
          ),
          IconButton(
            icon: Icon(
              _showDebugInfo ? Icons.bug_report : Icons.bug_report_outlined,
              color: Colors.grey,
            ),
            tooltip: 'Toggle debug info',
            onPressed: () => setState(() => _showDebugInfo = !_showDebugInfo),
          ),
          PopupMenuButton<String>(
            onSelected: (value) {
              switch (value) {
                case 'add_samples':
                  _addSampleDocuments();
                  break;
                case 'new_chat':
                  _startNewChat();
                  break;
                case 'clear_chat':
                  setState(() => _messages.clear());
                  break;
                case 'compression_0':
                  setState(() => _compressionLevel = 0);
                  break;
                case 'compression_1':
                  setState(() => _compressionLevel = 1);
                  break;
                case 'compression_2':
                  setState(() => _compressionLevel = 2);
                  break;
                case 'rebuild_index':
                  _rebuildHnswIndex();
                  break;
                case 'regenerate_embeddings':
                  _regenerateAllEmbeddings();
                  break;
              }
            },
            itemBuilder: (context) => [
              const PopupMenuItem(
                value: 'add_samples',
                child: ListTile(
                  leading: Icon(Icons.dataset),
                  title: Text('Add Sample Docs'),
                  contentPadding: EdgeInsets.zero,
                ),
              ),
              const PopupMenuItem(
                value: 'new_chat',
                child: ListTile(
                  leading: Icon(Icons.refresh),
                  title: Text('New Chat'),
                  subtitle: Text('Clear chat history'),
                  contentPadding: EdgeInsets.zero,
                ),
              ),
              const PopupMenuItem(
                value: 'clear_chat',
                child: ListTile(
                  leading: Icon(Icons.clear_all),
                  title: Text('Clear Chat'),
                  contentPadding: EdgeInsets.zero,
                ),
              ),
              const PopupMenuItem(
                value: 'rebuild_index',
                child: ListTile(
                  leading: Icon(Icons.refresh),
                  title: Text('Rebuild Index'),
                  subtitle: Text('Sync HNSW with DB'),
                  contentPadding: EdgeInsets.zero,
                ),
              ),
              const PopupMenuItem(
                value: 'regenerate_embeddings',
                child: ListTile(
                  leading: Icon(Icons.autorenew),
                  title: Text('Regenerate Embeddings'),
                  subtitle: Text('Fix model mismatch'),
                  contentPadding: EdgeInsets.zero,
                ),
              ),
              const PopupMenuDivider(),
              PopupMenuItem(
                enabled: false,
                child: Text(
                  'Compression Level',
                  style: TextStyle(
                    fontWeight: FontWeight.bold,
                    color: Colors.grey[600],
                  ),
                ),
              ),
              PopupMenuItem(
                value: 'compression_0',
                child: ListTile(
                  leading: Radio<int>(
                    value: 0,
                    groupValue: _compressionLevel,
                    onChanged: null,
                  ),
                  title: const Text('Minimal'),
                  subtitle: const Text('Max context'),
                  contentPadding: EdgeInsets.zero,
                ),
              ),
              PopupMenuItem(
                value: 'compression_1',
                child: ListTile(
                  leading: Radio<int>(
                    value: 1,
                    groupValue: _compressionLevel,
                    onChanged: null,
                  ),
                  title: const Text('Balanced'),
                  subtitle: const Text('Default'),
                  contentPadding: EdgeInsets.zero,
                ),
              ),
              PopupMenuItem(
                value: 'compression_2',
                child: ListTile(
                  leading: Radio<int>(
                    value: 2,
                    groupValue: _compressionLevel,
                    onChanged: null,
                  ),
                  title: const Text('Aggressive'),
                  subtitle: const Text('Less context'),
                  contentPadding: EdgeInsets.zero,
                ),
              ),
            ],
          ),
        ],
      ),
      body: Row(
        children: [
          // Left: Chat area (flex 5)
          Expanded(
            flex: 5,
            child: Container(
              color: const Color(0xFF121212),
              child: Column(
                children: [
                  // Status bar
                  if (_showDebugInfo)
                    Container(
                      padding: const EdgeInsets.symmetric(
                        horizontal: 16,
                        vertical: 8,
                      ),
                      color: const Color(0xFF1A1A1A),
                      child: Row(
                        children: [
                          if (_isLoading)
                            const SizedBox(
                              width: 16,
                              height: 16,
                              child: CircularProgressIndicator(strokeWidth: 2),
                            )
                          else
                            Icon(
                              _isInitialized ? Icons.check_circle : Icons.error,
                              size: 16,
                              color: _isInitialized ? Colors.green : Colors.red,
                            ),
                          const SizedBox(width: 8),
                          Expanded(
                            child: Text(
                              _status,
                              style: const TextStyle(
                                fontSize: 12,
                                color: Colors.white70,
                              ),
                              overflow: TextOverflow.ellipsis,
                            ),
                          ),
                          Text(
                            '📄$_totalSources 📦$_totalChunks',
                            style: const TextStyle(
                              fontSize: 12,
                              color: Colors.white70,
                            ),
                          ),
                        ],
                      ),
                    ),

                  // Suggestion chips
                  SuggestionChipsPanel(
                    suggestions: _suggestedQuestions,
                    isLoading: _isLoadingSuggestions,
                    isExpanded: _isSuggestionsExpanded,
                    isDisabled: _isGenerating,
                    onToggleExpanded: () => setState(
                      () => _isSuggestionsExpanded = !_isSuggestionsExpanded,
                    ),
                    onRefresh: () {
                      _topicService.invalidateCache();
                      _generateTopicSuggestions();
                    },
                    onQuestionSelected: _sendSuggestedQuestion,
                  ),

                  // Messages
                  Expanded(child: _buildMessageList()),

                  // Input area
                  ChatInputArea(
                    controller: _messageController,
                    focusNode: _focusNode,
                    isEnabled: _isInitialized,
                    isGenerating: _isGenerating,
                    onSend: _sendMessage,
                    onAttach: _showAddDocumentDialog,
                  ),
                ],
              ),
            ),
          ),

          // Divider
          if (_showGraphPanel) Container(width: 1, color: Colors.grey[800]),

          // Middle: Graph panel (flex 3-4)
          if (_showGraphPanel)
            Expanded(
              flex: 4,
              child: KnowledgeGraphPanel(
                query: _lastQuery,
                chunks: _lastChunks,
                similarityThreshold: _minSimilarityThreshold,
                selectedChunk: _selectedChunk,
                onChunkSelected: (chunk) {
                  setState(() => _selectedChunk = chunk);
                },
              ),
            ),

          // Divider
          if (_showGraphPanel) Container(width: 1, color: Colors.grey[800]),

          // Right: Chunk detail sidebar (flex 1-2)
          if (_showGraphPanel)
            Expanded(
              flex: 2,
              child: ChunkDetailSidebar(
                chunks: _lastChunks,
                selectedChunk: _selectedChunk,
                searchQuery: _lastQuery,
                onChunkSelected: (chunk) {
                  setState(() => _selectedChunk = chunk);
                },
              ),
            ),
        ],
      ),
    );
  }

  Widget _buildMessageList() {
    if (_messages.isEmpty) {
      return Center(
        child: Column(
          mainAxisAlignment: MainAxisAlignment.center,
          children: [
            Icon(Icons.chat_bubble_outline, size: 64, color: Colors.grey[400]),
            const SizedBox(height: 16),
            Text('No messages yet', style: TextStyle(color: Colors.grey[600])),
            const SizedBox(height: 8),
            TextButton.icon(
              onPressed: _addSampleDocuments,
              icon: const Icon(Icons.add),
              label: const Text('Add sample documents to start'),
            ),
          ],
        ),
      );
    }

    return ListView.builder(
      controller: _scrollController,
      padding: const EdgeInsets.symmetric(horizontal: 16, vertical: 16),
      reverse: true,
      itemCount: _messages.length,
      itemBuilder: (context, index) {
        final message = _messages[index];
        // Use document style for AI responses, bubble for user
        if (message.isUser) {
          return UserMessageBubble(message: message);
        } else {
          return DocumentStyleResponse(
            message: message,
            showDebugInfo: _showDebugInfo,
            isGraphActive: _showGraphPanel && _activeGraphMessageIndex == index,
            onViewGraph:
                message.retrievedChunks != null &&
                    message.retrievedChunks!.isNotEmpty
                ? () {
                    setState(() {
                      _lastQuery = _getQueryForMessage(index);
                      _lastChunks = message.retrievedChunks!;
                      _activeGraphMessageIndex = index;
                      _showGraphPanel = true;
                    });
                  }
                : null,
          );
        }
      },
    );
  }

  /// Get the user query that generated this AI response
  String? _getQueryForMessage(int aiMessageIndex) {
    // The user message is typically the next one (since list is reversed)
    for (var i = aiMessageIndex + 1; i < _messages.length; i++) {
      if (_messages[i].isUser) {
        return _messages[i].content;
      }
    }
    return null;
  }

  /// Build suggestion chips for topic-based questions (collapsible)
  Widget _buildSuggestionChips() {
    if (_isLoadingSuggestions) {
      return Container(
        padding: const EdgeInsets.symmetric(horizontal: 16, vertical: 12),
        child: Row(
          children: [
            const SizedBox(
              width: 16,
              height: 16,
              child: CircularProgressIndicator(strokeWidth: 2),
            ),
            const SizedBox(width: 8),
            Text(
              '추천 질문 생성 중...',
              style: TextStyle(fontSize: 13, color: Colors.grey[600]),
            ),
          ],
        ),
      );
    }

    if (_suggestedQuestions.isEmpty) {
      return const SizedBox.shrink();
    }

    return Container(
      decoration: BoxDecoration(
        color: Theme.of(
          context,
        ).colorScheme.surfaceContainerHighest.withValues(alpha: 0.5),
        border: Border(
          bottom: BorderSide(color: Colors.grey.withValues(alpha: 0.2)),
        ),
      ),
      child: Column(
        crossAxisAlignment: CrossAxisAlignment.start,
        children: [
          // Header row (always visible, tappable to expand/collapse)
          InkWell(
            onTap: () {
              setState(() => _isSuggestionsExpanded = !_isSuggestionsExpanded);
            },
            child: Padding(
              padding: const EdgeInsets.fromLTRB(16, 10, 16, 10),
              child: Row(
                children: [
                  Icon(
                    Icons.lightbulb_outline,
                    size: 16,
                    color: Colors.amber[700],
                  ),
                  const SizedBox(width: 6),
                  Text(
                    '추천 질문',
                    style: TextStyle(
                      fontSize: 12,
                      fontWeight: FontWeight.w600,
                      color: Colors.grey[700],
                    ),
                  ),
                  const SizedBox(width: 4),
                  // Question count badge
                  Container(
                    padding: const EdgeInsets.symmetric(
                      horizontal: 6,
                      vertical: 2,
                    ),
                    decoration: BoxDecoration(
                      color: Colors.amber.withValues(alpha: 0.2),
                      borderRadius: BorderRadius.circular(10),
                    ),
                    child: Text(
                      '${_suggestedQuestions.length}',
                      style: TextStyle(
                        fontSize: 10,
                        fontWeight: FontWeight.bold,
                        color: Colors.amber[800],
                      ),
                    ),
                  ),
                  const Spacer(),
                  // Refresh button
                  GestureDetector(
                    onTap: () {
                      _topicService.invalidateCache();
                      _generateTopicSuggestions();
                    },
                    child: Icon(
                      Icons.refresh,
                      size: 16,
                      color: Colors.grey[500],
                    ),
                  ),
                  const SizedBox(width: 8),
                  // Expand/collapse icon
                  AnimatedRotation(
                    turns: _isSuggestionsExpanded ? 0.5 : 0,
                    duration: const Duration(milliseconds: 200),
                    child: Icon(
                      Icons.expand_more,
                      size: 20,
                      color: Colors.grey[600],
                    ),
                  ),
                ],
              ),
            ),
          ),
          // Expandable content
          AnimatedCrossFade(
            firstChild: Padding(
              padding: const EdgeInsets.fromLTRB(16, 0, 16, 12),
              child: Wrap(
                spacing: 8,
                runSpacing: 8,
                children: _suggestedQuestions.map((q) {
                  return ActionChip(
                    label: Text(
                      q.question,
                      style: TextStyle(
                        fontSize: 13,
                        color: _isGenerating ? Colors.grey : null,
                      ),
                      maxLines: 2,
                      overflow: TextOverflow.ellipsis,
                    ),
                    backgroundColor: _isGenerating
                        ? Colors.grey.withValues(alpha: 0.2)
                        : Theme.of(
                            context,
                          ).colorScheme.primaryContainer.withValues(alpha: 0.5),
                    side: BorderSide(
                      color: _isGenerating
                          ? Colors.grey.withValues(alpha: 0.3)
                          : Theme.of(
                              context,
                            ).colorScheme.primary.withValues(alpha: 0.3),
                    ),
                    onPressed: _isGenerating
                        ? null // Disable during generation
                        : () => _sendSuggestedQuestion(q),
                  );
                }).toList(),
              ),
            ),
            secondChild: const SizedBox.shrink(),
            crossFadeState: _isSuggestionsExpanded
                ? CrossFadeState.showFirst
                : CrossFadeState.showSecond,
            duration: const Duration(milliseconds: 200),
          ),
        ],
      ),
    );
  }

  Widget _buildMessageBubble(ChatMessage message) {
    final isUser = message.isUser;
    final messageIndex = _messages.indexOf(message);
    final hasSourceChunks =
        !isUser &&
        message.retrievedChunks != null &&
        message.retrievedChunks!.isNotEmpty;
    final isSourceExpanded = _expandedSourceMessages.contains(messageIndex);

    // Get best similarity for display
    double? bestSimilarity;
    if (hasSourceChunks) {
      final validSims = message.retrievedChunks!
          .map((c) => c.similarity)
          .where((s) => s > 0)
          .toList();
      if (validSims.isNotEmpty) {
        bestSimilarity = validSims.reduce((a, b) => a > b ? a : b);
      }
    }

    return Padding(
      padding: const EdgeInsets.only(bottom: 12),
      child: Row(
        mainAxisAlignment: isUser
            ? MainAxisAlignment.end
            : MainAxisAlignment.start,
        crossAxisAlignment: CrossAxisAlignment.start,
        children: [
          if (!isUser) ...[
            CircleAvatar(
              radius: 18,
              backgroundColor: Theme.of(context).colorScheme.primaryContainer,
              child: const Icon(Icons.smart_toy, size: 20),
            ),
            const SizedBox(width: 8),
          ],
          Flexible(
            child: Column(
              crossAxisAlignment: isUser
                  ? CrossAxisAlignment.end
                  : CrossAxisAlignment.start,
              children: [
                Container(
                  padding: const EdgeInsets.symmetric(
                    horizontal: 14,
                    vertical: 10,
                  ),
                  decoration: BoxDecoration(
                    // User: pink bubble, AI: white bubble (matching existing style)
                    color: isUser ? const Color(0xFFFFE6E6) : Colors.white,
                    borderRadius: BorderRadius.circular(16),
                    border: Border.all(
                      color: isUser
                          ? const Color(0xFFFFDADA)
                          : const Color(0xFFE5E5E5),
                    ),
                    boxShadow: [
                      BoxShadow(
                        color: Colors.grey.withValues(alpha: 0.1),
                        blurRadius: 4,
                        offset: const Offset(0, 2),
                      ),
                    ],
                  ),
                  child: SelectableText.rich(
                    _parseMarkdown(message.content),
                    style: const TextStyle(fontSize: 14, color: Colors.black87),
                  ),
                ),

                // Source chunks section (collapsible) for AI messages
                if (hasSourceChunks)
                  Padding(
                    padding: const EdgeInsets.only(top: 6),
                    child: Container(
                      decoration: BoxDecoration(
                        color: Colors.blue.withValues(alpha: 0.05),
                        borderRadius: BorderRadius.circular(12),
                        border: Border.all(
                          color: Colors.blue.withValues(alpha: 0.2),
                        ),
                      ),
                      child: Column(
                        crossAxisAlignment: CrossAxisAlignment.start,
                        children: [
                          // Header (always visible)
                          InkWell(
                            onTap: () {
                              setState(() {
                                if (isSourceExpanded) {
                                  _expandedSourceMessages.remove(messageIndex);
                                } else {
                                  _expandedSourceMessages.add(messageIndex);
                                }
                              });
                            },
                            borderRadius: BorderRadius.circular(12),
                            child: Padding(
                              padding: const EdgeInsets.symmetric(
                                horizontal: 10,
                                vertical: 8,
                              ),
                              child: Row(
                                mainAxisSize: MainAxisSize.min,
                                children: [
                                  Icon(
                                    Icons.description_outlined,
                                    size: 14,
                                    color: Colors.blue[700],
                                  ),
                                  const SizedBox(width: 6),
                                  Text(
                                    '참고 문서',
                                    style: TextStyle(
                                      fontSize: 11,
                                      fontWeight: FontWeight.w600,
                                      color: Colors.blue[700],
                                    ),
                                  ),
                                  const SizedBox(width: 6),
                                  // Similarity badge
                                  if (bestSimilarity != null)
                                    Container(
                                      padding: const EdgeInsets.symmetric(
                                        horizontal: 5,
                                        vertical: 2,
                                      ),
                                      decoration: BoxDecoration(
                                        color: _getSimilarityColor(
                                          bestSimilarity,
                                        ).withValues(alpha: 0.2),
                                        borderRadius: BorderRadius.circular(6),
                                      ),
                                      child: Text(
                                        '${(bestSimilarity * 100).toStringAsFixed(0)}%',
                                        style: TextStyle(
                                          fontSize: 10,
                                          fontWeight: FontWeight.bold,
                                          color: _getSimilarityColor(
                                            bestSimilarity,
                                          ),
                                        ),
                                      ),
                                    ),
                                  const SizedBox(width: 4),
                                  AnimatedRotation(
                                    turns: isSourceExpanded ? 0.5 : 0,
                                    duration: const Duration(milliseconds: 150),
                                    child: Icon(
                                      Icons.expand_more,
                                      size: 16,
                                      color: Colors.blue[600],
                                    ),
                                  ),
                                ],
                              ),
                            ),
                          ),
                          // Expandable chunk previews
                          if (isSourceExpanded)
                            Padding(
                              padding: const EdgeInsets.fromLTRB(10, 0, 10, 10),
                              child: Column(
                                crossAxisAlignment: CrossAxisAlignment.start,
                                children: _buildChunkPreviews(
                                  message.retrievedChunks!,
                                ),
                              ),
                            ),
                        ],
                      ),
                    ),
                  ),

                // Debug info for AI messages
                if (!isUser && _showDebugInfo && message.tokensUsed != null)
                  Padding(
                    padding: const EdgeInsets.only(top: 4),
                    child: Column(
                      crossAxisAlignment: CrossAxisAlignment.start,
                      children: [
                        Text(
                          '~${message.tokensUsed} tokens • ${message.retrievedChunks?.length ?? 0} chunks',
                          style: TextStyle(
                            fontSize: 11,
                            color: Colors.grey[500],
                          ),
                        ),
                        if (message.ragSearchTime != null)
                          Text(
                            '⚡ RAG: ${message.ragSearchTime!.inMilliseconds}ms • '
                            'LLM: ${message.llmGenerationTime?.inMilliseconds ?? 0}ms • '
                            'Total: ${message.totalTime?.inMilliseconds ?? 0}ms',
                            style: TextStyle(
                              fontSize: 11,
                              color: Colors.blue[400],
                              fontFamily: 'monospace',
                            ),
                          ),
                      ],
                    ),
                  ),
                // Timestamp
                Padding(
                  padding: const EdgeInsets.only(top: 4),
                  child: Text(
                    _formatTime(message.timestamp),
                    style: TextStyle(fontSize: 11, color: Colors.grey[500]),
                  ),
                ),
              ],
            ),
          ),
          if (isUser) const SizedBox(width: 8),
        ],
      ),
    );
  }

  /// Get color based on similarity score
  Color _getSimilarityColor(double similarity) {
    if (similarity >= 0.7) return Colors.green;
    if (similarity >= 0.5) return Colors.orange;
    return Colors.red;
  }

  /// Build chunk preview widgets (top 3 most relevant)
  List<Widget> _buildChunkPreviews(List<ChunkSearchResult> chunks) {
    // Sort by similarity (descending) and take top 3
    final sortedChunks = chunks.where((c) => c.similarity > 0).toList()
      ..sort((a, b) => b.similarity.compareTo(a.similarity));
    final topChunks = sortedChunks.take(3).toList();

    return topChunks.map((chunk) {
      final preview = chunk.content.length > 120
          ? '${chunk.content.substring(0, 120)}...'
          : chunk.content;

      return Padding(
        padding: const EdgeInsets.only(top: 6),
        child: Container(
          padding: const EdgeInsets.all(8),
          decoration: BoxDecoration(
            color: Colors.white,
            borderRadius: BorderRadius.circular(8),
            border: Border.all(color: Colors.grey.withValues(alpha: 0.2)),
          ),
          child: Column(
            crossAxisAlignment: CrossAxisAlignment.start,
            children: [
              Row(
                children: [
                  Container(
                    padding: const EdgeInsets.symmetric(
                      horizontal: 5,
                      vertical: 2,
                    ),
                    decoration: BoxDecoration(
                      color: _getSimilarityColor(
                        chunk.similarity,
                      ).withValues(alpha: 0.15),
                      borderRadius: BorderRadius.circular(4),
                    ),
                    child: Text(
                      '${(chunk.similarity * 100).toStringAsFixed(0)}%',
                      style: TextStyle(
                        fontSize: 9,
                        fontWeight: FontWeight.bold,
                        color: _getSimilarityColor(chunk.similarity),
                      ),
                    ),
                  ),
                  const Spacer(),
                  Text(
                    'Chunk #${chunk.chunkId}',
                    style: TextStyle(fontSize: 9, color: Colors.grey[500]),
                  ),
                ],
              ),
              const SizedBox(height: 4),
              Text(
                preview.replaceAll('\n', ' '),
                style: TextStyle(
                  fontSize: 11,
                  color: Colors.grey[700],
                  height: 1.3,
                ),
              ),
            ],
          ),
        ),
      );
    }).toList();
  }

  String _formatTime(DateTime timestamp) {
    final hour = timestamp.hour;
    final period = hour < 12 ? '오전' : '오후';
    final hour12 = hour == 12 ? 12 : hour % 12;
    final minute = timestamp.minute.toString().padLeft(2, '0');
    return '$period $hour12:$minute';
  }

  Widget _buildInputArea() {
    return Container(
      padding: const EdgeInsets.fromLTRB(16, 8, 12, 8),
      decoration: BoxDecoration(
        color: Colors.white,
        boxShadow: [
          BoxShadow(
            color: Colors.grey.withValues(alpha: 0.12),
            spreadRadius: 10,
            blurRadius: 15,
            offset: const Offset(0, -2),
          ),
        ],
        borderRadius: const BorderRadius.only(
          topLeft: Radius.circular(20),
          topRight: Radius.circular(20),
        ),
      ),
      child: SafeArea(
        child: Row(
          children: [
            // Add document button
            IconButton(
              onPressed: _isInitialized ? _showAddDocumentDialog : null,
              icon: Icon(
                Icons.attach_file,
                color: _isInitialized ? Colors.grey[700] : Colors.grey[400],
              ),
            ),
            // Text input
            Expanded(
              child: TextField(
                controller: _messageController,
                focusNode: _focusNode,
                enabled: _isInitialized && !_isGenerating,
                maxLines: 5,
                minLines: 1,
                textInputAction: TextInputAction.send,
                onSubmitted: (_) => _sendMessage(),
                style: const TextStyle(color: Colors.black87, fontSize: 15),
                decoration: InputDecoration(
                  hintText: 'Ask a question...',
                  hintStyle: TextStyle(color: Colors.grey[500]),
                  filled: true,
                  fillColor: Colors.grey[100],
                  border: OutlineInputBorder(
                    borderRadius: BorderRadius.circular(24),
                    borderSide: BorderSide.none,
                  ),
                  contentPadding: const EdgeInsets.symmetric(
                    horizontal: 16,
                    vertical: 10,
                  ),
                ),
              ),
            ),
            const SizedBox(width: 8),
            // Send button
            GestureDetector(
              onTap: _isGenerating ? null : _sendMessage,
              child: Container(
                width: 40,
                height: 40,
                decoration: BoxDecoration(
                  color: _isGenerating
                      ? Colors.grey
                      : Theme.of(context).colorScheme.primary,
                  shape: BoxShape.circle,
                ),
                child: _isGenerating
                    ? const Padding(
                        padding: EdgeInsets.all(10),
                        child: CircularProgressIndicator(
                          strokeWidth: 2,
                          color: Colors.white,
                        ),
                      )
                    : const Icon(
                        Icons.arrow_upward,
                        color: Colors.white,
                        size: 22,
                      ),
              ),
            ),
          ],
        ),
      ),
    );
  }

  void _showAddDocumentDialog() {
    final controller = TextEditingController();

    showModalBottomSheet(
      context: context,
      isScrollControlled: true,
      shape: const RoundedRectangleBorder(
        borderRadius: BorderRadius.vertical(top: Radius.circular(20)),
      ),
      builder: (context) => Padding(
        padding: EdgeInsets.only(
          bottom: MediaQuery.of(context).viewInsets.bottom,
        ),
        child: Container(
          padding: const EdgeInsets.all(20),
          child: Column(
            mainAxisSize: MainAxisSize.min,
            crossAxisAlignment: CrossAxisAlignment.stretch,
            children: [
              Text(
                'Add Document',
                style: Theme.of(context).textTheme.titleLarge,
              ),
              const SizedBox(height: 16),
              TextField(
                controller: controller,
                maxLines: 6,
                decoration: const InputDecoration(
                  hintText: 'Paste or type document content...',
                  border: OutlineInputBorder(),
                ),
              ),
              const SizedBox(height: 16),
              Row(
                mainAxisAlignment: MainAxisAlignment.end,
                children: [
                  TextButton(
                    onPressed: () => Navigator.pop(context),
                    child: const Text('Cancel'),
                  ),
                  const SizedBox(width: 8),
                  FilledButton(
                    onPressed: () async {
                      final text = controller.text.trim();
                      if (text.isEmpty) return;

                      Navigator.pop(context);

                      setState(() {
                        _isLoading = true;
                        _status = 'Adding document...';
                      });

                      try {
                        final result = await _ragService!.addSourceWithChunking(
                          text,
                        );
                        await _ragService!.rebuildIndex();

                        final stats = await _ragService!.getStats();
                        _totalSources = stats.sourceCount.toInt();
                        _totalChunks = stats.chunkCount.toInt();

                        setState(() {
                          _isLoading = false;
                          _status =
                              'Document added! Chunks: ${result.chunkCount}';
                        });

                        _addSystemMessage(
                          '✅ Document added with ${result.chunkCount} chunks.',
                        );

                        // Regenerate topic suggestions with new content
                        if (!widget.mockLlm) {
                          _topicService.invalidateCache();
                          _generateTopicSuggestions();
                        }
                      } catch (e) {
                        setState(() {
                          _isLoading = false;
                          _status = 'Error: $e';
                        });
                      }
                    },
                    child: const Text('Add'),
                  ),
                ],
              ),
            ],
          ),
        ),
      ),
    );
  }

  @override
  void dispose() {
    _messageController.dispose();
    _scrollController.dispose();
    _focusNode.dispose();
    EmbeddingService.dispose();
    super.dispose();
  }
}
