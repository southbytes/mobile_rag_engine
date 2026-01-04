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

import 'package:local_gemma_macos/services/topic_suggestion_service.dart';
import 'package:local_gemma_macos/services/query_understanding_service.dart';
import 'package:local_gemma_macos/services/ollama_response_service.dart';
import 'package:local_gemma_macos/models/chat_models.dart';
import 'package:local_gemma_macos/widgets/knowledge_graph_panel.dart';
import 'package:local_gemma_macos/widgets/chunk_detail_sidebar.dart';
import 'package:local_gemma_macos/widgets/suggestion_chips.dart';
import 'package:local_gemma_macos/widgets/chat_input_area.dart';
import 'package:local_gemma_macos/widgets/document_style_response.dart';

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

  // Query understanding service
  QueryUnderstandingService? _queryService;

  // Ollama response service
  OllamaResponseService? _ollamaResponseService;

  // Similarity threshold for RAG
  final double _minSimilarityThreshold = 0.35;

  // Topic suggestion service
  final TopicSuggestionService _topicService = TopicSuggestionService();
  List<SuggestedQuestion> _suggestedQuestions = [];
  bool _isLoadingSuggestions = false;

  // UI state for collapsible sections
  bool _isSuggestionsExpanded = true; // Collapsible suggestions panel

  // Knowledge Graph panel state
  bool _showGraphPanel = true; // Toggle graph panel visibility
  ChunkSearchResult? _selectedChunk; // Currently selected chunk in graph
  String? _lastQuery; // Last query for graph display
  List<ChunkSearchResult> _lastChunks = []; // Last chunks for graph display
  int? _activeGraphMessageIndex; // Track which message's chunks are in graph

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

      // 4. Initialize Query Understanding service
      _queryService = QueryUnderstandingService(
        ollamaClient: _ollamaClient,
        modelName: widget.modelName,
      );

      // 5. Initialize Ollama Response service
      _ollamaResponseService = OllamaResponseService(
        ollamaClient: _ollamaClient,
        modelName: widget.modelName ?? 'gemma3:4b',
      );

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

      // === Stage 1: Query Understanding (NEW) ===
      final understanding = await _queryService!.analyze(text);

      // 🚫 Reject invalid queries
      if (!understanding.isValid) {
        setState(() {
          _messages.insert(
            0,
            ChatMessage(
              content: understanding.rejectionReason ?? '질문을 이해하지 못했습니다.',
              isUser: false,
            ),
          );
          _isGenerating = false;
        });
        return;
      }

      debugPrint('✅ Query validated: ${understanding.type.name}');
      debugPrint('   Normalized: "${understanding.normalizedQuery}"');
      debugPrint('   Keywords: ${understanding.keywords}');

      // === Stage 2: Map QueryType to RAG parameters ===
      final (adjacentChunks, tokenBudget, topK) = switch (understanding.type) {
        QueryType.definition => (1, 1000, 5),
        QueryType.explanation => (2, 2500, 10),
        QueryType.factual => (1, 1500, 5),
        QueryType.comparison => (2, 3000, 12),
        QueryType.listing => (3, 4000, 15),
        QueryType.summary => (1, 1500, 5),
        _ => (2, 2000, 10),
      };

      // === Stage 3: RAG Search ===
      final ragStopwatch = Stopwatch()..start();
      debugPrint(
        '📐 Using: type=${understanding.type.name}, adjacent=$adjacentChunks, budget=$tokenBudget, topK=$topK',
      );
      debugPrint('🔎 Search query: "${understanding.normalizedQuery}"');

      final ragResult = await _ragService!.search(
        understanding.normalizedQuery, // Use normalized query
        topK: topK,
        tokenBudget: tokenBudget,
        strategy: ContextStrategy.relevanceFirst,
        adjacentChunks: adjacentChunks,
        singleSourceMode: false,
      );
      ragStopwatch.stop();
      final ragSearchTime = ragStopwatch.elapsed;

      // DEBUG: Log search results
      debugPrint('🔍 BGE-m3 search for: "${understanding.normalizedQuery}"');
      debugPrint('   Found ${ragResult.chunks.length} chunks');
      for (var i = 0; i < ragResult.chunks.length && i < 5; i++) {
        final c = ragResult.chunks[i];
        final preview = c.content.length > 50
            ? '${c.content.substring(0, 50)}...'
            : c.content;
        debugPrint('   [$i] sim=${c.similarity.toStringAsFixed(3)}: $preview');
      }

      // Filter low similarity chunks
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

      final bool hasRelevantContext = relevantChunks.isNotEmpty;
      final contextText = ragResult.context.text;
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

  Future<String> _generateOllamaResponse(
    String query,
    String contextText,
    RagSearchResult ragResult,
    bool hasRelevantContext,
  ) async {
    final result = await _ollamaResponseService!.generateResponse(
      query: query,
      contextText: contextText,
      ragResult: ragResult,
      hasRelevantContext: hasRelevantContext,
      chatHistory: _chatHistory,
      onHistoryUpdate: (message) => _chatHistory.add(message),
    );

    return result.response;
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
                case 'new_chat':
                  _startNewChat();
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
              }
            },
            itemBuilder: (context) => [
              const PopupMenuItem(
                value: 'new_chat',
                child: ListTile(
                  leading: Icon(Icons.refresh),
                  title: Text('New Chat'),
                  subtitle: Text('Clear chat history'),
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
                  leading: Icon(
                    _compressionLevel == 0
                        ? Icons.radio_button_checked
                        : Icons.radio_button_unchecked,
                    color: _compressionLevel == 0
                        ? Theme.of(context).colorScheme.primary
                        : null,
                  ),
                  title: const Text('Minimal'),
                  subtitle: const Text('Max context'),
                  contentPadding: EdgeInsets.zero,
                ),
              ),
              PopupMenuItem(
                value: 'compression_1',
                child: ListTile(
                  leading: Icon(
                    _compressionLevel == 1
                        ? Icons.radio_button_checked
                        : Icons.radio_button_unchecked,
                    color: _compressionLevel == 1
                        ? Theme.of(context).colorScheme.primary
                        : null,
                  ),
                  title: const Text('Balanced'),
                  subtitle: const Text('Default'),
                  contentPadding: EdgeInsets.zero,
                ),
              ),
              PopupMenuItem(
                value: 'compression_2',
                child: ListTile(
                  leading: Icon(
                    _compressionLevel == 2
                        ? Icons.radio_button_checked
                        : Icons.radio_button_unchecked,
                    color: _compressionLevel == 2
                        ? Theme.of(context).colorScheme.primary
                        : null,
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
              onPressed: _showAddDocumentDialog,
              icon: const Icon(Icons.add),
              label: const Text('Add documents to start'),
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
