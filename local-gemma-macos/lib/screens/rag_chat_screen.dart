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

/// Message model for chat
class ChatMessage {
  final String content;
  final bool isUser;
  final DateTime timestamp;
  final List<ChunkSearchResult>? retrievedChunks;
  final int? tokensUsed;
  final double? compressionRatio;  // 0.0-1.0, lower = more compressed
  final int? originalTokens;  // Before compression
  
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
    this.ragSearchTime,
    this.llmGenerationTime,
    this.totalTime,
  }) : timestamp = timestamp ?? DateTime.now();
}

class RagChatScreen extends StatefulWidget {
  final bool mockLlm;
  final String? modelName;

  const RagChatScreen({
    super.key,
    this.mockLlm = false,
    this.modelName,
  });

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

      // 1. Copy tokenizer
      await _copyAsset('assets/tokenizer.json', tokenizerPath);
      await initTokenizer(tokenizerPath: tokenizerPath);

      // 2. Load ONNX model
      setState(() => _status = 'Loading embedding model...');
      final modelBytes = await rootBundle.load('assets/model.onnx');
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
      _messages.insert(0, ChatMessage(
        content: content,
        isUser: false,
      ));
    });
  }

  Future<void> _sendMessage() async {
    final text = _messageController.text.trim();
    if (text.isEmpty || !_isInitialized || _isGenerating) return;

    _messageController.clear();
    _focusNode.unfocus();

    // Add user message
    setState(() {
      _messages.insert(0, ChatMessage(
        content: text,
        isUser: true,
      ));
      _isGenerating = true;
    });

    try {
      final totalStopwatch = Stopwatch()..start();
      
      // 1. RAG Search with timing
      final ragStopwatch = Stopwatch()..start();
      // 벡터 검색 + 인접 청크 포함 + 단일 소스 모드
      final ragResult = await _ragService!.search(
        text,
        topK: 10,
        tokenBudget: 4000,  // 압축 전 더 많은 컨텍스트 수집
        strategy: ContextStrategy.relevanceFirst,
        adjacentChunks: 2,  // 앞뒤 2개 청크 포함
        singleSourceMode: true,  // 가장 관련 높은 소스만 사용
      );
      ragStopwatch.stop();
      final ragSearchTime = ragStopwatch.elapsed;

      // 2. Apply actual compression using PromptCompressor
      final originalText = ragResult.context.text;
      final originalTokens = ragResult.context.estimatedTokens;
      
      // Map compression level to budget: 0=3000, 1=2000, 2=1500
      final maxChars = _compressionLevel == 2 ? 6000 : _compressionLevel == 1 ? 8000 : 12000;
      
      final compressed = await compressText(
        text: originalText,
        maxChars: maxChars,
        options: CompressionOptions(
          removeStopwords: false,  // Disabled - damages context
          removeDuplicates: true,
          language: 'ko',
          level: _compressionLevel,
        ),
      );
      
      // Calculate actual compression stats
      final compressedTokens = (compressed.compressedChars / 4).ceil();
      final savedPercent = ((1 - compressed.ratio) * 100).toStringAsFixed(1);
      final chunkCount = ragResult.chunks.length;
      
      // Debug log actual compression stats - show breakdown of each step
      debugPrint('📊 Compression: $originalTokens → $compressedTokens tokens (-$savedPercent%)');
      if (_showDebugInfo) {
        debugPrint('   📝 Duplicates: ${compressed.sentencesRemoved} sentences removed');
        if (compressed.charsSavedStopwords > 0) {
          debugPrint('   🔤 Stopwords: ${compressed.charsSavedStopwords} chars removed');
        }
        if (compressed.charsSavedTruncation > 0) {
          debugPrint('   ✂️ Truncation: ${compressed.charsSavedTruncation} chars cut');
        }
      }
      
      // Debug: Show removed/duplicate sentences (only in debug mode with _showDebugInfo)
      if (_showDebugInfo && compressed.sentencesRemoved > 0) {
        final originalSentences = await splitSentences(text: originalText);
        
        // Count sentence occurrences to find duplicates
        final sentenceCounts = <String, int>{};
        for (final sentence in originalSentences) {
          final normalized = sentence.trim();
          if (normalized.isNotEmpty) {
            sentenceCounts[normalized] = (sentenceCounts[normalized] ?? 0) + 1;
          }
        }
        
        // Find sentences that appeared more than once (duplicates)
        final duplicates = sentenceCounts.entries
            .where((e) => e.value > 1)
            .map((e) => '${e.key} (x${e.value})')
            .toList();
        
        if (duplicates.isNotEmpty) {
          debugPrint('   🗑️ Duplicates:');
          for (final dup in duplicates.take(3)) {
            final preview = dup.length > 60 
                ? '${dup.substring(0, 60)}...' 
                : dup;
            debugPrint('      • $preview');
          }
          if (duplicates.length > 3) {
            debugPrint('      ... and ${duplicates.length - 3} more');
          }
        }
      }

      // 3. LLM Generation with timing
      final llmStopwatch = Stopwatch()..start();
      String response;
      if (widget.mockLlm) {
        // Mock mode - show the compressed context stats
        response = _generateMockResponse(text, ragResult, savedPercent);
      } else {
        // Real LLM generation with Ollama - use compressed text
        response = await _generateOllamaResponseWithCompressedText(text, compressed.text, ragResult);
      }
      llmStopwatch.stop();
      final llmGenerationTime = llmStopwatch.elapsed;
      
      totalStopwatch.stop();

      // Add AI response with stats
      setState(() {
        _messages.insert(0, ChatMessage(
          content: response,
          isUser: false,
          retrievedChunks: ragResult.chunks,
          tokensUsed: compressedTokens,
          ragSearchTime: ragSearchTime,
          llmGenerationTime: llmGenerationTime,
          totalTime: totalStopwatch.elapsed,
        ));
      });
    } catch (e) {
      setState(() {
        _messages.insert(0, ChatMessage(
          content: '❌ Error: $e',
          isUser: false,
        ));
      });
    } finally {
      setState(() => _isGenerating = false);
    }

    _scrollToBottom();
  }

  String _generateMockResponse(String query, RagSearchResult ragResult, String savedPercent) {
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
    buffer.writeln('💡 This is a mock response. Install an LLM model for real answers.');
    
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
        messages.add(Message(
          role: MessageRole.system,
          content: '''당신은 주어진 문맥을 기반으로 질문에 답변하는 도우미입니다.

규칙:
1. 아래 문맥의 정보만을 기반으로 답변하세요.
2. 문맥에 관련 정보가 없으면 "문서에서 해당 정보를 찾을 수 없습니다"라고 답변하세요.
3. 질문과 동일한 언어로 답변하세요.
4. 조항 번호(제X조)는 문맥에 있는 그대로 인용하세요.

문맥:
$compressedContext''',
        ));
      } else {
        messages.add(const Message(
          role: MessageRole.system,
          content: 'You are a helpful assistant.',
        ));
      }
      
      // Add current user message
      messages.add(Message(
        role: MessageRole.user,
        content: query,
      ));
      
      // Save to history
      _chatHistory.add(Message(
        role: MessageRole.user,
        content: query,
      ));

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
      _chatHistory.add(Message(
        role: MessageRole.assistant,
        content: response,
      ));

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

  Future<String> _generateOllamaResponse(String query, RagSearchResult ragResult) async {
    try {
      // Build messages with RAG context
      final messages = <Message>[];
      
      // System message with RAG context
      if (ragResult.chunks.isNotEmpty) {
        messages.add(Message(
          role: MessageRole.system,
          content: '''당신은 주어진 문맥을 기반으로 질문에 답변하는 도우미입니다.

규칙:
1. 아래 문맥의 정보만을 기반으로 답변하세요.
2. 문맥에 관련 정보가 없으면 명확히 말씀해주세요.
3. 질문과 동일한 언어로 답변하세요.
4. 조항 번호(제X조)는 문맥에 있는 그대로 인용하세요.

문맥:
${ragResult.context.text}''',
        ));
      } else {
        messages.add(const Message(
          role: MessageRole.system,
          content: 'You are a helpful assistant.',
        ));
      }
      
      // Add chat history (last 10 messages for context)
      final historyStart = _chatHistory.length > 10 ? _chatHistory.length - 10 : 0;
      messages.addAll(_chatHistory.sublist(historyStart));
      
      // Add current user message
      messages.add(Message(
        role: MessageRole.user,
        content: query,
      ));
      
      // Save to history
      _chatHistory.add(Message(
        role: MessageRole.user,
        content: query,
      ));

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
      _chatHistory.add(Message(
        role: MessageRole.assistant,
        content: response,
      ));

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
      spans.add(TextSpan(
        text: match.group(1),
        style: const TextStyle(fontWeight: FontWeight.bold),
      ));
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
        setState(() => _status = 'Adding document ${i + 1}/${samples.length}...');
        await _ragService!.addSourceWithChunking(samples[i]);
      }

      await _ragService!.rebuildIndex();

      final stats = await _ragService!.getStats();
      _totalSources = stats.sourceCount.toInt();
      _totalChunks = stats.chunkCount.toInt();

      setState(() {
        _isLoading = false;
        _status = 'Added ${samples.length} documents! Total chunks: $_totalChunks';
      });

      _addSystemMessage('✅ Added ${samples.length} sample documents with $_totalChunks chunks.');
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
      appBar: AppBar(
        title: const Text('RAG Chat'),
        centerTitle: true,
        actions: [
          IconButton(
            icon: Icon(_showDebugInfo ? Icons.bug_report : Icons.bug_report_outlined),
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
      body: Column(
        children: [
          // Status bar
          if (_showDebugInfo)
            Container(
              padding: const EdgeInsets.symmetric(horizontal: 16, vertical: 8),
              color: Theme.of(context).colorScheme.surfaceContainerHighest,
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
                      style: const TextStyle(fontSize: 12),
                      overflow: TextOverflow.ellipsis,
                    ),
                  ),
                  Text(
                    '📄$_totalSources 📦$_totalChunks',
                    style: const TextStyle(fontSize: 12),
                  ),
                ],
              ),
            ),

          // Messages
          Expanded(
            child: _buildMessageList(),
          ),

          // Input area
          _buildInputArea(),
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
            Icon(
              Icons.chat_bubble_outline,
              size: 64,
              color: Colors.grey[400],
            ),
            const SizedBox(height: 16),
            Text(
              'No messages yet',
              style: TextStyle(color: Colors.grey[600]),
            ),
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
        return _buildMessageBubble(message);
      },
    );
  }

  Widget _buildMessageBubble(ChatMessage message) {
    final isUser = message.isUser;
    
    return Padding(
      padding: const EdgeInsets.only(bottom: 12),
      child: Row(
        mainAxisAlignment: isUser ? MainAxisAlignment.end : MainAxisAlignment.start,
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
              crossAxisAlignment: isUser ? CrossAxisAlignment.end : CrossAxisAlignment.start,
              children: [
                Container(
                  padding: const EdgeInsets.symmetric(horizontal: 14, vertical: 10),
                  decoration: BoxDecoration(
                    // User: pink bubble, AI: white bubble (matching existing style)
                    color: isUser 
                        ? const Color(0xFFFFE6E6)
                        : Colors.white,
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
                    style: TextStyle(
                      fontSize: 11,
                      color: Colors.grey[500],
                    ),
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
                style: const TextStyle(
                  color: Colors.black87,
                  fontSize: 15,
                ),
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
                        final result = await _ragService!.addSourceWithChunking(text);
                        await _ragService!.rebuildIndex();
                        
                        final stats = await _ragService!.getStats();
                        _totalSources = stats.sourceCount.toInt();
                        _totalChunks = stats.chunkCount.toInt();
                        
                        setState(() {
                          _isLoading = false;
                          _status = 'Document added! Chunks: ${result.chunkCount}';
                        });
                        
                        _addSystemMessage(
                          '✅ Document added with ${result.chunkCount} chunks.'
                        );
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
