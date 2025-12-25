// test_app/lib/screens/rag_chat_screen.dart
//
// RAG + LLM Chat Screen
// Uses mobile_rag_engine for retrieval and flutter_gemma for generation

import 'dart:io';
import 'package:flutter/material.dart';
import 'package:flutter/services.dart';
import 'package:path_provider/path_provider.dart';

import 'package:mobile_rag_engine/mobile_rag_engine.dart';
import 'package:flutter_gemma/flutter_gemma.dart';

/// Message model for chat
class ChatMessage {
  final String content;
  final bool isUser;
  final DateTime timestamp;
  final List<ChunkSearchResult>? retrievedChunks;
  final int? tokensUsed;
  final double? compressionRatio;  // 0.0-1.0, lower = more compressed
  final int? originalTokens;  // Before compression

  ChatMessage({
    required this.content,
    required this.isUser,
    DateTime? timestamp,
    this.retrievedChunks,
    this.tokensUsed,
    this.compressionRatio,
    this.originalTokens,
  }) : timestamp = timestamp ?? DateTime.now();
}

class RagChatScreen extends StatefulWidget {
  final bool mockLlm;

  const RagChatScreen({
    super.key,
    this.mockLlm = false,
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
  
  // LLM model and chat session (persistent for conversation history)
  InferenceModel? _llmModel;
  InferenceChat? _chatSession;
  
  // Debug info
  bool _showDebugInfo = true;
  int _totalChunks = 0;
  int _totalSources = 0;

  // Compression settings (Phase 1)
  int _compressionLevel = 1; // 0=minimal, 1=balanced, 2=aggressive
  bool _compressionEnabled = true;


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
      // Reset any existing LLM session to prevent context overflow on app restart
      await _resetChatSession();
      final dir = await getApplicationDocumentsDirectory();
      final dbPath = '${dir.path}/test_rag_chat.db';
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
        '• ${widget.mockLlm ? "(Mock mode - no LLM)" : "Using local Gemma LLM"}',
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
      // 1. RAG Search with compression level applied
      // Use lower token budget to leave room for LLM generation
      final searchBudget = _compressionLevel == 2 ? 800 : _compressionLevel == 1 ? 1000 : 1200;
      
      final ragResult = await _ragService!.search(
        text,
        topK: 5,
        tokenBudget: searchBudget,
        strategy: ContextStrategy.relevanceFirst,
      );

      // Calculate token stats for display
      final compressedTokens = ragResult.context.estimatedTokens;
      final chunkCount = ragResult.chunks.length;
      
      // Debug log stats
      debugPrint('📊 Context: $compressedTokens tokens from $chunkCount chunks');

      String response;
      if (widget.mockLlm) {
        // Mock mode - show the context
        response = _generateMockResponse(text, ragResult);
      } else {
        // Real LLM generation
        response = await _generateLlmResponse(text, ragResult);
      }

      // Add AI response
      setState(() {
        _messages.insert(0, ChatMessage(
          content: response,
          isUser: false,
          retrievedChunks: ragResult.chunks,
          tokensUsed: compressedTokens,
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

  String _generateMockResponse(String query, RagSearchResult ragResult) {
    if (ragResult.chunks.isEmpty) {
      return '📭 No relevant documents found.\n\nPlease add some documents using the menu.';
    }

    final buffer = StringBuffer();
    buffer.writeln('📚 Found ${ragResult.chunks.length} relevant chunks:');
    buffer.writeln('📊 Using ~${ragResult.context.estimatedTokens} tokens\n');
    
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

  Future<String> _generateLlmResponse(String query, RagSearchResult ragResult) async {
    try {
      // For RAG queries with documents, use fresh session to avoid context overflow
      // The 2048 token limit is shared between context + history + generation
      final hasRagContext = ragResult.chunks.isNotEmpty;
      
      if (hasRagContext) {
        // Reset session for fresh RAG context - prevents previous chat from taking up tokens
        await _resetChatSession();
        _llmModel = await FlutterGemma.getActiveModel(
          maxTokens: 2048,
          preferredBackend: PreferredBackend.gpu,
        );
        _chatSession = await _llmModel!.createChat();
      } else if (_llmModel == null) {
        // Initialize model for non-RAG questions
        _llmModel = await FlutterGemma.getActiveModel(
          maxTokens: 2048,
          preferredBackend: PreferredBackend.gpu,
        );
        _chatSession = await _llmModel!.createChat();
      }

      // Build prompt - RAG context or simple query
      String prompt;
      if (hasRagContext) {
        // Include RAG context with strict document-only instruction
        prompt = _ragService!.formatPrompt(query, ragResult);
      } else {
        // Simple follow-up question
        prompt = query;
      }

      // Add user message to chat history
      await _chatSession!.addQueryChunk(Message.text(
        text: prompt,
        isUser: true,
      ));

      // Generate response
      final response = await _chatSession!.generateChatResponse();

      // Extract text from response
      String responseText = '';
      if (response != null) {
        // Handle different response types
        if (response is TextResponse) {
          responseText = response.token;
        } else {
          // Fallback: try to extract text from toString()
          final raw = response.toString();
          // Parse TextResponse("...") format if needed
          final match = RegExp(r'TextResponse\("(.*)"\)$', dotAll: true).firstMatch(raw);
          if (match != null) {
            responseText = match.group(1) ?? raw;
          } else {
            responseText = raw;
          }
        }
      }


      // Disabled: _cleanRepetition was truncating valid content
      // responseText = _cleanRepetition(responseText);

      // Handle empty response - likely context overflow, reset and retry
      if (responseText.trim().isEmpty) {
        debugPrint('🟡 Empty response detected, resetting session and retrying...');
        await _resetChatSession();
        
        // Retry with fresh session
        _llmModel = await FlutterGemma.getActiveModel(
          maxTokens: 2048,
          preferredBackend: PreferredBackend.gpu,
        );
        _chatSession = await _llmModel!.createChat();
        
        // Use full RAG prompt for fresh session
        final freshPrompt = _ragService!.formatPrompt(query, ragResult);
        await _chatSession!.addQueryChunk(Message.text(
          text: freshPrompt,
          isUser: true,
        ));
        
        final retryResponse = await _chatSession!.generateChatResponse();
        if (retryResponse != null && retryResponse is TextResponse) {
          responseText = retryResponse.token; // Disabled: _cleanRepetition
        }
        
        if (responseText.trim().isEmpty) {
          return '⚠️ The model could not generate a response.\n\n'
                 'This may happen when the context is too long.\n'
                 'Try asking a simpler question.';
        }
      }

      return responseText;
    } catch (e, stackTrace) {
      // Log detailed error to terminal
      debugPrint('🔴 LLM Error: $e');
      debugPrint('🔴 Stack Trace: $stackTrace');
      
      // Reset session on error (will be recreated on next message)
      await _resetChatSession();
      
      // Get error type for user-friendly message
      String errorType = 'Unknown error';
      if (e.toString().contains('PlatformException')) {
        errorType = 'Model session error';
      } else if (e.toString().contains('StateError')) {
        errorType = 'Model not initialized';
      } else if (e.toString().contains('timeout')) {
        errorType = 'Request timed out';
      } else if (e.toString().contains('OUT_OF_RANGE')) {
        errorType = 'Context too long';
      }
      
      return '⚠️ LLM Error: $errorType\n\n'
             'The model encountered an issue. Please try again.\n'
             '(Check console for details)';
    }
  }

  Future<void> _resetChatSession() async {
    if (_llmModel != null) {
      await _llmModel!.close();
      _llmModel = null;
    }
    _chatSession = null;
  }

  /// Start a new chat session - clears messages and resets LLM context
  Future<void> _startNewChat() async {
    setState(() {
      _isLoading = true;
      _status = 'Starting new chat...';
    });

    await _resetChatSession();

    setState(() {
      _messages.clear();
      _isLoading = false;
      _status = 'Ready! Sources: $_totalSources, Chunks: $_totalChunks';
    });

    _addSystemMessage(
      '🔄 New chat started! LLM session has been reset.\n\n'
      '• Previous conversation context cleared\n'
      '• Ask me questions about your documents',
    );
  }

  /// Clean up repetition loops in LLM output
  /// Small LLMs sometimes get stuck repeating the same phrase
  /// Preserves original formatting (newlines, markdown)
  String _cleanRepetition(String text) {
    if (text.length < 200) return text;
    
    // Only detect actual repeated substrings (30+ chars, 2+ times)
    // This preserves the original text structure
    for (int len = 40; len <= 100; len += 10) {
      for (int i = 0; i < text.length - len * 2; i++) {
        final pattern = text.substring(i, i + len);
        
        // Skip patterns that are mostly whitespace
        if (pattern.trim().length < 20) continue;
        
        final rest = text.substring(i + len);
        final idx = rest.indexOf(pattern);
        
        if (idx != -1 && idx < len * 2) {
          // Found repetition close to original - truncate here
          debugPrint('🔄 Repetition detected, truncating...');
          return '${text.substring(0, i + len).trim()}...';
        }
      }
    }
    
    return text;
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

        '''Gemma is a family of lightweight, state-of-the-art open models from 
Google, built from the same research and technology used to create the 
Gemini models. Gemma models can run on-device, providing privacy-preserving 
AI capabilities without requiring cloud connectivity.''',
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
                  subtitle: Text('Reset LLM session'),
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
                  subtitle: const Text('Duplicates only'),
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
                  subtitle: const Text('+ Light filtering'),
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
                  subtitle: const Text('+ Stopwords'),
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

          // Input area (inspired by existing chat UI)
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
                        color: Colors.grey.withOpacity(0.1),
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
                    child: Text(
                      '~${message.tokensUsed} tokens • ${message.retrievedChunks?.length ?? 0} chunks',
                      style: TextStyle(
                        fontSize: 11,
                        color: Colors.grey[500],
                      ),
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
            color: Colors.grey.withOpacity(0.12),
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
    _resetChatSession(); // Cleanup LLM session
    EmbeddingService.dispose();
    super.dispose();
  }
}
