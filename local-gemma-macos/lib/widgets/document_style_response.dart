// local-gemma-macos/lib/widgets/document_style_response.dart
//
// Document-style AI response widget (Gemini/NotebookLM style)
// Replaces the chat bubble for AI responses

import 'package:flutter/material.dart';
import 'package:flutter_markdown/flutter_markdown.dart';
import 'package:flutter_streaming_text_markdown/flutter_streaming_text_markdown.dart';
import '../models/chat_models.dart';

/// Document-style response widget for AI messages
/// Shows content in a full-width document format with source attribution
class DocumentStyleResponse extends StatefulWidget {
  final ChatMessage message;
  final bool showDebugInfo;
  final VoidCallback? onViewGraph;
  final bool isGraphActive;
  final bool shouldAnimate; // Whether to animate this message

  const DocumentStyleResponse({
    super.key,
    required this.message,
    this.showDebugInfo = false,
    this.onViewGraph,
    this.isGraphActive = false,
    this.shouldAnimate = false,
  });

  @override
  State<DocumentStyleResponse> createState() => _DocumentStyleResponseState();
}

class _DocumentStyleResponseState extends State<DocumentStyleResponse> {
  bool _animationComplete = false;

  @override
  Widget build(BuildContext context) {
    final hasChunks =
        widget.message.retrievedChunks != null &&
        widget.message.retrievedChunks!.isNotEmpty;

    // Get best similarity for display
    double? bestSimilarity;
    if (hasChunks) {
      final validSims = widget.message.retrievedChunks!
          .map((c) => c.similarity)
          .where((s) => s > 0)
          .toList();
      if (validSims.isNotEmpty) {
        bestSimilarity = validSims.reduce((a, b) => a > b ? a : b);
      }
    }

    return Container(
      margin: const EdgeInsets.only(bottom: 16),
      decoration: BoxDecoration(
        color: const Color(0xFF1E1E1E),
        borderRadius: BorderRadius.circular(12),
        border: Border.all(color: Colors.grey[800]!, width: 0.5),
      ),
      child: Column(
        crossAxisAlignment: CrossAxisAlignment.start,
        children: [
          // Header
          Container(
            padding: const EdgeInsets.symmetric(horizontal: 16, vertical: 10),
            decoration: BoxDecoration(
              border: Border(bottom: BorderSide(color: Colors.grey[800]!)),
            ),
            child: Row(
              children: [
                Container(
                  width: 24,
                  height: 24,
                  decoration: BoxDecoration(
                    gradient: LinearGradient(
                      colors: [Colors.purple[400]!, Colors.blue[400]!],
                    ),
                    borderRadius: BorderRadius.circular(6),
                  ),
                  child: const Icon(
                    Icons.auto_awesome,
                    color: Colors.white,
                    size: 14,
                  ),
                ),
                const SizedBox(width: 10),
                Text(
                  'AI Response',
                  style: TextStyle(
                    color: Colors.grey[300],
                    fontSize: 13,
                    fontWeight: FontWeight.w500,
                  ),
                ),
                const Spacer(),
                // Timestamp
                Text(
                  _formatTime(widget.message.timestamp),
                  style: TextStyle(color: Colors.grey[600], fontSize: 11),
                ),
              ],
            ),
          ),

          // Content (Markdown with optional streaming animation)
          Padding(padding: const EdgeInsets.all(16), child: _buildContent()),

          // Footer with sources and debug info
          Container(
            padding: const EdgeInsets.symmetric(horizontal: 16, vertical: 10),
            decoration: BoxDecoration(
              color: Colors.grey[900],
              borderRadius: const BorderRadius.vertical(
                bottom: Radius.circular(12),
              ),
            ),
            child: Column(
              crossAxisAlignment: CrossAxisAlignment.start,
              children: [
                // Source info and graph button row
                Row(
                  children: [
                    // Left: info chips
                    if (hasChunks)
                      Expanded(
                        child: Wrap(
                          spacing: 12,
                          runSpacing: 6,
                          crossAxisAlignment: WrapCrossAlignment.center,
                          children: [
                            _buildInfoChip(
                              Icons.layers_outlined,
                              '${widget.message.retrievedChunks!.length} chunks',
                            ),
                            if (widget.message.tokensUsed != null)
                              _buildInfoChip(
                                Icons.token_outlined,
                                '~${widget.message.tokensUsed} tokens',
                              ),
                            if (bestSimilarity != null)
                              _buildSimilarityChip(bestSimilarity),
                          ],
                        ),
                      ),
                    // Right: graph button
                    if (hasChunks && widget.onViewGraph != null)
                      Container(
                        decoration: widget.isGraphActive
                            ? BoxDecoration(
                                color: Colors.purple.withValues(alpha: 0.2),
                                borderRadius: BorderRadius.circular(6),
                                border: Border.all(
                                  color: Colors.purple.withValues(alpha: 0.5),
                                ),
                              )
                            : null,
                        child: IconButton(
                          onPressed: widget.onViewGraph,
                          icon: Icon(
                            widget.isGraphActive
                                ? Icons.hub
                                : Icons.hub_outlined,
                            size: 18,
                            color: widget.isGraphActive
                                ? Colors.purple
                                : Colors.grey[500],
                          ),
                          tooltip: widget.isGraphActive
                              ? 'Viewing in Graph'
                              : 'View in Graph',
                          padding: const EdgeInsets.all(4),
                          constraints: const BoxConstraints(
                            minWidth: 32,
                            minHeight: 32,
                          ),
                        ),
                      ),
                  ],
                ),

                // Processing pipeline summary (always visible)
                if (hasChunks || widget.message.queryType != null)
                  Padding(
                    padding: const EdgeInsets.only(top: 8),
                    child: _buildProcessingSummary(bestSimilarity),
                  ),
              ],
            ),
          ),
        ],
      ),
    );
  }

  /// Build content with optional streaming animation
  Widget _buildContent() {
    // Use streaming animation for new messages
    if (widget.shouldAnimate && !_animationComplete) {
      return StreamingText(
        text: widget.message.content,
        style: const TextStyle(color: Colors.white, fontSize: 14, height: 1.6),
        typingSpeed: const Duration(milliseconds: 15),
        onComplete: () {
          if (mounted) {
            setState(() => _animationComplete = true);
          }
        },
      );
    }

    // Use standard markdown for completed or older messages
    return MarkdownBody(
      data: widget.message.content,
      styleSheet: MarkdownStyleSheet(
        p: const TextStyle(color: Colors.white, fontSize: 14, height: 1.6),
        h1: const TextStyle(
          color: Colors.white,
          fontSize: 20,
          fontWeight: FontWeight.bold,
        ),
        h2: const TextStyle(
          color: Colors.white,
          fontSize: 18,
          fontWeight: FontWeight.bold,
        ),
        h3: const TextStyle(
          color: Colors.white,
          fontSize: 16,
          fontWeight: FontWeight.w600,
        ),
        strong: const TextStyle(
          color: Colors.white,
          fontWeight: FontWeight.bold,
        ),
        em: const TextStyle(color: Colors.white, fontStyle: FontStyle.italic),
        listBullet: TextStyle(color: Colors.grey[400]),
        code: TextStyle(
          backgroundColor: Colors.grey[850],
          color: Colors.green[300],
          fontSize: 13,
        ),
        codeblockDecoration: BoxDecoration(
          color: Colors.grey[850],
          borderRadius: BorderRadius.circular(8),
        ),
      ),
      selectable: true,
    );
  }

  /// Build compact processing pipeline summary
  /// Format: 🔍 explanation | 📚 42 chunks (91.8%) | ⚡ 4.5s
  Widget _buildProcessingSummary(double? bestSimilarity) {
    final parts = <Widget>[];

    // Query type badge
    if (widget.message.queryType != null) {
      parts.add(
        _buildPipelineBadge(
          icon: Icons.search,
          label: widget.message.queryType!,
          color: Colors.blue[400]!,
        ),
      );
    }

    // Chunk count with similarity
    if (widget.message.retrievedChunks != null &&
        widget.message.retrievedChunks!.isNotEmpty) {
      final chunkCount = widget.message.retrievedChunks!.length;
      final simText = bestSimilarity != null && bestSimilarity > 0
          ? ' (${(bestSimilarity * 100).toStringAsFixed(0)}%)'
          : '';
      parts.add(
        _buildPipelineBadge(
          icon: Icons.library_books,
          label: '$chunkCount chunks$simText',
          color: Colors.green[400]!,
        ),
      );
    }

    // Total time
    if (widget.message.totalTime != null) {
      final seconds = widget.message.totalTime!.inMilliseconds / 1000;
      parts.add(
        _buildPipelineBadge(
          icon: Icons.bolt,
          label: '${seconds.toStringAsFixed(1)}s',
          color: Colors.orange[400]!,
        ),
      );
    }

    if (parts.isEmpty) return const SizedBox.shrink();

    // Add separators between parts
    final separatedParts = <Widget>[];
    for (var i = 0; i < parts.length; i++) {
      separatedParts.add(parts[i]);
      if (i < parts.length - 1) {
        separatedParts.add(
          Padding(
            padding: const EdgeInsets.symmetric(horizontal: 6),
            child: Text(
              '•',
              style: TextStyle(color: Colors.grey[600], fontSize: 10),
            ),
          ),
        );
      }
    }

    return Row(mainAxisSize: MainAxisSize.min, children: separatedParts);
  }

  Widget _buildPipelineBadge({
    required IconData icon,
    required String label,
    required Color color,
  }) {
    return Row(
      mainAxisSize: MainAxisSize.min,
      children: [
        Icon(icon, size: 12, color: color),
        const SizedBox(width: 4),
        Text(
          label,
          style: TextStyle(
            color: color,
            fontSize: 10,
            fontWeight: FontWeight.w500,
          ),
        ),
      ],
    );
  }

  Widget _buildInfoChip(IconData icon, String label) {
    return Row(
      mainAxisSize: MainAxisSize.min,
      children: [
        Icon(icon, size: 14, color: Colors.grey[500]),
        const SizedBox(width: 4),
        Text(label, style: TextStyle(color: Colors.grey[400], fontSize: 11)),
      ],
    );
  }

  Widget _buildSimilarityChip(double similarity) {
    final color = similarity >= 0.7
        ? Colors.green
        : similarity >= 0.5
        ? Colors.orange
        : Colors.red;

    return Container(
      padding: const EdgeInsets.symmetric(horizontal: 6, vertical: 2),
      decoration: BoxDecoration(
        color: color.withValues(alpha: 0.2),
        borderRadius: BorderRadius.circular(4),
      ),
      child: Row(
        mainAxisSize: MainAxisSize.min,
        children: [
          Icon(Icons.analytics_outlined, size: 12, color: color),
          const SizedBox(width: 4),
          Text(
            '${(similarity * 100).toStringAsFixed(0)}% match',
            style: TextStyle(
              color: color,
              fontSize: 10,
              fontWeight: FontWeight.w600,
            ),
          ),
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
}

/// User message bubble (simple style)
class UserMessageBubble extends StatelessWidget {
  final ChatMessage message;

  const UserMessageBubble({super.key, required this.message});

  @override
  Widget build(BuildContext context) {
    return Padding(
      padding: const EdgeInsets.only(bottom: 12),
      child: Row(
        mainAxisAlignment: MainAxisAlignment.end,
        children: [
          Flexible(
            child: Container(
              padding: const EdgeInsets.symmetric(horizontal: 14, vertical: 10),
              decoration: BoxDecoration(
                color: Colors.blue[600],
                borderRadius: BorderRadius.circular(16),
              ),
              child: Text(
                message.content,
                style: const TextStyle(color: Colors.white, fontSize: 14),
              ),
            ),
          ),
        ],
      ),
    );
  }
}
