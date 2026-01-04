// local-gemma-macos/lib/widgets/document_style_response.dart
//
// Document-style AI response widget (Gemini/NotebookLM style)
// Replaces the chat bubble for AI responses

import 'package:flutter/material.dart';
import 'package:flutter_markdown/flutter_markdown.dart';
import '../models/chat_models.dart';

/// Document-style response widget for AI messages
/// Shows content in a full-width document format with source attribution
class DocumentStyleResponse extends StatelessWidget {
  final ChatMessage message;
  final bool showDebugInfo;
  final VoidCallback? onViewGraph;
  final bool isGraphActive;

  const DocumentStyleResponse({
    super.key,
    required this.message,
    this.showDebugInfo = false,
    this.onViewGraph,
    this.isGraphActive = false,
  });

  @override
  Widget build(BuildContext context) {
    final hasChunks =
        message.retrievedChunks != null && message.retrievedChunks!.isNotEmpty;

    // Get best similarity for display
    double? bestSimilarity;
    if (hasChunks) {
      final validSims = message.retrievedChunks!
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
                  _formatTime(message.timestamp),
                  style: TextStyle(color: Colors.grey[600], fontSize: 11),
                ),
              ],
            ),
          ),

          // Content (Markdown)
          Padding(
            padding: const EdgeInsets.all(16),
            child: MarkdownBody(
              data: message.content,
              styleSheet: MarkdownStyleSheet(
                p: const TextStyle(
                  color: Colors.white,
                  fontSize: 14,
                  height: 1.6,
                ),
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
                em: const TextStyle(
                  color: Colors.white,
                  fontStyle: FontStyle.italic,
                ),
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
            ),
          ),

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
                              '${message.retrievedChunks!.length} chunks',
                            ),
                            if (message.tokensUsed != null)
                              _buildInfoChip(
                                Icons.token_outlined,
                                '~${message.tokensUsed} tokens',
                              ),
                            if (bestSimilarity != null)
                              _buildSimilarityChip(bestSimilarity),
                          ],
                        ),
                      ),
                    // Right: graph button
                    if (hasChunks && onViewGraph != null)
                      Container(
                        decoration: isGraphActive
                            ? BoxDecoration(
                                color: Colors.purple.withValues(alpha: 0.2),
                                borderRadius: BorderRadius.circular(6),
                                border: Border.all(
                                  color: Colors.purple.withValues(alpha: 0.5),
                                ),
                              )
                            : null,
                        child: IconButton(
                          onPressed: onViewGraph,
                          icon: Icon(
                            isGraphActive ? Icons.hub : Icons.hub_outlined,
                            size: 18,
                            color: isGraphActive
                                ? Colors.purple
                                : Colors.grey[500],
                          ),
                          tooltip: isGraphActive
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

                // Debug timing info
                if (showDebugInfo && message.ragSearchTime != null)
                  Padding(
                    padding: const EdgeInsets.only(top: 8),
                    child: Text(
                      '⚡ RAG: ${message.ragSearchTime!.inMilliseconds}ms • '
                      'LLM: ${message.llmGenerationTime?.inMilliseconds ?? 0}ms • '
                      'Total: ${message.totalTime?.inMilliseconds ?? 0}ms',
                      style: TextStyle(
                        fontSize: 10,
                        color: Colors.blue[400],
                        fontFamily: 'monospace',
                      ),
                    ),
                  ),
              ],
            ),
          ),
        ],
      ),
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
