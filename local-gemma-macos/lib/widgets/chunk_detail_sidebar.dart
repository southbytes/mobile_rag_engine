// local-gemma-macos/lib/widgets/chunk_detail_sidebar.dart
//
// Obsidian-style sidebar showing chunk details
// Uses scrollable_positioned_list for reliable scroll-to-index

import 'package:flutter/material.dart';
import 'package:mobile_rag_engine/mobile_rag_engine.dart';
import 'package:scrollable_positioned_list/scrollable_positioned_list.dart';

/// Obsidian-style chunk detail sidebar with auto-scroll to selected chunk
class ChunkDetailSidebar extends StatefulWidget {
  final List<ChunkSearchResult> chunks;
  final ChunkSearchResult? selectedChunk;
  final void Function(ChunkSearchResult chunk)? onChunkSelected;
  final String? searchQuery;

  const ChunkDetailSidebar({
    super.key,
    required this.chunks,
    this.selectedChunk,
    this.onChunkSelected,
    this.searchQuery,
  });

  @override
  State<ChunkDetailSidebar> createState() => _ChunkDetailSidebarState();
}

class _ChunkDetailSidebarState extends State<ChunkDetailSidebar> {
  final ItemScrollController _itemScrollController = ItemScrollController();
  final ItemPositionsListener _itemPositionsListener =
      ItemPositionsListener.create();

  @override
  void didUpdateWidget(ChunkDetailSidebar oldWidget) {
    super.didUpdateWidget(oldWidget);
    // Auto-scroll when selected chunk changes
    if (widget.selectedChunk != oldWidget.selectedChunk &&
        widget.selectedChunk != null) {
      _scrollToSelectedChunk();
    }
  }

  void _scrollToSelectedChunk() {
    if (widget.selectedChunk == null) return;

    // Find target index
    final targetIndex = widget.chunks.indexWhere(
      (c) => c.chunkId == widget.selectedChunk!.chunkId,
    );
    if (targetIndex == -1) return;

    // Check if target is already visible - skip scroll if so
    final visiblePositions = _itemPositionsListener.itemPositions.value;
    final isAlreadyVisible = visiblePositions.any((pos) {
      if (pos.index != targetIndex) return false;
      // Check if item is fully or mostly visible (between 0% and 80% of viewport)
      return pos.itemLeadingEdge >= 0 && pos.itemTrailingEdge <= 1.0;
    });

    if (isAlreadyVisible) return; // Already visible, no scroll needed

    // Use ItemScrollController to scroll to index
    if (_itemScrollController.isAttached) {
      _itemScrollController.scrollTo(
        index: targetIndex,
        duration: const Duration(milliseconds: 300),
        curve: Curves.easeInOut,
        alignment: 0.3, // Position at 30% from top
      );
    }
  }

  @override
  Widget build(BuildContext context) {
    if (widget.chunks.isEmpty) {
      return Container(
        color: const Color(0xFF1E1E1E),
        child: Center(
          child: Column(
            mainAxisAlignment: MainAxisAlignment.center,
            children: [
              Icon(Icons.article_outlined, size: 32, color: Colors.grey[600]),
              const SizedBox(height: 8),
              Text(
                'No chunks',
                style: TextStyle(color: Colors.grey[500], fontSize: 12),
              ),
            ],
          ),
        ),
      );
    }

    return Container(
      color: const Color(0xFF1E1E1E),
      child: Column(
        children: [
          // Header with count
          Container(
            padding: const EdgeInsets.all(8),
            decoration: BoxDecoration(
              color: const Color(0xFF252525),
              border: Border(bottom: BorderSide(color: Colors.grey[800]!)),
            ),
            child: Row(
              children: [
                Icon(Icons.search, size: 14, color: Colors.grey[400]),
                const SizedBox(width: 6),
                Text(
                  '${widget.chunks.length} results',
                  style: TextStyle(
                    color: Colors.grey[400],
                    fontSize: 11,
                    fontWeight: FontWeight.w500,
                  ),
                ),
              ],
            ),
          ),
          // Chunk list using ScrollablePositionedList
          Expanded(
            child: ScrollablePositionedList.builder(
              itemScrollController: _itemScrollController,
              itemPositionsListener: _itemPositionsListener,
              padding: const EdgeInsets.symmetric(vertical: 4),
              itemCount: widget.chunks.length,
              itemBuilder: (context, index) {
                final chunk = widget.chunks[index];
                final isSelected =
                    widget.selectedChunk?.chunkId == chunk.chunkId;

                return _ChunkCard(
                  chunk: chunk,
                  isSelected: isSelected,
                  searchQuery: widget.searchQuery,
                  onTap: () => widget.onChunkSelected?.call(chunk),
                );
              },
            ),
          ),
        ],
      ),
    );
  }
}

class _ChunkCard extends StatelessWidget {
  final ChunkSearchResult chunk;
  final bool isSelected;
  final String? searchQuery;
  final VoidCallback? onTap;

  const _ChunkCard({
    required this.chunk,
    required this.isSelected,
    this.searchQuery,
    this.onTap,
  });

  @override
  Widget build(BuildContext context) {
    final isAdjacent = chunk.similarity == 0.0;
    final similarity = chunk.similarity;

    return GestureDetector(
      onTap: onTap,
      child: AnimatedContainer(
        duration: const Duration(milliseconds: 200),
        margin: const EdgeInsets.symmetric(horizontal: 6, vertical: 2),
        padding: const EdgeInsets.all(8),
        decoration: BoxDecoration(
          color: isSelected
              ? Colors.blue.withValues(alpha: 0.25)
              : const Color(0xFF2A2A2A),
          borderRadius: BorderRadius.circular(6),
          border: Border.all(
            color: isSelected ? Colors.blue : Colors.transparent,
            width: 1.5,
          ),
          boxShadow: isSelected
              ? [
                  BoxShadow(
                    color: Colors.blue.withValues(alpha: 0.4),
                    blurRadius: 10,
                    spreadRadius: 2,
                  ),
                ]
              : null,
        ),
        child: Column(
          crossAxisAlignment: CrossAxisAlignment.start,
          children: [
            // Source info and similarity
            Row(
              children: [
                // Similarity badge
                if (!isAdjacent)
                  Container(
                    padding: const EdgeInsets.symmetric(
                      horizontal: 4,
                      vertical: 1,
                    ),
                    decoration: BoxDecoration(
                      color: _getSimilarityColor(
                        similarity,
                      ).withValues(alpha: 0.2),
                      borderRadius: BorderRadius.circular(3),
                    ),
                    child: Text(
                      '${(similarity * 100).toStringAsFixed(0)}%',
                      style: TextStyle(
                        color: _getSimilarityColor(similarity),
                        fontSize: 9,
                        fontWeight: FontWeight.w600,
                      ),
                    ),
                  )
                else
                  Container(
                    padding: const EdgeInsets.symmetric(
                      horizontal: 4,
                      vertical: 1,
                    ),
                    decoration: BoxDecoration(
                      color: Colors.grey.withValues(alpha: 0.2),
                      borderRadius: BorderRadius.circular(3),
                    ),
                    child: Text(
                      'adj',
                      style: TextStyle(color: Colors.grey[500], fontSize: 9),
                    ),
                  ),
                const Spacer(),
                // Selected indicator
                if (isSelected)
                  Container(
                    padding: const EdgeInsets.symmetric(
                      horizontal: 4,
                      vertical: 1,
                    ),
                    decoration: BoxDecoration(
                      color: Colors.blue.withValues(alpha: 0.3),
                      borderRadius: BorderRadius.circular(3),
                    ),
                    child: Row(
                      mainAxisSize: MainAxisSize.min,
                      children: [
                        Icon(
                          Icons.visibility,
                          size: 10,
                          color: Colors.blue[300],
                        ),
                        const SizedBox(width: 2),
                        Text(
                          'viewing',
                          style: TextStyle(
                            color: Colors.blue[300],
                            fontSize: 8,
                            fontWeight: FontWeight.w500,
                          ),
                        ),
                      ],
                    ),
                  ),
                const SizedBox(width: 4),
                // Chunk index
                Text(
                  '#${chunk.chunkIndex}',
                  style: TextStyle(color: Colors.grey[600], fontSize: 9),
                ),
              ],
            ),
            const SizedBox(height: 6),
            // Content preview with keyword highlighting
            _buildHighlightedText(
              chunk.content,
              searchQuery,
              isSelected: isSelected,
            ),
          ],
        ),
      ),
    );
  }

  Widget _buildHighlightedText(
    String text,
    String? query, {
    bool isSelected = false,
  }) {
    // Show full content when selected, otherwise truncate
    final preview = isSelected
        ? text
        : (text.length > 150 ? '${text.substring(0, 150)}...' : text);

    // Selected: 5% larger font, semi-bold, brighter color
    final baseFontSize = isSelected ? 11.55 : 11.0;
    final fontWeight = isSelected ? FontWeight.w500 : FontWeight.normal;
    final baseStyle = TextStyle(
      color: isSelected ? Colors.white : Colors.grey[300],
      fontSize: baseFontSize,
      height: 1.4,
      fontWeight: fontWeight,
    );

    final maxLines = isSelected ? null : 4;

    if (query == null || query.isEmpty) {
      return Text(
        preview,
        maxLines: maxLines,
        overflow: isSelected ? TextOverflow.visible : TextOverflow.ellipsis,
        style: baseStyle,
      );
    }

    // Highlight query keywords
    final keywords = query
        .split(RegExp(r'\s+'))
        .where((w) => w.length > 2)
        .toList();

    if (keywords.isEmpty) {
      return Text(
        preview,
        maxLines: maxLines,
        overflow: isSelected ? TextOverflow.visible : TextOverflow.ellipsis,
        style: baseStyle,
      );
    }

    final pattern = keywords.map(RegExp.escape).join('|');
    final regex = RegExp('($pattern)', caseSensitive: false);
    final spans = <TextSpan>[];
    var lastEnd = 0;

    for (final match in regex.allMatches(preview)) {
      if (match.start > lastEnd) {
        spans.add(TextSpan(text: preview.substring(lastEnd, match.start)));
      }
      spans.add(
        TextSpan(
          text: match.group(0),
          style: TextStyle(
            backgroundColor: isSelected
                ? const Color(0xFF6B5200)
                : const Color(0xFF4A3B00),
            color: Colors.yellow,
            fontWeight: FontWeight.bold,
          ),
        ),
      );
      lastEnd = match.end;
    }

    if (lastEnd < preview.length) {
      spans.add(TextSpan(text: preview.substring(lastEnd)));
    }

    return RichText(
      maxLines: maxLines,
      overflow: isSelected ? TextOverflow.visible : TextOverflow.ellipsis,
      text: TextSpan(style: baseStyle, children: spans),
    );
  }

  Color _getSimilarityColor(double sim) {
    if (sim >= 0.7) return Colors.green;
    if (sim >= 0.5) return Colors.orange;
    return Colors.red;
  }
}
