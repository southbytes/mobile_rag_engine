// local-gemma-macos/lib/widgets/suggestion_chips.dart
//
// Collapsible suggestion chips for topic-based questions

import 'package:flutter/material.dart';
import '../services/topic_suggestion_service.dart';

/// Collapsible suggestion chips panel
class SuggestionChipsPanel extends StatelessWidget {
  final List<SuggestedQuestion> suggestions;
  final bool isLoading;
  final bool isExpanded;
  final bool isDisabled;
  final VoidCallback onToggleExpanded;
  final VoidCallback onRefresh;
  final void Function(SuggestedQuestion question) onQuestionSelected;

  const SuggestionChipsPanel({
    super.key,
    required this.suggestions,
    required this.isLoading,
    required this.isExpanded,
    required this.isDisabled,
    required this.onToggleExpanded,
    required this.onRefresh,
    required this.onQuestionSelected,
  });

  @override
  Widget build(BuildContext context) {
    if (isLoading) {
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

    if (suggestions.isEmpty) {
      return const SizedBox.shrink();
    }

    return Container(
      decoration: BoxDecoration(
        color: Theme.of(
          context,
        ).colorScheme.surfaceContainerHighest.withOpacity(0.5),
        border: Border(bottom: BorderSide(color: Colors.grey.withOpacity(0.2))),
      ),
      child: Column(
        crossAxisAlignment: CrossAxisAlignment.start,
        children: [
          // Header row (always visible, tappable to expand/collapse)
          InkWell(
            onTap: onToggleExpanded,
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
                      color: Colors.amber.withOpacity(0.2),
                      borderRadius: BorderRadius.circular(10),
                    ),
                    child: Text(
                      '${suggestions.length}',
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
                    onTap: onRefresh,
                    child: Icon(
                      Icons.refresh,
                      size: 16,
                      color: Colors.grey[500],
                    ),
                  ),
                  const SizedBox(width: 8),
                  // Expand/collapse icon
                  AnimatedRotation(
                    turns: isExpanded ? 0.5 : 0,
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
                children: suggestions.map((q) {
                  return ActionChip(
                    label: Text(
                      q.question,
                      style: TextStyle(
                        fontSize: 13,
                        color: isDisabled ? Colors.grey : null,
                      ),
                      maxLines: 2,
                      overflow: TextOverflow.ellipsis,
                    ),
                    backgroundColor: isDisabled
                        ? Colors.grey.withOpacity(0.2)
                        : Theme.of(
                            context,
                          ).colorScheme.primaryContainer.withOpacity(0.5),
                    side: BorderSide(
                      color: isDisabled
                          ? Colors.grey.withOpacity(0.3)
                          : Theme.of(
                              context,
                            ).colorScheme.primary.withOpacity(0.3),
                    ),
                    onPressed: isDisabled ? null : () => onQuestionSelected(q),
                  );
                }).toList(),
              ),
            ),
            secondChild: const SizedBox.shrink(),
            crossFadeState: isExpanded
                ? CrossFadeState.showFirst
                : CrossFadeState.showSecond,
            duration: const Duration(milliseconds: 200),
          ),
        ],
      ),
    );
  }
}
