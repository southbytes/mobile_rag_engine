// local-gemma-macos/lib/widgets/chat_input_area.dart
//
// Chat input area widget with text field and action buttons

import 'package:flutter/material.dart';
import 'package:flutter/services.dart';
import 'slash_command_overlay.dart';

/// Chat input area with text field, attachment button, and send button
class ChatInputArea extends StatelessWidget {
  final TextEditingController controller;
  final FocusNode focusNode;
  final bool isEnabled;
  final bool isGenerating;
  final VoidCallback onSend;
  final VoidCallback onAttach;

  /// Called when user types '/' - passes (showPopup, filterText)
  final void Function(bool showPopup, String filter)? onSlashInput;

  /// Currently selected slash command (shown as chip)
  final SlashCommand? selectedCommand;

  /// Called when user clears the selected command
  final VoidCallback? onClearCommand;

  /// Whether slash popup is currently visible
  final bool isSlashPopupVisible;

  /// Called when Enter is pressed while popup is visible
  final VoidCallback? onConfirmSlashSelection;

  /// Called when arrow up/down is pressed while popup is visible
  final void Function(bool isUp)? onArrowKey;

  const ChatInputArea({
    super.key,
    required this.controller,
    required this.focusNode,
    required this.isEnabled,
    required this.isGenerating,
    required this.onSend,
    required this.onAttach,
    this.onSlashInput,
    this.selectedCommand,
    this.onClearCommand,
    this.isSlashPopupVisible = false,
    this.onConfirmSlashSelection,
    this.onArrowKey,
  });

  void _onTextChanged(String text) {
    if (onSlashInput == null) return;

    // Don't show slash popup if a command is already selected
    if (selectedCommand != null) {
      onSlashInput!(false, '');
      return;
    }

    // Check if text starts with '/'
    if (text.startsWith('/')) {
      // Extract the command portion (before space or end)
      final spaceIndex = text.indexOf(' ');
      final filter = spaceIndex == -1 ? text : text.substring(0, spaceIndex);

      // Only show popup for partial commands (no space yet)
      if (spaceIndex == -1) {
        onSlashInput!(true, filter);
      } else {
        onSlashInput!(false, '');
      }
    } else {
      onSlashInput!(false, '');
    }
  }

  KeyEventResult _handleKeyEvent(FocusNode node, KeyEvent event) {
    if (!isSlashPopupVisible) return KeyEventResult.ignored;
    if (event is! KeyDownEvent) return KeyEventResult.ignored;

    // Intercept Enter key when popup is visible
    if (event.logicalKey == LogicalKeyboardKey.enter) {
      onConfirmSlashSelection?.call();
      return KeyEventResult.handled;
    }

    // Intercept arrow keys when popup is visible
    if (event.logicalKey == LogicalKeyboardKey.arrowUp) {
      onArrowKey?.call(true);
      return KeyEventResult.handled;
    }
    if (event.logicalKey == LogicalKeyboardKey.arrowDown) {
      onArrowKey?.call(false);
      return KeyEventResult.handled;
    }

    return KeyEventResult.ignored;
  }

  @override
  Widget build(BuildContext context) {
    return Container(
      padding: const EdgeInsets.fromLTRB(16, 8, 12, 8),
      decoration: BoxDecoration(
        color: const Color(0xFF1A1A1A),
        boxShadow: [
          BoxShadow(
            color: Colors.black.withValues(alpha: 0.3),
            spreadRadius: 1,
            blurRadius: 10,
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
              onPressed: isEnabled ? onAttach : null,
              icon: Icon(
                Icons.attach_file,
                color: isEnabled ? Colors.grey[400] : Colors.grey[700],
              ),
            ),
            // Selected command chip (if any)
            if (selectedCommand != null) ...[
              _buildCommandChip(),
              const SizedBox(width: 8),
            ],
            // Text input with key interception
            Expanded(
              child: Focus(
                onKeyEvent: _handleKeyEvent,
                child: TextField(
                  controller: controller,
                  focusNode: focusNode,
                  enabled: isEnabled && !isGenerating,
                  maxLines: 5,
                  minLines: 1,
                  textInputAction: TextInputAction.send,
                  onSubmitted: (_) {
                    // Only send if popup is not visible
                    if (!isSlashPopupVisible) {
                      onSend();
                    }
                  },
                  onChanged: _onTextChanged,
                  style: const TextStyle(color: Colors.white, fontSize: 15),
                  decoration: InputDecoration(
                    hintText: selectedCommand != null
                        ? _getHintForCommand(selectedCommand!)
                        : 'Ask a question... (/ for commands)',
                    hintStyle: TextStyle(color: Colors.grey[600]),
                    filled: true,
                    fillColor: Colors.grey[850],
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
            ),
            const SizedBox(width: 8),
            // Send button
            GestureDetector(
              onTap: isGenerating ? null : onSend,
              child: Container(
                width: 40,
                height: 40,
                decoration: BoxDecoration(
                  gradient: isGenerating
                      ? null
                      : LinearGradient(
                          colors: [Colors.purple[400]!, Colors.blue[400]!],
                        ),
                  color: isGenerating ? Colors.grey[700] : null,
                  shape: BoxShape.circle,
                ),
                child: isGenerating
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

  Widget _buildCommandChip() {
    final cmd = selectedCommand!;
    return Container(
      padding: const EdgeInsets.symmetric(horizontal: 10, vertical: 6),
      decoration: BoxDecoration(
        color: cmd.color.withValues(alpha: 0.2),
        borderRadius: BorderRadius.circular(16),
        border: Border.all(color: cmd.color.withValues(alpha: 0.5)),
      ),
      child: Row(
        mainAxisSize: MainAxisSize.min,
        children: [
          Icon(cmd.icon, size: 14, color: cmd.color),
          const SizedBox(width: 4),
          Text(
            cmd.label,
            style: TextStyle(
              color: cmd.color,
              fontWeight: FontWeight.w600,
              fontSize: 12,
            ),
          ),
          const SizedBox(width: 4),
          GestureDetector(
            onTap: onClearCommand,
            child: Icon(
              Icons.close,
              size: 14,
              color: cmd.color.withValues(alpha: 0.7),
            ),
          ),
        ],
      ),
    );
  }

  String _getHintForCommand(SlashCommand cmd) {
    switch (cmd.command) {
      case '/summary':
        return '요약할 내용을 입력하세요...';
      case '/define':
        return '정의할 용어를 입력하세요...';
      case '/more':
        return '확장할 질문을 입력하세요...';
      default:
        return '질문을 입력하세요...';
    }
  }
}
