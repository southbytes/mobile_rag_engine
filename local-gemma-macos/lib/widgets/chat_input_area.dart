// local-gemma-macos/lib/widgets/chat_input_area.dart
//
// Chat input area widget with text field and action buttons

import 'package:flutter/material.dart';

/// Chat input area with text field, attachment button, and send button
class ChatInputArea extends StatelessWidget {
  final TextEditingController controller;
  final FocusNode focusNode;
  final bool isEnabled;
  final bool isGenerating;
  final VoidCallback onSend;
  final VoidCallback onAttach;

  const ChatInputArea({
    super.key,
    required this.controller,
    required this.focusNode,
    required this.isEnabled,
    required this.isGenerating,
    required this.onSend,
    required this.onAttach,
  });

  @override
  Widget build(BuildContext context) {
    return Container(
      padding: const EdgeInsets.fromLTRB(16, 8, 12, 8),
      decoration: BoxDecoration(
        color: const Color(0xFF1A1A1A),
        boxShadow: [
          BoxShadow(
            color: Colors.black.withOpacity(0.3),
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
            // Text input
            Expanded(
              child: TextField(
                controller: controller,
                focusNode: focusNode,
                enabled: isEnabled && !isGenerating,
                maxLines: 5,
                minLines: 1,
                textInputAction: TextInputAction.send,
                onSubmitted: (_) => onSend(),
                style: const TextStyle(color: Colors.white, fontSize: 15),
                decoration: InputDecoration(
                  hintText: 'Ask a question...',
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
}
