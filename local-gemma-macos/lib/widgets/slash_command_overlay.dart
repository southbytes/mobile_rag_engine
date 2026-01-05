// local-gemma-macos/lib/widgets/slash_command_overlay.dart
//
// Slash command auto-complete overlay widget

import 'package:flutter/material.dart';

/// Slash command definition
class SlashCommand {
  final String command;
  final String label;
  final String description;
  final IconData icon;
  final Color color;

  const SlashCommand({
    required this.command,
    required this.label,
    required this.description,
    required this.icon,
    this.color = Colors.purple,
  });
}

/// Available slash commands
const kSlashCommands = [
  SlashCommand(
    command: '/summary',
    label: '요약',
    description: 'RAG 결과를 핵심만 요약',
    icon: Icons.summarize,
    color: Colors.blue,
  ),
  SlashCommand(
    command: '/define',
    label: '정의',
    description: '용어의 학술적/실무적 정의 제공',
    icon: Icons.menu_book,
    color: Colors.green,
  ),
  SlashCommand(
    command: '/more',
    label: '확장',
    description: 'RAG + LLM 지식 결합',
    icon: Icons.auto_awesome,
    color: Colors.orange,
  ),
];

/// Slash command overlay popup widget
/// Now controlled externally - no internal keyboard handling
class SlashCommandOverlay extends StatelessWidget {
  final String filter;
  final int selectedIndex;
  final void Function(SlashCommand) onSelect;
  final VoidCallback onDismiss;

  const SlashCommandOverlay({
    super.key,
    required this.filter,
    required this.onSelect,
    required this.onDismiss,
    this.selectedIndex = 0,
  });

  List<SlashCommand> get _filteredCommands {
    if (filter.isEmpty || filter == '/') {
      return kSlashCommands;
    }
    final filterLower = filter.toLowerCase();
    return kSlashCommands.where((cmd) {
      return cmd.command.toLowerCase().startsWith(filterLower) ||
          cmd.label.toLowerCase().contains(filterLower.replaceFirst('/', ''));
    }).toList();
  }

  @override
  Widget build(BuildContext context) {
    final commands = _filteredCommands;

    if (commands.isEmpty) {
      return const SizedBox.shrink();
    }

    return Material(
      color: Colors.transparent,
      elevation: 8,
      borderRadius: BorderRadius.circular(12),
      child: Container(
        constraints: const BoxConstraints(maxHeight: 200, maxWidth: 300),
        decoration: BoxDecoration(
          color: const Color(0xFF2A2A2A),
          borderRadius: BorderRadius.circular(12),
          border: Border.all(color: Colors.grey.shade700),
        ),
        child: ClipRRect(
          borderRadius: BorderRadius.circular(12),
          child: ListView.builder(
            shrinkWrap: true,
            padding: const EdgeInsets.symmetric(vertical: 4),
            itemCount: commands.length,
            itemBuilder: (context, index) {
              final cmd = commands[index];
              final isSelected = index == selectedIndex;

              return GestureDetector(
                behavior:
                    HitTestBehavior.opaque, // Important: capture all touches
                onTapDown: (_) => onSelect(cmd),
                onTap: () => onSelect(cmd), // Backup handler
                child: MouseRegion(
                  cursor: SystemMouseCursors.click,
                  child: Container(
                    padding: const EdgeInsets.symmetric(
                      horizontal: 12,
                      vertical: 10,
                    ),
                    decoration: BoxDecoration(
                      color: isSelected
                          ? cmd.color.withValues(alpha: 0.2)
                          : Colors.transparent,
                    ),
                    child: Row(
                      children: [
                        Container(
                          width: 32,
                          height: 32,
                          decoration: BoxDecoration(
                            color: cmd.color.withValues(alpha: 0.2),
                            borderRadius: BorderRadius.circular(8),
                          ),
                          child: Icon(cmd.icon, size: 18, color: cmd.color),
                        ),
                        const SizedBox(width: 12),
                        Expanded(
                          child: Column(
                            crossAxisAlignment: CrossAxisAlignment.start,
                            children: [
                              Row(
                                children: [
                                  Text(
                                    cmd.command,
                                    style: TextStyle(
                                      color: cmd.color,
                                      fontWeight: FontWeight.bold,
                                      fontSize: 13,
                                    ),
                                  ),
                                  const SizedBox(width: 8),
                                  Text(
                                    cmd.label,
                                    style: TextStyle(
                                      color: Colors.grey[400],
                                      fontSize: 12,
                                    ),
                                  ),
                                ],
                              ),
                              const SizedBox(height: 2),
                              Text(
                                cmd.description,
                                style: TextStyle(
                                  color: Colors.grey[600],
                                  fontSize: 11,
                                ),
                                maxLines: 1,
                                overflow: TextOverflow.ellipsis,
                              ),
                            ],
                          ),
                        ),
                        if (isSelected)
                          Icon(
                            Icons.keyboard_return,
                            size: 14,
                            color: Colors.grey[500],
                          ),
                      ],
                    ),
                  ),
                ),
              );
            },
          ),
        ),
      ),
    );
  }
}
