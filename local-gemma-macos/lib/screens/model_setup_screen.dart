// local-gemma-macos/lib/screens/model_setup_screen.dart
//
// Screen for downloading Ollama models

import 'package:flutter/material.dart';
import 'package:ollama_dart/ollama_dart.dart';

class ModelSetupScreen extends StatefulWidget {
  const ModelSetupScreen({super.key});

  @override
  State<ModelSetupScreen> createState() => _ModelSetupScreenState();
}

class _ModelSetupScreenState extends State<ModelSetupScreen> {
  double _progress = 0;
  String _status = 'Ready to download';
  bool _isDownloading = false;
  bool _isComplete = false;
  String? _error;
  
  final OllamaClient _client = OllamaClient();

  // Model options for Ollama
  final List<_ModelOption> _models = [
    _ModelOption(
      name: 'Gemma 3 4B ⭐ (Best for RAG)',
      description: 'Google Gemma 3 4B - balanced quality/speed (~2.6GB)',
      modelId: 'gemma3:4b',
    ),
    _ModelOption(
      name: 'Gemma 3 1B (Lightweight)',
      description: 'Google Gemma 3 1B - fast, lower quality (~815MB)',
      modelId: 'gemma3:1b',
    ),
    _ModelOption(
      name: 'Gemma 3 12B (High Quality)',
      description: 'Google Gemma 3 12B - best quality, slower (~7.8GB)',
      modelId: 'gemma3:12b',
    ),
    _ModelOption(
      name: 'Llama 3.2 3B',
      description: 'Meta Llama 3.2 3B - good general purpose (~2GB)',
      modelId: 'llama3.2:3b',
    ),
    _ModelOption(
      name: 'Mistral 7B',
      description: 'Mistral 7B - excellent for reasoning (~4.1GB)',
      modelId: 'mistral:latest',
    ),
    _ModelOption(
      name: 'Qwen 3 4B',
      description: 'Alibaba Qwen 3 4B - multilingual support (~2.6GB)',
      modelId: 'qwen3:4b',
    ),
  ];

  int _selectedModelIndex = 0;

  Future<void> _downloadModel() async {
    final model = _models[_selectedModelIndex];

    setState(() {
      _isDownloading = true;
      _progress = 0;
      _status = 'Starting download...';
      _error = null;
    });

    try {
      final stream = _client.pullModelStream(
        request: PullModelRequest(model: model.modelId),
      );

      await for (final event in stream) {
        if (event.total != null && event.total! > 0) {
          final progress = (event.completed ?? 0) / event.total!;
          setState(() {
            _progress = progress;
            _status = '${event.status ?? "Downloading"}: ${(progress * 100).toStringAsFixed(1)}%';
          });
        } else {
          setState(() {
            _status = (event.status ?? 'Processing...').toString();
          });
        }
      }

      setState(() {
        _isComplete = true;
        _status = 'Installation complete!';
        _isDownloading = false;
      });
    } catch (e) {
      setState(() {
        _error = e.toString();
        _status = 'Download failed';
        _isDownloading = false;
      });
    }
  }

  @override
  Widget build(BuildContext context) {
    return Scaffold(
      appBar: AppBar(
        title: const Text('Download LLM Model'),
      ),
      body: SingleChildScrollView(
        padding: const EdgeInsets.all(24),
        child: Column(
          crossAxisAlignment: CrossAxisAlignment.stretch,
          children: [
            // Model selection
            Card(
              child: Padding(
                padding: const EdgeInsets.all(16),
                child: Column(
                  crossAxisAlignment: CrossAxisAlignment.start,
                  children: [
                    Text(
                      'Select Model',
                      style: Theme.of(context).textTheme.titleMedium,
                    ),
                    const SizedBox(height: 16),
                    ...List.generate(_models.length, (index) {
                      final model = _models[index];
                      return RadioListTile<int>(
                        value: index,
                        groupValue: _selectedModelIndex,
                        onChanged: _isDownloading
                            ? null
                            : (value) {
                                setState(() => _selectedModelIndex = value!);
                              },
                        title: Text(model.name),
                        subtitle: Text(model.description),
                      );
                    }),
                  ],
                ),
              ),
            ),

            const SizedBox(height: 24),

            // Progress section
            Card(
              child: Padding(
                padding: const EdgeInsets.all(16),
                child: Column(
                  children: [
                    if (_isDownloading) ...[
                      LinearProgressIndicator(value: _progress),
                      const SizedBox(height: 16),
                    ],
                    Row(
                      children: [
                        Icon(
                          _isComplete
                              ? Icons.check_circle
                              : _error != null
                                  ? Icons.error
                                  : Icons.info,
                          color: _isComplete
                              ? Colors.green
                              : _error != null
                                  ? Colors.red
                                  : Colors.blue,
                        ),
                        const SizedBox(width: 12),
                        Expanded(
                          child: Text(_status),
                        ),
                      ],
                    ),
                    if (_error != null) ...[
                      const SizedBox(height: 8),
                      Text(
                        _error!,
                        style: TextStyle(
                          color: Colors.red[700],
                          fontSize: 12,
                        ),
                      ),
                    ],
                  ],
                ),
              ),
            ),

            const SizedBox(height: 24),

            // Action buttons
            if (!_isComplete)
              FilledButton.icon(
                onPressed: _isDownloading ? null : _downloadModel,
                icon: _isDownloading
                    ? const SizedBox(
                        width: 20,
                        height: 20,
                        child: CircularProgressIndicator(
                          strokeWidth: 2,
                          color: Colors.white,
                        ),
                      )
                    : const Icon(Icons.download),
                label: Text(_isDownloading ? 'Downloading...' : 'Download Model'),
              ),
            if (_isComplete)
              FilledButton.icon(
                onPressed: () => Navigator.pop(
                  context, 
                  _models[_selectedModelIndex].modelId,
                ),
                icon: const Icon(Icons.check),
                label: const Text('Continue'),
              ),

            const SizedBox(height: 24),

            // Info card
            Card(
              color: Theme.of(context).colorScheme.surfaceContainerHighest,
              child: const Padding(
                padding: EdgeInsets.all(16),
                child: Column(
                  crossAxisAlignment: CrossAxisAlignment.start,
                  children: [
                    Row(
                      children: [
                        Icon(Icons.info_outline, size: 20),
                        SizedBox(width: 8),
                        Text('Note'),
                      ],
                    ),
                    SizedBox(height: 8),
                    Text(
                      '• Models are managed by Ollama\n'
                      '• Ollama server must be running (ollama serve)\n'
                      '• Models are stored in ~/.ollama/models',
                      style: TextStyle(fontSize: 13),
                    ),
                  ],
                ),
              ),
            ),
          ],
        ),
      ),
    );
  }
}

class _ModelOption {
  final String name;
  final String description;
  final String modelId;

  const _ModelOption({
    required this.name,
    required this.description,
    required this.modelId,
  });
}
