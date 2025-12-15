// test_app/lib/screens/model_setup_screen.dart
//
// Screen for downloading and installing flutter_gemma models

import 'package:flutter/foundation.dart' show kIsWeb;
import 'package:flutter/material.dart';
import 'package:flutter_gemma/flutter_gemma.dart';

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

  // Model options
  final List<_ModelOption> _allModels = [
    // 4B Models - Web Only (Coming Soon for Mobile)
    _ModelOption(
      name: 'Gemma3 4B IT INT4 ⭐ (Best for RAG)',
      description: 'Better context understanding for RAG (~2.6GB)',
      url: 'https://huggingface.co/litert-community/Gemma3-4B-IT/resolve/main/gemma3-4b-it-int4-web.task',
      modelType: ModelType.gemmaIt,
      requiresToken: true,
      webOnly: true,
    ),
    _ModelOption(
      name: 'Gemma3 4B IT INT8',
      description: 'Higher quality 4B (~3.9GB)',
      url: 'https://huggingface.co/litert-community/Gemma3-4B-IT/resolve/main/gemma3-4b-it-int8-web.task',
      modelType: ModelType.gemmaIt,
      requiresToken: true,
      webOnly: true,
    ),
    // E2B Model - Mobile & Web (Better RAG than 1B)
    _ModelOption(
      name: 'Gemma 3n E2B IT INT4 ⭐ (Recommended)',
      description: 'Effective 2B params, better RAG performance (~3.1GB)',
      url: 'https://huggingface.co/google/gemma-3n-E2B-it-litert-preview/resolve/main/gemma-3n-E2B-it-int4.task',
      modelType: ModelType.gemmaIt,
      requiresToken: true,
      webOnly: false,
    ),
    // 1B Models - Mobile & Web (Lightweight)
    _ModelOption(
      name: 'Gemma3 1B IT Q4',
      description: 'Lightest model (~555MB, limited RAG performance)',
      url: 'https://huggingface.co/litert-community/Gemma3-1B-IT/resolve/main/Gemma3-1B-IT_multi-prefill-seq_q4_ekv2048.task',
      modelType: ModelType.gemmaIt,
      requiresToken: true,
      webOnly: false,
    ),
    _ModelOption(
      name: 'Gemma3 1B IT Q4 (Block32)',
      description: 'Better quality, larger cache (~722MB)',
      url: 'https://huggingface.co/litert-community/Gemma3-1B-IT/resolve/main/Gemma3-1B-IT_multi-prefill-seq_q4_block32_ekv4096.task',
      modelType: ModelType.gemmaIt,
      requiresToken: true,
      webOnly: false,
    ),
    _ModelOption(
      name: 'Gemma3 1B IT Q4 (Block128)',
      description: 'Best quality 1B quantized (~689MB)',
      url: 'https://huggingface.co/litert-community/Gemma3-1B-IT/resolve/main/Gemma3-1B-IT_multi-prefill-seq_q4_block128_ekv4096.task',
      modelType: ModelType.gemmaIt,
      requiresToken: true,
      webOnly: false,
    ),
  ];
  
  /// Filtered models based on platform
  List<_ModelOption> get _models =>
      _allModels.where((m) => kIsWeb || !m.webOnly).toList();


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
      await FlutterGemma.installModel(
        modelType: model.modelType,
      ).fromNetwork(
        model.url,
      ).withProgress((progress) {
        setState(() {
          _progress = progress / 100;
          _status = 'Downloading: ${progress.toStringAsFixed(1)}%';
        });
      }).install();

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
        title: const Text('Install LLM Model'),
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
                onPressed: () => Navigator.pop(context, true),
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
                      '• Models are stored locally on your device\n'
                      '• GPU acceleration is recommended\n'
                      '• iOS Simulator may not support GPU models',
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
  final String url;
  final ModelType modelType;
  final bool requiresToken;
  final bool webOnly;

  const _ModelOption({
    required this.name,
    required this.description,
    required this.url,
    required this.modelType,
    required this.requiresToken,
    this.webOnly = false,
  });
}
