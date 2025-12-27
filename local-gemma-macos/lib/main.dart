// local-gemma-macos/lib/main.dart
//
// RAG + Ollama LLM Integration for macOS
// Uses mobile_rag_engine for RAG and ollama_dart for LLM
// Auto-manages Ollama server lifecycle

import 'dart:io';
import 'package:flutter/material.dart';
import 'package:mobile_rag_engine/mobile_rag_engine.dart';
import 'package:flutter_rust_bridge/flutter_rust_bridge_for_generated.dart';
import 'package:ollama_dart/ollama_dart.dart';
import 'package:path_provider/path_provider.dart';
import 'screens/model_setup_screen.dart';
import 'screens/rag_chat_screen.dart';

/// Manages the Ollama server process lifecycle
class OllamaServerManager {
  static Process? _ollamaProcess;
  static bool _weStartedOllama = false;
  
  /// Check if Ollama is already running
  static Future<bool> isOllamaRunning() async {
    try {
      final client = OllamaClient();
      await client.getVersion();
      return true;
    } catch (e) {
      return false;
    }
  }
  
  /// Start Ollama server if not already running
  static Future<bool> ensureOllamaRunning() async {
    // Check if already running
    if (await isOllamaRunning()) {
      debugPrint('✅ Ollama is already running');
      return true;
    }
    
    debugPrint('🚀 Starting Ollama server...');
    
    try {
      // Start ollama serve in background
      _ollamaProcess = await Process.start(
        'ollama',
        ['serve'],
        mode: ProcessStartMode.detached,
      );
      _weStartedOllama = true;
      
      debugPrint('🚀 Ollama process started (PID: ${_ollamaProcess!.pid})');
      
      // Wait for server to be ready (max 10 seconds)
      for (int i = 0; i < 20; i++) {
        await Future.delayed(const Duration(milliseconds: 500));
        if (await isOllamaRunning()) {
          debugPrint('✅ Ollama server is ready');
          return true;
        }
      }
      
      debugPrint('⚠️ Ollama server started but not responding');
      return false;
    } catch (e) {
      debugPrint('🔴 Failed to start Ollama: $e');
      return false;
    }
  }
  
  /// Stop Ollama server if we started it
  static Future<void> stopOllama() async {
    if (!_weStartedOllama || _ollamaProcess == null) {
      debugPrint('ℹ️ Ollama was not started by us, not stopping');
      return;
    }
    
    debugPrint('🛑 Stopping Ollama server...');
    
    try {
      // Try graceful shutdown first via API
      final result = await Process.run('pkill', ['-f', 'ollama serve']);
      debugPrint('🛑 Ollama stop result: ${result.exitCode}');
      _ollamaProcess = null;
      _weStartedOllama = false;
    } catch (e) {
      debugPrint('⚠️ Error stopping Ollama: $e');
    }
  }
}


Future<void> main() async {
  WidgetsFlutterBinding.ensureInitialized();
  
  // For macOS, use DynamicLibrary.process() since the Rust library
  // is statically linked via Cargokit's -force_load mechanism.
  if (Platform.isMacOS) {
    await RustLib.init(
      externalLibrary: ExternalLibrary.process(iKnowHowToUseIt: true),
    );
  } else {
    await RustLib.init();
  }
  
  runApp(const LocalGemmaApp());
}


class LocalGemmaApp extends StatefulWidget {
  const LocalGemmaApp({super.key});

  @override
  State<LocalGemmaApp> createState() => _LocalGemmaAppState();
}

class _LocalGemmaAppState extends State<LocalGemmaApp> with WidgetsBindingObserver {
  @override
  void initState() {
    super.initState();
    WidgetsBinding.instance.addObserver(this);
  }

  @override
  void dispose() {
    WidgetsBinding.instance.removeObserver(this);
    // Stop Ollama when app closes
    OllamaServerManager.stopOllama();
    super.dispose();
  }

  @override
  void didChangeAppLifecycleState(AppLifecycleState state) {
    if (state == AppLifecycleState.detached) {
      // App is being terminated
      OllamaServerManager.stopOllama();
    }
  }

  @override
  Widget build(BuildContext context) {
    return MaterialApp(
      title: 'RAG + Ollama',
      debugShowCheckedModeBanner: false,
      theme: ThemeData(
        colorSchemeSeed: Colors.deepPurple,
        useMaterial3: true,
        brightness: Brightness.light,
      ),
      darkTheme: ThemeData(
        colorSchemeSeed: Colors.deepPurple,
        useMaterial3: true,
        brightness: Brightness.dark,
      ),
      home: const HomeScreen(),
    );
  }
}

class HomeScreen extends StatefulWidget {
  const HomeScreen({super.key});

  @override
  State<HomeScreen> createState() => _HomeScreenState();
}

class _HomeScreenState extends State<HomeScreen> {
  bool _isOllamaConnected = false;
  bool _hasModel = false;
  bool _isChecking = true;
  String? _statusMessage;
  String? _selectedModel;
  final OllamaClient _client = OllamaClient();

  @override
  void initState() {
    super.initState();
    _initializeOllama();
  }

  Future<void> _initializeOllama() async {
    setState(() {
      _isChecking = true;
      _statusMessage = 'Starting Ollama server...';
    });

    // Try to start Ollama automatically
    final started = await OllamaServerManager.ensureOllamaRunning();
    
    if (started) {
      await _checkOllamaStatus();
    } else {
      setState(() {
        _isOllamaConnected = false;
        _isChecking = false;
        _statusMessage = 'Failed to start Ollama server';
      });
    }
  }

  Future<void> _checkOllamaStatus() async {
    setState(() {
      _isChecking = true;
      _statusMessage = 'Connecting to Ollama...';
    });

    try {
      // Check if Ollama server is running
      final version = await _client.getVersion();
      debugPrint('✅ Ollama version: ${version.version}');
      
      setState(() => _isOllamaConnected = true);
      
      // Check installed models
      final models = await _client.listModels();
      final modelList = models.models ?? [];
      
      if (modelList.isNotEmpty) {
        // Find a suitable model (prefer gemma, then llama, then any)
        String? preferredModel;
        for (final model in modelList) {
          final name = model.model ?? '';
          if (name.contains('gemma')) {
            preferredModel = name;
            break;
          }
        }
        preferredModel ??= modelList.first.model;
        
        setState(() {
          _hasModel = true;
          _selectedModel = preferredModel;
          _isChecking = false;
          _statusMessage = null;
        });
      } else {
        setState(() {
          _hasModel = false;
          _isChecking = false;
          _statusMessage = null;
        });
      }
    } catch (e) {
      debugPrint('🔴 Ollama connection error: $e');
      setState(() {
        _isOllamaConnected = false;
        _isChecking = false;
        _statusMessage = 'Cannot connect to Ollama server';
      });
    }
  }

  @override
  Widget build(BuildContext context) {
    if (_isChecking) {
      return Scaffold(
        body: Center(
          child: Column(
            mainAxisAlignment: MainAxisAlignment.center,
            children: [
              const CircularProgressIndicator(),
              const SizedBox(height: 16),
              Text(_statusMessage ?? 'Checking status...'),
            ],
          ),
        ),
      );
    }

    return Scaffold(
      appBar: AppBar(
        title: const Text('🤖 RAG + Ollama'),
        centerTitle: true,
        actions: [
          PopupMenuButton<String>(
            icon: const Icon(Icons.settings),
            onSelected: (value) => _handleMenuAction(value),
            itemBuilder: (context) => [
              const PopupMenuItem(
                value: 'clear_documents',
                child: ListTile(
                  leading: Icon(Icons.delete_sweep),
                  title: Text('Clear Documents'),
                  subtitle: Text('Reset RAG database'),
                  contentPadding: EdgeInsets.zero,
                ),
              ),
              const PopupMenuItem(
                value: 'change_model',
                child: ListTile(
                  leading: Icon(Icons.swap_horiz),
                  title: Text('Change Model'),
                  subtitle: Text('Select different model'),
                  contentPadding: EdgeInsets.zero,
                ),
              ),
              const PopupMenuItem(
                value: 'refresh',
                child: ListTile(
                  leading: Icon(Icons.refresh),
                  title: Text('Refresh Status'),
                  contentPadding: EdgeInsets.zero,
                ),
              ),
              const PopupMenuItem(
                value: 'restart_ollama',
                child: ListTile(
                  leading: Icon(Icons.restart_alt),
                  title: Text('Restart Ollama'),
                  subtitle: Text('Restart Ollama server'),
                  contentPadding: EdgeInsets.zero,
                ),
              ),
            ],
          ),
        ],
      ),
      body: Center(
        child: Padding(
          padding: const EdgeInsets.all(24),
          child: Column(
            mainAxisAlignment: MainAxisAlignment.center,
            children: [
              // Ollama connection status
              if (!_isOllamaConnected) ...[
                const Icon(
                  Icons.cloud_off,
                  size: 80,
                  color: Colors.red,
                ),
                const SizedBox(height: 24),
                Text(
                  'Ollama Not Running',
                  style: Theme.of(context).textTheme.headlineSmall,
                ),
                const SizedBox(height: 8),
                Text(
                  'Make sure Ollama is installed:\nbrew install ollama',
                  textAlign: TextAlign.center,
                  style: Theme.of(context).textTheme.bodyMedium?.copyWith(
                        color: Colors.grey[600],
                        fontFamily: 'monospace',
                      ),
                ),
                const SizedBox(height: 32),
                FilledButton.icon(
                  onPressed: _initializeOllama,
                  icon: const Icon(Icons.refresh),
                  label: const Text('Retry Connection'),
                ),
              ] else ...[
                Icon(
                  _hasModel ? Icons.check_circle : Icons.download,
                  size: 80,
                  color: _hasModel ? Colors.green : Colors.grey,
                ),
                const SizedBox(height: 24),
                Text(
                  _hasModel
                      ? 'Model Ready!'
                      : 'LLM Model Required',
                  style: Theme.of(context).textTheme.headlineSmall,
                ),
                const SizedBox(height: 8),
                Text(
                  _hasModel
                      ? 'Using: $_selectedModel\nYou can start chatting with RAG-powered responses.'
                      : 'Download a model from Ollama to enable AI responses.',
                  textAlign: TextAlign.center,
                  style: Theme.of(context).textTheme.bodyMedium?.copyWith(
                        color: Colors.grey[600],
                      ),
                ),
                const SizedBox(height: 32),
                if (!_hasModel)
                  FilledButton.icon(
                    onPressed: () async {
                      final result = await Navigator.push<String>(
                        context,
                        MaterialPageRoute(
                          builder: (_) => const ModelSetupScreen(),
                        ),
                      );
                      if (result != null) {
                        setState(() {
                          _hasModel = true;
                          _selectedModel = result;
                        });
                      }
                    },
                    icon: const Icon(Icons.download),
                    label: const Text('Download Model'),
                  ),
                if (_hasModel)
                  FilledButton.icon(
                    onPressed: () {
                      Navigator.push(
                        context,
                        MaterialPageRoute(
                          builder: (_) => RagChatScreen(
                            modelName: _selectedModel!,
                          ),
                        ),
                      );
                    },
                    icon: const Icon(Icons.chat),
                    label: const Text('Start Chat'),
                  ),
                const SizedBox(height: 16),
                // Skip model installation for testing RAG only
                TextButton(
                  onPressed: () {
                    Navigator.push(
                      context,
                      MaterialPageRoute(
                        builder: (_) => const RagChatScreen(mockLlm: true),
                      ),
                    );
                  },
                  child: const Text('Skip (Test RAG only)'),
                ),
              ],
            ],
          ),
        ),
      ),
    );
  }

  Future<void> _handleMenuAction(String action) async {
    switch (action) {
      case 'clear_documents':
        await _clearDocuments();
        break;
      case 'change_model':
        await _changeModel();
        break;
      case 'refresh':
        await _checkOllamaStatus();
        break;
      case 'restart_ollama':
        await _restartOllama();
        break;
    }
  }

  Future<void> _restartOllama() async {
    setState(() {
      _isChecking = true;
      _statusMessage = 'Restarting Ollama...';
    });

    await OllamaServerManager.stopOllama();
    await Future.delayed(const Duration(seconds: 1));
    await _initializeOllama();
  }

  Future<void> _clearDocuments() async {
    final confirm = await showDialog<bool>(
      context: context,
      builder: (context) => AlertDialog(
        title: const Text('Clear Documents?'),
        content: const Text(
          'This will delete all stored documents and RAG data. This action cannot be undone.',
        ),
        actions: [
          TextButton(
            onPressed: () => Navigator.pop(context, false),
            child: const Text('Cancel'),
          ),
          FilledButton(
            onPressed: () => Navigator.pop(context, true),
            child: const Text('Clear'),
          ),
        ],
      ),
    );

    if (confirm == true) {
      try {
        final dir = await getApplicationDocumentsDirectory();
        final dbPath = '${dir.path}/local_gemma_rag.db';
        final dbFile = File(dbPath);
        
        if (await dbFile.exists()) {
          await dbFile.delete();
        }
        
        if (mounted) {
          ScaffoldMessenger.of(context).showSnackBar(
            const SnackBar(content: Text('✅ Documents cleared successfully')),
          );
        }
      } catch (e) {
        debugPrint('🔴 Error clearing documents: $e');
        if (mounted) {
          ScaffoldMessenger.of(context).showSnackBar(
            SnackBar(content: Text('❌ Error: $e')),
          );
        }
      }
    }
  }

  Future<void> _changeModel() async {
    final result = await Navigator.push<String>(
      context,
      MaterialPageRoute(
        builder: (_) => const ModelSetupScreen(),
      ),
    );
    if (result != null) {
      setState(() {
        _hasModel = true;
        _selectedModel = result;
      });
    }
  }
}
