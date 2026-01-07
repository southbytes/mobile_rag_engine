// example/lib/main.dart
//
// Simple example demonstrating Mobile RAG Engine usage
//
import 'dart:io';
import 'package:flutter/material.dart';
import 'package:flutter/services.dart';
import 'package:path_provider/path_provider.dart';

// Import from mobile_rag_engine package
import 'package:mobile_rag_engine/src/rust/api/simple_rag.dart';
import 'package:mobile_rag_engine/src/rust/api/tokenizer.dart';
import 'package:mobile_rag_engine/src/rust/frb_generated.dart';
import 'package:mobile_rag_engine/services/embedding_service.dart';

void main() async {
  WidgetsFlutterBinding.ensureInitialized();
  await RustLib.init();
  runApp(const ExampleApp());
}

class ExampleApp extends StatelessWidget {
  const ExampleApp({super.key});

  @override
  Widget build(BuildContext context) {
    return MaterialApp(
      title: 'Mobile RAG Engine Example',
      theme: ThemeData(colorSchemeSeed: Colors.blue, useMaterial3: true),
      home: const RagExampleScreen(),
    );
  }
}

class RagExampleScreen extends StatefulWidget {
  const RagExampleScreen({super.key});

  @override
  State<RagExampleScreen> createState() => _RagExampleScreenState();
}

class _RagExampleScreenState extends State<RagExampleScreen> {
  String _status = 'Not initialized';
  String _dbPath = '';
  bool _isReady = false;
  bool _isLoading = false;

  final _queryController = TextEditingController();
  List<String> _results = [];

  @override
  void initState() {
    super.initState();
    _initialize();
  }

  Future<void> _initialize() async {
    setState(() {
      _isLoading = true;
      _status = 'Initializing...';
    });

    try {
      final dir = await getApplicationDocumentsDirectory();
      _dbPath = '${dir.path}/example_rag.db';

      // 1. Copy and initialize tokenizer
      final tokenizerPath = '${dir.path}/tokenizer.json';
      await _copyAsset('assets/tokenizer.json', tokenizerPath);
      await initTokenizer(tokenizerPath: tokenizerPath);

      // 2. Load ONNX model
      setState(() => _status = 'Loading ONNX model...');
      final modelBytes = await rootBundle.load('assets/model.onnx');
      await EmbeddingService.init(modelBytes.buffer.asUint8List());

      // 3. Initialize database
      await initDb(dbPath: _dbPath);

      // 4. Add sample documents
      await _addSampleDocuments();

      setState(() {
        _isReady = true;
        _isLoading = false;
        _status = 'Ready! Try searching.';
      });
    } catch (e) {
      setState(() {
        _isLoading = false;
        _status = 'Error: $e';
      });
    }
  }

  Future<void> _copyAsset(String assetPath, String targetPath) async {
    final file = File(targetPath);
    // Always overwrite to ensure latest assets are used
    final data = await rootBundle.load(assetPath);
    await file.writeAsBytes(data.buffer.asUint8List());
  }

  Future<void> _addSampleDocuments() async {
    final samples = [
      'Apple is a red fruit that is sweet and crunchy.',
      'Banana is a yellow tropical fruit rich in potassium.',
      'Tesla is an electric vehicle company founded by Elon Musk.',
      'Google is a technology company known for its search engine.',
      'Python is a programming language popular for data science.',
      'The Great Wall of China is one of the Seven Wonders of the World.',
    ];

    for (final doc in samples) {
      final embedding = await EmbeddingService.embed(doc);
      await addDocument(dbPath: _dbPath, content: doc, embedding: embedding);
    }

    // Rebuild HNSW index after bulk insert
    await rebuildHnswIndex(dbPath: _dbPath);
  }

  Future<void> _search() async {
    final query = _queryController.text.trim();
    if (query.isEmpty) return;

    setState(() {
      _isLoading = true;
      _status = 'Searching...';
    });

    try {
      final queryEmbedding = await EmbeddingService.embed(query);
      final results = await searchSimilar(
        dbPath: _dbPath,
        queryEmbedding: queryEmbedding,
        topK: 3,
      );

      setState(() {
        _results = results;
        _isLoading = false;
        _status = 'Found ${results.length} results';
      });
    } catch (e) {
      setState(() {
        _isLoading = false;
        _status = 'Search error: $e';
      });
    }
  }

  @override
  Widget build(BuildContext context) {
    return Scaffold(
      appBar: AppBar(
        title: const Text('RAG Engine Example'),
        centerTitle: true,
      ),
      body: Padding(
        padding: const EdgeInsets.all(16),
        child: Column(
          crossAxisAlignment: CrossAxisAlignment.stretch,
          children: [
            // Status
            Card(
              child: Padding(
                padding: const EdgeInsets.all(12),
                child: Row(
                  children: [
                    if (_isLoading)
                      const SizedBox(
                        width: 20,
                        height: 20,
                        child: CircularProgressIndicator(strokeWidth: 2),
                      )
                    else
                      Icon(
                        _isReady ? Icons.check_circle : Icons.info,
                        color: _isReady ? Colors.green : Colors.grey,
                      ),
                    const SizedBox(width: 12),
                    Expanded(child: Text(_status)),
                  ],
                ),
              ),
            ),

            const SizedBox(height: 24),

            // Search input
            TextField(
              controller: _queryController,
              decoration: const InputDecoration(
                labelText: 'Search Query',
                hintText: 'e.g., "fruit" or "technology company"',
                border: OutlineInputBorder(),
              ),
              enabled: _isReady,
              onSubmitted: (_) => _search(),
            ),

            const SizedBox(height: 12),

            ElevatedButton.icon(
              onPressed: _isReady && !_isLoading ? _search : null,
              icon: const Icon(Icons.search),
              label: const Text('Search'),
            ),

            const SizedBox(height: 24),

            // Results
            if (_results.isNotEmpty) ...[
              Text('Results:', style: Theme.of(context).textTheme.titleMedium),
              const SizedBox(height: 8),
              Expanded(
                child: ListView.builder(
                  itemCount: _results.length,
                  itemBuilder: (context, index) {
                    return Card(
                      child: ListTile(
                        leading: CircleAvatar(child: Text('${index + 1}')),
                        title: Text(_results[index]),
                      ),
                    );
                  },
                ),
              ),
            ],
          ],
        ),
      ),
    );
  }

  @override
  void dispose() {
    _queryController.dispose();
    EmbeddingService.dispose();
    super.dispose();
  }
}
