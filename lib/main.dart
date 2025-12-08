// lib/main.dart
import 'dart:io';
import 'package:flutter/material.dart';
import 'package:flutter/services.dart';
import 'package:path_provider/path_provider.dart';
import 'package:mobile_rag_engine/src/rust/api/simple_rag.dart';
import 'package:mobile_rag_engine/src/rust/api/tokenizer.dart';
import 'package:mobile_rag_engine/src/rust/frb_generated.dart';
import 'package:mobile_rag_engine/services/embedding_service.dart';
import 'package:mobile_rag_engine/screens/benchmark_screen.dart';
import 'package:mobile_rag_engine/screens/quality_test_screen.dart';
import 'package:mobile_rag_engine/screens/chunking_test_screen.dart';

Future<void> main() async {
  WidgetsFlutterBinding.ensureInitialized();
  await RustLib.init();
  runApp(const MyApp());
}

class MyApp extends StatefulWidget {
  const MyApp({super.key});
  @override
  State<MyApp> createState() => _MyAppState();
}

class _MyAppState extends State<MyApp> {
  String _status = "Initializing...";
  String _dbPath = "";
  bool _isReady = false;
  bool _isLoading = false;
  
  final TextEditingController _docController = TextEditingController();
  final TextEditingController _queryController = TextEditingController();
  List<String> _searchResults = [];

  @override
  void initState() {
    super.initState();
    _setup();
  }

  Future<void> _setup() async {
    setState(() {
      _status = "Copying files...";
      _isLoading = true;
    });
    
    try {
      final dir = await getApplicationDocumentsDirectory();
      _dbPath = "${dir.path}/rag_db.sqlite";
      final tokenizerPath = "${dir.path}/tokenizer.json";
      
      // 1. Copy and initialize tokenizer
      await _copyAssetToFile('assets/tokenizer.json', tokenizerPath);
      await initTokenizer(tokenizerPath: tokenizerPath);
      final vocabSize = getVocabSize();
      setState(() => _status = "Tokenizer loaded (Vocab: $vocabSize)");
      
      // 2. Load ONNX model (Flutter onnxruntime)
      setState(() => _status = "Loading ONNX model (90MB)...");
      final modelBytes = await rootBundle.load('assets/model.onnx');
      await EmbeddingService.init(modelBytes.buffer.asUint8List());
      setState(() => _status = "ONNX model loaded!");
      
      // 3. Initialize DB
      await initDb(dbPath: _dbPath);
      setState(() => _status = "SQLite DB initialized");
      
      _isReady = true;
      setState(() {
        _status = "✅ Ready!\nVocab: $vocabSize | Embedding: 384 dims";
        _isLoading = false;
      });
    } catch (e, st) {
      setState(() {
        _status = "❌ Init error: $e\n$st";
        _isLoading = false;
      });
    }
  }

  Future<void> _copyAssetToFile(String assetPath, String filePath) async {
    final file = File(filePath);
    if (!await file.exists()) {
      final data = await rootBundle.load(assetPath);
      await file.writeAsBytes(data.buffer.asUint8List());
    }
  }

  /// Save document: text -> embedding -> DB storage
  Future<void> _saveDocument() async {
    final text = _docController.text.trim();
    if (text.isEmpty) return;
    
    setState(() {
      _isLoading = true;
      _status = "Processing document...";
    });
    
    try {
      // Generate embedding in Dart (Rust tokenizer + Flutter ONNX)
      final embedding = await EmbeddingService.embed(text);
      
      // Convert to f32 list
      final embeddingF32 = embedding.map((e) => e.toDouble()).toList();
      
      // Save to DB (with deduplication)
      final result = await addDocument(
        dbPath: _dbPath, 
        content: text, 
        embedding: embeddingF32,
      );
      
      if (result.isDuplicate) {
        setState(() {
          _status = "⚠️ Duplicate detected!\nDocument already exists in database.";
          _isLoading = false;
        });
      } else {
        setState(() {
          _status = "✅ Saved!\nEmbedding: ${embedding.length} dims\n"
              "First 3 values: ${embedding.take(3).map((e) => e.toStringAsFixed(4)).toList()}";
          _isLoading = false;
        });
        _docController.clear();
      }
    } catch (e) {
      setState(() {
        _status = "❌ Save error: $e";
        _isLoading = false;
      });
    }
  }

  /// Search: query -> embedding -> similarity search
  Future<void> _searchDocuments() async {
    final query = _queryController.text.trim();
    if (query.isEmpty) return;
    
    setState(() {
      _isLoading = true;
      _status = "Searching...";
    });
    
    try {
      // Generate query embedding
      final queryEmbedding = await EmbeddingService.embed(query);
      
      // Search similar documents
      final results = await searchSimilar(
        dbPath: _dbPath,
        queryEmbedding: queryEmbedding,
        topK: 3,
      );
      
      setState(() {
        _searchResults = results;
        _status = "✅ Search complete! Found ${results.length} results";
        _isLoading = false;
      });
    } catch (e) {
      setState(() {
        _status = "❌ Search error: $e";
        _isLoading = false;
      });
    }
  }

  @override
  Widget build(BuildContext context) {
    return MaterialApp(
      theme: ThemeData(
        useMaterial3: true,
        colorSchemeSeed: Colors.blue,
      ),
      home: Builder(
        builder: (context) => Scaffold(
        appBar: AppBar(
          title: const Text('🔍 Local RAG Engine'),
          centerTitle: true,
          actions: [
            IconButton(
              icon: const Icon(Icons.auto_awesome),
              tooltip: 'Chunking Test',
              onPressed: _isReady ? () {
                Navigator.of(context).push(
                  MaterialPageRoute(builder: (_) => const ChunkingTestScreen()),
                );
              } : null,
            ),
            IconButton(
              icon: const Icon(Icons.speed),
              tooltip: 'Benchmark',
              onPressed: _isReady ? () {
                Navigator.of(context).push(
                  MaterialPageRoute(builder: (_) => const BenchmarkScreen()),
                );
              } : null,
            ),
            IconButton(
              icon: const Icon(Icons.checklist),
              tooltip: 'Quality Test',
              onPressed: _isReady ? () {
                Navigator.of(context).push(
                  MaterialPageRoute(builder: (_) => const QualityTestScreen()),
                );
              } : null,
            ),
          ],
        ),
        body: SingleChildScrollView(
          padding: const EdgeInsets.all(16.0),
          child: Column(
            crossAxisAlignment: CrossAxisAlignment.stretch,
            children: [
              // Status display
              Card(
                child: Padding(
                  padding: const EdgeInsets.all(12.0),
                  child: Row(
                    children: [
                      if (_isLoading)
                        const SizedBox(
                          width: 20, height: 20,
                          child: CircularProgressIndicator(strokeWidth: 2),
                        )
                      else
                        Icon(
                          _isReady ? Icons.check_circle : Icons.error,
                          color: _isReady ? Colors.green : Colors.red,
                        ),
                      const SizedBox(width: 12),
                      Expanded(
                        child: Text(_status, style: const TextStyle(fontSize: 13)),
                      ),
                    ],
                  ),
                ),
              ),
              
              const SizedBox(height: 24),
              
              // Document save section
              const Text('📄 Save Document', style: TextStyle(fontWeight: FontWeight.bold, fontSize: 16)),
              const SizedBox(height: 8),
              TextField(
                controller: _docController,
                decoration: const InputDecoration(
                  border: OutlineInputBorder(),
                  hintText: 'Enter document content to save',
                ),
                maxLines: 3,
                enabled: _isReady,
              ),
              const SizedBox(height: 8),
              ElevatedButton.icon(
                onPressed: _isReady && !_isLoading ? _saveDocument : null,
                icon: const Icon(Icons.save),
                label: const Text('Save Document (Auto Embed)'),
              ),
              
              const Divider(height: 40),
              
              // Search section
              const Text('🔎 Semantic Search', style: TextStyle(fontWeight: FontWeight.bold, fontSize: 16)),
              const SizedBox(height: 8),
              TextField(
                controller: _queryController,
                decoration: const InputDecoration(
                  border: OutlineInputBorder(),
                  hintText: 'Enter search query',
                ),
                enabled: _isReady,
              ),
              const SizedBox(height: 8),
              ElevatedButton.icon(
                onPressed: _isReady && !_isLoading ? _searchDocuments : null,
                icon: const Icon(Icons.search),
                label: const Text('Search Similar Documents'),
              ),
              
              const SizedBox(height: 16),
              
              // Search results
              if (_searchResults.isNotEmpty) ...[
                const Text('Search Results:', style: TextStyle(fontWeight: FontWeight.bold)),
                const SizedBox(height: 8),
                ...List.generate(_searchResults.length, (i) => Card(
                  child: ListTile(
                    leading: CircleAvatar(child: Text('${i + 1}')),
                    title: Text(_searchResults[i]),
                  ),
                )),
              ],
              
              const Divider(height: 40),
              
              // Sample data load button
              OutlinedButton.icon(
                onPressed: _isReady && !_isLoading ? () async {
                  setState(() => _isLoading = true);
                  try {
                    final samples = [
                      "Apple is a red fruit.",
                      "Banana is yellow and sweet.",
                      "Tesla is an electric car company.",
                      "Apple is a company that makes iPhones.",
                      "Orange is a fruit rich in vitamin C.",
                    ];
                    for (final sample in samples) {
                      final emb = await EmbeddingService.embed(sample);
                      await addDocument(dbPath: _dbPath, content: sample, embedding: emb);
                    }
                    setState(() {
                      _status = "✅ Saved ${samples.length} sample documents!";
                      _isLoading = false;
                    });
                  } catch (e) {
                    setState(() {
                      _status = "❌ Sample save error: $e";
                      _isLoading = false;
                    });
                  }
                } : null,
                icon: const Icon(Icons.dataset),
                label: const Text('Load 5 Sample Documents'),
              ),
            ],
          ),
        ),
      ),
      ),
    );
  }

  @override
  void dispose() {
    EmbeddingService.dispose();
    super.dispose();
  }
}