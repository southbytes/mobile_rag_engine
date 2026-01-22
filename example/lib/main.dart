// lib/main.dart
import 'dart:io';
import 'package:flutter/material.dart';
import 'package:file_picker/file_picker.dart';
import 'package:mobile_rag_engine/mobile_rag_engine.dart';

import 'screens/benchmark_screen.dart';
import 'screens/quality_test_screen.dart';
import 'screens/chunking_test_screen.dart';

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
  bool _isReady = false;
  bool _isLoading = false;
  RagEngine? _engine;

  final TextEditingController _docController = TextEditingController();
  final TextEditingController _queryController = TextEditingController();
  List<String> _searchResults = [];
  int _topK = 5; // Adjustable topK for search

  @override
  void initState() {
    super.initState();
    _setup();
  }

  Future<void> _setup() async {
    setState(() {
      _status = "Initializing...";
      _isLoading = true;
    });

    try {
      _engine = await RagEngine.initialize(
        config: RagConfig.fromAssets(
          tokenizerAsset: 'assets/tokenizer.json',
          modelAsset: 'assets/model.onnx',
          databaseName: 'rag_db.sqlite',
        ),
        onProgress: (status) => setState(() => _status = status),
      );

      _isReady = true;
      setState(() {
        _status =
            "✅ Ready!\nVocab: ${_engine!.vocabSize} | Embedding: 384 dims";
        _isLoading = false;
      });
    } catch (e, st) {
      setState(() {
        _status = "❌ Init error: $e\n$st";
        _isLoading = false;
      });
    }
  }

  /// Save document: text -> chunking -> embedding -> DB storage
  Future<void> _saveDocument() async {
    final text = _docController.text.trim();
    if (text.isEmpty) return;

    setState(() {
      _isLoading = true;
      _status = "Processing document (chunking + embedding)...";
    });

    try {
      // Add document with automatic chunking and embedding
      final result = await _engine!.addDocument(
        text,
        onProgress: (done, total) {
          setState(() => _status = "Embedding chunks: $done/$total");
        },
      );

      if (result.isDuplicate) {
        setState(() {
          _status =
              "⚠️ Duplicate detected!\nDocument already exists in database.";
          _isLoading = false;
        });
      } else {
        // Rebuild HNSW index after adding
        await _engine!.rebuildIndex();

        setState(() {
          _status =
              "✅ Saved!\n"
              "📄 Source ID: ${result.sourceId}\n"
              "📦 Chunks created: ${result.chunkCount}\n"
              "(Each chunk ~500 chars with 50 char overlap)";
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

  /// Search: query -> embedding -> chunk similarity search
  Future<void> _searchDocuments() async {
    final query = _queryController.text.trim();
    if (query.isEmpty) return;

    setState(() {
      _isLoading = true;
      _status = "Searching chunks...";
    });

    try {
      // Search using RagEngine (searches chunks, not full documents)
      final ragResult = await _engine!.search(
        query,
        topK: _topK,
        tokenBudget: 2000,
      );

      // Extract chunk contents for display
      final results = ragResult.chunks.map((c) => c.content).toList();

      setState(() {
        _searchResults = results;
        _status =
            "✅ Search complete!\n"
            "Found ${ragResult.chunks.length} relevant chunks\n"
            "Context: ${ragResult.context.estimatedTokens} tokens used";
        _isLoading = false;
      });
    } catch (e) {
      setState(() {
        _status = "❌ Search error: $e";
        _isLoading = false;
      });
    }
  }

  /// Import PDF/DOCX/Markdown file and embed: file -> text extraction -> chunking -> embedding
  Future<void> _importAndEmbedDocument() async {
    setState(() {
      _isLoading = true;
      _status = "Selecting file...";
    });

    try {
      // Pick a PDF, DOCX, or Markdown file
      final result = await FilePicker.platform.pickFiles(
        type: FileType.custom,
        allowedExtensions: ['pdf', 'docx', 'md', 'markdown'],
      );

      if (result == null || result.files.isEmpty) {
        setState(() {
          _status = "⚠️ No file selected";
          _isLoading = false;
        });
        return;
      }

      final file = result.files.first;
      final filePath = file.path;
      if (filePath == null) {
        setState(() {
          _status = "❌ Could not get file path";
          _isLoading = false;
        });
        return;
      }

      setState(() => _status = "Reading file: ${file.name}");

      String extractedText;
      final ext = filePath.split('.').last.toLowerCase();

      if (ext == 'md' || ext == 'markdown') {
        // Markdown: read as text directly
        extractedText = await File(filePath).readAsString();
        setState(
          () => _status = "Markdown loaded! (${extractedText.length} chars)",
        );
      } else {
        // PDF/DOCX: use Rust DTT extractor
        final bytes = await File(filePath).readAsBytes();
        setState(() => _status = "Extracting text from ${file.name}...");

        extractedText = await extractTextFromDocument(
          fileBytes: bytes.toList(),
        );
      }

      if (extractedText.isEmpty) {
        setState(() {
          _status = "⚠️ No text extracted from file";
          _isLoading = false;
        });
        return;
      }

      setState(
        () => _status =
            "Text extracted! (${extractedText.length} chars)\nProcessing chunks...",
      );

      // Add to RAG with chunking and embedding (auto-detect strategy from filePath)
      final addResult = await _engine!.addDocument(
        extractedText,
        metadata: '{"filename": "${file.name}"}',
        filePath: filePath, // <-- Auto-detect chunking strategy
        onProgress: (done, total) {
          setState(() => _status = "Embedding chunks: $done/$total");
        },
      );

      if (addResult.isDuplicate) {
        setState(() {
          _status =
              "⚠️ Duplicate document detected!\n${file.name} already exists.";
          _isLoading = false;
        });
      } else {
        // Rebuild HNSW index
        await _engine!.rebuildIndex();

        setState(() {
          _status =
              "✅ Document imported!\n"
              "📄 File: ${file.name}\n"
              "📝 Text: ${extractedText.length} chars\n"
              "📦 ${addResult.message}";
          _isLoading = false;
        });
      }
    } catch (e, st) {
      setState(() {
        _status = "❌ Import error: $e\n$st";
        _isLoading = false;
      });
    }
  }

  @override
  Widget build(BuildContext context) {
    return MaterialApp(
      theme: ThemeData(useMaterial3: true, colorSchemeSeed: Colors.blue),
      home: Builder(
        builder: (context) => Scaffold(
          appBar: AppBar(
            title: const Text('🔍 Local RAG Engine'),
            centerTitle: true,
            actions: [
              IconButton(
                icon: const Icon(Icons.science),
                tooltip: 'Chunking Test',
                onPressed: _isReady
                    ? () {
                        Navigator.of(context).push(
                          MaterialPageRoute(
                            builder: (_) => const ChunkingTestScreen(),
                          ),
                        );
                      }
                    : null,
              ),
              IconButton(
                icon: const Icon(Icons.speed),
                tooltip: 'Benchmark',
                onPressed: _isReady
                    ? () {
                        Navigator.of(context).push(
                          MaterialPageRoute(
                            builder: (_) => const BenchmarkScreen(),
                          ),
                        );
                      }
                    : null,
              ),
              IconButton(
                icon: const Icon(Icons.checklist),
                tooltip: 'Quality Test',
                onPressed: _isReady
                    ? () {
                        Navigator.of(context).push(
                          MaterialPageRoute(
                            builder: (_) => const QualityTestScreen(),
                          ),
                        );
                      }
                    : null,
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
                            width: 20,
                            height: 20,
                            child: CircularProgressIndicator(strokeWidth: 2),
                          )
                        else
                          Icon(
                            _isReady ? Icons.check_circle : Icons.error,
                            color: _isReady ? Colors.green : Colors.red,
                          ),
                        const SizedBox(width: 12),
                        Expanded(
                          child: Text(
                            _status,
                            style: const TextStyle(fontSize: 13),
                          ),
                        ),
                      ],
                    ),
                  ),
                ),

                const SizedBox(height: 24),

                // Document save section
                const Text(
                  '📄 Save Document',
                  style: TextStyle(fontWeight: FontWeight.bold, fontSize: 16),
                ),
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
                const SizedBox(height: 8),
                ElevatedButton.icon(
                  onPressed: _isReady && !_isLoading
                      ? _importAndEmbedDocument
                      : null,
                  icon: const Icon(Icons.upload_file),
                  label: const Text('Import PDF/DOCX'),
                  style: ElevatedButton.styleFrom(
                    backgroundColor: Colors.orange,
                    foregroundColor: Colors.white,
                  ),
                ),

                const Divider(height: 40),

                // Search section
                const Text(
                  '🔎 Semantic Search',
                  style: TextStyle(fontWeight: FontWeight.bold, fontSize: 16),
                ),
                const SizedBox(height: 8),
                // TopK slider
                Row(
                  children: [
                    const Text('Top K: '),
                    Expanded(
                      child: Slider(
                        value: _topK.toDouble(),
                        min: 1,
                        max: 20,
                        divisions: 19,
                        label: _topK.toString(),
                        onChanged: _isReady
                            ? (value) {
                                setState(() => _topK = value.round());
                              }
                            : null,
                      ),
                    ),
                    SizedBox(
                      width: 40,
                      child: Text(
                        '$_topK',
                        style: const TextStyle(fontWeight: FontWeight.bold),
                      ),
                    ),
                  ],
                ),
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
                  const Text(
                    'Search Results:',
                    style: TextStyle(fontWeight: FontWeight.bold),
                  ),
                  const SizedBox(height: 8),
                  ...List.generate(
                    _searchResults.length,
                    (i) => Card(
                      child: ListTile(
                        leading: CircleAvatar(child: Text('${i + 1}')),
                        title: Text(_searchResults[i]),
                      ),
                    ),
                  ),
                ],

                const Divider(height: 40),

                // Sample data load button
                OutlinedButton.icon(
                  onPressed: _isReady && !_isLoading
                      ? () async {
                          setState(() => _isLoading = true);
                          try {
                            final samples = [
                              "Apple is a red fruit.",
                              "Banana is yellow and sweet.",
                              "Tesla is an electric car company.",
                              "Apple is a company that makes iPhones.",
                              "Orange is a fruit rich in vitamin C.",
                            ];
                            int totalChunks = 0;
                            for (var i = 0; i < samples.length; i++) {
                              setState(
                                () => _status =
                                    "Adding sample ${i + 1}/${samples.length}...",
                              );
                              final result = await _engine!.addDocument(
                                samples[i],
                              );
                              totalChunks += result.chunkCount;
                            }
                            // Rebuild index after all samples added
                            await _engine!.rebuildIndex();

                            setState(() {
                              _status =
                                  "✅ Saved ${samples.length} sample documents!\n"
                                  "📦 Total chunks: $totalChunks";
                              _isLoading = false;
                            });
                          } catch (e) {
                            setState(() {
                              _status = "❌ Sample save error: $e";
                              _isLoading = false;
                            });
                          }
                        }
                      : null,
                  icon: const Icon(Icons.dataset),
                  label: const Text('Load 5 Sample Documents'),
                ),
                const SizedBox(height: 8),
                OutlinedButton.icon(
                  onPressed: _isReady && !_isLoading
                      ? () async {
                          // Confirm dialog
                          final confirmed = await showDialog<bool>(
                            context: context,
                            builder: (ctx) => AlertDialog(
                              title: const Text('Delete All Documents'),
                              content: const Text(
                                'Are you sure you want to delete all documents? This cannot be undone.',
                              ),
                              actions: [
                                TextButton(
                                  onPressed: () => Navigator.pop(ctx, false),
                                  child: const Text('Cancel'),
                                ),
                                TextButton(
                                  onPressed: () => Navigator.pop(ctx, true),
                                  style: TextButton.styleFrom(
                                    foregroundColor: Colors.red,
                                  ),
                                  child: const Text('Delete All'),
                                ),
                              ],
                            ),
                          );

                          if (confirmed != true) return;

                          setState(() {
                            _isLoading = true;
                            _status = "Deleting all documents...";
                          });

                          try {
                            // Get stats before deletion
                            final stats = await _engine!.getStats();
                            // Delete DB file and re-initialize
                            final dbFile = File(_engine!.dbPath);
                            if (await dbFile.exists()) {
                              await dbFile.delete();
                            }
                            // Re-initialize via engine recreation
                            _engine = await RagEngine.initialize(
                              config: RagConfig.fromAssets(
                                tokenizerAsset: 'assets/tokenizer.json',
                                modelAsset: 'assets/model.onnx',
                                databaseName: 'rag_db.sqlite',
                              ),
                            );
                            _searchResults.clear();

                            setState(() {
                              _status =
                                  "✅ Deleted all documents!\n"
                                  "Previously had ${stats.sourceCount} sources.";
                              _isLoading = false;
                            });
                          } catch (e) {
                            setState(() {
                              _status = "❌ Delete error: $e";
                              _isLoading = false;
                            });
                          }
                        }
                      : null,
                  icon: const Icon(Icons.delete_forever, color: Colors.red),
                  label: const Text(
                    'Delete All Documents',
                    style: TextStyle(color: Colors.red),
                  ),
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
    RagEngine.dispose();
    super.dispose();
  }
}
