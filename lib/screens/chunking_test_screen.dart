import 'dart:io';
import 'package:flutter/material.dart';
import 'package:flutter/services.dart';
import 'package:path_provider/path_provider.dart';
import 'package:mobile_rag_engine/mobile_rag_engine.dart';

void main() async {
  WidgetsFlutterBinding.ensureInitialized();
  await RustLib.init();
  runApp(const ChunkingTestApp());
}

class ChunkingTestApp extends StatelessWidget {
  const ChunkingTestApp({super.key});

  @override
  Widget build(BuildContext context) {
    return MaterialApp(
      title: 'Chunking Integration Test',
      theme: ThemeData(
        colorSchemeSeed: Colors.deepPurple,
        useMaterial3: true,
      ),
      home: const ChunkingTestScreen(),
    );
  }
}

class ChunkingTestScreen extends StatefulWidget {
  const ChunkingTestScreen({super.key});

  @override
  State<ChunkingTestScreen> createState() => _ChunkingTestScreenState();
}

class _ChunkingTestScreenState extends State<ChunkingTestScreen> {
  final List<String> _logs = [];
  bool _isRunning = false;
  String _status = 'Ready to test';
  SourceRagService? _rag;

  void _log(String message) {
    setState(() {
      _logs.add('[${DateTime.now().toString().substring(11, 19)}] $message');
      if (_logs.length > 50) _logs.removeAt(0);
    });
    print(message);
  }

  Future<void> _runFullTest() async {
    if (_isRunning) return;
    
    setState(() {
      _isRunning = true;
      _status = 'Running...';
      _logs.clear();
    });

    try {
      await _testInitialization();
      await _testChunking();
      await _testSourceRag();
      await _testSearch();
      await _testContextBuilder();
      
      setState(() => _status = '✅ All tests passed!');
    } catch (e, st) {
      _log('❌ ERROR: $e');
      _log('Stack: $st');
      setState(() => _status = '❌ Test failed: $e');
    } finally {
      setState(() => _isRunning = false);
    }
  }

  Future<void> _testInitialization() async {
    _log('=== Test 1: Initialization ===');
    
    final dir = await getApplicationDocumentsDirectory();
    
    // Initialize tokenizer
    _log('Loading tokenizer...');
    final tokenizerPath = '${dir.path}/tokenizer.json';
    await _copyAsset('assets/tokenizer.json', tokenizerPath);
    await initTokenizer(tokenizerPath: tokenizerPath);
    _log('✓ Tokenizer loaded');
    
    // Initialize embedding service
    _log('Loading ONNX model...');
    final modelBytes = await rootBundle.load('assets/model.onnx');
    await EmbeddingService.init(modelBytes.buffer.asUint8List());
    _log('✓ ONNX model loaded');
    
    // Initialize source RAG service
    _log('Initializing SourceRagService...');
    final dbPath = '${dir.path}/chunking_test.db';
    
    // Delete old DB for clean test
    final dbFile = File(dbPath);
    if (await dbFile.exists()) {
      await dbFile.delete();
    }
    
    _rag = SourceRagService(
      dbPath: dbPath,
      maxChunkChars: 500,
      overlapChars: 50,
    );
    await _rag!.init();
    _log('✓ SourceRagService initialized');
  }

  Future<void> _testChunking() async {
    _log('');
    _log('=== Test 2: Rust Semantic Chunker ===');
    
    const longText = '''
Flutter is Google's UI toolkit for building beautiful, natively compiled applications for mobile, web, and desktop from a single codebase. It was first released in May 2017 and has since become one of the most popular frameworks for cross-platform development.

The key features of Flutter include:
1. Fast Development with Hot Reload
2. Expressive and Flexible UI
3. Native Performance on both iOS and Android
4. Single Codebase for Multiple Platforms

Dart is the programming language used by Flutter. It is optimized for building user interfaces with features such as async/await for asynchronous programming, strong typing, and a garbage collector for memory management.

Flutter uses a reactive framework, where the UI is rebuilt whenever the application state changes. This is similar to React's approach but with some key differences in how widgets are composed and rendered.
''';

    _log('Input text: ${longText.length} chars');
    
    // Test with Rust semantic chunking (Unicode sentence/word boundaries)
    final chunks = semanticChunkWithOverlap(
      text: longText,
      maxChars: 500,
      overlapChars: 50,
    );
    
    _log('Chunks created: ${chunks.length}');
    for (var i = 0; i < chunks.length; i++) {
      final chunk = chunks[i];
      _log('  Chunk $i: ${chunk.content.length} chars');
    }
    
    // Verify chunks
    assert(chunks.isNotEmpty, 'Should create at least one chunk');
    
    _log('✓ Rust Semantic Chunker works correctly');
  }

  Future<void> _testSourceRag() async {
    _log('');
    _log('=== Test 3: SourceRagService.addSourceWithChunking ===');
    
    const documents = [
      '''
Apple Inc. is an American multinational technology company headquartered in Cupertino, California. Apple is the world's largest technology company by revenue, with US\$394.3 billion in 2022 revenue. The company was founded by Steve Jobs, Steve Wozniak, and Ronald Wayne in April 1976. Apple develops, manufactures, and sells consumer electronics, computer software, and online services. The company's hardware products include the iPhone smartphone, the iPad tablet computer, the Mac personal computer, the iPod portable media player, the Apple Watch smartwatch, and Apple TV. Apple's software includes the macOS, iOS, iPadOS, watchOS, and tvOS operating systems.
''',
      '''
Google LLC is an American multinational technology company focusing on search engine technology, online advertising, cloud computing, computer software, quantum computing, e-commerce, and consumer electronics. It has been referred to as the most powerful company in the world and is one of the world's most valuable brands due to its market dominance and data collection. Google was founded on September 4, 1998, by Larry Page and Sergey Brin while they were PhD students at Stanford University in California.
''',
      '''
Microsoft Corporation is an American multinational technology corporation headquartered in Redmond, Washington. Microsoft's best-known software products are the Windows line of operating systems, the Microsoft Office suite, and the Edge web browser. Its flagship hardware products are the Xbox video game consoles and the Microsoft Surface lineup of touchscreen personal computers. Microsoft ranked No. 14 in the 2022 Fortune 500 rankings of the largest United States corporations by total revenue.
''',
    ];
    
    for (var i = 0; i < documents.length; i++) {
      _log('Adding document ${i + 1}...');
      
      final result = await _rag!.addSourceWithChunking(
        documents[i],
        metadata: '{"index": $i}',
        onProgress: (done, total) {
          if (done == total) {
            _log('  Embedded $total chunks');
          }
        },
      );
      
      _log('  Source ID: ${result.sourceId}, Chunks: ${result.chunkCount}, Duplicate: ${result.isDuplicate}');
    }
    
    // Rebuild index
    _log('Rebuilding HNSW index...');
    await _rag!.rebuildIndex();
    _log('✓ Index rebuilt');
    
    // Check stats
    final stats = await _rag!.getStats();
    _log('Stats: ${stats.sourceCount} sources, ${stats.chunkCount} chunks');
    
    assert(stats.sourceCount == 3, 'Should have 3 sources');
    assert(stats.chunkCount > 0, 'Should have chunks');
    
    _log('✓ SourceRagService.addSourceWithChunking works');
  }

  Future<void> _testSearch() async {
    _log('');
    _log('=== Test 4: Search ===');
    
    final queries = [
      'smartphone company',
      'search engine',
      'operating system',
    ];
    
    for (final query in queries) {
      _log('Query: "$query"');
      
      final result = await _rag!.search(
        query,
        topK: 3,
        tokenBudget: 1000,
      );
      
      _log('  Found ${result.chunks.length} chunks');
      for (var i = 0; i < result.chunks.length && i < 2; i++) {
        final chunk = result.chunks[i];
        final preview = chunk.content.substring(0, chunk.content.length.clamp(0, 50));
        _log('  ${i + 1}. [sim: ${chunk.similarity.toStringAsFixed(3)}] "$preview..."');
      }
    }
    
    _log('✓ Search works');
  }

  Future<void> _testContextBuilder() async {
    _log('');
    _log('=== Test 5: ContextBuilder ===');
    
    final result = await _rag!.search(
      'technology company products',
      topK: 5,
      tokenBudget: 500,
      strategy: ContextStrategy.diverseSources,
    );
    
    _log('Context assembled:');
    _log('  Chunks included: ${result.context.includedChunks.length}');
    _log('  Estimated tokens: ${result.context.estimatedTokens}');
    _log('  Remaining budget: ${result.context.remainingBudget}');
    
    // Test prompt formatting
    final prompt = _rag!.formatPrompt('What products do these companies make?', result);
    _log('');
    _log('Generated prompt (${prompt.length} chars):');
    _log('---');
    _log(prompt.substring(0, prompt.length.clamp(0, 300)));
    if (prompt.length > 300) _log('...[truncated]');
    _log('---');
    
    assert(result.context.estimatedTokens <= 500, 'Should stay within token budget');
    
    _log('✓ ContextBuilder works');
  }

  Future<void> _copyAsset(String assetPath, String targetPath) async {
    final file = File(targetPath);
    if (!await file.exists()) {
      final data = await rootBundle.load(assetPath);
      await file.writeAsBytes(data.buffer.asUint8List());
    }
  }

  @override
  void dispose() {
    EmbeddingService.dispose();
    super.dispose();
  }

  @override
  Widget build(BuildContext context) {
    return Scaffold(
      appBar: AppBar(
        title: const Text('Chunking Integration Test'),
        centerTitle: true,
      ),
      body: Padding(
        padding: const EdgeInsets.all(16),
        child: Column(
          crossAxisAlignment: CrossAxisAlignment.stretch,
          children: [
            // Status
            Card(
              color: _status.contains('✅') 
                  ? Colors.green.shade50 
                  : _status.contains('❌')
                      ? Colors.red.shade50
                      : Colors.grey.shade100,
              child: Padding(
                padding: const EdgeInsets.all(16),
                child: Row(
                  children: [
                    if (_isRunning)
                      const SizedBox(
                        width: 24,
                        height: 24,
                        child: CircularProgressIndicator(strokeWidth: 2),
                      )
                    else
                      Icon(
                        _status.contains('✅') 
                            ? Icons.check_circle 
                            : _status.contains('❌')
                                ? Icons.error
                                : Icons.info,
                        color: _status.contains('✅') 
                            ? Colors.green 
                            : _status.contains('❌')
                                ? Colors.red
                                : Colors.grey,
                      ),
                    const SizedBox(width: 12),
                    Expanded(
                      child: Text(
                        _status,
                        style: const TextStyle(fontWeight: FontWeight.bold),
                      ),
                    ),
                  ],
                ),
              ),
            ),
            
            const SizedBox(height: 16),
            
            ElevatedButton.icon(
              onPressed: _isRunning ? null : _runFullTest,
              icon: const Icon(Icons.play_arrow),
              label: const Text('Run Integration Test'),
              style: ElevatedButton.styleFrom(
                padding: const EdgeInsets.all(16),
              ),
            ),
            
            const SizedBox(height: 16),
            
            const Text('Test Output:', style: TextStyle(fontWeight: FontWeight.bold)),
            const SizedBox(height: 8),
            
            Expanded(
              child: Container(
                padding: const EdgeInsets.all(12),
                decoration: BoxDecoration(
                  color: Colors.grey.shade900,
                  borderRadius: BorderRadius.circular(8),
                ),
                child: ListView.builder(
                  itemCount: _logs.length,
                  itemBuilder: (context, index) {
                    final log = _logs[index];
                    Color color = Colors.white70;
                    if (log.contains('✓')) color = Colors.greenAccent;
                    if (log.contains('❌') || log.contains('ERROR')) color = Colors.redAccent;
                    if (log.contains('===')) color = Colors.cyanAccent;
                    
                    return Text(
                      log,
                      style: TextStyle(
                        fontFamily: 'monospace',
                        fontSize: 11,
                        color: color,
                      ),
                    );
                  },
                ),
              ),
            ),
          ],
        ),
      ),
    );
  }
}
