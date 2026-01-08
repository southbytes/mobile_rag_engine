// lib/screens/benchmark_screen.dart
import 'package:flutter/material.dart';
import 'package:path_provider/path_provider.dart';
import 'package:mobile_rag_engine/services/benchmark_service.dart';

class BenchmarkScreen extends StatefulWidget {
  const BenchmarkScreen({super.key});

  @override
  State<BenchmarkScreen> createState() => _BenchmarkScreenState();
}

class _BenchmarkScreenState extends State<BenchmarkScreen> {
  bool _isRunning = false;
  String _progress = "";
  List<BenchmarkResult> _results = [];
  String _dbPath = "";

  @override
  void initState() {
    super.initState();
    _initDbPath();
  }

  Future<void> _initDbPath() async {
    final dir = await getApplicationDocumentsDirectory();
    _dbPath = "${dir.path}/rag_db.sqlite";
  }

  Future<void> _runBenchmark() async {
    setState(() {
      _isRunning = true;
      _results = [];
      _progress = "Starting benchmark...";
    });

    try {
      final results = await BenchmarkService.runFullBenchmark(
        dbPath: _dbPath,
        onProgress: (msg) {
          setState(() => _progress = msg);
        },
      );

      setState(() {
        _results = results;
        _isRunning = false;
        _progress = "Complete!";
      });
    } catch (e) {
      setState(() {
        _isRunning = false;
        _progress = "Error: $e";
      });
    }
  }

  @override
  Widget build(BuildContext context) {
    return Scaffold(
      appBar: AppBar(
        title: const Text('🧪 Performance Benchmark'),
      ),
      body: Padding(
        padding: const EdgeInsets.all(16.0),
        child: Column(
          crossAxisAlignment: CrossAxisAlignment.stretch,
          children: [
            // Run button
            ElevatedButton.icon(
              onPressed: _isRunning ? null : _runBenchmark,
              icon: _isRunning
                  ? const SizedBox(
                      width: 20,
                      height: 20,
                      child: CircularProgressIndicator(strokeWidth: 2),
                    )
                  : const Icon(Icons.play_arrow),
              label: Text(_isRunning ? 'Running...' : 'Run Benchmark'),
              style: ElevatedButton.styleFrom(
                padding: const EdgeInsets.all(16),
              ),
            ),

            const SizedBox(height: 16),

            // Progress status
            if (_progress.isNotEmpty)
              Card(
                child: Padding(
                  padding: const EdgeInsets.all(12.0),
                  child: Text(_progress, style: const TextStyle(fontSize: 14)),
                ),
              ),

            const SizedBox(height: 16),

            // Results
            if (_results.isNotEmpty) ...[
              const Text(
                '📊 Results',
                style: TextStyle(fontWeight: FontWeight.bold, fontSize: 18),
              ),
              const SizedBox(height: 8),
              Expanded(
                child: ListView.builder(
                  itemCount: _results.length,
                  itemBuilder: (context, index) {
                    final r = _results[index];
                    return Card(
                      child: ListTile(
                        title: Text(r.name),
                        subtitle: Text(
                          'avg: ${r.avgMs.toStringAsFixed(2)}ms | '
                          'min: ${r.minMs.toStringAsFixed(2)}ms | '
                          'max: ${r.maxMs.toStringAsFixed(2)}ms',
                        ),
                        trailing: Text(
                          '${r.avgMs.toStringAsFixed(1)}ms',
                          style: TextStyle(
                            fontSize: 16,
                            fontWeight: FontWeight.bold,
                            color: r.avgMs < 50
                                ? Colors.green
                                : r.avgMs < 200
                                    ? Colors.orange
                                    : Colors.red,
                          ),
                        ),
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
}
