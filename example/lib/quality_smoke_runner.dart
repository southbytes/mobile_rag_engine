import 'dart:io';

import 'package:flutter/material.dart';
import 'package:mobile_rag_engine/mobile_rag_engine.dart';

void main() {
  WidgetsFlutterBinding.ensureInitialized();
  runApp(const _QualitySmokeApp());
}

class _QualitySmokeApp extends StatefulWidget {
  const _QualitySmokeApp();

  @override
  State<_QualitySmokeApp> createState() => _QualitySmokeAppState();
}

class _QualitySmokeAppState extends State<_QualitySmokeApp> {
  @override
  void initState() {
    super.initState();
    _run();
  }

  Future<void> _run() async {
    try {
      await MobileRag.initialize(
        tokenizerAsset: 'assets/tokenizer.json',
        modelAsset: 'assets/model.onnx',
        databaseName: 'quality_smoke_runner.sqlite',
        threadLevel: ThreadUseLevel.low,
      );

      final summary = await QualityTestService.runQualityTest(
        onProgress: (message, current, total) {
          if (total == 0) return;
          if (current == 0 || current == total || current % 5 == 0) {
            stdout.writeln(
              'QUALITY_TEST_PROGRESS message="$message" '
              'current=$current total=$total',
            );
          }
        },
      );

      stdout.writeln(
        'QUALITY_TEST_RESULT '
        'total=${summary.total} '
        'passed=${summary.passed} '
        'pass_rate=${summary.passRate.toStringAsFixed(6)} '
        'avg_recall=${summary.avgRecallAt3.toStringAsFixed(6)} '
        'avg_precision=${summary.avgPrecision.toStringAsFixed(6)}',
      );

      await Future<void>.delayed(const Duration(milliseconds: 300));
      exit(0);
    } catch (e, st) {
      stderr.writeln('QUALITY_TEST_ERROR $e');
      stderr.writeln(st);
      await Future<void>.delayed(const Duration(milliseconds: 300));
      exit(1);
    }
  }

  @override
  Widget build(BuildContext context) {
    return const MaterialApp(
      home: Scaffold(
        body: Center(
          child: Text('Running quality smoke test...'),
        ),
      ),
    );
  }
}
