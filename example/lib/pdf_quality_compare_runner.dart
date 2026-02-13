import 'dart:io';

import 'package:flutter/material.dart';
import 'package:flutter/services.dart';
import 'package:mobile_rag_engine/mobile_rag_engine.dart';
import 'package:path_provider/path_provider.dart';

void main() {
  WidgetsFlutterBinding.ensureInitialized();
  runApp(const _PdfQualityCompareApp());
}

class _PdfQualityCompareApp extends StatefulWidget {
  const _PdfQualityCompareApp();

  @override
  State<_PdfQualityCompareApp> createState() => _PdfQualityCompareAppState();
}

class _PdfQualityCompareAppState extends State<_PdfQualityCompareApp> {
  @override
  void initState() {
    super.initState();
    _run();
  }

  Future<void> _run() async {
    final tempFiles = <String>[];
    final enableWeightSweep =
        const String.fromEnvironment('RAG_WEIGHT_SWEEP', defaultValue: '0') ==
        '1';

    try {
      await MobileRag.initialize(
        tokenizerAsset: 'assets/tokenizer.json',
        modelAsset: 'assets/model.onnx',
        databaseName: 'pdf_quality_compare.sqlite',
        threadLevel: ThreadUseLevel.low,
      );

      await MobileRag.instance.clearAllData();

      final fixtures = <_Fixture>[
        _Fixture(path: 'assets/sample_data/sample_eng.pdf', label: 'eng'),
        _Fixture(path: 'assets/sample_data/sample_kor.pdf', label: 'kor'),
      ];

      final tempDir = await getTemporaryDirectory();

      for (final fixture in fixtures) {
        final pdfAsset = await rootBundle.load(fixture.path);
        final bytes = pdfAsset.buffer.asUint8List();
        final tempPdfPath = '${tempDir.path}/${fixture.path.split('/').last}';
        await File(tempPdfPath).writeAsBytes(bytes, flush: true);
        tempFiles.add(tempPdfPath);

        final text = await DocumentParser.parse(bytes.toList());
        fixture.text = text;

        final addResult = await MobileRag.instance.addDocument(
          text,
          name: fixture.path.split('/').last,
          filePath: tempPdfPath,
          metadata: '{"label":"${fixture.label}"}',
        );
        fixture.sourceId = addResult.sourceId;
      }

      await MobileRag.instance.rebuildIndex();

      final engFixture = fixtures.firstWhere((f) => f.label == 'eng');
      final korFixture = fixtures.firstWhere((f) => f.label == 'kor');

      final engPositiveQueries = _buildQueries(
        engFixture.text,
        maxQueries: 4,
        preferKorean: false,
      );
      final korPositiveQueries = _buildQueries(
        korFixture.text,
        maxQueries: 10,
        preferKorean: true,
      );

      final hardKorQueries = _buildHardNegativeQueries(
        targetQueries: korPositiveQueries,
        distractorQueries: engPositiveQueries,
        expectedSourceId: korFixture.sourceId,
        label: 'hard_neg_kor',
        maxQueries: 4,
        isKorean: true,
      );
      final hardEngQueries = _buildHardNegativeQueries(
        targetQueries: engPositiveQueries,
        distractorQueries: korPositiveQueries,
        expectedSourceId: engFixture.sourceId,
        label: 'hard_neg_eng',
        maxQueries: 2,
        isKorean: false,
      );

      final evalQueries = <_EvalQuery>[
        ...engPositiveQueries.map(
          (q) => _EvalQuery(
            query: q,
            expectedSourceId: engFixture.sourceId,
            label: 'positive_eng',
            isHardNegative: false,
            isKorean: false,
          ),
        ),
        ...korPositiveQueries.map(
          (q) => _EvalQuery(
            query: q,
            expectedSourceId: korFixture.sourceId,
            label: 'positive_kor',
            isHardNegative: false,
            isKorean: true,
          ),
        ),
        ...hardKorQueries,
        ...hardEngQueries,
      ];

      if (evalQueries.isEmpty) {
        throw StateError('No evaluation queries were generated.');
      }

      stdout.writeln(
        'PDF_EVALSET_RESULT '
        'total=${evalQueries.length} '
        'positive_eng=${engPositiveQueries.length} '
        'positive_kor=${korPositiveQueries.length} '
        'hard_neg_kor=${hardKorQueries.length} '
        'hard_neg_eng=${hardEngQueries.length} '
        'weight_sweep=${enableWeightSweep ? 1 : 0}',
      );

      final defaultSummary = await _evaluateQueries(
        evalQueries,
        vectorWeight: null,
        bm25Weight: null,
        emitQueryLogs: true,
      );
      _printQualityResult(defaultSummary, evalQueries.length);

      if (enableWeightSweep) {
        final sweepConfigs = <_WeightConfig>[
          const _WeightConfig(vectorWeight: 0.2, bm25Weight: 0.8),
          const _WeightConfig(vectorWeight: 0.3, bm25Weight: 0.7),
          const _WeightConfig(vectorWeight: 0.4, bm25Weight: 0.6),
          const _WeightConfig(vectorWeight: 0.5, bm25Weight: 0.5),
          const _WeightConfig(vectorWeight: 0.6, bm25Weight: 0.4),
          const _WeightConfig(vectorWeight: 0.7, bm25Weight: 0.3),
          const _WeightConfig(vectorWeight: 0.8, bm25Weight: 0.2),
        ];

        final sweepSummaries = <_EvalSummary>[];
        for (final config in sweepConfigs) {
          final summary = await _evaluateQueries(
            evalQueries,
            vectorWeight: config.vectorWeight,
            bm25Weight: config.bm25Weight,
            emitQueryLogs: false,
          );
          sweepSummaries.add(summary);
          _printWeightResult(summary, evalQueries.length);
        }

        final bestOverall = _pickBest(
          sweepSummaries,
          (s) =>
              s.allStats.hitAt1 * 1000 +
              s.allStats.mrrAt3 * 10 +
              s.hardStats.hitAt1,
        );
        final bestHard = _pickBest(
          sweepSummaries,
          (s) =>
              s.hardStats.hitAt1 * 1000 +
              s.hardStats.mrrAt3 * 10 +
              s.allStats.hitAt1,
        );
        final bestKor = _pickBest(
          sweepSummaries,
          (s) =>
              s.korStats.hitAt1 * 1000 +
              s.korStats.mrrAt3 * 10 +
              s.allStats.hitAt1,
        );

        stdout.writeln(
          'PDF_WEIGHT_BEST scope=overall '
          'vector_weight=${_weightLabel(bestOverall.vectorWeight)} '
          'bm25_weight=${_weightLabel(bestOverall.bm25Weight)} '
          'hit_at_1=${bestOverall.allStats.hitAt1.toStringAsFixed(6)} '
          'mrr_at_3=${bestOverall.allStats.mrrAt3.toStringAsFixed(6)} '
          'hard_hit_at_1=${bestOverall.hardStats.hitAt1.toStringAsFixed(6)} '
          'kor_hit_at_1=${bestOverall.korStats.hitAt1.toStringAsFixed(6)}',
        );
        stdout.writeln(
          'PDF_WEIGHT_BEST scope=hard_negative '
          'vector_weight=${_weightLabel(bestHard.vectorWeight)} '
          'bm25_weight=${_weightLabel(bestHard.bm25Weight)} '
          'hard_hit_at_1=${bestHard.hardStats.hitAt1.toStringAsFixed(6)} '
          'hard_mrr_at_3=${bestHard.hardStats.mrrAt3.toStringAsFixed(6)} '
          'hit_at_1=${bestHard.allStats.hitAt1.toStringAsFixed(6)}',
        );
        stdout.writeln(
          'PDF_WEIGHT_BEST scope=korean '
          'vector_weight=${_weightLabel(bestKor.vectorWeight)} '
          'bm25_weight=${_weightLabel(bestKor.bm25Weight)} '
          'kor_hit_at_1=${bestKor.korStats.hitAt1.toStringAsFixed(6)} '
          'kor_mrr_at_3=${bestKor.korStats.mrrAt3.toStringAsFixed(6)} '
          'hit_at_1=${bestKor.allStats.hitAt1.toStringAsFixed(6)}',
        );
      }

      await Future<void>.delayed(const Duration(milliseconds: 300));
      exit(0);
    } catch (e, st) {
      stderr.writeln('PDF_QUALITY_ERROR $e');
      stderr.writeln(st);
      await Future<void>.delayed(const Duration(milliseconds: 300));
      exit(1);
    } finally {
      for (final tempFile in tempFiles) {
        final file = File(tempFile);
        if (file.existsSync()) {
          try {
            await file.delete();
          } catch (_) {
            // Best-effort cleanup only.
          }
        }
      }
    }
  }

  Future<_EvalSummary> _evaluateQueries(
    List<_EvalQuery> evalQueries, {
    required double? vectorWeight,
    required double? bm25Weight,
    required bool emitQueryLogs,
  }) async {
    final allStats = _MetricStats();
    final hardStats = _MetricStats();
    final korStats = _MetricStats();

    for (final eval in evalQueries) {
      final results = (vectorWeight == null || bm25Weight == null)
          ? await MobileRag.instance.searchHybrid(eval.query, topK: 3)
          : await MobileRag.instance.searchHybrid(
              eval.query,
              topK: 3,
              vectorWeight: vectorWeight,
              bm25Weight: bm25Weight,
            );
      final sourceIds = results.map((r) => r.sourceId.toInt()).toList();
      final rank = sourceIds.indexOf(eval.expectedSourceId);

      allStats.record(rank);
      if (eval.isHardNegative) hardStats.record(rank);
      if (eval.isKorean) korStats.record(rank);

      if (emitQueryLogs) {
        final top1Source = results.isEmpty
            ? -1
            : results.first.sourceId.toInt();
        final top1Score = results.isEmpty
            ? 0.0
            : double.parse(results.first.score.toStringAsFixed(6));

        stdout.writeln(
          'PDF_QUERY_RESULT type=${eval.label} '
          'lang=${eval.isKorean ? 'ko' : 'en'} '
          'expected=${eval.expectedSourceId} '
          'rank=${rank < 0 ? -1 : rank + 1} '
          'top1=$top1Source '
          'top1_score=$top1Score '
          'vector_weight=${_weightLabel(vectorWeight)} '
          'bm25_weight=${_weightLabel(bm25Weight)} '
          'query="${_clip(eval.query, 90)}"',
        );
      }
    }

    return _EvalSummary(
      vectorWeight: vectorWeight,
      bm25Weight: bm25Weight,
      allStats: allStats,
      hardStats: hardStats,
      korStats: korStats,
    );
  }

  String _weightLabel(double? weight) {
    if (weight == null) return 'default';
    return weight.toStringAsFixed(1);
  }

  void _printQualityResult(_EvalSummary summary, int evalTotal) {
    stdout.writeln(
      'PDF_QUALITY_RESULT '
      'total=${summary.allStats.total} '
      'vector_weight=${_weightLabel(summary.vectorWeight)} '
      'bm25_weight=${_weightLabel(summary.bm25Weight)} '
      'hit_at_1=${summary.allStats.hitAt1.toStringAsFixed(6)} '
      'hit_at_3=${summary.allStats.hitAt3.toStringAsFixed(6)} '
      'mrr_at_3=${summary.allStats.mrrAt3.toStringAsFixed(6)} '
      'hard_total=${summary.hardStats.total} '
      'hard_hit_at_1=${summary.hardStats.hitAt1.toStringAsFixed(6)} '
      'hard_hit_at_3=${summary.hardStats.hitAt3.toStringAsFixed(6)} '
      'hard_mrr_at_3=${summary.hardStats.mrrAt3.toStringAsFixed(6)} '
      'kor_total=${summary.korStats.total} '
      'kor_hit_at_1=${summary.korStats.hitAt1.toStringAsFixed(6)} '
      'kor_hit_at_3=${summary.korStats.hitAt3.toStringAsFixed(6)} '
      'kor_mrr_at_3=${summary.korStats.mrrAt3.toStringAsFixed(6)} '
      'kor_query_ratio=${(evalTotal == 0 ? 0.0 : summary.korStats.total / evalTotal).toStringAsFixed(6)}',
    );
  }

  void _printWeightResult(_EvalSummary summary, int evalTotal) {
    stdout.writeln(
      'PDF_WEIGHT_RESULT '
      'vector_weight=${_weightLabel(summary.vectorWeight)} '
      'bm25_weight=${_weightLabel(summary.bm25Weight)} '
      'total=${summary.allStats.total} '
      'hit_at_1=${summary.allStats.hitAt1.toStringAsFixed(6)} '
      'mrr_at_3=${summary.allStats.mrrAt3.toStringAsFixed(6)} '
      'hard_hit_at_1=${summary.hardStats.hitAt1.toStringAsFixed(6)} '
      'hard_mrr_at_3=${summary.hardStats.mrrAt3.toStringAsFixed(6)} '
      'kor_hit_at_1=${summary.korStats.hitAt1.toStringAsFixed(6)} '
      'kor_mrr_at_3=${summary.korStats.mrrAt3.toStringAsFixed(6)} '
      'kor_query_ratio=${(evalTotal == 0 ? 0.0 : summary.korStats.total / evalTotal).toStringAsFixed(6)}',
    );
  }

  _EvalSummary _pickBest(
    List<_EvalSummary> candidates,
    double Function(_EvalSummary summary) scoreFn,
  ) {
    var best = candidates.first;
    var bestScore = scoreFn(best);
    for (var i = 1; i < candidates.length; i++) {
      final score = scoreFn(candidates[i]);
      if (score > bestScore) {
        best = candidates[i];
        bestScore = score;
      }
    }
    return best;
  }

  List<String> _buildQueries(
    String text, {
    required int maxQueries,
    required bool preferKorean,
  }) {
    final normalized = text.replaceAll(RegExp(r'\s+'), ' ').trim();
    if (normalized.isEmpty) return [];

    final allSentences = normalized
        .split(RegExp(r'[.!?。\n]'))
        .map((s) => s.trim())
        .where((s) => s.length >= 24)
        .toList();

    if (allSentences.isEmpty) return [];

    final preferredSentences = allSentences.where((s) {
      final hasHangul = _containsHangul(s);
      return preferKorean ? hasHangul : !hasHangul;
    }).toList();

    final candidates = preferredSentences.isEmpty
        ? allSentences
        : preferredSentences;

    final queries = <String>[];
    final seen = <String>{};

    for (final sentence in candidates) {
      final query = _makeQuery(sentence, preferKorean: preferKorean);
      if (query.length < 12) continue;
      if (seen.add(query)) {
        queries.add(query);
      }
      if (queries.length >= maxQueries) break;
    }

    if (queries.length >= maxQueries) return queries;

    for (final sentence in allSentences) {
      final query = _makeQuery(sentence, preferKorean: preferKorean);
      if (query.length < 12) continue;
      if (seen.add(query)) {
        queries.add(query);
      }
      if (queries.length >= maxQueries) break;
    }

    return queries;
  }

  List<_EvalQuery> _buildHardNegativeQueries({
    required List<String> targetQueries,
    required List<String> distractorQueries,
    required int expectedSourceId,
    required String label,
    required int maxQueries,
    required bool isKorean,
  }) {
    if (targetQueries.isEmpty || distractorQueries.isEmpty || maxQueries <= 0) {
      return [];
    }

    final hardQueries = <_EvalQuery>[];
    final seen = <String>{};

    for (var i = 0; i < maxQueries; i++) {
      final target = targetQueries[i % targetQueries.length];
      final distractor = distractorQueries[i % distractorQueries.length];
      final mixed =
          '${_takeWords(target, isKorean ? 6 : 7)} ${_takeWords(distractor, 2)}'
              .trim();

      if (mixed.length < 12) continue;
      if (!seen.add(mixed)) continue;

      hardQueries.add(
        _EvalQuery(
          query: mixed,
          expectedSourceId: expectedSourceId,
          label: label,
          isHardNegative: true,
          isKorean: isKorean,
        ),
      );
    }

    return hardQueries;
  }

  String _makeQuery(String sentence, {required bool preferKorean}) {
    final s = sentence.replaceAll(RegExp(r'\s+'), ' ').trim();
    final words = s.split(' ').where((w) => w.isNotEmpty).toList();
    final desiredWords = preferKorean ? 6 : 7;
    if (words.length >= desiredWords) {
      return words.take(desiredWords).join(' ');
    }
    if (s.length > 48) {
      return s.substring(0, 48);
    }
    return s;
  }

  String _takeWords(String text, int count) {
    final words = text.split(' ').where((w) => w.isNotEmpty).toList();
    if (words.isEmpty) return text;
    return words.take(count).join(' ');
  }

  bool _containsHangul(String text) => RegExp(r'[가-힣]').hasMatch(text);

  String _clip(String s, int max) {
    if (s.length <= max) return s;
    return '${s.substring(0, max - 3)}...';
  }

  @override
  Widget build(BuildContext context) {
    return const MaterialApp(
      home: Scaffold(
        body: Center(child: Text('Running PDF quality comparison...')),
      ),
    );
  }
}

class _Fixture {
  final String path;
  final String label;
  int sourceId = -1;
  String text = '';

  _Fixture({required this.path, required this.label});
}

class _EvalQuery {
  final String query;
  final int expectedSourceId;
  final String label;
  final bool isHardNegative;
  final bool isKorean;

  _EvalQuery({
    required this.query,
    required this.expectedSourceId,
    required this.label,
    required this.isHardNegative,
    required this.isKorean,
  });
}

class _MetricStats {
  int total = 0;
  int _hitAt1 = 0;
  int _hitAt3 = 0;
  double _mrrAt3 = 0;

  void record(int rank) {
    total += 1;
    if (rank == 0) _hitAt1 += 1;
    if (rank >= 0) {
      _hitAt3 += 1;
      _mrrAt3 += 1.0 / (rank + 1);
    }
  }

  double get hitAt1 => total == 0 ? 0.0 : _hitAt1 / total;
  double get hitAt3 => total == 0 ? 0.0 : _hitAt3 / total;
  double get mrrAt3 => total == 0 ? 0.0 : _mrrAt3 / total;
}

class _EvalSummary {
  final double? vectorWeight;
  final double? bm25Weight;
  final _MetricStats allStats;
  final _MetricStats hardStats;
  final _MetricStats korStats;

  _EvalSummary({
    required this.vectorWeight,
    required this.bm25Weight,
    required this.allStats,
    required this.hardStats,
    required this.korStats,
  });
}

class _WeightConfig {
  final double vectorWeight;
  final double bm25Weight;

  const _WeightConfig({required this.vectorWeight, required this.bm25Weight});
}
