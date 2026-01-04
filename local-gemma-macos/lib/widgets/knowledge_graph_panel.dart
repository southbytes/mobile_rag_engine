// local-gemma-macos/lib/widgets/knowledge_graph_panel.dart
//
// Custom graph visualization for RAG search results
// Shows query as center node connected to retrieved chunks

import 'dart:math' as math;

import 'package:flutter/material.dart';
import 'package:mobile_rag_engine/src/rust/api/source_rag.dart';

/// Node types in the knowledge graph
enum GraphNodeType {
  query, // Center node - the search query
  selected, // Chunk that passed similarity threshold
  candidate, // Chunk that was searched but filtered out
  adjacent, // Adjacent chunk (similarity = 0.0)
}

/// Data for each node in the graph
class ChunkNode {
  final String id;
  final String label;
  final GraphNodeType type;
  final double? similarity;
  final ChunkSearchResult? chunk;
  Offset position;

  ChunkNode({
    required this.id,
    required this.label,
    required this.type,
    this.similarity,
    this.chunk,
    this.position = Offset.zero,
  });

  @override
  bool operator ==(Object other) =>
      identical(this, other) || other is ChunkNode && id == other.id;

  @override
  int get hashCode => id.hashCode;
}

/// Edge between nodes
class GraphEdge {
  final ChunkNode source;
  final ChunkNode target;
  final bool isDashed;

  const GraphEdge({
    required this.source,
    required this.target,
    this.isDashed = false,
  });
}

/// Knowledge Graph Panel for RAG visualization
class KnowledgeGraphPanel extends StatefulWidget {
  final String? query;
  final List<ChunkSearchResult> chunks;
  final double similarityThreshold;
  final void Function(ChunkSearchResult chunk)? onChunkSelected;
  final ChunkSearchResult? selectedChunk;

  const KnowledgeGraphPanel({
    super.key,
    this.query,
    this.chunks = const [],
    this.similarityThreshold = 0.35,
    this.onChunkSelected,
    this.selectedChunk,
  });

  @override
  State<KnowledgeGraphPanel> createState() => _KnowledgeGraphPanelState();
}

class _KnowledgeGraphPanelState extends State<KnowledgeGraphPanel>
    with SingleTickerProviderStateMixin {
  final List<ChunkNode> _nodes = [];
  final List<GraphEdge> _edges = [];
  ChunkNode? _queryNode;

  // Pan and zoom
  Offset _offset = Offset.zero;
  double _scale = 1.0;
  Size? _containerSize; // Store container size for centering

  // Animation
  late AnimationController _animationController;

  @override
  void initState() {
    super.initState();
    _animationController = AnimationController(
      vsync: this,
      duration: const Duration(milliseconds: 1500),
    );
    _buildGraph();
    _animationController.forward();
  }

  @override
  void didUpdateWidget(KnowledgeGraphPanel oldWidget) {
    super.didUpdateWidget(oldWidget);
    if (oldWidget.query != widget.query ||
        oldWidget.chunks.length != widget.chunks.length) {
      _buildGraph();
      // Reset offset to center the new graph
      _centerGraph();
      _animationController.forward(from: 0);
    }
  }

  void _centerGraph() {
    if (_containerSize != null && _queryNode != null) {
      // Calculate offset to center the query node
      final centerX = _containerSize!.width / 2;
      final centerY = _containerSize!.height / 2;
      _offset = Offset(
        centerX - _queryNode!.position.dx,
        centerY - _queryNode!.position.dy,
      );
    }
  }

  void _buildGraph() {
    _nodes.clear();
    _edges.clear();
    _queryNode = null;

    if (widget.query == null || widget.query!.isEmpty) return;

    final random = math.Random(42);
    const centerX = 180.0;
    const centerY = 180.0;

    // Create query node (center)
    _queryNode = ChunkNode(
      id: 'query',
      label: widget.query!.length > 15
          ? '${widget.query!.substring(0, 15)}...'
          : widget.query!,
      type: GraphNodeType.query,
      position: const Offset(centerX, centerY),
    );
    _nodes.add(_queryNode!);

    // Separate primary chunks (with similarity > 0) from adjacent chunks
    final primaryChunks = <int>[]; // Chunks with actual similarity
    final adjacentMap =
        <String, List<int>>{}; // Map sourceId+chunkIndex -> adjacent indices

    for (var i = 0; i < widget.chunks.length; i++) {
      final chunk = widget.chunks[i];
      if (chunk.similarity > 0.0) {
        primaryChunks.add(i);
      }
    }

    // Build map of which adjacent chunks belong to which primary chunk
    for (var i = 0; i < widget.chunks.length; i++) {
      final chunk = widget.chunks[i];
      if (chunk.similarity == 0.0) {
        // Find parent primary chunk (same source, adjacent index)
        for (final primaryIdx in primaryChunks) {
          final primary = widget.chunks[primaryIdx];
          if (primary.sourceId == chunk.sourceId &&
              (primary.chunkIndex - chunk.chunkIndex).abs() <= 2) {
            final key = '${primary.sourceId}_${primary.chunkIndex}';
            adjacentMap.putIfAbsent(key, () => []).add(i);
            break;
          }
        }
      }
    }

    // Position primary chunks with more variance
    final primaryCount = primaryChunks.length;
    final baseAngleStep = primaryCount > 0 ? (2 * math.pi) / primaryCount : 0;
    final nodeMap = <String, ChunkNode>{};

    for (var i = 0; i < primaryCount; i++) {
      final chunkIndex = primaryChunks[i];
      final chunk = widget.chunks[chunkIndex];
      final isSelected = chunk.similarity >= widget.similarityThreshold;

      // Add angle variance to break uniform pattern
      final angleVariance = (random.nextDouble() - 0.5) * 0.3;
      final angle = baseAngleStep * i + angleVariance;

      // More radius variance based on similarity
      final baseRadius = isSelected ? 90.0 : 130.0;
      final radiusVariance = 30.0 + random.nextDouble() * 40;
      final radius = baseRadius + radiusVariance;
      final x = centerX + radius * math.cos(angle);
      final y = centerY + radius * math.sin(angle);

      final preview = chunk.content.length > 20
          ? '${chunk.content.substring(0, 20)}...'
          : chunk.content;

      final node = ChunkNode(
        id: 'chunk_${chunk.chunkId}',
        label: preview,
        type: isSelected ? GraphNodeType.selected : GraphNodeType.candidate,
        similarity: chunk.similarity,
        chunk: chunk,
        position: Offset(x, y),
      );
      _nodes.add(node);
      nodeMap['${chunk.sourceId}_${chunk.chunkIndex}'] = node;

      // Edge from query to primary chunk
      _edges.add(GraphEdge(source: _queryNode!, target: node));

      // Position adjacent chunks as satellites - spread outward from center
      final key = '${chunk.sourceId}_${chunk.chunkIndex}';
      final adjacentIndices = adjacentMap[key] ?? [];

      for (var j = 0; j < adjacentIndices.length; j++) {
        final adjChunkIndex = adjacentIndices[j];
        final adjChunk = widget.chunks[adjChunkIndex];

        // Position outward from parent (away from center)
        // Use different angles to spread around parent
        final spreadAngle = angle + (j - adjacentIndices.length / 2) * 0.6;
        final spreadVariance = (random.nextDouble() - 0.5) * 0.4;
        final satAngle = spreadAngle + spreadVariance;

        // Distance from parent, further out
        final satRadius = 50.0 + random.nextDouble() * 25;
        final satX = x + satRadius * math.cos(satAngle);
        final satY = y + satRadius * math.sin(satAngle);

        final adjPreview = adjChunk.content.length > 20
            ? '${adjChunk.content.substring(0, 20)}...'
            : adjChunk.content;

        final adjNode = ChunkNode(
          id: 'chunk_${adjChunk.chunkId}',
          label: adjPreview,
          type: GraphNodeType.adjacent,
          similarity: adjChunk.similarity,
          chunk: adjChunk,
          position: Offset(satX, satY),
        );
        _nodes.add(adjNode);

        // Edge from parent to adjacent (short, dashed)
        _edges.add(GraphEdge(source: node, target: adjNode, isDashed: true));
      }
    }
  }

  @override
  void dispose() {
    _animationController.dispose();
    super.dispose();
  }

  Color _getNodeColor(GraphNodeType type) {
    switch (type) {
      case GraphNodeType.query:
        return Colors.purple;
      case GraphNodeType.selected:
        return Colors.green;
      case GraphNodeType.candidate:
        return Colors.orange.withOpacity(0.7);
      case GraphNodeType.adjacent:
        return Colors.grey.withOpacity(0.5);
    }
  }

  double _getNodeSize(GraphNodeType type, double? similarity) {
    switch (type) {
      case GraphNodeType.query:
        return 32;
      case GraphNodeType.selected:
        return 16 + (similarity ?? 0.5) * 10;
      case GraphNodeType.candidate:
        return 14;
      case GraphNodeType.adjacent:
        return 10;
    }
  }

  @override
  Widget build(BuildContext context) {
    if (widget.query == null || widget.chunks.isEmpty) {
      return Container(
        color: Colors.grey[900],
        child: Center(
          child: Column(
            mainAxisAlignment: MainAxisAlignment.center,
            children: [
              Icon(Icons.hub_outlined, size: 48, color: Colors.grey[600]),
              const SizedBox(height: 16),
              Text(
                'Knowledge Graph',
                style: TextStyle(color: Colors.grey[500], fontSize: 16),
              ),
              const SizedBox(height: 8),
              Text(
                '질문을 하면 검색 결과가\n그래프로 표시됩니다',
                textAlign: TextAlign.center,
                style: TextStyle(color: Colors.grey[600], fontSize: 12),
              ),
            ],
          ),
        ),
      );
    }

    return Container(
      color: Colors.grey[900],
      child: Column(
        children: [
          // Header
          Container(
            padding: const EdgeInsets.symmetric(horizontal: 12, vertical: 8),
            decoration: BoxDecoration(
              color: Colors.grey[850],
              border: Border(bottom: BorderSide(color: Colors.grey[800]!)),
            ),
            child: Row(
              children: [
                const Icon(Icons.hub, size: 16, color: Colors.purple),
                const SizedBox(width: 8),
                const Text(
                  'Graph View',
                  style: TextStyle(
                    fontWeight: FontWeight.bold,
                    fontSize: 12,
                    color: Colors.white,
                  ),
                ),
                const Spacer(),
                Text(
                  '${widget.chunks.length} nodes',
                  style: TextStyle(color: Colors.grey[500], fontSize: 11),
                ),
              ],
            ),
          ),
          Expanded(
            child: ClipRect(
              child: LayoutBuilder(
                builder: (context, constraints) {
                  // Store container size and center graph on first build
                  if (_containerSize != constraints.biggest) {
                    WidgetsBinding.instance.addPostFrameCallback((_) {
                      if (mounted && _containerSize != constraints.biggest) {
                        _containerSize = constraints.biggest;
                        if (_offset == Offset.zero && _queryNode != null) {
                          setState(() => _centerGraph());
                        }
                      }
                    });
                  }

                  return GestureDetector(
                    onScaleStart: (details) {
                      // Store initial values for scale gesture
                    },
                    onScaleUpdate: (details) {
                      setState(() {
                        // Handle both pan (focalPointDelta) and scale
                        _offset += details.focalPointDelta;
                        if (details.scale != 1.0) {
                          _scale = (_scale * details.scale).clamp(0.5, 3.0);
                        }
                      });
                    },
                    child: AnimatedBuilder(
                      animation: _animationController,
                      builder: (context, child) {
                        return CustomPaint(
                          painter: _GraphPainter(
                            nodes: _nodes,
                            edges: _edges,
                            offset: _offset,
                            scale: _scale,
                            selectedChunk: widget.selectedChunk,
                            animationValue: _animationController.value,
                            getNodeColor: _getNodeColor,
                            getNodeSize: _getNodeSize,
                          ),
                          size: Size.infinite,
                          child: Stack(
                            children: _nodes.map((node) {
                              final pos = (node.position + _offset) * _scale;
                              final size = _getNodeSize(
                                node.type,
                                node.similarity,
                              );
                              final isSelected =
                                  widget.selectedChunk != null &&
                                  node.chunk?.chunkId ==
                                      widget.selectedChunk!.chunkId;

                              return Positioned(
                                left: pos.dx - size / 2,
                                top: pos.dy - size / 2,
                                child: GestureDetector(
                                  onTap: () {
                                    if (node.chunk != null &&
                                        widget.onChunkSelected != null) {
                                      widget.onChunkSelected!(node.chunk!);
                                    }
                                  },
                                  child: Tooltip(
                                    message: node.label,
                                    child: AnimatedContainer(
                                      duration: const Duration(
                                        milliseconds: 200,
                                      ),
                                      width: size,
                                      height: size,
                                      decoration: BoxDecoration(
                                        shape: BoxShape.circle,
                                        color: _getNodeColor(node.type),
                                        border: isSelected
                                            ? Border.all(
                                                color: Colors.white,
                                                width: 2,
                                              )
                                            : null,
                                        boxShadow: [
                                          BoxShadow(
                                            color: _getNodeColor(
                                              node.type,
                                            ).withOpacity(0.5),
                                            blurRadius: isSelected ? 12 : 6,
                                          ),
                                        ],
                                      ),
                                      child: node.type == GraphNodeType.query
                                          ? const Center(
                                              child: Icon(
                                                Icons.search,
                                                color: Colors.white,
                                                size: 18,
                                              ),
                                            )
                                          : null,
                                    ),
                                  ),
                                ),
                              );
                            }).toList(),
                          ),
                        );
                      },
                    ),
                  );
                },
              ),
            ),
          ),
          // Legend
          Container(
            padding: const EdgeInsets.symmetric(horizontal: 12, vertical: 6),
            decoration: BoxDecoration(
              color: Colors.grey[850],
              border: Border(top: BorderSide(color: Colors.grey[800]!)),
            ),
            child: Row(
              mainAxisAlignment: MainAxisAlignment.center,
              children: [
                _buildLegendItem(Colors.purple, 'Query'),
                const SizedBox(width: 16),
                _buildLegendItem(Colors.green, 'Selected'),
                const SizedBox(width: 16),
                _buildLegendItem(Colors.orange.withOpacity(0.7), 'Low'),
              ],
            ),
          ),
        ],
      ),
    );
  }

  Widget _buildLegendItem(Color color, String label) {
    return Row(
      mainAxisSize: MainAxisSize.min,
      children: [
        Container(
          width: 8,
          height: 8,
          decoration: BoxDecoration(shape: BoxShape.circle, color: color),
        ),
        const SizedBox(width: 4),
        Text(label, style: TextStyle(color: Colors.grey[400], fontSize: 10)),
      ],
    );
  }
}

/// Custom painter for graph edges
class _GraphPainter extends CustomPainter {
  final List<ChunkNode> nodes;
  final List<GraphEdge> edges;
  final Offset offset;
  final double scale;
  final ChunkSearchResult? selectedChunk;
  final double animationValue;
  final Color Function(GraphNodeType) getNodeColor;
  final double Function(GraphNodeType, double?) getNodeSize;

  _GraphPainter({
    required this.nodes,
    required this.edges,
    required this.offset,
    required this.scale,
    required this.selectedChunk,
    required this.animationValue,
    required this.getNodeColor,
    required this.getNodeSize,
  });

  @override
  void paint(Canvas canvas, Size size) {
    // Draw edges
    for (final edge in edges) {
      final start = (edge.source.position + offset) * scale;
      final end = (edge.target.position + offset) * scale;

      // Animate edge drawing
      final animatedEnd = Offset.lerp(start, end, animationValue.clamp(0, 1))!;

      final paint = Paint()
        ..color = Colors.grey.withOpacity(0.3)
        ..strokeWidth = edge.isDashed ? 1.0 : 1.5
        ..style = PaintingStyle.stroke;

      if (edge.isDashed) {
        _drawDashedLine(canvas, start, animatedEnd, paint);
      } else {
        canvas.drawLine(start, animatedEnd, paint);
      }
    }
  }

  void _drawDashedLine(Canvas canvas, Offset start, Offset end, Paint paint) {
    const dashLength = 5.0;
    const gapLength = 3.0;
    final dx = end.dx - start.dx;
    final dy = end.dy - start.dy;
    final distance = math.sqrt(dx * dx + dy * dy);
    final dashCount = (distance / (dashLength + gapLength)).floor();

    for (var i = 0; i < dashCount; i++) {
      final t1 = i * (dashLength + gapLength) / distance;
      final t2 = (i * (dashLength + gapLength) + dashLength) / distance;
      canvas.drawLine(
        Offset(start.dx + dx * t1, start.dy + dy * t1),
        Offset(start.dx + dx * t2, start.dy + dy * t2),
        paint,
      );
    }
  }

  @override
  bool shouldRepaint(_GraphPainter oldDelegate) {
    return oldDelegate.offset != offset ||
        oldDelegate.scale != scale ||
        oldDelegate.animationValue != animationValue ||
        oldDelegate.selectedChunk != selectedChunk;
  }
}
