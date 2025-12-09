// rust/src/api/hnsw_index.rs
//! HNSW (Hierarchical Navigable Small Worlds) vector indexing
//! O(log n) search for high-speed large-scale document search

use instant_distance::{Builder, HnswMap, Search};
use std::sync::RwLock;
use once_cell::sync::Lazy;
use log::{info, debug, warn};

/// Custom point type: 384-dimensional embedding with cached norm
#[derive(Clone, Debug)]
pub struct EmbeddingPoint {
    pub id: i64,
    pub embedding: Vec<f32>,
    /// Pre-computed L2 norm for efficient distance calculation
    pub norm: f32,
}

impl EmbeddingPoint {
    /// Create a new EmbeddingPoint with pre-computed norm
    pub fn new(id: i64, embedding: Vec<f32>) -> Self {
        let norm = embedding.iter().map(|x| x * x).sum::<f32>().sqrt();
        Self { id, embedding, norm }
    }
}

impl instant_distance::Point for EmbeddingPoint {
    fn distance(&self, other: &Self) -> f32 {
        // Cosine distance = 1 - cosine similarity
        // HNSW uses smaller distance = closer, so we use 1 - similarity
        let dot: f32 = self.embedding.iter()
            .zip(other.embedding.iter())
            .map(|(a, b)| a * b)
            .sum();
        
        // Use cached norms instead of recalculating
        if self.norm == 0.0 || other.norm == 0.0 {
            return 1.0; // Maximum distance
        }
        
        let similarity = dot / (self.norm * other.norm);
        1.0 - similarity // Cosine distance
    }
}

/// Global HNSW index (in-memory cache)
/// Using RwLock for improved read concurrency (searches can run in parallel)
static HNSW_INDEX: Lazy<RwLock<Option<HnswMap<EmbeddingPoint, i64>>>> = 
    Lazy::new(|| RwLock::new(None));

/// Build HNSW index
pub fn build_hnsw_index(points: Vec<(i64, Vec<f32>)>) -> anyhow::Result<()> {
    info!("[hnsw] Building index with {} points", points.len());
    
    if points.is_empty() {
        warn!("[hnsw] No points provided");
        return Ok(());
    }
    
    // Map EmbeddingPoint to value(id) - using constructor for norm caching
    let embedding_points: Vec<EmbeddingPoint> = points.iter()
        .map(|(id, emb)| EmbeddingPoint::new(*id, emb.clone()))
        .collect();
    
    let values: Vec<i64> = points.iter().map(|(id, _)| *id).collect();
    
    // Create HNSW index
    let hnsw_map = Builder::default().build(embedding_points, values);
    
    // Store in global index (write lock)
    let mut index_guard = HNSW_INDEX.write().unwrap();
    *index_guard = Some(hnsw_map);
    
    info!("[hnsw] Index build complete");
    Ok(())
}

/// HNSW search result
#[derive(Debug)]
pub struct HnswSearchResult {
    pub id: i64,
    pub distance: f32,
}

/// Search in HNSW index
pub fn search_hnsw(query_embedding: Vec<f32>, top_k: usize) -> anyhow::Result<Vec<HnswSearchResult>> {
    debug!("[hnsw] Starting search, top_k: {}", top_k);
    
    // Read lock allows concurrent searches
    let index_guard = HNSW_INDEX.read().unwrap();
    let hnsw_map = index_guard.as_ref()
        .ok_or_else(|| anyhow::anyhow!("HNSW index not initialized"))?;
    
    // Use constructor for pre-computed norm
    let query_point = EmbeddingPoint::new(-1, query_embedding);
    
    let mut search = Search::default();
    let neighbors = hnsw_map.search(&query_point, &mut search);
    
    let results: Vec<HnswSearchResult> = neighbors
        .take(top_k)
        .map(|item| HnswSearchResult {
            id: *item.value,
            distance: item.distance,
        })
        .collect();
    
    debug!("[hnsw] Returning {} results", results.len());
    Ok(results)
}

/// Check if HNSW index is loaded
pub fn is_hnsw_index_loaded() -> bool {
    let index_guard = HNSW_INDEX.read().unwrap();
    index_guard.is_some()
}

/// Clear HNSW index
pub fn clear_hnsw_index() {
    let mut index_guard = HNSW_INDEX.write().unwrap();
    *index_guard = None;
    info!("[hnsw] Index cleared");
}
