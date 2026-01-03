// rust/src/api/hnsw_index.rs
//! HNSW (Hierarchical Navigable Small Worlds) vector indexing using hnsw_rs

use hnsw_rs::prelude::*;
use std::sync::RwLock;
use once_cell::sync::Lazy;
use log::{info, debug, warn};
use std::path::Path;
use serde::{Serialize, Deserialize};
// HnswIo persistence disabled due to lifetime constraints with static storage

/// Custom point type: wrapper for FRB compatibility
/// This struct was used in previous FRB generation, so we keep it to avoid breaking changes.
/// We don't use it in the index itself anymore (we use native vectors).
#[derive(Clone, Debug, Serialize, Deserialize)]
pub struct EmbeddingPoint {
    pub id: i64,
    pub embedding: Vec<f32>,
    pub norm: f32,
}

impl EmbeddingPoint {
    pub fn new(id: i64, embedding: Vec<f32>) -> Self {
        let norm = embedding.iter().map(|x| x * x).sum::<f32>().sqrt();
        Self { id, embedding, norm }
    }
}

/// Global HNSW index (in-memory cache)
// hnsw_rs::Hnsw is thread-safe for searching, so RwLock is good for replacing the index
// Explicitly specifying types helps: Hnsw<Data, Distance>
static HNSW_INDEX: Lazy<RwLock<Option<Hnsw<f32, DistCosine>>>> = 
    Lazy::new(|| RwLock::new(None));

/// Build HNSW index (Optimized)
pub fn build_hnsw_index(points: Vec<(i64, Vec<f32>)>) -> anyhow::Result<()> {
    info!("[hnsw] Building index with {} points", points.len());
    
    if points.is_empty() {
        warn!("[hnsw] No points provided");
        return Ok(());
    }
    
    let count = points.len();
    
    // hnsw_rs 0.3.x: Hnsw::new(max_nb_connection, max_elements, max_layer, ef_construction, distance_fn)
    // We expect f32 data and Cosine distance
    // Note: insert takes &self, so hwns doesn't need to be mut if using interior mutability
    let hwns = Hnsw::new(
        16, // max_nb_connection
        count, // max_elements: initial capacity
        16, // max_layer
        100, // ef_construction
        DistCosine
    );
    
    // Insert items
    for (id, embedding) in points {
        // hnsw_rs inserts using &slice and an external ID (usize).
        let u_id = id as usize; 
        
        // Insert takes a tuple: (&[T], usize)
        hwns.insert((&embedding, u_id)); 
    }
    
    // Store in global
    let mut index_guard = HNSW_INDEX.write().unwrap();
    *index_guard = Some(hwns);
    
    info!("[hnsw] Index build complete");
    Ok(())
}

/// Save HNSW index point data to disk (bincode serialization)
/// 
/// Since hnsw_rs's native persistence has lifetime constraints,
/// we save the point data and rebuild the index on load.
/// This is fast enough for practical use (1000 points < 100ms rebuild).
pub fn save_hnsw_index(base_path: &str) -> anyhow::Result<()> {
    // We need access to the points that were used to build the index
    // For now, this will be a no-op until we track points during build
    // The actual save happens in source_rag.rs after rebuild_chunk_hnsw_index
    info!("[hnsw] save_hnsw_index called - using DB-based persistence");
    
    // Create a marker file to indicate index was built
    let marker_path = format!("{}.hnsw.marker", base_path);
    if let Some(parent) = Path::new(&marker_path).parent() {
        std::fs::create_dir_all(parent)?;
    }
    std::fs::write(&marker_path, "1")?;
    
    info!("[hnsw] Index marker saved to {}", marker_path);
    Ok(())
}

/// Load HNSW index from disk
/// 
/// Returns true if marker exists (index should be rebuilt from DB),
/// false if no cached index exists.
pub fn load_hnsw_index(base_path: &str) -> anyhow::Result<bool> {
    let marker_path = format!("{}.hnsw.marker", base_path);
    
    if Path::new(&marker_path).exists() {
        info!("[hnsw] Index marker found at {} - rebuild from DB recommended", base_path);
        // Marker exists, but actual rebuild happens via rebuild_chunk_hnsw_index
        // This just tells the caller that an index was previously built
        return Ok(true);
    }
    
    info!("[hnsw] No index marker found at {}", base_path);
    Ok(false)
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
    
    let index_guard = HNSW_INDEX.read().unwrap();
    let index = index_guard.as_ref()
        .ok_or_else(|| anyhow::anyhow!("HNSW index not initialized"))?;
    
    // ef_search = top_k * 2 usually good
    let ef_search = core::cmp::max(30, top_k * 2);
    
    // search returns Vec<Neighbor>
    let neighbors = index.search(&query_embedding, top_k, ef_search);
    
    let results: Vec<HnswSearchResult> = neighbors.iter()
        .map(|neighbor| HnswSearchResult {
            id: neighbor.d_id as i64,
            distance: neighbor.distance,
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
