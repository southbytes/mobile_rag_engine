// Copyright 2025 mobile_rag_engine contributors
// SPDX-License-Identifier: MIT
//
// Licensed under the MIT License. You may obtain a copy of the License at
// https://opensource.org/licenses/MIT
//
// This software is provided "AS IS", without warranty of any kind, express or
// implied, including but not limited to the warranties of merchantability,
// fitness for a particular purpose, and noninfringement. In no event shall the
// authors or copyright holders be liable for any claim, damages, or other
// liability arising from the use of this software.
//
// CONTRIBUTOR GUIDELINES:
// This file is part of the core engine. Any modifications require owner approval.
// Please submit a PR with detailed explanation of changes before modifying.
//
//! HNSW (Hierarchical Navigable Small Worlds) vector indexing module.

use hnsw_rs::prelude::*;
use std::sync::RwLock;
use once_cell::sync::Lazy;
use log::{info, debug, warn};
use std::path::Path;
use serde::{Serialize, Deserialize};

/// Embedding point wrapper for FRB compatibility (legacy support).
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

/// Global HNSW index (thread-safe in-memory cache).
static HNSW_INDEX: Lazy<RwLock<Option<Hnsw<f32, DistCosine>>>> = 
    Lazy::new(|| RwLock::new(None));

/// Build HNSW index from embedding points.
/// 
/// Parameters are tuned for optimal recall vs speed tradeoff:
/// - M (max connections per node): 16-24 based on dataset size
/// - M0 (layer 0 connections): 2*M for better recall
/// - efConstruction: 100-200 based on dataset size
pub fn build_hnsw_index(points: Vec<(i64, Vec<f32>)>) -> anyhow::Result<()> {
    info!("[hnsw] Building index with {} points", points.len());
    
    if points.is_empty() {
        warn!("[hnsw] No points provided");
        return Ok(());
    }
    
    let count = points.len();
    
    // Adaptive parameters based on dataset size
    // - Small datasets (<1000): faster build, adequate recall
    // - Large datasets (>10000): higher quality, better recall
    let (m, m0, ef_construction, size_category) = if count > 10_000 {
        (24, 48, 200, "large (>10K)")
    } else if count > 1_000 {
        (20, 40, 150, "medium (1K-10K)")
    } else {
        (16, 32, 100, "small (<1K)")
    };
    
    // Debug output for Flutter console (only in debug builds)
    #[cfg(debug_assertions)]
    {
        println!("[HNSW] Dataset size: {} points ({})", count, size_category);
        println!("[HNSW] Parameters: M={}, M0={}, efConstruction={}", m, m0, ef_construction);
        println!("[HNSW] Expected recall: ~{}%", if count > 10_000 { "97" } else if count > 1_000 { "95" } else { "92" });
    }
    
    debug!("[hnsw] Using M={}, M0={}, efConstruction={}", m, m0, ef_construction);
    
    let hnsw = Hnsw::new(m, count, m0, ef_construction, DistCosine);
    
    for (id, embedding) in points {
        hnsw.insert((&embedding, id as usize));
    }
    
    let mut index_guard = HNSW_INDEX.write().unwrap();
    *index_guard = Some(hnsw);
    
    #[cfg(debug_assertions)]
    println!("[HNSW] ✅ Index build complete");
    
    info!("[hnsw] Index build complete (M={}, M0={}, efC={})", m, m0, ef_construction);
    Ok(())
}

/// Save HNSW index marker to disk (uses DB-based persistence).
pub fn save_hnsw_index(base_path: &str) -> anyhow::Result<()> {
    info!("[hnsw] save_hnsw_index called - using DB-based persistence");
    
    let marker_path = format!("{}.hnsw.marker", base_path);
    if let Some(parent) = Path::new(&marker_path).parent() {
        std::fs::create_dir_all(parent)?;
    }
    std::fs::write(&marker_path, "1")?;
    
    info!("[hnsw] Index marker saved to {}", marker_path);
    Ok(())
}

/// Load HNSW index marker. Returns true if marker exists.
pub fn load_hnsw_index(base_path: &str) -> anyhow::Result<bool> {
    let marker_path = format!("{}.hnsw.marker", base_path);
    
    if Path::new(&marker_path).exists() {
        info!("[hnsw] Index marker found at {} - rebuild from DB recommended", base_path);
        return Ok(true);
    }
    
    info!("[hnsw] No index marker found at {}", base_path);
    Ok(false)
}

/// HNSW search result containing doc ID and distance.
#[derive(Debug)]
pub struct HnswSearchResult {
    pub id: i64,
    pub distance: f32,
}

/// Search in HNSW index.
/// 
/// ef_search parameter controls accuracy vs speed:
/// - Higher ef_search = better recall but slower
/// - Lower ef_search = faster but may miss relevant results
/// 
/// Current tuning targets ~95% recall for most use cases.
pub fn search_hnsw(query_embedding: Vec<f32>, top_k: usize) -> anyhow::Result<Vec<HnswSearchResult>> {
    debug!("[hnsw] Starting search, top_k: {}", top_k);
    
    let index_guard = HNSW_INDEX.read().unwrap();
    let index = index_guard.as_ref()
        .ok_or_else(|| anyhow::anyhow!("HNSW index not initialized"))?;
    
    // ef_search should be >= top_k, higher values improve recall
    // Rule of thumb: ef_search = max(100, top_k * 5) for ~95% recall
    let ef_search = core::cmp::max(100, top_k * 5);
    
    #[cfg(debug_assertions)]
    println!("[HNSW] Search: top_k={}, ef_search={} (recall target: ~95%)", top_k, ef_search);
    
    debug!("[hnsw] Using ef_search={}", ef_search);
    
    let neighbors = index.search(&query_embedding, top_k, ef_search);
    
    let results: Vec<HnswSearchResult> = neighbors.iter()
        .map(|neighbor| HnswSearchResult {
            id: neighbor.d_id as i64,
            distance: neighbor.distance,
        })
        .collect();
    
    #[cfg(debug_assertions)]
    println!("[HNSW] Found {} results", results.len());
    
    debug!("[hnsw] Returning {} results", results.len());
    Ok(results)
}

/// Check if HNSW index is loaded.
pub fn is_hnsw_index_loaded() -> bool {
    let index_guard = HNSW_INDEX.read().unwrap();
    index_guard.is_some()
}

/// Clear HNSW index from memory.
pub fn clear_hnsw_index() {
    let mut index_guard = HNSW_INDEX.write().unwrap();
    *index_guard = None;
    info!("[hnsw] Index cleared");
}
