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
pub fn build_hnsw_index(points: Vec<(i64, Vec<f32>)>) -> anyhow::Result<()> {
    info!("[hnsw] Building index with {} points", points.len());
    
    if points.is_empty() {
        warn!("[hnsw] No points provided");
        return Ok(());
    }
    
    let count = points.len();
    let hwns = Hnsw::new(16, count, 16, 100, DistCosine);
    
    for (id, embedding) in points {
        hwns.insert((&embedding, id as usize));
    }
    
    let mut index_guard = HNSW_INDEX.write().unwrap();
    *index_guard = Some(hwns);
    
    info!("[hnsw] Index build complete");
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
pub fn search_hnsw(query_embedding: Vec<f32>, top_k: usize) -> anyhow::Result<Vec<HnswSearchResult>> {
    debug!("[hnsw] Starting search, top_k: {}", top_k);
    
    let index_guard = HNSW_INDEX.read().unwrap();
    let index = index_guard.as_ref()
        .ok_or_else(|| anyhow::anyhow!("HNSW index not initialized"))?;
    
    let ef_search = core::cmp::max(30, top_k * 2);
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
