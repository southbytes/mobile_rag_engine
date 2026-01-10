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
//! Hybrid Search: Vector + Keyword with Reciprocal Rank Fusion.

use std::collections::HashMap;
use log::{info, debug};
use rusqlite::{params, Connection};
use crate::api::hnsw_index::{search_hnsw, is_hnsw_index_loaded};
use crate::api::bm25_search::{bm25_search, Bm25SearchResult};

#[derive(Debug, Clone)]
pub struct HybridSearchResult {
    pub doc_id: i64,
    pub content: String,
    pub score: f64,
    pub vector_rank: u32,
    pub bm25_rank: u32,
}

#[derive(Debug, Clone)]
pub struct RrfConfig {
    pub k: u32,
    pub vector_weight: f64,
    pub bm25_weight: f64,
}

impl Default for RrfConfig {
    fn default() -> Self { Self { k: 60, vector_weight: 0.5, bm25_weight: 0.5 } }
}

fn rrf_score(rank: usize, k: u32) -> f64 { 1.0 / (k as f64 + rank as f64) }

/// Perform hybrid search combining vector and keyword search.
pub fn search_hybrid(
    db_path: String,
    query_text: String,
    query_embedding: Vec<f32>,
    top_k: u32,
    config: Option<RrfConfig>,
) -> anyhow::Result<Vec<HybridSearchResult>> {
    let config = config.unwrap_or_default();
    info!("[hybrid] Starting hybrid search, top_k: {}", top_k);
    
    let candidate_k = (top_k * 2) as usize;
    
    let vector_results = if is_hnsw_index_loaded() {
        search_hnsw(query_embedding, candidate_k)?
    } else {
        debug!("[hybrid] HNSW index not loaded, skipping vector search");
        vec![]
    };
    
    let bm25_results: Vec<Bm25SearchResult> = bm25_search(query_text.clone(), candidate_k as u32);
    info!("[hybrid] Vector results: {}, BM25 results: {}", vector_results.len(), bm25_results.len());
    
    let mut vector_ranks: HashMap<i64, usize> = HashMap::new();
    for (rank, result) in vector_results.iter().enumerate() {
        vector_ranks.insert(result.id, rank + 1);
    }
    
    let mut bm25_ranks: HashMap<i64, usize> = HashMap::new();
    for (rank, result) in bm25_results.iter().enumerate() {
        bm25_ranks.insert(result.doc_id, rank + 1);
    }
    
    let mut all_doc_ids: Vec<i64> = vector_ranks.keys().chain(bm25_ranks.keys()).copied().collect();
    all_doc_ids.sort();
    all_doc_ids.dedup();
    
    if all_doc_ids.is_empty() { return Ok(vec![]); }
    
    let mut rrf_scores: Vec<(i64, f64, u32, u32)> = Vec::new();
    for doc_id in &all_doc_ids {
        let vec_rank = vector_ranks.get(doc_id).copied();
        let bm25_rank = bm25_ranks.get(doc_id).copied();
        
        let mut combined_score = 0.0;
        if let Some(rank) = vec_rank { combined_score += config.vector_weight * rrf_score(rank, config.k); }
        if let Some(rank) = bm25_rank { combined_score += config.bm25_weight * rrf_score(rank, config.k); }
        
        rrf_scores.push((*doc_id, combined_score, vec_rank.unwrap_or(0) as u32, bm25_rank.unwrap_or(0) as u32));
    }
    
    rrf_scores.sort_by(|a, b| b.1.partial_cmp(&a.1).unwrap_or(std::cmp::Ordering::Equal));
    rrf_scores.truncate(top_k as usize);
    
    let conn = Connection::open(&db_path)?;
    let mut results: Vec<HybridSearchResult> = Vec::new();
    
    for (doc_id, score, vec_rank, bm25_rank) in rrf_scores {
        let content: Option<String> = conn.query_row("SELECT content FROM docs WHERE id = ?1", params![doc_id], |row| row.get(0))
            .ok()
            .or_else(|| conn.query_row("SELECT content FROM chunks WHERE id = ?1", params![doc_id], |row| row.get(0)).ok());
        
        if let Some(content) = content {
            results.push(HybridSearchResult { doc_id, content, score, vector_rank: vec_rank, bm25_rank });
        }
    }
    
    info!("[hybrid] Returning {} results", results.len());
    Ok(results)
}

/// Simplified hybrid search returning content strings only.
pub fn search_hybrid_simple(db_path: String, query_text: String, query_embedding: Vec<f32>, top_k: u32) -> anyhow::Result<Vec<String>> {
    Ok(search_hybrid(db_path, query_text, query_embedding, top_k, None)?.into_iter().map(|r| r.content).collect())
}

/// Search with custom weights (vector_weight + bm25_weight = 1.0 recommended).
pub fn search_hybrid_weighted(
    db_path: String,
    query_text: String,
    query_embedding: Vec<f32>,
    top_k: u32,
    vector_weight: f64,
    bm25_weight: f64,
) -> anyhow::Result<Vec<HybridSearchResult>> {
    let config = RrfConfig { k: 60, vector_weight: vector_weight.clamp(0.0, 1.0), bm25_weight: bm25_weight.clamp(0.0, 1.0) };
    search_hybrid(db_path, query_text, query_embedding, top_k, Some(config))
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_rrf_score() {
        let score = rrf_score(1, 60);
        assert!((score - 0.0164).abs() < 0.001);
    }

    #[test]
    fn test_rrf_config_default() {
        let config = RrfConfig::default();
        assert_eq!(config.k, 60);
    }
}
