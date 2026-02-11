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

use crate::api::hnsw_index::{search_hnsw, is_hnsw_index_loaded, HnswSearchResult};
use crate::api::bm25_search::{bm25_search};
use crate::api::db_pool::{get_connection};
use crate::api::error::RagError;
use ndarray::Array1;
use rusqlite::params;

#[derive(Debug, Clone)]
pub struct SearchFilter {
    pub source_ids: Option<Vec<i64>>,
    pub metadata_like: Option<String>, // SQL LIKE pattern
}

#[derive(Debug, Clone)]
pub struct HybridSearchResult {
    pub doc_id: i64,
    pub content: String,
    pub score: f64,
    pub vector_rank: u32,
    pub bm25_rank: u32,
    pub source_id: i64,
    pub metadata: Option<String>,
    pub chunk_index: u32,
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
    query_text: String,
    query_embedding: Vec<f32>,
    top_k: u32,
    config: Option<RrfConfig>,
    filter: Option<SearchFilter>,
) -> Result<Vec<HybridSearchResult>, RagError> {
    let config = config.unwrap_or_default();
    info!("[hybrid] Starting hybrid search, top_k: {}", top_k);
    
    // Fetch more candidates if filtering is active to maintain recall
    let multiplier = if filter.is_some() { 4 } else { 2 };
    let candidate_k = (top_k * multiplier) as usize;
    
    // 1. Parallel Execution: Run Vector and BM25 search simultaneously
    let (mut vector_results, mut bm25_results) = std::thread::scope(|s| {
        let handle_vec = s.spawn(|| {
            if is_hnsw_index_loaded() {
                search_hnsw(query_embedding.clone(), candidate_k).unwrap_or_else(|e| {
                    log::error!("[hybrid] Vector search failed: {}", e);
                    vec![]
                })
            } else {
                debug!("[hybrid] HNSW index not loaded, skipping vector search");
                vec![]
            }
        });

        let handle_bm25 = s.spawn(|| {
            bm25_search(query_text.clone(), candidate_k as u32)
        });

        let vec_res = handle_vec.join().unwrap_or_else(|e| {
            log::error!("[hybrid] Vector search thread panicked: {:?}", e);
            vec![]
        });
        
        let bm25_res = handle_bm25.join().unwrap_or_else(|e| {
            log::error!("[hybrid] BM25 search thread panicked: {:?}", e);
            vec![]
        });

        (vec_res, bm25_res)
    });

    info!("[hybrid] Raw candidates - Vector: {}, BM25: {}", vector_results.len(), bm25_results.len());


    // 2. Filter-Aware Search Strategy
    // If filtering by source_id, performing a global HNSW search and then filtering is inefficient 
    // and prone to low recall (if source is small/obscure).
    // Instead, we perform an EXACT SCAN (Brute Force) on the target source's chunks.
    if let Some(f) = &filter {
        if let Some(sids) = &f.source_ids {
            if !sids.is_empty() {
                info!("[hybrid] Source filter active ({:?}), switching to EXACT SCAN", sids);
                
                let conn = get_connection().map_err(|e| RagError::DatabaseError(e.to_string()))?;
                let sids_str = sids.iter().map(|id| id.to_string()).collect::<Vec<_>>().join(",");
                
                // Fetch ALL chunks for these sources
                // This is O(N_source) which is typically small (<10k) and very fast.
                let query = format!(
                    "SELECT c.id, c.embedding FROM chunks c WHERE c.source_id IN ({})", 
                    sids_str
                );
                
                let mut stmt = conn.prepare(&query).map_err(|e| RagError::DatabaseError(e.to_string()))?;
                let chunk_iter = stmt.query_map([], |row| {
                    Ok((row.get::<_, i64>(0)?, row.get::<_, Vec<u8>>(1)?))
                }).map_err(|e| RagError::DatabaseError(e.to_string()))?;

                let query_vec = ndarray::Array1::from(query_embedding.clone());
                let query_norm = query_vec.mapv(|x| x * x).sum().sqrt();
                
                // Clear global results and replace with exact scan results
                vector_results.clear();
                bm25_results.clear(); // We temporarily disable BM25 in this mode or need to re-run it scoped via SQL
                // Note: For "Independent Source Search", vector-only is often sufficient/desired. 
                // Creating a simplified hybrid result from vector only.

                for row in chunk_iter {
                    if let Ok((id, embedding_blob)) = row {
                        let embedding: Vec<f32> = embedding_blob.chunks(4)
                            .map(|c| f32::from_ne_bytes(c.try_into().unwrap()))
                            .collect();
                        
                        if embedding.len() == query_embedding.len() {
                            let target_vec = ndarray::Array1::from(embedding);
                            let target_norm = target_vec.mapv(|x| x * x).sum().sqrt();
                            let dot = query_vec.dot(&target_vec);
                            let sim = if query_norm == 0.0 || target_norm == 0.0 { 0.0 } else { dot / (query_norm * target_norm) };
                            
                            // Emulate a HnswSearchResult derived purely from Vector
                            // In this mode, we treat vector score as the primary ranker.
                            vector_results.push(HnswSearchResult {
                                id: id,
                                distance: (1.0 - sim) as f32, // Convert similarity to distance (lower is better)
                            });
                        }
                    }
                }
                
                // Sort by distance ASCENDING (Best match first)
                vector_results.sort_by(|a, b| a.distance.partial_cmp(&b.distance).unwrap_or(std::cmp::Ordering::Equal));
                vector_results.truncate(top_k as usize);
                
                info!("[hybrid] Exact scan found {} candidates from sources {:?}", vector_results.len(), sids);
                
                // Skip the standard post-filtering since we only fetched valid candidates
                // Logic continues to RRF Ranking (Step 3). 
                // Since bm25_results is empty, RRF will just use vector ranks.
            }
        }
    } else {
        // Standard Global Search Post-Filtering (Old Logic)
        if let Some(f) = &filter {
             let mut all_doc_ids: Vec<i64> = vector_results.iter().map(|r| r.id)
                .chain(bm25_results.iter().map(|r| r.doc_id))
                .collect();
             all_doc_ids.sort();
             all_doc_ids.dedup();
    
             if !all_doc_ids.is_empty() {
                 let conn = get_connection().map_err(|e| RagError::DatabaseError(e.to_string()))?;
                 let id_list = all_doc_ids.iter().map(|id| id.to_string()).collect::<Vec<_>>().join(",");
                 
                 let mut sql_conditions = Vec::new();
                 sql_conditions.push(format!("c.id IN ({})", id_list));
    
                 if let Some(sids) = &f.source_ids {
                     if !sids.is_empty() {
                         let sids_str = sids.iter().map(|id| id.to_string()).collect::<Vec<_>>().join(",");
                         sql_conditions.push(format!("c.source_id IN ({})", sids_str));
                     }
                 }
    
                 if let Some(pattern) = &f.metadata_like {
                     sql_conditions.push(format!("s.metadata LIKE '{}'", pattern.replace("'", "''")));
                 }
                 
                 let query = format!(
                     "SELECT c.id FROM chunks c 
                      LEFT JOIN sources s ON c.source_id = s.id 
                      WHERE {}", 
                     sql_conditions.join(" AND ")
                 );
    
                 debug!("[hybrid] Filter query: {}", query);
                 
                 let mut stmt = conn.prepare(&query).map_err(|e| RagError::DatabaseError(e.to_string()))?;
                 let valid_ids: std::collections::HashSet<i64> = stmt.query_map([], |row| row.get(0))
                     .map_err(|e| RagError::DatabaseError(e.to_string()))?
                     .filter_map(|r| r.ok())
                     .collect();
                 
                 info!("[hybrid] Filter maintained {}/{} candidates", valid_ids.len(), all_doc_ids.len());
    
                 vector_results.retain(|r| valid_ids.contains(&r.id));
                 bm25_results.retain(|r| valid_ids.contains(&r.doc_id));
             }
        }
    }
    
    // 3. RRF Ranking
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
    
    let mut rrf_scores: Vec<(i64, f64, u32, u32)> = Vec::with_capacity(all_doc_ids.len());
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
    
    // 4. Batch Content Fetch
    if rrf_scores.is_empty() { return Ok(vec![]); }

    let target_ids: Vec<String> = rrf_scores.iter().map(|(id, _, _, _)| id.to_string()).collect();
    let id_list = target_ids.join(",");
    
    let conn = get_connection().map_err(|e| RagError::DatabaseError(e.to_string()))?;
    // Map: id -> (content, source_id, metadata, chunk_index)
    let mut content_map: HashMap<i64, (String, i64, Option<String>, u32)> = HashMap::new();
    
    // First try docs table (Simple RAG) - assume source_id=id, metadata=None, chunk_index=0
    // BUT if filter was active, we likely filtered these out.
    if filter.is_none() {
        let query_docs = format!("SELECT id, content FROM docs WHERE id IN ({})", id_list);
        if let Ok(mut stmt) = conn.prepare(&query_docs) {
            let found_docs = stmt.query_map([], |row| Ok((row.get::<_, i64>(0)?, row.get::<_, String>(1)?)));
            if let Ok(rows) = found_docs {
                for row in rows {
                     if let Ok((id, content)) = row {
                        content_map.insert(id, (content, id, None, 0));
                    }
                }
            }
        }
    }
    
    // If missing, try chunks table
    let missing_ids: Vec<String> = rrf_scores.iter()
        .filter(|(id, _, _, _)| !content_map.contains_key(id))
        .map(|(id, _, _, _)| id.to_string())
        .collect();

    if !missing_ids.is_empty() {
        let missing_list = missing_ids.join(",");
        let query_chunks = format!(
            "SELECT c.id, c.content, c.source_id, s.metadata, c.chunk_index 
             FROM chunks c 
             LEFT JOIN sources s ON c.source_id = s.id 
             WHERE c.id IN ({})", 
            missing_list
        );
        
        if let Ok(mut stmt) = conn.prepare(&query_chunks) {
            let found_chunks = stmt.query_map([], |row| {
                Ok((
                    row.get::<_, i64>(0)?, 
                    row.get::<_, String>(1)?,
                    row.get::<_, i64>(2)?,
                    row.get::<_, Option<String>>(3)?,
                    row.get::<_, u32>(4)?
                ))
            });
            
            if let Ok(results_iter) = found_chunks {
                for row in results_iter {
                    if let Ok((id, content, source_id, metadata, chunk_index)) = row {
                        content_map.insert(id, (content, source_id, metadata, chunk_index));
                    }
                }
            }
        }
    }

    let mut results: Vec<HybridSearchResult> = Vec::with_capacity(rrf_scores.len());
    
    for (doc_id, score, vec_rank, bm25_rank) in rrf_scores {
        if let Some((content, source_id, metadata, chunk_index)) = content_map.remove(&doc_id) { 
            results.push(HybridSearchResult { 
                doc_id, 
                content, 
                score, 
                vector_rank: vec_rank, 
                bm25_rank,
                source_id,
                metadata,
                chunk_index
            });
        }
    }
    
    info!("[hybrid] Returning {} results", results.len());
    Ok(results)
}

/// Simplified hybrid search returning content strings only.
pub fn search_hybrid_simple(query_text: String, query_embedding: Vec<f32>, top_k: u32) -> Result<Vec<String>, RagError> {
    Ok(search_hybrid(query_text, query_embedding, top_k, None, None)?.into_iter().map(|r| r.content).collect())
}

/// Search with custom weights (vector_weight + bm25_weight = 1.0 recommended).
pub fn search_hybrid_weighted(
    query_text: String,
    query_embedding: Vec<f32>,
    top_k: u32,
    vector_weight: f64,
    bm25_weight: f64,
) -> Result<Vec<HybridSearchResult>, RagError> {
    let config = RrfConfig { k: 60, vector_weight: vector_weight.clamp(0.0, 1.0), bm25_weight: bm25_weight.clamp(0.0, 1.0) };
    search_hybrid(query_text, query_embedding, top_k, Some(config), None)
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::api::db_pool::{init_db_pool, close_db_pool, get_connection};
    use crate::api::simple_rag::init_db;
    use crate::api::hnsw_index::{build_hnsw_index, clear_hnsw_index};
    use crate::api::bm25_search::{bm25_add_document, bm25_clear_index};
    use rusqlite::params;

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

    #[test]
    fn test_hybrid_search_integration() {
        // 1. Setup
        let db_path = std::env::temp_dir().join("test_hybrid_search.db");
        let _ = std::fs::remove_file(&db_path); // Ensure clean state
        
        // Initialize pool with 1 connection to avoid locking issues in test
        init_db_pool(db_path.to_str().unwrap().to_string(), 1).unwrap();
        
        // Create tables
        init_db().unwrap();
        
        // Clear indices
        clear_hnsw_index();
        bm25_clear_index();

        // 2. Add Data
        {
            let conn = get_connection().unwrap();
            // We use dummy embedding blobs for DB, as search_hybrid uses HNSW index for vectors
            let dummy_blob = vec![0u8; 4]; 
            
            conn.execute("INSERT INTO docs (id, content, content_hash, embedding) VALUES (1, 'Apple iPhone is great', 'h1', ?1)", params![dummy_blob]).unwrap();
            conn.execute("INSERT INTO docs (id, content, content_hash, embedding) VALUES (2, 'Banana is a yellow fruit', 'h2', ?1)", params![dummy_blob]).unwrap();
            conn.execute("INSERT INTO docs (id, content, content_hash, embedding) VALUES (3, 'Apple pie recipe', 'h3', ?1)", params![dummy_blob]).unwrap();
        }

        // 3. Populate Indices
        // Vector: 
        // Doc 1 (Apple): [1, 0]
        // Doc 2 (Banana): [0, 1]
        // Doc 3 (Apple): [0.9, 0.1]
        let points = vec![
            (1, vec![1.0, 0.0]),
            (2, vec![0.0, 1.0]),
            (3, vec![0.9, 0.1]),
        ];
        build_hnsw_index(points).unwrap();

        // BM25
        bm25_add_document(1, "Apple iPhone is great".to_string());
        bm25_add_document(2, "Banana is a yellow fruit".to_string());
        bm25_add_document(3, "Apple pie recipe".to_string());

        // 4. Test Search
        // Query: "Apple" with vector [1, 0]
        // Should return Doc 1 and Doc 3 as top results
        let results = search_hybrid(
            "Apple".to_string(), 
            vec![1.0, 0.0], 
            2, 
            None,
            None // No filter
        ).unwrap();

        // Verify results
        assert_eq!(results.len(), 2);
        // Doc 1 should match both Vector and BM25 'Apple'
        assert!(results.iter().any(|r| r.doc_id == 1)); 
        assert!(results.iter().any(|r| r.doc_id == 3));

        // 5. Cleanup
        close_db_pool();
        let _ = std::fs::remove_file(db_path);
    }
}
