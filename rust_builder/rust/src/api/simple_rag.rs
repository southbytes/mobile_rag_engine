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
//! Simple RAG API for basic document storage and similarity search.

use flutter_rust_bridge::frb;
use rusqlite::{params, Connection};
use ndarray::Array1;
use log::{info, warn, error, debug};
use sha2::{Sha256, Digest};
use crate::api::hnsw_index::{build_hnsw_index, search_hnsw, is_hnsw_index_loaded, clear_hnsw_index};
use crate::api::bm25_search::{bm25_add_document, bm25_add_documents, bm25_clear_index};
use crate::api::incremental_index::{incremental_add, clear_buffer};

fn truncate_str(s: &str, max_chars: usize) -> &str {
    match s.char_indices().nth(max_chars) {
        Some((idx, _)) => &s[..idx],
        None => s,
    }
}

fn calculate_content_hash(content: &str) -> String {
    let mut hasher = Sha256::new();
    hasher.update(content.as_bytes());
    format!("{:x}", hasher.finalize())
}

/// Calculate cosine similarity between two vectors.
#[frb(sync)]
pub fn calculate_cosine_similarity(vec_a: Vec<f32>, vec_b: Vec<f32>) -> f64 {
    if vec_a.len() != vec_b.len() || vec_a.is_empty() {
        warn!("[cosine] Vector length mismatch: a={}, b={}", vec_a.len(), vec_b.len());
        return 0.0;
    }

    let a = Array1::from(vec_a);
    let b = Array1::from(vec_b);

    let dot_product = a.dot(&b);
    let norm_a = a.mapv(|x| x * x).sum().sqrt();
    let norm_b = b.mapv(|x| x * x).sum().sqrt();

    if norm_a == 0.0 || norm_b == 0.0 { return 0.0; }
    (dot_product / (norm_a * norm_b)) as f64
}

/// Initialize database with docs table.
pub fn init_db(db_path: String) -> anyhow::Result<()> {
    info!("[init_db] DB path: {}", db_path);
    let conn = Connection::open(&db_path)?;
    
    conn.execute(
        "CREATE TABLE IF NOT EXISTS docs (
            id INTEGER PRIMARY KEY,
            content TEXT NOT NULL,
            content_hash TEXT UNIQUE,
            embedding BLOB NOT NULL
        )",
        [],
    )?;
    
    let has_hash_column: bool = conn.prepare("SELECT content_hash FROM docs LIMIT 1").is_ok();
    
    if !has_hash_column {
        info!("[init_db] Migrating: adding content_hash column");
        conn.execute("ALTER TABLE docs ADD COLUMN content_hash TEXT", [])?;
        
        let mut stmt = conn.prepare("SELECT id, content FROM docs WHERE content_hash IS NULL")?;
        let rows: Vec<(i64, String)> = stmt.query_map([], |row| Ok((row.get(0)?, row.get(1)?)))?.filter_map(|r| r.ok()).collect();
        
        for (id, content) in rows {
            let hash = calculate_content_hash(&content);
            conn.execute("UPDATE docs SET content_hash = ?1 WHERE id = ?2", params![hash, id])?;
        }
        
        conn.execute("CREATE UNIQUE INDEX IF NOT EXISTS idx_content_hash ON docs(content_hash)", [])?;
    }
    
    rebuild_hnsw_index_internal(&conn)?;
    rebuild_bm25_index_internal(&conn)?;
    
    info!("[init_db] Initialization complete");
    Ok(())
}

fn rebuild_hnsw_index_internal(conn: &Connection) -> anyhow::Result<()> {
    let mut stmt = conn.prepare("SELECT id, embedding FROM docs")?;
    let points: Vec<(i64, Vec<f32>)> = stmt.query_map([], |row| {
        let id: i64 = row.get(0)?;
        let embedding_blob: Vec<u8> = row.get(1)?;
        let embedding: Vec<f32> = embedding_blob.chunks(4).map(|chunk| f32::from_ne_bytes(chunk.try_into().unwrap())).collect();
        Ok((id, embedding))
    })?.filter_map(|r| r.ok()).collect();
    
    if !points.is_empty() { build_hnsw_index(points)?; }
    Ok(())
}

/// Rebuild HNSW index.
pub fn rebuild_hnsw_index(db_path: String) -> anyhow::Result<()> {
    info!("[rebuild_hnsw] Starting index rebuild");
    let conn = Connection::open(&db_path)?;
    rebuild_hnsw_index_internal(&conn)?;
    info!("[rebuild_hnsw] Index rebuild complete");
    Ok(())
}

fn rebuild_bm25_index_internal(conn: &Connection) -> anyhow::Result<()> {
    let mut stmt = conn.prepare("SELECT id, content FROM docs")?;
    let docs: Vec<(i64, String)> = stmt.query_map([], |row| Ok((row.get(0)?, row.get(1)?)))?.filter_map(|r| r.ok()).collect();
    if !docs.is_empty() {
        info!("[bm25] Building index from {} documents", docs.len());
        bm25_add_documents(docs);
    }
    Ok(())
}

/// Rebuild BM25 index.
pub fn rebuild_bm25_index(db_path: String) -> anyhow::Result<()> {
    info!("[rebuild_bm25] Starting index rebuild");
    let conn = Connection::open(&db_path)?;
    bm25_clear_index();
    rebuild_bm25_index_internal(&conn)?;
    info!("[rebuild_bm25] Index rebuild complete");
    Ok(())
}

#[derive(Debug, Clone)]
pub struct AddDocumentResult {
    pub success: bool,
    pub is_duplicate: bool,
    pub message: String,
}

/// Add document with embedding vector (with deduplication).
pub fn add_document(db_path: String, content: String, embedding: Vec<f32>) -> anyhow::Result<AddDocumentResult> {
    info!("[add_document] Saving document");
    debug!("[add_document] content length: {} chars, embedding dims: {}", content.chars().count(), embedding.len());
    
    if embedding.is_empty() {
        error!("[add_document] Embedding is empty!");
        return Ok(AddDocumentResult { success: false, is_duplicate: false, message: "Embedding vector is empty".to_string() });
    }
    
    let content_hash = calculate_content_hash(&content);
    let conn = Connection::open(&db_path)?;
    
    let existing: Option<i64> = conn.query_row("SELECT id FROM docs WHERE content_hash = ?1", params![content_hash], |row| row.get(0)).ok();
    
    if let Some(id) = existing {
        info!("[add_document] Duplicate found (id={})", id);
        return Ok(AddDocumentResult { success: true, is_duplicate: true, message: format!("Document already exists (id={})", id) });
    }

    let mut embedding_bytes: Vec<u8> = Vec::with_capacity(embedding.len() * 4);
    for f in &embedding { embedding_bytes.extend_from_slice(&f.to_ne_bytes()); }

    conn.execute("INSERT INTO docs (content, content_hash, embedding) VALUES (?1, ?2, ?3)", params![content, content_hash, embedding_bytes])?;
    
    let doc_id = conn.last_insert_rowid();
    bm25_add_document(doc_id, content.clone());
    incremental_add(doc_id, embedding);
    
    info!("[add_document] Document saved (id={})", doc_id);
    Ok(AddDocumentResult { success: true, is_duplicate: false, message: "Document saved successfully".to_string() })
}

/// Legacy add_document for backward compatibility.
pub fn add_document_simple(db_path: String, content: String, embedding: Vec<f32>) -> anyhow::Result<()> {
    let result = add_document(db_path, content, embedding)?;
    if result.success { Ok(()) } else { Err(anyhow::anyhow!(result.message)) }
}

/// Similarity-based search (uses HNSW).
pub fn search_similar(db_path: String, query_embedding: Vec<f32>, top_k: u32) -> anyhow::Result<Vec<String>> {
    info!("[search] Starting search, query dims: {}, top_k: {}", query_embedding.len(), top_k);
    
    if query_embedding.is_empty() { return Err(anyhow::anyhow!("Query embedding is empty")); }
    
    if is_hnsw_index_loaded() {
        info!("[search] Using HNSW index");
        return search_with_hnsw(&db_path, query_embedding, top_k);
    }
    
    info!("[search] No HNSW index, attempting to build...");
    let conn = Connection::open(&db_path)?;
    
    if let Ok(()) = rebuild_hnsw_index_internal(&conn) {
        if is_hnsw_index_loaded() { return search_with_hnsw(&db_path, query_embedding, top_k); }
    }
    
    info!("[search] Using Linear Scan (no HNSW index)");
    search_with_linear_scan(&db_path, query_embedding, top_k)
}

fn search_with_hnsw(db_path: &str, query_embedding: Vec<f32>, top_k: u32) -> anyhow::Result<Vec<String>> {
    let hnsw_results = search_hnsw(query_embedding, top_k as usize)?;
    if hnsw_results.is_empty() { return Ok(Vec::new()); }
    
    let conn = Connection::open(db_path)?;
    let mut results: Vec<String> = Vec::new();
    
    for result in hnsw_results {
        if let Ok(content) = conn.query_row("SELECT content FROM docs WHERE id = ?1", params![result.id], |row| row.get::<_, String>(0)) {
            let similarity = 1.0 - result.distance;
            info!("[search] HNSW result: similarity={:.4}, content='{}...'", similarity, truncate_str(&content, 15));
            results.push(content);
        }
    }
    
    info!("[search] HNSW search complete, {} results", results.len());
    Ok(results)
}

fn search_with_linear_scan(db_path: &str, query_embedding: Vec<f32>, top_k: u32) -> anyhow::Result<Vec<String>> {
    let conn = Connection::open(db_path)?;
    let mut stmt = conn.prepare("SELECT content, embedding FROM docs")?;
    
    let query_vec = Array1::from(query_embedding.clone());
    let query_norm = query_vec.mapv(|x| x * x).sum().sqrt();
    let mut candidates: Vec<(f64, String)> = Vec::new();

    let rows = stmt.query_map([], |row| {
        let content: String = row.get(0)?;
        let embedding_blob: Vec<u8> = row.get(1)?;
        Ok((content, embedding_blob))
    })?;

    for row in rows {
        let (content, embedding_blob) = row?;
        if embedding_blob.len() % 4 != 0 { continue; }
        
        let embedding_vec: Vec<f32> = embedding_blob.chunks(4).map(|chunk| f32::from_ne_bytes(chunk.try_into().unwrap())).collect();
        if embedding_vec.len() != query_embedding.len() { continue; }
            
        let target_vec = Array1::from(embedding_vec);
        let target_norm = target_vec.mapv(|x| x * x).sum().sqrt();
        let dot_product = query_vec.dot(&target_vec);
        let similarity = if query_norm == 0.0 || target_norm == 0.0 { 0.0 } else { dot_product / (query_norm * target_norm) };

        candidates.push((similarity as f64, content));
    }

    candidates.sort_by(|a, b| b.0.partial_cmp(&a.0).unwrap());
    let result: Vec<String> = candidates.into_iter().take(top_k as usize).map(|(_, content)| content).collect();
    
    info!("[search] Linear search complete, {} results", result.len());
    Ok(result)
}

/// Get document count.
pub fn get_document_count(db_path: String) -> anyhow::Result<i64> {
    let conn = Connection::open(&db_path)?;
    Ok(conn.query_row("SELECT COUNT(*) FROM docs", [], |row| row.get(0))?)
}

/// Clear all documents.
pub fn clear_all_documents(db_path: String) -> anyhow::Result<()> {
    let conn = Connection::open(&db_path)?;
    conn.execute("DELETE FROM docs", [])?;
    clear_hnsw_index();
    bm25_clear_index();
    clear_buffer();
    info!("[clear] All documents and indexes deleted");
    Ok(())
}