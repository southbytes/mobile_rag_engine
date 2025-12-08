// rust/src/api/source_rag.rs
//
// Extended RAG API with sources and chunks for LLM-optimized context.
// Builds on simple_rag.rs but adds hierarchical document structure.

use flutter_rust_bridge::frb;
use rusqlite::{params, Connection};
use ndarray::Array1;
use log::{info, debug, error};
use sha2::{Sha256, Digest};
use crate::api::hnsw_index::{build_hnsw_index, search_hnsw, is_hnsw_index_loaded};

/// Calculate SHA256 hash
fn hash_content(content: &str) -> String {
    let mut hasher = Sha256::new();
    hasher.update(content.as_bytes());
    format!("{:x}", hasher.finalize())
}

/// Initialize extended database with sources and chunks tables.
pub fn init_source_db(db_path: String) -> anyhow::Result<()> {
    info!("[init_source_db] Initializing: {}", db_path);
    let conn = Connection::open(&db_path)?;
    
    // Sources table: original documents
    conn.execute(
        "CREATE TABLE IF NOT EXISTS sources (
            id INTEGER PRIMARY KEY,
            content TEXT NOT NULL,
            content_hash TEXT UNIQUE,
            metadata TEXT,
            created_at INTEGER DEFAULT (strftime('%s', 'now'))
        )",
        [],
    )?;
    
    // Chunks table: split pieces with embeddings
    conn.execute(
        "CREATE TABLE IF NOT EXISTS chunks (
            id INTEGER PRIMARY KEY,
            source_id INTEGER NOT NULL,
            chunk_index INTEGER NOT NULL,
            content TEXT NOT NULL,
            start_pos INTEGER NOT NULL,
            end_pos INTEGER NOT NULL,
            embedding BLOB NOT NULL,
            FOREIGN KEY (source_id) REFERENCES sources(id) ON DELETE CASCADE
        )",
        [],
    )?;
    
    // Index for fast source lookup
    conn.execute(
        "CREATE INDEX IF NOT EXISTS idx_chunks_source_id ON chunks(source_id)",
        [],
    )?;
    
    info!("[init_source_db] Tables created");
    Ok(())
}

/// Result of adding a source document.
#[derive(Debug, Clone)]
pub struct AddSourceResult {
    pub source_id: i64,
    pub is_duplicate: bool,
    pub chunk_count: i32,
    pub message: String,
}

/// Add a source document (without chunks - chunks added separately).
pub fn add_source(
    db_path: String,
    content: String,
    metadata: Option<String>,
) -> anyhow::Result<AddSourceResult> {
    info!("[add_source] Adding source, {} chars", content.len());
    
    let content_hash = hash_content(&content);
    let conn = Connection::open(&db_path)?;
    
    // Check for duplicate
    let existing: Option<i64> = conn
        .query_row(
            "SELECT id FROM sources WHERE content_hash = ?1",
            params![content_hash],
            |row| row.get(0),
        )
        .ok();
    
    if let Some(id) = existing {
        info!("[add_source] Duplicate found: {}", id);
        return Ok(AddSourceResult {
            source_id: id,
            is_duplicate: true,
            chunk_count: 0,
            message: format!("Source already exists (id={})", id),
        });
    }
    
    // Insert new source
    conn.execute(
        "INSERT INTO sources (content, content_hash, metadata) VALUES (?1, ?2, ?3)",
        params![content, content_hash, metadata],
    )?;
    
    let source_id = conn.last_insert_rowid();
    info!("[add_source] Created source: {}", source_id);
    
    Ok(AddSourceResult {
        source_id,
        is_duplicate: false,
        chunk_count: 0,
        message: "Source created".to_string(),
    })
}

/// Chunk data for batch insertion.
#[derive(Debug, Clone)]
pub struct ChunkData {
    pub content: String,
    pub chunk_index: i32,
    pub start_pos: i32,
    pub end_pos: i32,
    pub embedding: Vec<f32>,
}

/// Add chunks for a source document.
pub fn add_chunks(
    db_path: String,
    source_id: i64,
    chunks: Vec<ChunkData>,
) -> anyhow::Result<i32> {
    info!("[add_chunks] Adding {} chunks for source {}", chunks.len(), source_id);
    
    let conn = Connection::open(&db_path)?;
    
    for chunk in &chunks {
        let embedding_bytes: Vec<u8> = chunk.embedding
            .iter()
            .flat_map(|f| f.to_ne_bytes().to_vec())
            .collect();
        
        conn.execute(
            "INSERT INTO chunks (source_id, chunk_index, content, start_pos, end_pos, embedding)
             VALUES (?1, ?2, ?3, ?4, ?5, ?6)",
            params![
                source_id,
                chunk.chunk_index,
                chunk.content,
                chunk.start_pos,
                chunk.end_pos,
                embedding_bytes
            ],
        )?;
    }
    
    info!("[add_chunks] Added {} chunks", chunks.len());
    Ok(chunks.len() as i32)
}

/// Rebuild HNSW index from chunks table.
pub fn rebuild_chunk_hnsw_index(db_path: String) -> anyhow::Result<()> {
    info!("[rebuild_chunk_hnsw] Starting");
    let conn = Connection::open(&db_path)?;
    
    let mut stmt = conn.prepare("SELECT id, embedding FROM chunks")?;
    
    let points: Vec<(i64, Vec<f32>)> = stmt.query_map([], |row| {
        let id: i64 = row.get(0)?;
        let embedding_blob: Vec<u8> = row.get(1)?;
        
        let embedding: Vec<f32> = embedding_blob
            .chunks(4)
            .map(|chunk| f32::from_ne_bytes(chunk.try_into().unwrap()))
            .collect();
        
        Ok((id, embedding))
    })?
    .filter_map(|r| r.ok())
    .collect();
    
    if !points.is_empty() {
        build_hnsw_index(points)?;
        info!("[rebuild_chunk_hnsw] Built index with {} chunks", stmt.column_count());
    }
    
    Ok(())
}

/// Search result with chunk and source info.
#[derive(Debug, Clone)]
pub struct ChunkSearchResult {
    pub chunk_id: i64,
    pub source_id: i64,
    pub chunk_index: i32,
    pub content: String,
    pub similarity: f64,
}

/// Search chunks by embedding similarity.
pub fn search_chunks(
    db_path: String,
    query_embedding: Vec<f32>,
    top_k: u32,
) -> anyhow::Result<Vec<ChunkSearchResult>> {
    info!("[search_chunks] Searching, top_k={}", top_k);
    
    if !is_hnsw_index_loaded() {
        // Try to build index
        rebuild_chunk_hnsw_index(db_path.clone())?;
    }
    
    if !is_hnsw_index_loaded() {
        // Fall back to linear scan
        return search_chunks_linear(&db_path, query_embedding, top_k);
    }
    
    let hnsw_results = search_hnsw(query_embedding, top_k as usize)?;
    let conn = Connection::open(&db_path)?;
    
    let mut results = Vec::new();
    for result in hnsw_results {
        let row: Option<(i64, i32, String)> = conn
            .query_row(
                "SELECT source_id, chunk_index, content FROM chunks WHERE id = ?1",
                params![result.id],
                |row| Ok((row.get(0)?, row.get(1)?, row.get(2)?)),
            )
            .ok();
        
        if let Some((source_id, chunk_index, content)) = row {
            results.push(ChunkSearchResult {
                chunk_id: result.id,
                source_id,
                chunk_index,
                content,
                similarity: 1.0 - result.distance as f64,
            });
        }
    }
    
    info!("[search_chunks] Found {} results", results.len());
    Ok(results)
}

/// Linear scan fallback for chunk search.
fn search_chunks_linear(
    db_path: &str,
    query_embedding: Vec<f32>,
    top_k: u32,
) -> anyhow::Result<Vec<ChunkSearchResult>> {
    let conn = Connection::open(db_path)?;
    let mut stmt = conn.prepare("SELECT id, source_id, chunk_index, content, embedding FROM chunks")?;
    
    let query_vec = Array1::from(query_embedding.clone());
    let query_norm = query_vec.mapv(|x| x * x).sum().sqrt();
    
    let mut candidates: Vec<(f64, i64, i64, i32, String)> = Vec::new();
    
    let rows = stmt.query_map([], |row| {
        let id: i64 = row.get(0)?;
        let source_id: i64 = row.get(1)?;
        let chunk_index: i32 = row.get(2)?;
        let content: String = row.get(3)?;
        let embedding_blob: Vec<u8> = row.get(4)?;
        Ok((id, source_id, chunk_index, content, embedding_blob))
    })?;
    
    for row in rows {
        let (id, source_id, chunk_index, content, embedding_blob) = row?;
        
        let embedding: Vec<f32> = embedding_blob
            .chunks(4)
            .map(|chunk| f32::from_ne_bytes(chunk.try_into().unwrap()))
            .collect();
        
        if embedding.len() != query_embedding.len() {
            continue;
        }
        
        let target_vec = Array1::from(embedding);
        let target_norm = target_vec.mapv(|x| x * x).sum().sqrt();
        let dot_product = query_vec.dot(&target_vec);
        
        let similarity = if query_norm == 0.0 || target_norm == 0.0 {
            0.0
        } else {
            (dot_product / (query_norm * target_norm)) as f64
        };
        
        candidates.push((similarity, id, source_id, chunk_index, content));
    }
    
    candidates.sort_by(|a, b| b.0.partial_cmp(&a.0).unwrap());
    
    let results: Vec<ChunkSearchResult> = candidates
        .into_iter()
        .take(top_k as usize)
        .map(|(sim, id, source_id, chunk_index, content)| ChunkSearchResult {
            chunk_id: id,
            source_id,
            chunk_index,
            content,
            similarity: sim,
        })
        .collect();
    
    Ok(results)
}

/// Get source document by ID.
pub fn get_source(db_path: String, source_id: i64) -> anyhow::Result<Option<String>> {
    let conn = Connection::open(&db_path)?;
    
    let content: Option<String> = conn
        .query_row(
            "SELECT content FROM sources WHERE id = ?1",
            params![source_id],
            |row| row.get(0),
        )
        .ok();
    
    Ok(content)
}

/// Get all chunks for a source.
pub fn get_source_chunks(db_path: String, source_id: i64) -> anyhow::Result<Vec<String>> {
    let conn = Connection::open(&db_path)?;
    let mut stmt = conn.prepare(
        "SELECT content FROM chunks WHERE source_id = ?1 ORDER BY chunk_index"
    )?;
    
    let chunks: Vec<String> = stmt
        .query_map(params![source_id], |row| row.get(0))?
        .filter_map(|r| r.ok())
        .collect();
    
    Ok(chunks)
}

/// Delete a source and all its chunks.
pub fn delete_source(db_path: String, source_id: i64) -> anyhow::Result<()> {
    let conn = Connection::open(&db_path)?;
    
    conn.execute("DELETE FROM chunks WHERE source_id = ?1", params![source_id])?;
    conn.execute("DELETE FROM sources WHERE id = ?1", params![source_id])?;
    
    info!("[delete_source] Deleted source {}", source_id);
    Ok(())
}

/// Get count of sources and chunks.
#[derive(Debug, Clone)]
pub struct SourceStats {
    pub source_count: i64,
    pub chunk_count: i64,
}

pub fn get_source_stats(db_path: String) -> anyhow::Result<SourceStats> {
    let conn = Connection::open(&db_path)?;
    
    let source_count: i64 = conn.query_row("SELECT COUNT(*) FROM sources", [], |row| row.get(0))?;
    let chunk_count: i64 = conn.query_row("SELECT COUNT(*) FROM chunks", [], |row| row.get(0))?;
    
    Ok(SourceStats { source_count, chunk_count })
}
