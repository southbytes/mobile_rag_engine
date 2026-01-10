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
//! BM25 Keyword Search for Hybrid RAG - lightweight implementation optimized for mobile.

use std::collections::HashMap;
use std::sync::RwLock;
use once_cell::sync::Lazy;
use log::{info, debug};
use crate::api::tokenizer::tokenize;

static INVERTED_INDEX: Lazy<RwLock<InvertedIndex>> = Lazy::new(|| RwLock::new(InvertedIndex::new()));

#[derive(Clone, Debug)]
struct DocMeta {
    length: usize,
    #[allow(dead_code)]
    id: i64,
}

#[flutter_rust_bridge::frb(ignore)]
#[derive(Debug)]
struct InvertedIndex {
    postings: HashMap<String, Vec<(i64, u32)>>,
    doc_meta: HashMap<i64, DocMeta>,
    doc_count: usize,
    avg_doc_length: f64,
    total_tokens: usize,
}

impl InvertedIndex {
    pub fn new() -> Self {
        Self {
            postings: HashMap::new(),
            doc_meta: HashMap::new(),
            doc_count: 0,
            avg_doc_length: 0.0,
            total_tokens: 0,
        }
    }

    pub fn add_document(&mut self, doc_id: i64, content: &str) {
        if self.doc_meta.contains_key(&doc_id) { return; }

        let tokens = tokenize_for_bm25(content);
        let doc_length = tokens.len();
        if doc_length == 0 { return; }

        let mut term_freqs: HashMap<String, u32> = HashMap::new();
        for token in &tokens {
            *term_freqs.entry(token.clone()).or_insert(0) += 1;
        }

        for (term, freq) in term_freqs {
            self.postings.entry(term).or_insert_with(Vec::new).push((doc_id, freq));
        }

        self.doc_meta.insert(doc_id, DocMeta { length: doc_length, id: doc_id });
        self.doc_count += 1;
        self.total_tokens += doc_length;
        self.avg_doc_length = self.total_tokens as f64 / self.doc_count as f64;
    }

    pub fn remove_document(&mut self, doc_id: i64) {
        if let Some(meta) = self.doc_meta.remove(&doc_id) {
            self.doc_count = self.doc_count.saturating_sub(1);
            self.total_tokens = self.total_tokens.saturating_sub(meta.length);
            self.avg_doc_length = if self.doc_count > 0 { self.total_tokens as f64 / self.doc_count as f64 } else { 0.0 };

            for postings_list in self.postings.values_mut() {
                postings_list.retain(|(id, _)| *id != doc_id);
            }
            self.postings.retain(|_, v| !v.is_empty());
        }
    }

    pub fn search(&self, query: &str, top_k: usize) -> Vec<(i64, f64)> {
        if self.doc_count == 0 { return vec![]; }

        let query_tokens = tokenize_for_bm25(query);
        if query_tokens.is_empty() { return vec![]; }

        let k1 = 1.2;
        let b = 0.75;
        let mut scores: HashMap<i64, f64> = HashMap::new();

        for token in &query_tokens {
            if let Some(postings) = self.postings.get(token) {
                let n = postings.len() as f64;
                let idf = ((self.doc_count as f64 - n + 0.5) / (n + 0.5) + 1.0).ln();

                for &(doc_id, tf) in postings {
                    if let Some(meta) = self.doc_meta.get(&doc_id) {
                        let tf_f = tf as f64;
                        let doc_len = meta.length as f64;
                        let tf_component = (tf_f * (k1 + 1.0)) / (tf_f + k1 * (1.0 - b + b * (doc_len / self.avg_doc_length)));
                        *scores.entry(doc_id).or_insert(0.0) += idf * tf_component;
                    }
                }
            }
        }

        let mut results: Vec<(i64, f64)> = scores.into_iter().collect();
        results.sort_by(|a, b| b.1.partial_cmp(&a.1).unwrap_or(std::cmp::Ordering::Equal));
        results.truncate(top_k);
        results
    }

    pub fn clear(&mut self) {
        self.postings.clear();
        self.doc_meta.clear();
        self.doc_count = 0;
        self.avg_doc_length = 0.0;
        self.total_tokens = 0;
    }

    pub fn len(&self) -> usize { self.doc_count }
    pub fn is_empty(&self) -> bool { self.doc_count == 0 }
}

fn tokenize_for_bm25(text: &str) -> Vec<String> {
    let _token_ids = tokenize(text.to_string());
    text.to_lowercase()
        .split(|c: char| !c.is_alphanumeric() && c != '_')
        .filter(|s| s.len() >= 2)
        .map(|s| s.to_string())
        .collect()
}

/// Add document to BM25 index.
pub fn bm25_add_document(doc_id: i64, content: String) {
    let mut index = INVERTED_INDEX.write().unwrap();
    index.add_document(doc_id, &content);
    debug!("[bm25] Added document {} to index", doc_id);
}

/// Add multiple documents to BM25 index (batch).
pub fn bm25_add_documents(docs: Vec<(i64, String)>) {
    let doc_count = docs.len();
    let mut index = INVERTED_INDEX.write().unwrap();
    for (doc_id, content) in docs {
        index.add_document(doc_id, &content);
    }
    info!("[bm25] Added {} documents to index", doc_count);
}

/// Remove document from BM25 index.
pub fn bm25_remove_document(doc_id: i64) {
    let mut index = INVERTED_INDEX.write().unwrap();
    index.remove_document(doc_id);
    debug!("[bm25] Removed document {} from index", doc_id);
}

#[derive(Debug, Clone)]
pub struct Bm25SearchResult {
    pub doc_id: i64,
    pub score: f64,
}

/// Search using BM25.
pub fn bm25_search(query: String, top_k: u32) -> Vec<Bm25SearchResult> {
    let index = INVERTED_INDEX.read().unwrap();
    let results = index.search(&query, top_k as usize);
    debug!("[bm25] Search for '{}' returned {} results", query, results.len());
    results.into_iter().map(|(doc_id, score)| Bm25SearchResult { doc_id, score }).collect()
}

/// Clear BM25 index.
pub fn bm25_clear_index() {
    let mut index = INVERTED_INDEX.write().unwrap();
    index.clear();
    info!("[bm25] Index cleared");
}

/// Check if BM25 index is loaded.
pub fn is_bm25_index_loaded() -> bool {
    let index = INVERTED_INDEX.read().unwrap();
    !index.is_empty()
}

/// Get BM25 index document count.
pub fn bm25_get_document_count() -> usize {
    let index = INVERTED_INDEX.read().unwrap();
    index.len()
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_bm25_basic() {
        let mut index = InvertedIndex::new();
        index.add_document(1, "The quick brown fox jumps over the lazy dog");
        index.add_document(2, "The lazy cat sleeps all day");
        index.add_document(3, "Quick brown cats are rare");
        let results = index.search("lazy cat", 10);
        assert!(!results.is_empty());
        assert_eq!(results[0].0, 2);
    }

    #[test]
    fn test_tokenize_for_bm25() {
        let tokens = tokenize_for_bm25("Hello, World! This is a test.");
        assert!(tokens.contains(&"hello".to_string()));
        assert!(tokens.contains(&"world".to_string()));
    }
}
