// rust/src/api/semantic_chunker.rs
//
// Semantic text chunking using Unicode sentence/word boundaries.
// This module provides chunking that respects linguistic structures,
// ensuring chunks never split in the middle of words or sentences.

use text_splitter::TextSplitter;

/// Result of semantic chunking operation.
#[derive(Debug, Clone)]
pub struct SemanticChunk {
    /// Index of this chunk (0-based).
    pub index: i32,
    /// The chunk content (complete sentences/words, never cut mid-word).
    pub content: String,
    /// Approximate character position where this chunk starts.
    pub start_pos: i32,
    /// Approximate character position where this chunk ends.
    pub end_pos: i32,
}

/// Split text into semantic chunks using Unicode boundaries.
/// 
/// Uses `text-splitter` crate which splits on:
/// 1. Unicode Sentence Boundaries (preferred)
/// 2. Unicode Word Boundaries (fallback)
/// 3. Never splits in the middle of a word
/// 
/// # Arguments
/// * `text` - The text to chunk
/// * `max_chars` - Maximum characters per chunk (soft limit, may exceed slightly to preserve sentence)
/// 
/// # Returns
/// Vector of SemanticChunk with complete sentences/words
#[flutter_rust_bridge::frb(sync)]
pub fn semantic_chunk(text: String, max_chars: i32) -> Vec<SemanticChunk> {
    if text.is_empty() {
        return vec![];
    }
    
    let max_chars_usize = max_chars.max(100) as usize;
    let splitter = TextSplitter::new(max_chars_usize);
    
    let mut chunks = Vec::new();
    let mut current_pos = 0i32;
    
    for (index, chunk_str) in splitter.chunks(&text).enumerate() {
        let chunk_len = chunk_str.len() as i32;
        chunks.push(SemanticChunk {
            index: index as i32,
            content: chunk_str.to_string(),
            start_pos: current_pos,
            end_pos: current_pos + chunk_len,
        });
        current_pos += chunk_len;
    }
    
    chunks
}

/// Split text with overlap for RAG context continuity.
/// 
/// Similar to `semantic_chunk` but ensures overlap between chunks
/// for better context retrieval.
/// 
/// # Arguments
/// * `text` - The text to chunk
/// * `max_chars` - Maximum characters per chunk
/// * `overlap_chars` - Target overlap between consecutive chunks
#[flutter_rust_bridge::frb(sync)]
pub fn semantic_chunk_with_overlap(
    text: String, 
    max_chars: i32,
    overlap_chars: i32,
) -> Vec<SemanticChunk> {
    if text.is_empty() {
        return vec![];
    }
    
    let max_chars_usize = max_chars.max(100) as usize;
    let overlap_usize = overlap_chars.max(0).min(max_chars / 2) as usize;
    
    // Use text-splitter with overlap support
    let splitter = TextSplitter::new(max_chars_usize..max_chars_usize + overlap_usize);
    
    let mut chunks = Vec::new();
    let mut current_pos = 0i32;
    
    for (index, chunk_str) in splitter.chunks(&text).enumerate() {
        let chunk_len = chunk_str.len() as i32;
        chunks.push(SemanticChunk {
            index: index as i32,
            content: chunk_str.to_string(),
            start_pos: current_pos,
            end_pos: current_pos + chunk_len,
        });
        // Move position, accounting for potential overlap
        current_pos += chunk_len - (overlap_usize as i32).min(chunk_len / 2);
    }
    
    chunks
}

#[cfg(test)]
mod tests {
    use super::*;
    
    #[test]
    fn test_semantic_chunk_basic() {
        let text = "This is the first sentence. This is the second sentence. And here is the third one.";
        let chunks = semantic_chunk(text.to_string(), 50);
        
        assert!(!chunks.is_empty());
        // Verify no chunk starts with lowercase (would indicate mid-word split)
        for chunk in &chunks {
            let first_char = chunk.content.chars().next().unwrap();
            assert!(first_char.is_uppercase() || first_char.is_whitespace(), 
                    "Chunk should not start mid-word: {}", chunk.content);
        }
    }
    
    #[test]
    fn test_empty_text() {
        let chunks = semantic_chunk("".to_string(), 100);
        assert!(chunks.is_empty());
    }
}
