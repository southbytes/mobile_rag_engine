// rust/src/api/compression_utils.rs
//! Lightweight text compression utilities for REFRAG-style prompt optimization.
//!
//! Provides fast, rule-based compression for reducing token count while preserving
//! key information. Designed for mobile RAG scenarios.

use std::collections::HashSet;

/// Compression options
#[derive(Debug, Clone)]
pub struct CompressionOptions {
    /// Remove stopwords (common words with low information value)
    pub remove_stopwords: bool,
    /// Remove duplicate sentences
    pub remove_duplicates: bool,
    /// Language for stopword filtering ("ko" or "en")
    pub language: String,
    /// Compression level: 0=minimal, 1=balanced, 2=aggressive
    pub level: i32,
}

impl Default for CompressionOptions {
    fn default() -> Self {
        Self {
            remove_stopwords: true,
            remove_duplicates: true,
            language: "en".to_string(),
            level: 1,
        }
    }
}

/// Result of text compression
#[derive(Debug, Clone)]
pub struct CompressedText {
    /// Compressed text content
    pub text: String,
    /// Original character count
    pub original_chars: i32,
    /// Compressed character count
    pub compressed_chars: i32,
    /// Compression ratio (0.0 - 1.0, lower = more compressed)
    pub ratio: f64,
    /// Number of duplicate sentences removed
    pub sentences_removed: i32,
    /// Characters saved by stopword removal
    pub chars_saved_stopwords: i32,
    /// Characters saved by truncation
    pub chars_saved_truncation: i32,
}

/// Split text into sentences using Unicode-aware boundaries.
/// Supports English (period, ?, !) and other languages.
pub fn split_sentences(text: String) -> Vec<String> {
    if text.is_empty() {
        return vec![];
    }

    let mut sentences = Vec::new();
    let mut current = String::new();

    for ch in text.chars() {
        current.push(ch);
        
        // Sentence endings: . ? !
        if ch == '.' || ch == '?' || ch == '!' || ch == '。' {
            let trimmed = current.trim().to_string();
            if !trimmed.is_empty() && trimmed.len() > 1 {
                sentences.push(trimmed);
            }
            current = String::new();
        }
    }

    // Add remaining text if any
    let trimmed = current.trim().to_string();
    if !trimmed.is_empty() {
        sentences.push(trimmed);
    }

    sentences
}

/// Calculate a fast hash for sentence deduplication.
/// Uses a simple but effective string hash.
pub fn sentence_hash(sentence: String) -> u64 {
    // Normalize: lowercase, remove extra whitespace
    let normalized: String = sentence
        .to_lowercase()
        .split_whitespace()
        .collect::<Vec<_>>()
        .join(" ");
    
    // Simple FNV-1a hash
    let mut hash: u64 = 0xcbf29ce484222325;
    for byte in normalized.bytes() {
        hash ^= byte as u64;
        hash = hash.wrapping_mul(0x100000001b3);
    }
    hash
}

// NOTE: Stopword filtering removed - it damages context and meaning.
// Modern LLM-based systems use more sophisticated methods like
// LLMLingua (perplexity-based) or REFRAG (embedding compression).

/// Compress text using the full pipeline.
///
/// # Arguments
/// * `text` - Text to compress
/// * `max_chars` - Maximum characters in result (0 = no limit)
/// * `options` - Compression options
pub fn compress_text(
    text: String,
    max_chars: i32,
    options: CompressionOptions,
) -> CompressedText {
    let original_chars = text.chars().count() as i32;
    
    if text.is_empty() {
        return CompressedText {
            text: String::new(),
            original_chars: 0,
            compressed_chars: 0,
            ratio: 1.0,
            sentences_removed: 0,
            chars_saved_stopwords: 0,
            chars_saved_truncation: 0,
        };
    }

    // Step 1: Split into sentences
    let sentences = split_sentences(text);
    let original_sentence_count = sentences.len();
    
    // Step 2: Remove duplicate sentences (only exact duplicates)
    let mut unique_sentences = Vec::new();
    let mut seen_hashes = HashSet::new();
    
    if options.remove_duplicates {
        for sentence in sentences {
            let hash = sentence_hash(sentence.clone());
            if !seen_hashes.contains(&hash) {
                seen_hashes.insert(hash);
                unique_sentences.push(sentence);
            }
        }
    } else {
        unique_sentences = sentences;
    }
    
    // Step 3: Rejoin sentences
    let mut result = unique_sentences.join(" ");
    
    // Step 4: Truncate if max_chars specified
    let chars_before_truncation = result.chars().count() as i32;
    if max_chars > 0 && result.chars().count() > max_chars as usize {
        result = result.chars().take(max_chars as usize).collect();
        // Try to end at sentence boundary
        if let Some(pos) = result.rfind(|c| c == '.' || c == '?' || c == '!' || c == '。') {
            result = result[..=pos].to_string();
        }
    }
    let chars_saved_truncation = chars_before_truncation - result.chars().count() as i32;
    
    let compressed_chars = result.chars().count() as i32;
    let sentences_removed = (original_sentence_count - unique_sentences.len()) as i32;
    
    CompressedText {
        text: result,
        original_chars,
        compressed_chars,
        ratio: if original_chars > 0 {
            compressed_chars as f64 / original_chars as f64
        } else {
            1.0
        },
        sentences_removed,
        chars_saved_stopwords: 0,  // No longer used
        chars_saved_truncation,
    }
}

/// Quick compress with default options.
pub fn compress_text_simple(text: String, level: i32) -> String {
    let options = CompressionOptions {
        level,
        ..Default::default()
    };
    compress_text(text, 0, options).text
}

/// Check if text needs compression based on token estimate.
/// Returns true if estimated tokens exceed threshold.
pub fn should_compress(text: String, token_threshold: i32) -> bool {
    // Rough estimate: ~4 chars per token for English
    let char_count = text.chars().count();
    let estimated_tokens = char_count / 4;
    
    estimated_tokens > token_threshold as usize
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_split_sentences_multilingual() {
        let text = "Hello. Nice to meet you! How are you doing?".to_string();
        let sentences = split_sentences(text);
        assert_eq!(sentences.len(), 3);
        assert_eq!(sentences[0], "Hello.");
        assert_eq!(sentences[1], "Nice to meet you!");
        assert_eq!(sentences[2], "How are you doing?");
    }

    #[test]
    fn test_split_sentences_english() {
        let text = "Hello world. How are you? I am fine!".to_string();
        let sentences = split_sentences(text);
        assert_eq!(sentences.len(), 3);
    }

    #[test]
    fn test_sentence_hash_identical() {
        let hash1 = sentence_hash("Hello World".to_string());
        let hash2 = sentence_hash("hello world".to_string());
        let hash3 = sentence_hash("  Hello   World  ".to_string());
        
        // Should be equal after normalization
        assert_eq!(hash1, hash2);
        assert_eq!(hash2, hash3);
    }

    #[test]
    fn test_sentence_hash_different() {
        let hash1 = sentence_hash("Hello World".to_string());
        let hash2 = sentence_hash("Goodbye World".to_string());
        
        assert_ne!(hash1, hash2);
    }



    #[test]
    fn test_compress_text_removes_duplicates() {
        let text = "First sentence. Second sentence. First sentence.".to_string();
        let options = CompressionOptions {
            remove_duplicates: true,
            remove_stopwords: false,
            level: 1,
            ..Default::default()
        };
        
        let result = compress_text(text, 0, options);
        
        // Should have 1 sentence removed (duplicate)
        assert_eq!(result.sentences_removed, 1);
        assert!(result.ratio < 1.0);
    }

    #[test]
    fn test_compress_text_respects_max_chars() {
        let text = "This is a very long text. It has multiple sentences. We want to truncate it.".to_string();
        let options = CompressionOptions::default();
        
        let result = compress_text(text, 30, options);
        
        assert!(result.compressed_chars <= 30);
    }

    #[test]
    fn test_should_compress() {
        let short_text = "Hello".to_string();
        let long_text = "This is a very long text that definitely exceeds the token threshold when we count all the words and characters in this sentence and beyond.".to_string();
        
        assert!(!should_compress(short_text, 100));
        assert!(should_compress(long_text, 10));
    }

    #[test]
    fn test_compress_text_simple() {
        let text = "Hello world. Hello world. Goodbye world.".to_string();
        let compressed = compress_text_simple(text.clone(), 1);
        
        // Should remove duplicate "Hello world."
        assert!(compressed.len() < text.len());
    }
}
