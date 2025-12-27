// rust/src/api/semantic_chunker.rs
//
// Semantic text chunking using paragraph boundaries first, then Unicode sentence/word boundaries.
// Enhanced for Korean and multilingual documents that use newlines as section separators.

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

/// Split text into semantic chunks using paragraph boundaries first.
/// 
/// Strategy:
/// 1. First split by double newlines (\n\n) - paragraph boundaries
/// 2. If a paragraph is too long, split by single newlines (\n)
/// 3. If still too long, use text-splitter for Unicode sentence/word boundaries
/// 
/// This approach works better for Korean and other languages where
/// newlines often indicate logical section boundaries.
/// 
/// # Arguments
/// * `text` - The text to chunk
/// * `max_chars` - Maximum characters per chunk (soft limit, may exceed slightly to preserve sentence)
/// 
/// # Returns
/// Vector of SemanticChunk with complete paragraphs/sentences/words
#[flutter_rust_bridge::frb(sync)]
pub fn semantic_chunk(text: String, max_chars: i32) -> Vec<SemanticChunk> {
    if text.is_empty() {
        return vec![];
    }
    
    let max_chars_usize = max_chars.max(100) as usize;
    let mut chunks = Vec::new();
    let mut current_pos = 0i32;
    let mut chunk_index = 0i32;
    
    // Step 1: Split by double newlines (paragraphs) first
    let paragraphs: Vec<&str> = text.split("\n\n").collect();
    
    for para in paragraphs {
        let para_trimmed = para.trim();
        if para_trimmed.is_empty() {
            continue;
        }
        
        // Step 2: Further split by article title patterns (Korean legal docs)
        // Pattern: "제X조" or "제 X 조" at start of line
        let split_chunks = split_by_article_titles(para_trimmed);
        
        for sub_para in split_chunks {
            if sub_para.is_empty() {
                continue;
            }
            
            if sub_para.len() <= max_chars_usize {
                // Chunk fits
                chunks.push(SemanticChunk {
                    index: chunk_index,
                    content: sub_para.clone(),
                    start_pos: current_pos,
                    end_pos: current_pos + sub_para.len() as i32,
                });
                chunk_index += 1;
                current_pos += sub_para.len() as i32 + 1;
            } else {
                // Still too long - split by single newlines
                let lines: Vec<&str> = sub_para.split('\n').collect();
                let mut line_buffer = String::new();
                
                for line in lines {
                    let line_trimmed = line.trim();
                    if line_trimmed.is_empty() {
                        continue;
                    }
                    
                    // Force split if line starts with article pattern
                    let is_article_start = is_article_title(line_trimmed);
                    
                    // Check if adding this line would exceed limit
                    let would_be_len = if line_buffer.is_empty() {
                        line_trimmed.len()
                    } else {
                        line_buffer.len() + 1 + line_trimmed.len()
                    };
                    
                    if would_be_len <= max_chars_usize && !is_article_start {
                        // Add to buffer
                        if !line_buffer.is_empty() {
                            line_buffer.push('\n');
                        }
                        line_buffer.push_str(line_trimmed);
                    } else {
                        // Flush buffer if not empty
                        if !line_buffer.is_empty() {
                            chunks.push(SemanticChunk {
                                index: chunk_index,
                                content: line_buffer.clone(),
                                start_pos: current_pos,
                                end_pos: current_pos + line_buffer.len() as i32,
                            });
                            chunk_index += 1;
                            current_pos += line_buffer.len() as i32 + 1;
                            line_buffer.clear();
                        }
                        
                        // Handle the line itself
                        if line_trimmed.len() <= max_chars_usize {
                            line_buffer.push_str(line_trimmed);
                        } else {
                            // Line is too long - use text-splitter
                            let splitter = TextSplitter::new(max_chars_usize);
                            for sub_chunk in splitter.chunks(line_trimmed) {
                                let sub_chunk_trimmed = sub_chunk.trim();
                                if !sub_chunk_trimmed.is_empty() {
                                    chunks.push(SemanticChunk {
                                        index: chunk_index,
                                        content: sub_chunk_trimmed.to_string(),
                                        start_pos: current_pos,
                                        end_pos: current_pos + sub_chunk_trimmed.len() as i32,
                                    });
                                    chunk_index += 1;
                                    current_pos += sub_chunk_trimmed.len() as i32;
                                }
                            }
                        }
                    }
                }
                
                // Flush remaining buffer
                if !line_buffer.is_empty() {
                    chunks.push(SemanticChunk {
                        index: chunk_index,
                        content: line_buffer.clone(),
                        start_pos: current_pos,
                        end_pos: current_pos + line_buffer.len() as i32,
                    });
                    chunk_index += 1;
                    current_pos += line_buffer.len() as i32 + 2;
                }
            }
        }
    }
    
    chunks
}

/// Check if a line starts with Korean article title pattern
fn is_article_title(line: &str) -> bool {
    let trimmed = line.trim();
    // Pattern: "제X조" or "제 X 조" or "제X장" etc.
    if trimmed.starts_with("제") {
        // Check for patterns like "제1조", "제 1 조", "제1장", "제 1 장"
        let chars: Vec<char> = trimmed.chars().collect();
        if chars.len() >= 3 {
            // "제X조" pattern (compact)
            if chars.get(1).map_or(false, |c| c.is_numeric()) {
                return true;
            }
            // "제 X 조" pattern (spaced) or "제 1장" etc.
            if chars.get(1) == Some(&' ') && chars.get(2).map_or(false, |c| c.is_numeric()) {
                return true;
            }
        }
    }
    false
}

/// Split text by article title patterns
fn split_by_article_titles(text: &str) -> Vec<String> {
    let mut result = Vec::new();
    let mut current = String::new();
    
    for line in text.lines() {
        let trimmed = line.trim();
        
        if is_article_title(trimmed) && !current.is_empty() {
            // Found new article, save current buffer
            result.push(current.trim().to_string());
            current = String::new();
        }
        
        if !current.is_empty() {
            current.push('\n');
        }
        current.push_str(line);
    }
    
    // Don't forget the last chunk
    if !current.is_empty() {
        result.push(current.trim().to_string());
    }
    
    result
}

/// Split text with overlap for RAG context continuity.
/// 
/// Similar to `semantic_chunk` but ensures overlap between chunks
/// for better context retrieval.
/// 
/// # Arguments
/// * `text` - The text to chunk
/// * `max_chars` - Maximum characters per chunk
/// * `overlap_chars` - Target overlap between consecutive chunks (not used in paragraph mode, kept for API compatibility)
#[flutter_rust_bridge::frb(sync)]
pub fn semantic_chunk_with_overlap(
    text: String, 
    max_chars: i32,
    _overlap_chars: i32,  // Not used in paragraph-first mode, but kept for API compatibility
) -> Vec<SemanticChunk> {
    // For paragraph-based chunking, overlap is handled differently
    // We preserve complete paragraphs/lines, so overlap isn't needed
    semantic_chunk(text, max_chars)
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
