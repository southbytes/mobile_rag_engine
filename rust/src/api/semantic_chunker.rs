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
//! Semantic text chunking with paragraph-first strategy for multilingual support.

use text_splitter::TextSplitter;

/// Chunk type classification.
#[derive(Debug, Clone, Copy, PartialEq)]
pub enum ChunkType {
    Definition,
    Example,
    List,
    Procedure,
    Comparison,
    General,
}

impl ChunkType {
    pub fn as_str(&self) -> &'static str {
        match self {
            ChunkType::Definition => "definition",
            ChunkType::Example => "example",
            ChunkType::List => "list",
            ChunkType::Procedure => "procedure",
            ChunkType::Comparison => "comparison",
            ChunkType::General => "general",
        }
    }
    
    pub fn from_str(s: &str) -> Self {
        match s {
            "definition" => ChunkType::Definition,
            "example" => ChunkType::Example,
            "list" => ChunkType::List,
            "procedure" => ChunkType::Procedure,
            "comparison" => ChunkType::Comparison,
            _ => ChunkType::General,
        }
    }
}

/// Classify chunk by rule-based pattern matching.
#[flutter_rust_bridge::frb(sync)]
pub fn classify_chunk(text: &str) -> ChunkType {
    let text_lower = text.to_lowercase();
    
    // List detection
    let bullet_count = text.lines()
        .filter(|l| {
            let trimmed = l.trim();
            trimmed.starts_with("•") || trimmed.starts_with("●")
                || trimmed.starts_with("-") || trimmed.starts_with("*")
                || trimmed.starts_with("①") || trimmed.starts_with("②")
                || trimmed.starts_with("③") || trimmed.starts_with("④")
                || (trimmed.len() > 2 && trimmed.chars().next().map_or(false, |c| c.is_numeric()) 
                    && (trimmed.chars().nth(1) == Some('.') || trimmed.chars().nth(1) == Some(')')))
        }).count();
    if bullet_count >= 3 { return ChunkType::List; }
    
    // Definition patterns
    let definition_patterns = ["is defined as", "refers to", "means that", "is a type of", "can be defined as", "is known as"];
    for pattern in definition_patterns {
        if text_lower.contains(pattern) { return ChunkType::Definition; }
    }
    
    // Example patterns
    let example_patterns = ["for example", "e.g.", "for instance", "such as", "example:"];
    for pattern in example_patterns {
        if text_lower.contains(pattern) { return ChunkType::Example; }
    }
    
    // Procedure patterns
    let procedure_patterns = ["step 1", "step 2", "first,", "then,", "finally,", "how to", "procedure", "instructions"];
    let procedure_matches = procedure_patterns.iter().filter(|p| text_lower.contains(*p)).count();
    if procedure_matches >= 2 { return ChunkType::Procedure; }
    
    // Comparison patterns
    let comparison_patterns = ["vs", "versus", "compared to", "in contrast", "on the other hand", "differs from", "difference between"];
    for pattern in comparison_patterns {
        if text_lower.contains(pattern) { return ChunkType::Comparison; }
    }
    
    ChunkType::General
}

/// Semantic chunk result.
#[derive(Debug, Clone)]
pub struct SemanticChunk {
    pub index: i32,
    pub content: String,
    pub start_pos: i32,
    pub end_pos: i32,
    pub chunk_type: String,
}

/// Split text into semantic chunks using paragraph-first strategy.
#[flutter_rust_bridge::frb(sync)]
pub fn semantic_chunk(text: String, max_chars: i32) -> Vec<SemanticChunk> {
    if text.is_empty() { return vec![]; }
    
    let max_chars_usize = max_chars.max(100) as usize;
    let mut chunks = Vec::new();
    let mut current_pos = 0i32;
    let mut chunk_index = 0i32;
    
    let paragraphs: Vec<&str> = text.split("\n\n").collect();
    
    for para in paragraphs {
        let para_trimmed = para.trim();
        if para_trimmed.is_empty() { continue; }
        
        if para_trimmed.len() <= max_chars_usize {
            let chunk_type = classify_chunk(para_trimmed);
            chunks.push(SemanticChunk {
                index: chunk_index, content: para_trimmed.to_string(),
                start_pos: current_pos, end_pos: current_pos + para_trimmed.len() as i32,
                chunk_type: chunk_type.as_str().to_string(),
            });
            chunk_index += 1;
            current_pos += para_trimmed.len() as i32 + 1;
        } else {
            let lines: Vec<&str> = para_trimmed.split('\n').collect();
            let mut line_buffer = String::new();
            
            for line in lines {
                let line_trimmed = line.trim();
                if line_trimmed.is_empty() { continue; }
                
                let is_article_start = is_article_title(line_trimmed);
                let would_be_len = if line_buffer.is_empty() { line_trimmed.len() }
                else { line_buffer.len() + 1 + line_trimmed.len() };
                
                if would_be_len <= max_chars_usize && !is_article_start {
                    if !line_buffer.is_empty() { line_buffer.push('\n'); }
                    line_buffer.push_str(line_trimmed);
                } else {
                    if !line_buffer.is_empty() {
                        let chunk_type = classify_chunk(&line_buffer);
                        chunks.push(SemanticChunk {
                            index: chunk_index, content: line_buffer.clone(),
                            start_pos: current_pos, end_pos: current_pos + line_buffer.len() as i32,
                            chunk_type: chunk_type.as_str().to_string(),
                        });
                        chunk_index += 1;
                        current_pos += line_buffer.len() as i32 + 1;
                        line_buffer.clear();
                    }
                    
                    if line_trimmed.len() <= max_chars_usize {
                        line_buffer.push_str(line_trimmed);
                    } else {
                        let splitter = TextSplitter::new(max_chars_usize);
                        for sub_chunk in splitter.chunks(line_trimmed) {
                            let sub_chunk_trimmed = sub_chunk.trim();
                            if !sub_chunk_trimmed.is_empty() {
                                let chunk_type = classify_chunk(sub_chunk_trimmed);
                                chunks.push(SemanticChunk {
                                    index: chunk_index, content: sub_chunk_trimmed.to_string(),
                                    start_pos: current_pos, end_pos: current_pos + sub_chunk_trimmed.len() as i32,
                                    chunk_type: chunk_type.as_str().to_string(),
                                });
                                chunk_index += 1;
                                current_pos += sub_chunk_trimmed.len() as i32;
                            }
                        }
                    }
                }
            }
            
            if !line_buffer.is_empty() {
                let chunk_type = classify_chunk(&line_buffer);
                chunks.push(SemanticChunk {
                    index: chunk_index, content: line_buffer.clone(),
                    start_pos: current_pos, end_pos: current_pos + line_buffer.len() as i32,
                    chunk_type: chunk_type.as_str().to_string(),
                });
                chunk_index += 1;
                current_pos += line_buffer.len() as i32 + 2;
            }
        }
    }
    
    chunks
}

#[allow(dead_code)]
fn is_article_title(_line: &str) -> bool { false }

/// Split text with overlap (API compatibility wrapper).
#[flutter_rust_bridge::frb(sync)]
pub fn semantic_chunk_with_overlap(text: String, max_chars: i32, _overlap_chars: i32) -> Vec<SemanticChunk> {
    semantic_chunk(text, max_chars)
}

#[cfg(test)]
mod tests {
    use super::*;
    
    #[test]
    fn test_semantic_chunk_basic() {
        let text = "This is the first sentence. This is the second sentence.";
        let chunks = semantic_chunk(text.to_string(), 50);
        assert!(!chunks.is_empty());
    }
    
    #[test]
    fn test_empty_text() {
        let chunks = semantic_chunk("".to_string(), 100);
        assert!(chunks.is_empty());
    }
    
    #[test]
    fn test_classify_chunk_definition() {
        assert_eq!(classify_chunk("A blockchain is defined as a distributed ledger."), ChunkType::Definition);
    }
    
    #[test]
    fn test_classify_chunk_list() {
        let list_text = "Features:\n• Item1\n• Item2\n• Item3";
        assert_eq!(classify_chunk(list_text), ChunkType::List);
    }
    
    #[test]
    fn test_chunk_type_string_conversion() {
        assert_eq!(ChunkType::Definition.as_str(), "definition");
        assert_eq!(ChunkType::from_str("definition"), ChunkType::Definition);
        assert_eq!(ChunkType::from_str("unknown"), ChunkType::General);
    }
}
