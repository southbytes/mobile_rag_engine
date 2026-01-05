// rust/src/api/semantic_chunker.rs
//
// Semantic text chunking using paragraph boundaries first, then Unicode sentence/word boundaries.
// Enhanced for Korean and multilingual documents that use newlines as section separators.

use text_splitter::TextSplitter;

/// Type classification for a chunk based on content analysis.
#[derive(Debug, Clone, Copy, PartialEq)]
pub enum ChunkType {
    /// Definition or overview content
    Definition,
    /// Example or illustration
    Example,
    /// Bulleted or numbered list
    List,
    /// Procedure or step-by-step instructions
    Procedure,
    /// Comparison between items
    Comparison,
    /// General content (default)
    General,
}

impl ChunkType {
    /// Convert to string for database storage.
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
    
    /// Parse from string (database retrieval).
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

/// Classify a chunk based on its content using rule-based patterns.
/// 
/// Strategy (in order of priority):
/// 1. List patterns (bullet points, numbered items)
/// 2. Definition patterns (formal definitions)
/// 3. Example patterns
/// 4. Procedure patterns (step-by-step)
/// 5. Comparison patterns
/// 6. Default to General
#[flutter_rust_bridge::frb(sync)]
pub fn classify_chunk(text: &str) -> ChunkType {
    let text_lower = text.to_lowercase();
    
    // 1. List detection: count bullet/numbered items
    let bullet_count = text.lines()
        .filter(|l| {
            let trimmed = l.trim();
            trimmed.starts_with("•") 
                || trimmed.starts_with("●")
                || trimmed.starts_with("-") 
                || trimmed.starts_with("*")
                || trimmed.starts_with("①") || trimmed.starts_with("②")
                || trimmed.starts_with("③") || trimmed.starts_with("④")
                || (trimmed.len() > 2 && trimmed.chars().next().map_or(false, |c| c.is_numeric()) 
                    && (trimmed.chars().nth(1) == Some('.') || trimmed.chars().nth(1) == Some(')')))
        })
        .count();
    if bullet_count >= 3 {
        return ChunkType::List;
    }
    
    // 2. Definition patterns (Korean & English)
    let definition_patterns = [
        // Korean
        "이란", "이라 함은", "을 말한다", "를 말한다",
        "를 의미한다", "을 의미한다", "으로 정의된다", "로 정의된다",
        "이다.", "(은)는 ", "(이)란",
        // English
        "is defined as", "refers to", "means that", "is a type of",
        "can be defined as", "is known as",
    ];
    for pattern in definition_patterns {
        if text.contains(pattern) || text_lower.contains(pattern) {
            return ChunkType::Definition;
        }
    }
    
    // 3. Example patterns
    let example_patterns = [
        // Korean
        "예를 들어", "예시:", "예:", "예를 들면", "사례:", "사례로",
        // English
        "for example", "e.g.", "for instance", "such as", "example:",
    ];
    for pattern in example_patterns {
        if text.contains(pattern) || text_lower.contains(pattern) {
            return ChunkType::Example;
        }
    }
    
    // 4. Procedure patterns
    let procedure_patterns = [
        // Korean
        "방법", "단계", "먼저", "그 다음", "그다음", "마지막으로",
        "첫째", "둘째", "셋째", "1단계", "2단계", "절차",
        // English
        "step 1", "step 2", "first,", "then,", "finally,",
        "how to", "procedure", "instructions",
    ];
    let procedure_matches = procedure_patterns.iter()
        .filter(|p| text.contains(*p) || text_lower.contains(*p))
        .count();
    if procedure_matches >= 2 {
        return ChunkType::Procedure;
    }
    
    // 5. Comparison patterns
    let comparison_patterns = [
        // Korean
        "반면", "차이점", "비교하면", "대비", "와 달리", "과 달리",
        "보다 더", "에 비해", "비교 분석",
        // English
        "vs", "versus", "compared to", "in contrast", "on the other hand",
        "differs from", "difference between",
    ];
    for pattern in comparison_patterns {
        if text.contains(pattern) || text_lower.contains(pattern) {
            return ChunkType::Comparison;
        }
    }
    
    ChunkType::General
}

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
    /// Classification type of this chunk.
    pub chunk_type: String,
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
                let chunk_type = classify_chunk(&sub_para);
                chunks.push(SemanticChunk {
                    index: chunk_index,
                    content: sub_para.clone(),
                    start_pos: current_pos,
                    end_pos: current_pos + sub_para.len() as i32,
                    chunk_type: chunk_type.as_str().to_string(),
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
                            let chunk_type = classify_chunk(&line_buffer);
                            chunks.push(SemanticChunk {
                                index: chunk_index,
                                content: line_buffer.clone(),
                                start_pos: current_pos,
                                end_pos: current_pos + line_buffer.len() as i32,
                                chunk_type: chunk_type.as_str().to_string(),
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
                                    let chunk_type = classify_chunk(sub_chunk_trimmed);
                                    chunks.push(SemanticChunk {
                                        index: chunk_index,
                                        content: sub_chunk_trimmed.to_string(),
                                        start_pos: current_pos,
                                        end_pos: current_pos + sub_chunk_trimmed.len() as i32,
                                        chunk_type: chunk_type.as_str().to_string(),
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
                    let chunk_type = classify_chunk(&line_buffer);
                    chunks.push(SemanticChunk {
                        index: chunk_index,
                        content: line_buffer.clone(),
                        start_pos: current_pos,
                        end_pos: current_pos + line_buffer.len() as i32,
                        chunk_type: chunk_type.as_str().to_string(),
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
    
    #[test]
    fn test_classify_chunk_definition() {
        // Korean definition patterns
        assert_eq!(classify_chunk("비트코인이란 분산형 디지털 화폐이다."), ChunkType::Definition);
        assert_eq!(classify_chunk("이 용어를 말한다."), ChunkType::Definition);
        assert_eq!(classify_chunk("프로토콜을 의미한다."), ChunkType::Definition);
        
        // English definition patterns
        assert_eq!(classify_chunk("A blockchain is defined as a distributed ledger."), ChunkType::Definition);
        assert_eq!(classify_chunk("This term refers to a specific concept."), ChunkType::Definition);
    }
    
    #[test]
    fn test_classify_chunk_list() {
        let list_text = "다음은 주요 특징입니다:\n• 분산화\n• 투명성\n• 불변성";
        assert_eq!(classify_chunk(list_text), ChunkType::List);
        
        let numbered_list = "Steps:\n1. First step\n2. Second step\n3. Third step";
        assert_eq!(classify_chunk(numbered_list), ChunkType::List);
    }
    
    #[test]
    fn test_classify_chunk_example() {
        assert_eq!(classify_chunk("예를 들어, 비트코인의 경우..."), ChunkType::Example);
        assert_eq!(classify_chunk("For example, consider the following case."), ChunkType::Example);
    }
    
    #[test]
    fn test_classify_chunk_procedure() {
        let procedure = "먼저 계정을 생성하세요. 그 다음 지갑을 연결합니다.";
        assert_eq!(classify_chunk(procedure), ChunkType::Procedure);
    }
    
    #[test]
    fn test_classify_chunk_comparison() {
        assert_eq!(classify_chunk("비트코인과 이더리움의 차이점은..."), ChunkType::Comparison);
        assert_eq!(classify_chunk("In contrast to traditional systems..."), ChunkType::Comparison);
    }
    
    #[test]
    fn test_classify_chunk_general() {
        assert_eq!(classify_chunk("오늘 시장이 활발하게 움직였다."), ChunkType::General);
        assert_eq!(classify_chunk("The market showed strong activity today."), ChunkType::General);
    }
    
    #[test]
    fn test_chunk_type_string_conversion() {
        assert_eq!(ChunkType::Definition.as_str(), "definition");
        assert_eq!(ChunkType::from_str("definition"), ChunkType::Definition);
        assert_eq!(ChunkType::from_str("unknown"), ChunkType::General);
    }
}
