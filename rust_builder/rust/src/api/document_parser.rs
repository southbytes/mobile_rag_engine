// Copyright 2025 mobile_rag_engine contributors
// SPDX-License-Identifier: MIT
//
// Document-to-Text (DTT) module for PDF and DOCX text extraction

use anyhow::{anyhow, Result};
use regex::Regex;

/// Remove page number from the end of a page text (if present)
/// Only removes if the last non-empty line is purely numeric
fn remove_trailing_page_number(page_text: &str) -> String {
    let lines: Vec<&str> = page_text.lines().collect();
    if lines.is_empty() {
        return page_text.to_string();
    }
    
    // Find last non-empty line
    let mut last_content_idx = lines.len() - 1;
    while last_content_idx > 0 && lines[last_content_idx].trim().is_empty() {
        last_content_idx -= 1;
    }
    
    let last_line = lines[last_content_idx].trim();
    
    // Check if last line is purely numeric (likely page number)
    if !last_line.is_empty() && last_line.chars().all(|c| c.is_ascii_digit()) {
        // Remove the page number line
        let mut result: Vec<&str> = lines[..last_content_idx].to_vec();
        result.extend_from_slice(&lines[last_content_idx + 1..]);
        result.join("\n")
    } else {
        page_text.to_string()
    }
}

/// Join hyphenated word at page boundary
/// If page ends with "word-" and next page starts with "continuation",
/// join them as "wordcontinuation"
fn join_pages(pages: Vec<String>) -> String {
    if pages.is_empty() {
        return String::new();
    }
    
    // First, clean all pages by removing trailing page numbers
    let cleaned_pages: Vec<String> = pages.iter()
        .map(|p| remove_trailing_page_number(p))
        .collect();
    
    let hyphen_end_re = Regex::new(r"(\w+)-\s*$").unwrap();
    let word_start_re = Regex::new(r"^\s*(\w+)").unwrap();
    
    let mut result = String::new();
    
    for (i, page) in cleaned_pages.iter().enumerate() {
        if i == 0 {
            result = page.clone();
            continue;
        }
        
        // Clone to check for hyphenation without borrow conflicts
        let result_for_check = result.clone();
        let result_trimmed = result_for_check.trim_end();
        
        if let Some(caps) = hyphen_end_re.captures(result_trimmed) {
            let word_part1 = caps.get(1).unwrap().as_str().to_string();
            let match_len = caps.get(0).unwrap().as_str().len();
            
            // Check if current page starts with word continuation
            let page_trimmed = page.trim_start();
            if let Some(next_caps) = word_start_re.captures(page_trimmed) {
                let word_part2 = next_caps.get(1).unwrap().as_str();
                
                // Remove trailing "word-" from result
                let match_start = result_trimmed.len() - match_len;
                result.truncate(match_start);
                result.push_str(&word_part1);
                result.push_str(word_part2);
                
                // Add rest of current page (after the first word)
                let rest_start = next_caps.get(1).unwrap().end();
                result.push_str(&page_trimmed[rest_start..]);
                continue;
            }
        }
        
        // No hyphenation case: just add space and continue
        result.push(' ');
        result.push_str(page);
    }
    
    // Handle in-line hyphenation (line breaks within pages)
    // Only join when: word- + newline + lowercase continuation
    // Preserves real compound words like "user-facing", "data-binding"
    let inline_hyphen_re = Regex::new(r"(\w+)-\s*[\r\n]+\s*([a-z]\w*)").unwrap();
    let dehyphenated = inline_hyphen_re.replace_all(&result, "$1$2");
    
    // Normalize whitespace
    let whitespace_re = Regex::new(r"\s+").unwrap();
    whitespace_re.replace_all(&dehyphenated, " ").trim().to_string()
}

/// Extract text content from a PDF file (bytes)
/// Uses page-by-page extraction for safe page number removal and hyphenation handling
pub fn extract_text_from_pdf(file_bytes: Vec<u8>) -> Result<String> {
    let pages = pdf_extract::extract_text_from_mem_by_pages(&file_bytes)
        .map_err(|e| anyhow!("PDF extraction failed: {:?}", e))?;
    Ok(join_pages(pages))
}

/// Extract text content from a DOCX file (bytes)
pub fn extract_text_from_docx(file_bytes: Vec<u8>) -> Result<String> {
    docx_lite::extract_text_from_bytes(&file_bytes)
        .map_err(|e| anyhow!("DOCX extraction failed: {}", e))
}

/// Auto-detect document type and extract text
/// Uses magic bytes to determine file format
pub fn extract_text_from_document(file_bytes: Vec<u8>) -> Result<String> {
    const MAX_FILE_SIZE: usize = 50 * 1024 * 1024; // 50MB
    
    if file_bytes.len() > MAX_FILE_SIZE {
        return Err(anyhow!("File too large ({} bytes). Maximum supported size is 50MB.", file_bytes.len()));
    }

    if file_bytes.len() < 4 {
        return Err(anyhow!("File too small to determine format"));
    }
    
    // PDF magic bytes: %PDF
    if file_bytes.starts_with(b"%PDF") {
        return extract_text_from_pdf(file_bytes);
    }
    
    // DOCX magic bytes: PK (ZIP archive)
    if file_bytes.starts_with(b"PK") {
        return extract_text_from_docx(file_bytes);
    }
    
    Err(anyhow!("Unsupported document format. Expected PDF or DOCX."))
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_remove_trailing_page_number() {
        let text = "Some content here.\n\n42";
        let result = remove_trailing_page_number(text);
        assert!(!result.contains("42"));
        assert!(result.contains("Some content here."));
    }

    #[test]
    fn test_remove_trailing_page_number_no_number() {
        let text = "Some content here.\nMore content.";
        let result = remove_trailing_page_number(text);
        assert_eq!(result, text);
    }

    #[test]
    fn test_join_pages_dehyphenation() {
        let pages = vec![
            "This is a hyphen-".to_string(),
            "ated word in the text.".to_string(),
        ];
        let result = join_pages(pages);
        assert!(result.contains("hyphenated"));
        assert!(!result.contains("hyphen-"));
    }

    #[test]
    fn test_extract_unsupported_format() {
        let bytes = vec![0x00, 0x01, 0x02, 0x03];
        let result = extract_text_from_document(bytes);
        assert!(result.is_err());
        assert!(result.unwrap_err().to_string().contains("Unsupported"));
    }

    #[test]
    fn test_file_too_small() {
        let bytes = vec![0x50, 0x4B]; // Only 2 bytes
        let result = extract_text_from_document(bytes);
        assert!(result.is_err());
        assert!(result.unwrap_err().to_string().contains("too small"));
    }

    #[test]
    fn test_file_too_large() {
        // Create a vector that exceeds MAX_FILE_SIZE
        let bytes = vec![0u8; 51 * 1024 * 1024]; // 51MB
        let result = extract_text_from_document(bytes);
        assert!(result.is_err());
        assert!(result.unwrap_err().to_string().contains("too large"));
    }
}
