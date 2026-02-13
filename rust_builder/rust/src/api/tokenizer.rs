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
//! HuggingFace tokenizers integration module.

use anyhow::Result;
use flutter_rust_bridge::frb;
use once_cell::sync::Lazy;
use std::sync::RwLock;
use tokenizers::Tokenizer;

static TOKENIZER: Lazy<RwLock<Option<Tokenizer>>> = Lazy::new(|| RwLock::new(None));
const TOKENIZER_BASE_TRUNCATION_MAX_LENGTH: usize = 256;
const TOKENIZER_MID_TRUNCATION_MAX_LENGTH: usize = 384;
const TOKENIZER_MAX_TRUNCATION_MAX_LENGTH: usize = 512;
const TOKENIZER_MID_TRUNCATION_CHAR_THRESHOLD: usize = 1200;
const TOKENIZER_MAX_TRUNCATION_CHAR_THRESHOLD: usize = 2400;

fn resolve_truncation_max_length(text: &str) -> usize {
    let char_len = text.chars().count();
    if char_len >= TOKENIZER_MAX_TRUNCATION_CHAR_THRESHOLD {
        TOKENIZER_MAX_TRUNCATION_MAX_LENGTH
    } else if char_len >= TOKENIZER_MID_TRUNCATION_CHAR_THRESHOLD {
        TOKENIZER_MID_TRUNCATION_MAX_LENGTH
    } else {
        TOKENIZER_BASE_TRUNCATION_MAX_LENGTH
    }
}

/// Initialize tokenizer with tokenizer.json file path.
pub fn init_tokenizer(tokenizer_path: String) -> Result<()> {
    let mut tokenizer = Tokenizer::from_file(&tokenizer_path)
        .map_err(|e| anyhow::anyhow!("Failed to load tokenizer: {}", e))?;

    tokenizer.with_padding(None);
    tokenizer.with_truncation(None).ok();

    let mut global_tokenizer = TOKENIZER.write().unwrap();
    *global_tokenizer = Some(tokenizer);
    Ok(())
}

/// Tokenize text (returns token IDs with CLS/SEP tokens).
#[frb(sync)]
pub fn tokenize(text: String) -> Result<Vec<u32>> {
    let tokenizer_guard = TOKENIZER.read().unwrap();
    let tokenizer = tokenizer_guard
        .as_ref()
        .ok_or_else(|| anyhow::anyhow!("Tokenizer not initialized. Call init_tokenizer first."))?;

    // Dynamically widen truncation for longer chunks while keeping
    // an upper bound for mobile runtime stability.
    let max_length = resolve_truncation_max_length(&text);
    let mut tokenizer = tokenizer.clone();
    tokenizer
        .with_truncation(Some(tokenizers::TruncationParams {
            max_length,
            ..Default::default()
        }))
        .ok();

    let encoding = tokenizer
        .encode(text, true)
        .map_err(|e| anyhow::anyhow!("Tokenization failed: {}", e))?;
    Ok(encoding.get_ids().to_vec())
}

/// Decode token IDs to text.
#[frb(sync)]
pub fn decode_tokens(token_ids: Vec<u32>) -> Result<String> {
    let tokenizer_guard = TOKENIZER.read().unwrap();
    let tokenizer = tokenizer_guard
        .as_ref()
        .ok_or_else(|| anyhow::anyhow!("Tokenizer not initialized."))?;

    let decoded = tokenizer
        .decode(&token_ids, true)
        .map_err(|e| anyhow::anyhow!("Decoding failed: {}", e))?;
    Ok(decoded)
}

/// Get vocab size.
#[frb(sync)]
pub fn get_vocab_size() -> Result<u32> {
    let tokenizer_guard = TOKENIZER.read().unwrap();
    let tokenizer = tokenizer_guard
        .as_ref()
        .ok_or_else(|| anyhow::anyhow!("Tokenizer not initialized."))?;
    Ok(tokenizer.get_vocab_size(true) as u32)
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_resolve_truncation_max_length_short() {
        assert_eq!(resolve_truncation_max_length("hello world"), 256);
    }

    #[test]
    fn test_resolve_truncation_max_length_mid() {
        let text = "가".repeat(1500);
        assert_eq!(resolve_truncation_max_length(&text), 384);
    }

    #[test]
    fn test_resolve_truncation_max_length_long() {
        let text = "x".repeat(3000);
        assert_eq!(resolve_truncation_max_length(&text), 512);
    }
}
