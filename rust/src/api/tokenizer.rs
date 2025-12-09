// rust/src/api/tokenizer.rs

use flutter_rust_bridge::frb;
use tokenizers::Tokenizer;
use anyhow::Result;
use std::sync::RwLock;
use once_cell::sync::Lazy;

// Global tokenizer instance (loaded once)
// Using RwLock for concurrent read access (tokenize/decode can run in parallel)
static TOKENIZER: Lazy<RwLock<Option<Tokenizer>>> = Lazy::new(|| RwLock::new(None));

/// Initialize tokenizer with tokenizer.json file path
pub fn init_tokenizer(tokenizer_path: String) -> Result<()> {
    let mut tokenizer = Tokenizer::from_file(&tokenizer_path)
        .map_err(|e| anyhow::anyhow!("Failed to load tokenizer: {}", e))?;
    
    // Disable padding - ONNX input has dynamic length
    tokenizer.with_padding(None);
    
    // Set truncation (max 256 tokens)
    tokenizer.with_truncation(Some(tokenizers::TruncationParams {
        max_length: 256,
        ..Default::default()
    })).ok();
    
    // Write lock for initialization
    let mut global_tokenizer = TOKENIZER.write().unwrap();
    *global_tokenizer = Some(tokenizer);
    
    Ok(())
}

/// Tokenize text (returns token IDs)
/// add_special_tokens=true to include CLS/SEP tokens
#[frb(sync)]
pub fn tokenize(text: String) -> Result<Vec<u32>> {
    // Read lock for concurrent tokenization
    let tokenizer_guard = TOKENIZER.read().unwrap();
    let tokenizer = tokenizer_guard
        .as_ref()
        .ok_or_else(|| anyhow::anyhow!("Tokenizer not initialized. Call init_tokenizer first."))?;
    
    // add_special_tokens=true: Add CLS(101) and SEP(102) tokens
    let encoding = tokenizer
        .encode(text, true)  // Changed: false -> true
        .map_err(|e| anyhow::anyhow!("Tokenization failed: {}", e))?;
    
    Ok(encoding.get_ids().to_vec())
}

/// Decode token IDs to text
#[frb(sync)]
pub fn decode_tokens(token_ids: Vec<u32>) -> Result<String> {
    // Read lock for concurrent decoding
    let tokenizer_guard = TOKENIZER.read().unwrap();
    let tokenizer = tokenizer_guard
        .as_ref()
        .ok_or_else(|| anyhow::anyhow!("Tokenizer not initialized."))?;
    
    let decoded = tokenizer
        .decode(&token_ids, true)
        .map_err(|e| anyhow::anyhow!("Decoding failed: {}", e))?;
    
    Ok(decoded)
}

/// Get tokenizer info (vocab size, etc.)
#[frb(sync)]
pub fn get_vocab_size() -> Result<u32> {
    // Read lock for concurrent access
    let tokenizer_guard = TOKENIZER.read().unwrap();
    let tokenizer = tokenizer_guard
        .as_ref()
        .ok_or_else(|| anyhow::anyhow!("Tokenizer not initialized."))?;
    
    Ok(tokenizer.get_vocab_size(true) as u32)
}
