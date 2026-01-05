// rust/src/api/user_intent.rs
//
// User intent parsing for slash commands

/// Represents the user's intent parsed from their input.
/// Slash commands like /summary, /define, /more are parsed into specific intents.
#[derive(Debug, Clone, PartialEq)]
pub enum UserIntent {
    /// /summary - Summarize RAG results
    Summary { query: String },
    
    /// /define <term> - Define a term
    Define { term: String },
    
    /// /more - Expand knowledge using LLM beyond RAG
    ExpandKnowledge { query: String },
    
    /// General query without any special command
    General { query: String },
    
    /// Invalid or unrecognized command
    InvalidCommand { command: String, reason: String },
}

impl UserIntent {
    /// Get the query/term part of the intent
    pub fn get_query(&self) -> &str {
        match self {
            UserIntent::Summary { query } => query,
            UserIntent::Define { term } => term,
            UserIntent::ExpandKnowledge { query } => query,
            UserIntent::General { query } => query,
            UserIntent::InvalidCommand { command, .. } => command,
        }
    }
    
    /// Get the intent type as a string for logging/debugging
    pub fn intent_type(&self) -> &str {
        match self {
            UserIntent::Summary { .. } => "summary",
            UserIntent::Define { .. } => "define",
            UserIntent::ExpandKnowledge { .. } => "more",
            UserIntent::General { .. } => "general",
            UserIntent::InvalidCommand { .. } => "invalid",
        }
    }
}

/// Parse user input into a UserIntent.
/// 
/// Supported commands:
/// - /summary [query] - Summarize the query results
/// - /define <term> - Define a specific term
/// - /more [query] - Expand knowledge beyond RAG
/// - Any other text - General query
#[flutter_rust_bridge::frb(sync)]
pub fn parse_user_intent(input: &str) -> UserIntent {
    let trimmed = input.trim();
    
    // Check for empty input
    if trimmed.is_empty() {
        return UserIntent::InvalidCommand {
            command: String::new(),
            reason: "Empty input".to_string(),
        };
    }
    
    // Not a slash command - return as general query
    if !trimmed.starts_with('/') {
        return UserIntent::General {
            query: trimmed.to_string(),
        };
    }
    
    // Parse slash command
    let parts: Vec<&str> = trimmed.splitn(2, ' ').collect();
    let command = parts[0].to_lowercase();
    let argument = parts.get(1).map(|s| s.trim()).unwrap_or("");
    
    match command.as_str() {
        "/summary" | "/요약" => {
            UserIntent::Summary {
                query: argument.to_string(),
            }
        }
        "/define" | "/정의" => {
            if argument.is_empty() {
                UserIntent::InvalidCommand {
                    command: command.to_string(),
                    reason: "Term required for /define. Usage: /define <term>".to_string(),
                }
            } else {
                UserIntent::Define {
                    term: argument.to_string(),
                }
            }
        }
        "/more" | "/확장" => {
            UserIntent::ExpandKnowledge {
                query: argument.to_string(),
            }
        }
        _ => {
            // Unknown command
            UserIntent::InvalidCommand {
                command: command.to_string(),
                reason: format!("Unknown command '{}'. Available: /summary, /define, /more", command),
            }
        }
    }
}

/// Parsed intent result for FRB serialization
#[derive(Debug, Clone)]
pub struct ParsedIntent {
    pub intent_type: String,
    pub query: String,
    pub is_valid: bool,
    pub error_message: Option<String>,
}

/// Parse user input and return a FRB-friendly struct
#[flutter_rust_bridge::frb(sync)]
pub fn parse_intent(input: String) -> ParsedIntent {
    let intent = parse_user_intent(&input);
    
    match intent {
        UserIntent::Summary { query } => ParsedIntent {
            intent_type: "summary".to_string(),
            query,
            is_valid: true,
            error_message: None,
        },
        UserIntent::Define { term } => ParsedIntent {
            intent_type: "define".to_string(),
            query: term,
            is_valid: true,
            error_message: None,
        },
        UserIntent::ExpandKnowledge { query } => ParsedIntent {
            intent_type: "more".to_string(),
            query,
            is_valid: true,
            error_message: None,
        },
        UserIntent::General { query } => ParsedIntent {
            intent_type: "general".to_string(),
            query,
            is_valid: true,
            error_message: None,
        },
        UserIntent::InvalidCommand { command, reason } => ParsedIntent {
            intent_type: "invalid".to_string(),
            query: command,
            is_valid: false,
            error_message: Some(reason),
        },
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    
    #[test]
    fn test_parse_general_query() {
        let intent = parse_user_intent("비트코인이란 무엇인가요?");
        assert!(matches!(intent, UserIntent::General { .. }));
        assert_eq!(intent.get_query(), "비트코인이란 무엇인가요?");
    }
    
    #[test]
    fn test_parse_summary_command() {
        let intent = parse_user_intent("/summary RWA에 대해");
        assert!(matches!(intent, UserIntent::Summary { .. }));
        assert_eq!(intent.get_query(), "RWA에 대해");
    }
    
    #[test]
    fn test_parse_summary_korean() {
        let intent = parse_user_intent("/요약 블록체인 기술");
        assert!(matches!(intent, UserIntent::Summary { .. }));
    }
    
    #[test]
    fn test_parse_define_command() {
        let intent = parse_user_intent("/define 스마트 컨트랙트");
        assert!(matches!(intent, UserIntent::Define { .. }));
        assert_eq!(intent.get_query(), "스마트 컨트랙트");
    }
    
    #[test]
    fn test_parse_define_without_term() {
        let intent = parse_user_intent("/define");
        assert!(matches!(intent, UserIntent::InvalidCommand { .. }));
    }
    
    #[test]
    fn test_parse_more_command() {
        let intent = parse_user_intent("/more 트레이딩 전략");
        assert!(matches!(intent, UserIntent::ExpandKnowledge { .. }));
        assert_eq!(intent.get_query(), "트레이딩 전략");
    }
    
    #[test]
    fn test_parse_unknown_command() {
        let intent = parse_user_intent("/unknown test");
        assert!(matches!(intent, UserIntent::InvalidCommand { .. }));
    }
    
    #[test]
    fn test_parse_empty_input() {
        let intent = parse_user_intent("");
        assert!(matches!(intent, UserIntent::InvalidCommand { .. }));
    }
}
