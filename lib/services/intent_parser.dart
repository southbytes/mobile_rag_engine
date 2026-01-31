/// Utility for parsing user intents using the RAG engine.
library;

import '../src/rust/api/user_intent.dart' as raw;

// Re-export types so users can match on UserIntent or access ParsedIntent fields
export '../src/rust/api/user_intent.dart' show UserIntent, ParsedIntent;

/// Utility class for analyzing user input and determining intent.
class IntentParser {
  IntentParser._();

  /// Classify user input into a semantic intent (e.g. Summary, Define, General).
  ///
  /// Returns a [UserIntent] union that can be pattern matched.
  static raw.UserIntent classify(String input) =>
      raw.parseUserIntent(input: input);

  /// Parse the structure of a user command.
  ///
  /// Returns a [ParsedIntent] struct containing raw fields.
  /// Use this if you need low-level access to the parsed components.
  static raw.ParsedIntent parse(String input) => raw.parseIntent(input: input);
}
