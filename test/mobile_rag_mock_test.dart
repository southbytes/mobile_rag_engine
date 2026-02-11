import 'package:flutter_test/flutter_test.dart';
import 'package:mobile_rag_engine/mobile_rag_engine.dart';
import 'package:mocktail/mocktail.dart';

/// A mock class for MobileRag using mocktail.
/// This allows us to simulate the RAG engine behavior in pure Dart tests.
class MockMobileRag extends Mock implements MobileRag {}

/// A mock class for RagEngine using mocktail.
/// This allows us to simulate the native engine behavior.
class MockRagEngine extends Mock implements RagEngine {}

/// Fake class for RagSearchResult
class FakeRagSearchResult extends Fake implements RagSearchResult {}

void main() {
  group('MobileRag Mocking Template', () {
    late MockMobileRag mockMobileRag;

    setUpAll(() {
      registerFallbackValue(FakeRagSearchResult());
    });

    setUp(() {
      mockMobileRag = MockMobileRag();
      when(() => mockMobileRag.engine).thenReturn(MockRagEngine());

      // Inject the mock instance before each test.
      // This prevents the code from trying to load the native Rust library.
      MobileRag.setMockInstance(mockMobileRag);
    });

    tearDown(() {
      // Clean up the singleton after tests.
      MobileRag.setMockInstance(null);
    });

    test('should return mocked search results without native engine', () async {
      // 1. Arrange: Define what the mock should return
      final expectedPrompt = "This is a mocked prompt for the LLM";

      when(
        () =>
            mockMobileRag.search(any(), tokenBudget: any(named: 'tokenBudget')),
      ).thenAnswer(
        (_) async => RagSearchResult(
          chunks: [],
          context: AssembledContext(
            text: "Mocked context data",
            estimatedTokens: 5,
            includedChunks: [],
            remainingBudget: 1000,
          ),
        ),
      );

      when(
        () => mockMobileRag.formatPrompt(any(), any()),
      ).thenReturn(expectedPrompt);

      // 2. Act: Call the code that uses MobileRag.instance
      // In a real app, this might be inside a ViewModel or Bloc.
      final result = await MobileRag.instance.search(
        "What is Flutter?",
        tokenBudget: 1000,
      );
      final prompt = MobileRag.instance.formatPrompt(
        "What is Flutter?",
        result,
      );

      // 3. Assert: Verify the results and that the mock was called correctly
      expect(prompt, equals(expectedPrompt));
      expect(result.context.text, contains("Mocked context"));

      verify(
        () => mockMobileRag.search("What is Flutter?", tokenBudget: 1000),
      ).called(1);
    });

    test('should handle engine errors gracefully using mock', () async {
      // Arrange: Simulate a database error
      when(
        () => mockMobileRag.addDocument(any()),
      ).thenThrow(StateError("Database is locked"));

      // Act & Assert
      expect(
        () => MobileRag.instance.addDocument("Some text"),
        throwsStateError,
      );
    });
  });
}

/// EXAMPLE: How to test a "Service" class that uses MobileRag
class MyAIService {
  Future<String> askAI(String question) async {
    // This code normally requires the Rust library,
    // but works in tests thanks to our mock injection.
    final searchResult = await MobileRag.instance.search(question);
    final prompt = MobileRag.instance.formatPrompt(question, searchResult);

    return "LLM Response based on: $prompt";
  }
}
