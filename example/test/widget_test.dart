// This smoke test verifies that the app can launch even if MobileRag isn't initialized
// (it should show an error message instead of crashing).

import 'package:flutter/material.dart';
import 'package:flutter_test/flutter_test.dart';

import 'package:mobile_rag_engine_example/main.dart';

void main() {
  testWidgets('MyApp smoke test - handles uninitialized state', (
    WidgetTester tester,
  ) async {
    // Build our app and trigger a frame.
    await tester.pumpWidget(const MyApp());

    // Verify that the app shows the uninitialized status message (from initState)
    // Note: Since we didn't call MobileRag.initialize() in main(), the app should detect this.
    expect(find.textContaining('MobileRag not initialzed'), findsOneWidget);

    // Verify basic UI structure exists (AppBar title)
    expect(find.text('🔍 Local RAG Engine'), findsOneWidget);

    // Verify buttons are disabled or present
    expect(find.byIcon(Icons.save), findsOneWidget);
  });
}
