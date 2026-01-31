#!/bin/bash
# Deep clean script for mobile_rag_engine Flutter+Rust project
# This script removes ALL cached native artifacts to force a complete rebuild
# Run this when you get "Content hash mismatch" errors

set -e

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
PROJECT_ROOT="$(cd "$SCRIPT_DIR/.." && pwd)"

echo "🧹 Deep Clean for Mobile RAG Engine"
echo "Project root: $PROJECT_ROOT"
echo ""

cd "$PROJECT_ROOT"

# 1. Clean example build artifacts only (NOT flutter clean from root - it breaks Runner)
echo "1️⃣ Cleaning example build artifacts..."
rm -rf example/build
rm -rf example/.dart_tool

# 2. Clean macOS Pods and caches (preserve Runner.xcodeproj)
if [ -d "example/macos" ]; then
    echo "2️⃣ Cleaning macOS pods..."
    rm -rf example/macos/Pods
    rm -rf example/macos/Podfile.lock
    rm -rf example/macos/Flutter/ephemeral
    # NOTE: Do NOT delete Runner.xcworkspace or Runner.xcodeproj
fi

# 3. Clean iOS Pods (if present)
if [ -d "example/ios" ]; then
    echo "3️⃣ Cleaning iOS pods..."
    rm -rf example/ios/Pods
    rm -rf example/ios/Podfile.lock
    rm -rf example/ios/Flutter/Flutter.framework
    rm -rf example/ios/Flutter/App.framework
fi

# 4. Clean Rust target (forces recompile with new hash)
echo "4️⃣ Cleaning Rust target..."
rm -rf rust_builder/rust/target

# 5. Clean Xcode DerivedData for this project
echo "5️⃣ Cleaning Xcode DerivedData..."
rm -rf ~/Library/Developer/Xcode/DerivedData/Runner-*
rm -rf ~/Library/Developer/Xcode/DerivedData/mobile_rag_engine*

# 6. Clean root level dart artifacts
echo "6️⃣ Cleaning root Dart artifacts..."
rm -rf .dart_tool
rm -rf build

echo ""
echo "✅ Clean complete!"
echo ""
echo "📌 Next steps:"
echo "   1. Make sure pubspec.yaml uses path dependency:"
echo "      rag_engine_flutter:"
echo "        path: rust_builder"
echo ""
echo "   2. If you modified rust_builder/rust/src/api/**, run:"
echo "      flutter_rust_bridge_codegen generate"
echo ""
echo "   3. Run the app (first build may fail, run again if it does):"
echo "      cd example && flutter run -d macos"
echo ""
echo "   ⚠️  NOTE: First build after clean may fail because Xcode"
echo "       hasn't compiled the Rust library yet. Just run again."
