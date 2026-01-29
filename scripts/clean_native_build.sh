#!/bin/bash
set -e

echo "🧹 Starting Deep Clean for Mobile RAG Engine..."

# 1. Clean Flutter artifacts
echo "flutter clean..."
flutter clean

# 2. Clean macOS Pods and specific cache
if [ -d "example/macos" ]; then
    echo "removing example/macos/Pods..."
    rm -rf example/macos/Pods
    echo "removing example/macos/Podfile.lock..."
    rm -rf example/macos/Podfile.lock
    echo "removing example/macos/Flutter/ephemeral..."
    rm -rf example/macos/Flutter/ephemeral
fi

# 3. Clean iOS Pods (if present)
if [ -d "example/ios" ]; then
    echo "removing example/ios/Pods..."
    rm -rf example/ios/Pods
    echo "removing example/ios/Podfile.lock..."
    rm -rf example/ios/Podfile.lock
    rm -rf example/ios/Flutter/Flutter.framework
    rm -rf example/ios/Flutter/App.framework
fi

# 4. Hint about DerivedData (optional, but good to mention)
echo "💡 Tip: If issues persist, you may also need to clear Xcode DerivedData:"
echo "   rm -rf ~/Library/Developer/Xcode/DerivedData"

echo "✅ App build environment cleaned. Please run 'pod install' in ios/macos folder or just run flutter run."
