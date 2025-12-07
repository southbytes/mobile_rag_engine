#!/bin/bash
# scripts/build_ios.sh
# Build Rust library for iOS (XCFramework)

set -e

# Configuration
RUST_LIB_NAME="rust_lib_mobile_rag_engine"
SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
PROJECT_ROOT="$(dirname "$SCRIPT_DIR")"
RUST_DIR="$PROJECT_ROOT/rust"
OUTPUT_DIR="$PROJECT_ROOT/platform-build/ios"

echo "▸ Building iOS XCFramework"
echo "  Project: $PROJECT_ROOT"
echo "  Rust dir: $RUST_DIR"

# Create output directory
mkdir -p "$OUTPUT_DIR"

cd "$RUST_DIR"

# Build for all iOS targets
echo ""
echo "▸ Building for aarch64-apple-ios (iOS device)..."
cargo build --release --target aarch64-apple-ios

echo ""
echo "▸ Building for aarch64-apple-ios-sim (iOS simulator - Apple Silicon)..."
cargo build --release --target aarch64-apple-ios-sim

echo ""
echo "▸ Building for x86_64-apple-ios (iOS simulator - Intel)..."
cargo build --release --target x86_64-apple-ios

# Get library paths
DEVICE_LIB="$RUST_DIR/target/aarch64-apple-ios/release/lib${RUST_LIB_NAME}.a"
SIM_ARM_LIB="$RUST_DIR/target/aarch64-apple-ios-sim/release/lib${RUST_LIB_NAME}.a"
SIM_X86_LIB="$RUST_DIR/target/x86_64-apple-ios/release/lib${RUST_LIB_NAME}.a"

# Verify libraries exist
for lib in "$DEVICE_LIB" "$SIM_ARM_LIB" "$SIM_X86_LIB"; do
    if [ ! -f "$lib" ]; then
        echo "Error: Library not found: $lib"
        exit 1
    fi
done

echo ""
echo "▸ Creating fat library for simulator..."

# Create fat library for simulator (arm64 + x86_64)
SIM_FAT_LIB="$OUTPUT_DIR/lib${RUST_LIB_NAME}_sim.a"
lipo -create "$SIM_ARM_LIB" "$SIM_X86_LIB" -output "$SIM_FAT_LIB"

echo ""
echo "▸ Creating XCFramework..."

# Remove old XCFramework if exists
XCFRAMEWORK_PATH="$OUTPUT_DIR/${RUST_LIB_NAME}.xcframework"
rm -rf "$XCFRAMEWORK_PATH"

# Create XCFramework
xcodebuild -create-xcframework \
    -library "$DEVICE_LIB" \
    -library "$SIM_FAT_LIB" \
    -output "$XCFRAMEWORK_PATH"

echo ""
echo "▸ Compressing XCFramework..."

cd "$OUTPUT_DIR"
zip -r "${RUST_LIB_NAME}.xcframework.zip" "${RUST_LIB_NAME}.xcframework"

echo ""
echo "✅ iOS build complete!"
echo "   XCFramework: $XCFRAMEWORK_PATH"
echo "   Archive: $OUTPUT_DIR/${RUST_LIB_NAME}.xcframework.zip"

# Print sizes
echo ""
echo "▸ Library sizes:"
ls -lh "$DEVICE_LIB" | awk '{print "   Device (arm64): " $5}'
ls -lh "$SIM_FAT_LIB" | awk '{print "   Simulator (fat): " $5}'
ls -lh "$OUTPUT_DIR/${RUST_LIB_NAME}.xcframework.zip" | awk '{print "   XCFramework.zip: " $5}'
