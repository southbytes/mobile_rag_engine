#!/bin/bash
# scripts/build_android.sh
# Build Rust library for Android (jniLibs)

set -e

# Configuration
RUST_LIB_NAME="rust_lib_mobile_rag_engine"
SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
PROJECT_ROOT="$(dirname "$SCRIPT_DIR")"
RUST_DIR="$PROJECT_ROOT/rust"
OUTPUT_DIR="$PROJECT_ROOT/platform-build/android/jniLibs"

echo "▸ Building Android libraries"
echo "  Project: $PROJECT_ROOT"
echo "  Rust dir: $RUST_DIR"

# Check for NDK
if [ -z "$ANDROID_NDK_HOME" ]; then
    # Try common paths
    if [ -d "$HOME/Library/Android/sdk/ndk" ]; then
        # Find latest NDK version
        ANDROID_NDK_HOME=$(ls -d "$HOME/Library/Android/sdk/ndk/"* 2>/dev/null | sort -V | tail -1)
    elif [ -d "$HOME/Android/Sdk/ndk" ]; then
        ANDROID_NDK_HOME=$(ls -d "$HOME/Android/Sdk/ndk/"* 2>/dev/null | sort -V | tail -1)
    fi
fi

if [ -z "$ANDROID_NDK_HOME" ] || [ ! -d "$ANDROID_NDK_HOME" ]; then
    echo "Error: ANDROID_NDK_HOME not set or NDK not found"
    echo "Please install Android NDK and set ANDROID_NDK_HOME"
    exit 1
fi

echo "  NDK: $ANDROID_NDK_HOME"

# Detect host platform
case "$(uname -s)" in
    Darwin*)
        HOST_TAG="darwin-x86_64"
        ;;
    Linux*)
        HOST_TAG="linux-x86_64"
        ;;
    *)
        echo "Unsupported host platform"
        exit 1
        ;;
esac

TOOLCHAIN="$ANDROID_NDK_HOME/toolchains/llvm/prebuilt/$HOST_TAG"

# Create output directories
mkdir -p "$OUTPUT_DIR/arm64-v8a"
mkdir -p "$OUTPUT_DIR/armeabi-v7a"
mkdir -p "$OUTPUT_DIR/x86_64"
mkdir -p "$OUTPUT_DIR/x86"

cd "$RUST_DIR"

# Set up cargo config for Android
mkdir -p .cargo
cat > .cargo/config.toml << EOF
[target.aarch64-linux-android]
linker = "$TOOLCHAIN/bin/aarch64-linux-android24-clang"

[target.armv7-linux-androideabi]
linker = "$TOOLCHAIN/bin/armv7a-linux-androideabi24-clang"

[target.x86_64-linux-android]
linker = "$TOOLCHAIN/bin/x86_64-linux-android24-clang"

[target.i686-linux-android]
linker = "$TOOLCHAIN/bin/i686-linux-android24-clang"
EOF

# Build for all Android targets with explicit CC environment
echo ""
echo "▸ Building for aarch64-linux-android (arm64-v8a)..."
CC_aarch64_linux_android="$TOOLCHAIN/bin/aarch64-linux-android24-clang" \
AR_aarch64_linux_android="$TOOLCHAIN/bin/llvm-ar" \
cargo build --release --target aarch64-linux-android

echo ""
echo "▸ Building for armv7-linux-androideabi (armeabi-v7a)..."
CC_armv7_linux_androideabi="$TOOLCHAIN/bin/armv7a-linux-androideabi24-clang" \
AR_armv7_linux_androideabi="$TOOLCHAIN/bin/llvm-ar" \
cargo build --release --target armv7-linux-androideabi

echo ""
echo "▸ Building for x86_64-linux-android (x86_64)..."
CC_x86_64_linux_android="$TOOLCHAIN/bin/x86_64-linux-android24-clang" \
AR_x86_64_linux_android="$TOOLCHAIN/bin/llvm-ar" \
cargo build --release --target x86_64-linux-android

echo ""
echo "▸ Building for i686-linux-android (x86)..."
CC_i686_linux_android="$TOOLCHAIN/bin/i686-linux-android24-clang" \
AR_i686_linux_android="$TOOLCHAIN/bin/llvm-ar" \
cargo build --release --target i686-linux-android

# Copy libraries to jniLibs structure
echo ""
echo "▸ Copying libraries..."

cp "$RUST_DIR/target/aarch64-linux-android/release/lib${RUST_LIB_NAME}.so" \
   "$OUTPUT_DIR/arm64-v8a/lib${RUST_LIB_NAME}.so"

cp "$RUST_DIR/target/armv7-linux-androideabi/release/lib${RUST_LIB_NAME}.so" \
   "$OUTPUT_DIR/armeabi-v7a/lib${RUST_LIB_NAME}.so"

cp "$RUST_DIR/target/x86_64-linux-android/release/lib${RUST_LIB_NAME}.so" \
   "$OUTPUT_DIR/x86_64/lib${RUST_LIB_NAME}.so"

cp "$RUST_DIR/target/i686-linux-android/release/lib${RUST_LIB_NAME}.so" \
   "$OUTPUT_DIR/x86/lib${RUST_LIB_NAME}.so"

# Create archive
echo ""
echo "▸ Creating archive..."
cd "$PROJECT_ROOT/platform-build/android"
tar -czvf jniLibs.tar.gz jniLibs

echo ""
echo "✅ Android build complete!"
echo "   Output: $OUTPUT_DIR"
echo "   Archive: $PROJECT_ROOT/platform-build/android/jniLibs.tar.gz"

# Print sizes
echo ""
echo "▸ Library sizes:"
ls -lh "$OUTPUT_DIR/arm64-v8a/lib${RUST_LIB_NAME}.so" | awk '{print "   arm64-v8a: " $5}'
ls -lh "$OUTPUT_DIR/armeabi-v7a/lib${RUST_LIB_NAME}.so" | awk '{print "   armeabi-v7a: " $5}'
ls -lh "$OUTPUT_DIR/x86_64/lib${RUST_LIB_NAME}.so" | awk '{print "   x86_64: " $5}'
ls -lh "$OUTPUT_DIR/x86/lib${RUST_LIB_NAME}.so" | awk '{print "   x86: " $5}'
ls -lh "$PROJECT_ROOT/platform-build/android/jniLibs.tar.gz" | awk '{print "   Archive: " $5}'
