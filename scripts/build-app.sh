#!/bin/bash
# Builds dist/SoL-Voxt.app and dist/SoL-Voxt.zip:
#   1. engines/asr        -> asr_engine + asr.metallib
#   2. engines/llm        -> voxt-llm-engine + mlx-swift_Cmlx.bundle
#   3. adapter            -> qwen3_asr_server (Voxt's server interface in front of asr_engine)
#   4. SGLang-Omni's native runtime (Silero VAD, Sortformer and Voxt's other local ASR models),
#      built from sgl-project/sglang-omni at the commit Voxt/ was taken from
#   5. Voxt (Release, Config/Fast.xcconfig), then everything above copied into its Resources
#      and the bundle signed ad hoc.
# Requirements: Apple Silicon, Xcode with the Metal toolchain, uv (for step 4), network access
# on the first build. Options (environment):
#   OMNI_RUNTIME=<dir>      use an existing SGLang-Omni runtime build (its bin/ directory's parent)
#   LLM_SOURCE_PACKAGES=<dir>, APP_SOURCE_PACKAGES=<dir>
#                           build offline from existing SwiftPM checkouts
set -euo pipefail
root="$(cd "$(dirname "$0")/.." && pwd)"
cd "$root"
export DEVELOPER_DIR="${DEVELOPER_DIR:-/Applications/Xcode.app/Contents/Developer}"
SGLANG_OMNI_COMMIT=8b81cda35
build="$root/build"
mkdir -p "$build"

echo "== asr engine"
engines/asr/build.sh

echo "== llm engine"
SOURCE_PACKAGES="${LLM_SOURCE_PACKAGES:-}" engines/llm/scripts/build.sh

echo "== adapter"
xcrun swiftc -O -swift-version 5 -target arm64-apple-macos15.0 adapter/main.swift \
    -o "$build/qwen3_asr_server"

echo "== sglang-omni runtime"
runtime="${OMNI_RUNTIME:-$build/omni-runtime}"
if [ ! -x "$runtime/bin/qwen3_asr_server" ]; then
    src="$build/sglang-omni"
    if [ ! -d "$src/.git" ]; then
        git init -q "$src"
        git -C "$src" fetch -q --depth 1 https://github.com/sgl-project/sglang-omni "$SGLANG_OMNI_COMMIT"
        git -C "$src" checkout -q FETCH_HEAD
    fi
    "$src/sglang_omni_mlx/native/scripts/build_runtime.sh" "$runtime"
fi

echo "== voxt"
offline=()
if [ -n "${APP_SOURCE_PACKAGES:-}" ]; then
    if [ ! -d "$build/app-packages" ]; then
        cp -c -R "$APP_SOURCE_PACKAGES" "$build/app-packages"
        chmod -R u+w "$build/app-packages"
    fi
    offline=(-clonedSourcePackagesDirPath "$build/app-packages" -disableAutomaticPackageResolution)
fi
if ! nice -n 10 xcodebuild build \
    -project Voxt/Voxt.xcodeproj \
    -scheme Voxt \
    -configuration Release \
    -destination 'platform=macOS,arch=arm64' \
    -xcconfig Voxt/Config/Fast.xcconfig \
    -derivedDataPath "$build/app" \
    -skipPackagePluginValidation \
    -skipMacroValidation \
    "${offline[@]}" > "$build/app.log" 2>&1; then
    grep -E 'error:' "$build/app.log" | head -40
    echo "Voxt build failed; full log in $build/app.log" >&2
    exit 1
fi

echo "== assemble"
app="$root/dist/SoL-Voxt.app"
rm -rf "$root/dist"
mkdir -p "$root/dist"
ditto "$build/app/Build/Products/Release/SoL-Voxt.app" "$app"
resources="$app/Contents/Resources"
bin="$resources/OmniRuntime/bin"
mkdir -p "$bin" "$resources/FastLLM"
ditto "$runtime/bin" "$bin"
mv "$bin/qwen3_asr_server" "$bin/qwen3_asr_server.sglang"
cp "$build/qwen3_asr_server" engines/asr/build/bin/asr_engine engines/asr/build/bin/asr.metallib "$bin/"
cp engines/llm/build/bin/voxt-llm-engine "$resources/FastLLM/"
ditto engines/llm/build/bin/mlx-swift_Cmlx.bundle "$resources/FastLLM/mlx-swift_Cmlx.bundle"

# The added executables need their own (ad hoc) signatures, and the added resources invalidate
# the app's resource seal: sign the helpers, then the app with its entitlements. The app is not
# notarized, so it is signed without the hardened runtime, whose library validation would refuse
# the embedded frameworks of an ad hoc signed app.
for exe in "$bin"/* "$resources/FastLLM/voxt-llm-engine"; do
    if [ -f "$exe" ] && file "$exe" | grep -q 'Mach-O'; then
        codesign --force --sign - --timestamp=none "$exe"
    fi
done
codesign --force --sign - --timestamp=none --preserve-metadata=entitlements "$app"
codesign --verify --deep --strict "$app"

ditto -c -k --keepParent "$app" "$root/dist/SoL-Voxt.zip"
echo "built $app"
ls -lh "$root/dist/SoL-Voxt.zip"
