#!/data/data/com.termux/files/usr/bin/bash
# Shannon-Prime — Termux bootstrap for S22 Ultra (LONE_WOLF mode).
#
# Run from inside Termux. Installs build deps, fetches the QNN runtime
# libs, and stages a writable workspace at $HOME/sp-engine/.
# Building the engine itself happens via the standard NDK cross-build
# (see CLAUDE.md "Build → Android" section); this script gets the
# device side of the deploy ready.

set -euo pipefail

SP_HOME="${SP_HOME:-$HOME/sp-engine}"
SP_DEV_ROOT="${SP_DEV_ROOT:-/data/local/tmp/sp-engine}"
SP_QNN_ROOT="${SP_QNN_ROOT:-/data/local/tmp/sp22u/qnn}"

echo "[bootstrap] installing Termux packages..."
pkg update -y
pkg install -y clang ninja cmake git wget curl python rsync

echo "[bootstrap] preparing workspace at $SP_HOME"
mkdir -p "$SP_HOME"/{bin,www,models,qnn}

if [ ! -d "$SP_HOME/repo" ]; then
    echo "[bootstrap] cloning shannon-prime-burnhard..."
    git clone https://github.com/nihilistau/shannon-prime-burnhard.git \
        "$SP_HOME/repo"
else
    echo "[bootstrap] repo already cloned — pulling latest"
    git -C "$SP_HOME/repo" pull --ff-only
fi

# The NDK build produces sp-engine + the QNN runtime stubs. Termux has
# no Android NDK out of the box, so the typical workflow is to build on
# a desktop and rsync the binary in. Wire the convenience target here.
echo
echo "[bootstrap] DONE. Next steps:"
echo "  1. From your desktop with NDK r26+ installed, build the engine:"
echo "       (see CLAUDE.md Build/Android section, set -DSP_TARGET_MOBILE=ON)"
echo
echo "  2. Push the binary + QNN libs to this device:"
echo "       adb push build-android/bin/sp-engine $SP_DEV_ROOT/"
echo "       adb push build-android/bin/lib*.so   $SP_DEV_ROOT/"
echo "       adb push <repo>/frontend/            $SP_DEV_ROOT/www/"
echo "       adb push <model>.gguf                /data/local/tmp/sp22u/model.gguf"
echo "       adb push <qnn>/qwen3_4b_*.bin        $SP_QNN_ROOT/"
echo
echo "  3. Launch from adb shell (or from this Termux):"
echo "       sh $SP_HOME/repo/mobile/standalone/sp-mobile-launch.sh"
echo
echo "  4. Open http://<phone-ip>:8080 in any browser."
