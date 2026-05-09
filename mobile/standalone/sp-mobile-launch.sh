#!/system/bin/sh
# Shannon-Prime — LONE_WOLF launch script for S22 Ultra (Snapdragon 8 Gen 1).
#
# Wraps `sp-engine serve` with the canonical on-device configuration
# documented in CLAUDE.md:
#   - QNN HTP backend (Qwen3-4B w4a16 splits)
#   - HTTP server bound to 0.0.0.0:8080
#   - Frontend served from /data/local/tmp/sp-engine/www/
#
# Invoke under adb shell after sp-engine + model + QNN splits are pushed.

set -eu

SP_ROOT="${SP_ROOT:-/data/local/tmp/sp-engine}"
SP_MODEL="${SP_MODEL:-/data/local/tmp/sp22u/model.gguf}"
SP_HOST="${SP_HOST:-0.0.0.0}"
SP_PORT="${SP_PORT:-8080}"

if [ ! -x "$SP_ROOT/sp-engine" ]; then
    echo "[lone_wolf] error: $SP_ROOT/sp-engine not found or not executable" >&2
    exit 1
fi
if [ ! -f "$SP_MODEL" ]; then
    echo "[lone_wolf] error: model file not found at $SP_MODEL" >&2
    exit 1
fi

export LD_LIBRARY_PATH="$SP_ROOT:/data/local/tmp/sp22u/qnn:${LD_LIBRARY_PATH:-}"
export ADSP_LIBRARY_PATH='/data/local/tmp/sp-engine;/vendor/lib/rfsa/adsp'
# Force the run-mode dispatcher into LONE_WOLF — the build already defines
# SP_TARGET_MOBILE, but the override is a useful belt-and-braces signal
# when running a desktop build on the device for testing.
export SP_RUN_MODE=LONE_WOLF

echo "[lone_wolf] starting sp-engine serve --model $SP_MODEL --host $SP_HOST --port $SP_PORT"
exec "$SP_ROOT/sp-engine" serve \
    --model "$SP_MODEL" \
    --host  "$SP_HOST" \
    --port  "$SP_PORT" \
    --www   "$SP_ROOT/www"
