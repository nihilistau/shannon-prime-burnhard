#!/usr/bin/env bash
# Shannon-Prime BurnHard — Linux component bench harness.
#
# Runs every CPU-only verify + smoke target (Shredder, CRT, engine kernels)
# and emits per-target rows to components.json. Mirrors the schema used by
# scripts/bench_furnace.ps1's summary.json so downstream consumers can stay
# platform-agnostic.
#
# Usage:
#   scripts/bench_components.sh [--build-dir build_linux] [--out bench/linux_<stamp>]
#
# The build directory is expected to contain ./bin/{shred_verify,
# residue_matmul_verify, burn_moe_verify, test_core, test_engine_smoke,
# test_sp_kernels, test_sp_quant_q5k, test_disk_cache_roundtrip,
# bench_disk_partial, sp-engine}.
#
# The script is intentionally tolerant: a missing or crashing target is
# recorded as a row with exit != 0, the harness continues, and the final
# exit code is 0 unless THE HARNESS itself fails. Per-target pass/fail is
# the consumer's call.

set -euo pipefail

BUILD_DIR="${BUILD_DIR:-build_linux}"
STAMP=$(date +%Y%m%d-%H%M%S)
OUT_DIR="${OUT_DIR:-bench/linux_${STAMP}}"
MODEL_PATH="${SP_ENGINE_TEST_MODEL:-}"

while [[ $# -gt 0 ]]; do
    case "$1" in
        --build-dir) BUILD_DIR="$2"; shift 2;;
        --out)       OUT_DIR="$2";   shift 2;;
        --model)     MODEL_PATH="$2"; shift 2;;
        -h|--help)
            sed -n '2,20p' "$0"; exit 0;;
        *) echo "unknown arg: $1" >&2; exit 2;;
    esac
done

mkdir -p "$OUT_DIR/components" "$OUT_DIR/cli_nogguf"
BIN="$BUILD_DIR/bin"

if [[ ! -x "$BIN/sp-engine" ]]; then
    echo "[bench_components] sp-engine not found at $BIN — build first." >&2
    exit 1
fi

# sp-engine needs a large stack; the CLI allocates several large stack
# arrays. 12.5 MB default on Linux is not enough — bump it for the session.
ulimit -s unlimited || true

# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

# components.json buffer — we hand-build it with jq for deterministic order.
COMPONENTS_JSON="$OUT_DIR/components.json"
echo "[" > "$COMPONENTS_JSON"
FIRST_ROW=1

emit_row() {
    local name="$1" tool="$2" exit_code="$3" wall_s="$4"
    local metric="${5:-}" value="${6:-}" log_rel="${7:-}"
    local comma
    if [[ $FIRST_ROW -eq 1 ]]; then comma=""; FIRST_ROW=0; else comma=","; fi
    cat >> "$COMPONENTS_JSON" <<EOF
${comma}
  {
    "name":   "${name}",
    "tool":   "${tool}",
    "exit":   ${exit_code},
    "wall_s": ${wall_s},
    "metric": "${metric}",
    "value":  ${value:-null},
    "log":    "${log_rel}"
  }
EOF
}

# run_and_time <name> <tool> <log-relpath> <metric-key> <regex> -- <cmd...>
# Runs the cmd, captures stdout+stderr to log, parses metric via regex
# (first numeric capture group), and appends a row.
run_and_time() {
    local name="$1" tool="$2" log_rel="$3" metric="$4" regex="$5"; shift 5
    [[ "$1" == "--" ]] && shift
    local log_path="$OUT_DIR/$log_rel"
    mkdir -p "$(dirname "$log_path")"
    local t0=$(date +%s.%N) ec=0
    if ! "$@" >"$log_path" 2>&1; then ec=$?; fi
    local t1=$(date +%s.%N)
    local wall=$(awk -v a=$t1 -v b=$t0 'BEGIN{printf "%.3f", a-b}')
    echo "exit=$ec wall=${wall}s" >> "$log_path"
    local value=""
    if [[ -n "$regex" ]]; then
        value=$(grep -oE "$regex" "$log_path" | head -1 | grep -oE '[0-9]+(\.[0-9]+)?' | head -1 || true)
    fi
    emit_row "$name" "$tool" "$ec" "$wall" "$metric" "${value:-null}" "$log_rel"
    printf "  %-26s [%s]  wall=%ss" "$name" "$([[ $ec -eq 0 ]] && echo OK || echo FAIL)" "$wall"
    [[ -n "$value" ]] && printf "  %s=%s" "$metric" "$value"
    printf "\n"
}

echo "=== Shannon-Prime BurnHard — Linux Component Bench ==="
echo "    build: $BUILD_DIR"
echo "    out:   $OUT_DIR"
echo

# ---------------------------------------------------------------------------
# Component verifies (CPU, no GGUF)
# ---------------------------------------------------------------------------
echo "[1/3] Component verifies"
run_and_time shred_verify           shred_verify          components/shred_verify.log \
    ns_per_elem "([0-9.]+) ns/element" -- "$BIN/shred_verify"

run_and_time residue_matmul_verify  residue_matmul_verify components/residue_matmul_verify.log \
    ms_at_2048  "\\[hidden\\][^=]*res=([0-9.]+)ms" -- "$BIN/residue_matmul_verify"

if [[ -n "$MODEL_PATH" && -f "$MODEL_PATH" ]]; then
    run_and_time burn_moe_verify    burn_moe_verify       components/burn_moe_verify.log \
        max_abs "max_abs=([0-9.e+-]+)" -- "$BIN/burn_moe_verify" "$MODEL_PATH" 0 0
else
    emit_row burn_moe_verify burn_moe_verify -2 0 fixture null components/burn_moe_verify.SKIPPED
    echo "  burn_moe_verify             [SKIP] no MoE GGUF (set --model or \$SP_ENGINE_TEST_MODEL)"
fi

# ---------------------------------------------------------------------------
# Engine smoke + kernel parity
# ---------------------------------------------------------------------------
echo
echo "[2/3] Engine smoke + kernel parity"
for t in test_core test_engine_smoke test_sp_kernels test_sp_quant_q5k \
         test_disk_cache_roundtrip test_gguf_loader bench_disk_partial; do
    if [[ -x "$BIN/$t" ]]; then
        run_and_time "$t" "$t" "components/${t}.log" "" "" -- "$BIN/$t"
    else
        emit_row "$t" "$t" -2 0 missing null "components/${t}.MISSING"
        echo "  $t  [MISSING]"
    fi
done

# ---------------------------------------------------------------------------
# CRT smoke sweep — math sanity at four matmul sizes (CPU reference).
# ---------------------------------------------------------------------------
echo
echo "[3/3] CRT smoke sweep"
for N in 64 256 512 1024; do
    run_and_time "crt_smoke_${N}" "sp-engine crt_smoke" \
        "cli_nogguf/crt_smoke_${N}.log" abs_err \
        "abs error: max=([0-9.e+-]+)" -- "$BIN/sp-engine" crt_smoke --dim "$N"
done

# ---------------------------------------------------------------------------
# Close JSON, print summary
# ---------------------------------------------------------------------------
echo "]" >> "$COMPONENTS_JSON"
echo
echo "=== Summary ==="
echo "    components.json: $COMPONENTS_JSON"
echo
jq -r '
    "  " + (.name | tostring | .[:28]) +
    "  " + (if .exit == 0 then "OK" elif .exit == -2 then "SKIP" else "FAIL" end) +
    "  wall=" + (.wall_s | tostring) + "s" +
    (if .value then "  " + .metric + "=" + (.value | tostring) else "" end)
' "$COMPONENTS_JSON" 2>/dev/null || cat "$COMPONENTS_JSON"

echo
echo "Done."
