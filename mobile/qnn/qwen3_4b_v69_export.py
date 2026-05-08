"""
Phase 2.4a — hijack qai-hub-models qwen3_4b export pipeline to target
V69 (Samsung Galaxy S22 Ultra 5G) with QNN_CONTEXT_BINARY runtime
instead of the default Genie+8-Elite-QRD path.

Strategy (per backends/qnn_aihub/artifacts/PHASE_2_4_ARCHITECTURE.md):
  - Reuse all the qai-hub-models heavy lifting: HuggingFace Qwen3-4B
    PyTorch download, AIMET w4a16 quantize, layer splitting via
    NUM_SPLITS=4 / NUM_LAYERS_PER_SPLIT=12, ONNX export per split.
  - Substitute our own (target, device) at the AI Hub submit step
    by overriding the supported_precision_runtimes dict that
    get_llm_parser uses to build the CLI argspec. With only
    QNN_CONTEXT_BINARY accepted, the wrapper produces 8 .bins
    runnable via our sp_qnn shim (libsp_qnn.so, commit 25df55e —
    35% faster than qnn-net-run).

Outputs:
  ~8 .bin files (4 PP + 4 TG splits × possibly multiple context lengths),
  each ~250-500 MB w4a16, written to qai-hub-models' default cache or
  the directory you set via --output-dir.

Run:
  py -3.10 qwen3_4b_v69_export.py \\
      --device "Samsung Galaxy S22 Ultra 5G" \\
      --target-runtime qnn_context_binary \\
      --precision w4a16 \\
      --context-length 2048 \\
      --skip-inferencing \\
      --skip-profiling

  (The --skip flags avoid the lab-side profile + inference verification
   steps which target-device-mismatch with V69. We just want the .bin.)

WARNING: First run downloads ~16 GB from HuggingFace (Qwen3-4B PyTorch
weights). Cached in $HF_HOME / ~/.cache/huggingface/. Subsequent runs
are local-only.

WARNING: Compile jobs to AI Hub: 4 PP + 4 TG = 8 jobs at minimum, more
if multiple context lengths. Each ~10-30 min. Stagger / use AI Hub's
queue handling -- the Python script polls.

LINUX REQUIRED (verified 2026-05-01): qai-hub-models for LLMs depends
on AIMET-ONNX which is Linux-only. Importing this script on Windows
prints:

    Quantized models require the AIMET-ONNX package, which is only
    supported on Linux.

The user has WSL Ubuntu-20.04 installed (run `wsl --list -v` to
confirm). To execute this script:

    wsl -d Ubuntu-20.04
    # Inside WSL:
    cd /mnt/d/F/shannon-prime-repos/shannon-prime/backends/qnn_aihub
    pip install --user qai-hub-models[qwen3-4b] aimet-onnx
    cp ~/.qai_hub/client.ini ~/.qai_hub/client.ini   # or copy from Windows
    python3 qwen3_4b_v69_export.py --device "Samsung Galaxy S22 Ultra 5G" \\
        --target-runtime qnn_context_binary --precision w4a16 \\
        --context-length 2048 --skip-inferencing --skip-profiling

The /mnt/d path is shared with Windows so the resulting .bin files
land in the same workspace and can be adb-pushed from either side.
"""
from __future__ import annotations

import sys

from qai_hub_models.models._shared.llm.export import export_main
from qai_hub_models.models.common import Precision, TargetRuntime
from qai_hub_models.models.qwen3_4b import (
    MODEL_ID,
    FP_Model,
    Model,
    PositionProcessor,
)
from qai_hub_models.models.qwen3_4b.model import (
    DEFAULT_PRECISION,
    MODEL_ASSET_VERSION,
    NUM_LAYERS_PER_SPLIT,
    NUM_SPLITS,
)

# NOTE 2026-05-01: the QNN_CONTEXT_BINARY override below DOES NOT WORK.
# qai_hub_models has GENIE hardcoded in multiple places beyond the
# argparse layer (LLM_AIMETOnnx.get_hub_compile_options raises RuntimeError,
# export.py declares VALID_TARGET_RUNTIMES = Literal[TargetRuntime.GENIE],
# prepare_genie_assets is the bundling step). A partial run on this script
# crashed at submit_compile_job with:
#   RuntimeError: Unsupported target_runtime provided: TargetRuntime.QNN_CONTEXT_BINARY.
#   Only Genie runtime is supported.
# Cache state preserved at /home/claude/.qaihm/.../qwen34_w4a16_adascale/
# (PP ONNX, model.data, encodings) — not lost.
# Strategy needs rethinking before next run; the dict-override hijack is
# too shallow.


# === The two strategic overrides ==========================================

# 1. Make QNN_CONTEXT_BINARY the only valid runtime for this export.
#    The argparse parser built from this dict will reject anything else.
#    GENIE is removed → no Qualcomm-managed runtime wrapping → raw .bins.
SUPPORTED_PRECISION_RUNTIMES = {
    Precision.w4a16: [TargetRuntime.QNN_CONTEXT_BINARY],
    # Keeping w8a16 as a fallback in case w4a16 fails op-coverage on V69.
    Precision.w8a16: [TargetRuntime.QNN_CONTEXT_BINARY],
}

# 2. Default device pinned to our actual hardware. Can be overridden
#    via --device on the CLI.
DEFAULT_EXPORT_DEVICE = "Samsung Galaxy S22 Ultra 5G"


def main() -> None:
    # NOTE: ASCII-only output here — Windows cmd.exe defaults to cp1252
    # which can't encode arrows or em-dashes. ASCII keeps it portable
    # across the Windows-WSL boundary we'll cross when running for real.
    print("=== Shannon-Prime Phase 2.4a -- Qwen3-4B -> V69 export hijack ===")
    print(f"  model_id         : {MODEL_ID}")
    print(f"  asset_version    : {MODEL_ASSET_VERSION}")
    print(f"  num_splits       : {NUM_SPLITS}")
    print(f"  layers/split     : {NUM_LAYERS_PER_SPLIT}")
    print(f"  total_layers     : {NUM_SPLITS * NUM_LAYERS_PER_SPLIT}")
    print(f"  default_device   : {DEFAULT_EXPORT_DEVICE}")
    print(f"  default_precision: {DEFAULT_PRECISION}")
    print(f"  target_runtime   : QNN_CONTEXT_BINARY (overrides Genie)")
    print()

    export_main(
        MODEL_ID,
        MODEL_ASSET_VERSION,
        SUPPORTED_PRECISION_RUNTIMES,
        NUM_SPLITS,
        NUM_LAYERS_PER_SPLIT,
        Model,
        FP_Model,
        PositionProcessor,
        DEFAULT_EXPORT_DEVICE,
        DEFAULT_PRECISION,
        # Start with seq=2048 only; can extend to [512, 1024, 2048, 3072, 4096]
        # once one context length validates end-to-end.
        default_context_lengths=[2048],
    )


if __name__ == "__main__":
    main()
