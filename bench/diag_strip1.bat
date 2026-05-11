@echo off
REM Strip 1: bare ggml, no BURN_MOE, no compression, no SP cache, -ngl 4
set CUDA_LAUNCH_BLOCKING=1
set GGML_CUDA_DISABLE_GRAPHS=1
set SP_ENGINE_BACKEND=cuda
set SHANNON_PRIME_GPU_CACHE=0
set SP_GDN_DIAG=1
"C:\Projects\shannon-prime-burnhard\build_beast\bin\sp-engine.exe" chat ^
  --model "D:\Files\Models\lmstudio-community\Qwen3.6-35B-A3B-GGUF\Qwen3.6-35B-A3B-Q4_K_M.gguf" ^
  --n-gpu-layers 4 ^
  --n-predict 1 ^
  --furnace-dispatch=off ^
  --no-compression ^
  "The capital of France is" 2>&1
echo === EXIT %ERRORLEVEL% ===
