@echo off
REM 35B with -ngl 4, n-predict 2 (forces decode path), under sanitizer
set CUDA_LAUNCH_BLOCKING=1
set GGML_CUDA_DISABLE_GRAPHS=1
set SP_ENGINE_BACKEND=cuda
set SHANNON_PRIME_GPU_CACHE=0
"C:\Program Files\NVIDIA GPU Computing Toolkit\CUDA\v13.2\bin\compute-sanitizer.bat" --tool=memcheck --launch-timeout=120 --target-processes=all --print-limit=4 ^
  "C:\Projects\shannon-prime-burnhard\build_beast\bin\sp-engine.exe" chat ^
  --model "D:\Files\Models\lmstudio-community\Qwen3.6-35B-A3B-GGUF\Qwen3.6-35B-A3B-Q4_K_M.gguf" ^
  --n-gpu-layers 4 ^
  --n-predict 2 ^
  --furnace-dispatch=off ^
  --no-compression ^
  "Hi" 2>&1
echo === EXIT %ERRORLEVEL% ===
