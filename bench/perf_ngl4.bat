@echo off
REM 35B baseline: -ngl 4, no debug flags, 64 tokens for clean tok/s
set SP_ENGINE_BACKEND=cuda
set SHANNON_PRIME_GPU_CACHE=0
"C:\Projects\shannon-prime-burnhard\build_beast\bin\sp-engine.exe" chat ^
  --model "D:\Files\Models\lmstudio-community\Qwen3.6-35B-A3B-GGUF\Qwen3.6-35B-A3B-Q4_K_M.gguf" ^
  --n-gpu-layers 4 ^
  --n-predict 64 ^
  --furnace-dispatch=off ^
  --no-compression ^
  "Once upon a time in a faraway kingdom" 2>&1
echo === EXIT %ERRORLEVEL% ===
