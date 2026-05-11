@echo off
set SP_ENGINE_BACKEND=cpu
set SHANNON_PRIME_GPU_CACHE=0
"C:\Projects\shannon-prime-burnhard\build_beast\bin\sp-engine.exe" chat ^
  --model "D:\Files\Models\lmstudio-community\Qwen3.6-35B-A3B-GGUF\Qwen3.6-35B-A3B-Q4_K_M.gguf" ^
  --n-gpu-layers 0 ^
  --n-predict 4 ^
  --furnace-dispatch=off ^
  --no-compression ^
  "Hi" 2>&1
echo === EXIT %ERRORLEVEL% ===
