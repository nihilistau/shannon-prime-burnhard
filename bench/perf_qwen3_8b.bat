@echo off
set SP_ENGINE_BACKEND=cuda
"C:\Projects\shannon-prime-burnhard\build_beast\bin\sp-engine.exe" chat ^
  --model "D:\Files\Models\lmstudio-community\Qwen3-8B-GGUF\Qwen3-8B-Q8_0.gguf" ^
  --n-gpu-layers 999 ^
  --n-predict 32 ^
  "Once upon a time" 2>&1
echo === EXIT %ERRORLEVEL% ===
