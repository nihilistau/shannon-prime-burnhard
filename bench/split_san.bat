@echo off
set CUDA_LAUNCH_BLOCKING=1
set GGML_CUDA_DISABLE_GRAPHS=1
set SP_ENGINE_BACKEND=cuda
"C:\Program Files\NVIDIA GPU Computing Toolkit\CUDA\v13.2\bin\compute-sanitizer.bat" --tool=memcheck --launch-timeout=120 --target-processes=all --print-limit=4 ^
  "C:\Projects\shannon-prime-burnhard\build_beast\bin\sp-engine.exe" chat ^
  --model "D:\Files\Models\lmstudio-community\Qwen3.6-35B-A3B-GGUF\Qwen3.6-35B-A3B-Q4_K_M.gguf" ^
  --n-gpu-layers 19 --n-predict 1 ^
  --n-experts-used 4 --experts-on-cpu 128 ^
  --no-compression --no-gpu-cache ^
  "Hi" 2>&1
echo === EXIT %ERRORLEVEL% ===
