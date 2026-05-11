@echo off
set SP_ENGINE_BACKEND=cuda
set SP_BEAST_ENABLE_SIDECAR=1
echo Starting sp-engine at %TIME% > C:\Projects\shannon-prime-burnhard\bench\testC.log
"C:\Projects\shannon-prime-burnhard\build_beast\bin\sp-engine.exe" chat --beast "D:\Files\Models\lmstudio-community\Qwen3.6-35B-A3B-GGUF\Qwen3.6-35B-A3B-Q4_K_M.gguf" --n-gpu-layers 20 --n-predict 1 --model "D:\Files\Models\lmstudio-community\Qwen3.6-35B-A3B-GGUF\Qwen3.6-35B-A3B-Q4_K_M.gguf" "Hi" 1>>C:\Projects\shannon-prime-burnhard\bench\testC.log 2>&1
echo Exit code: %ERRORLEVEL% >> C:\Projects\shannon-prime-burnhard\bench\testC.log
echo Finished at %TIME% >> C:\Projects\shannon-prime-burnhard\bench\testC.log
