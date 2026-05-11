@echo off
set SP_ENGINE_BACKEND=cuda
powershell -Command "$sw=[Diagnostics.Stopwatch]::StartNew(); & 'C:\Projects\shannon-prime-burnhard\build_beast\bin\sp-engine.exe' chat --model 'D:\Files\Models\lmstudio-community\Qwen3.6-35B-A3B-GGUF\Qwen3.6-35B-A3B-Q4_K_M.gguf' --n-gpu-layers 19 --n-predict 8 --n-experts-used 4 --no-compression --no-gpu-cache 'Hi' 2>&1; $sw.Stop(); Write-Host ('=== WALL_MS=' + $sw.ElapsedMilliseconds)"
