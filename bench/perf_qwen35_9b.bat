@echo off
set SP_ENGINE_BACKEND=cuda
set SHANNON_PRIME_GPU_CACHE=0
set SP_N_EXPERTS_USED=4
powershell -Command "$sw=[Diagnostics.Stopwatch]::StartNew(); & 'C:\Projects\shannon-prime-burnhard\build_beast\bin\sp-engine.exe' chat --model 'D:\Files\Models\lmstudio-community\Qwen3.6-35B-A3B-GGUF\Qwen3.6-35B-A3B-Q4_K_M.gguf' --n-gpu-layers 19 --n-predict 32 --furnace-dispatch=off --no-compression 'The quick brown fox jumps over the lazy dog. Once upon a time in a faraway kingdom, there was a wise old wizard who' 2>&1; $sw.Stop(); Write-Host ('=== WALL_MS=' + $sw.ElapsedMilliseconds)"
