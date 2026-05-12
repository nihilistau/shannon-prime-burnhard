@echo off
set SP_ENGINE_BACKEND=cuda
set OMP_NUM_THREADS=16
powershell -Command "$sw=[Diagnostics.Stopwatch]::StartNew(); & 'C:\Projects\shannon-prime-burnhard\build_beast\bin\sp-engine.exe' chat --model 'F:\Qwen3.6-35B-A3B-Q4_K_M.gguf' --beast 'F:\Qwen3.6-35B-A3B-Q4_K_M.gguf' --n-predict 8 'The capital of France is' 2>&1; $sw.Stop(); Write-Host ('=== WALL_MS=' + $sw.ElapsedMilliseconds)"
