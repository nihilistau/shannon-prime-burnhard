@echo off
REM Beast (Optane F:) + top-K=4 + fp32 pre-stage + sidecar
set SP_ENGINE_BACKEND=cuda
set SP_BEAST_ENABLE_SIDECAR=1
adb forward tcp:9876 tcp:9876 2>nul
powershell -Command "$sw=[Diagnostics.Stopwatch]::StartNew(); & 'C:\Projects\shannon-prime-burnhard\build_beast\bin\sp-engine.exe' chat --model 'F:\Qwen3.6-35B-A3B-Q4_K_M.gguf' --beast 'F:\Qwen3.6-35B-A3B-Q4_K_M.gguf' --n-experts-used 4 --n-predict 16 'Once upon a time in a faraway kingdom, there was a wise old wizard who' 2>&1; $sw.Stop(); Write-Host ('=== WALL_MS=' + $sw.ElapsedMilliseconds)"
