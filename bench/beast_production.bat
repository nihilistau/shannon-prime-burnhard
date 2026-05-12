@echo off
REM Production path: --beast (boots layer-split policy + arms CRT) +
REM --furnace-dispatch=on (Furnace intercepts mul_mat_id for routed MoE).
REM Chat runs through the standard ForwardContext + ggml-cuda + MMQ path
REM since SP_BEAST_OWN_CHAT is NOT set.
set SP_ENGINE_BACKEND=cuda
set SP_BEAST_ENABLE_SIDECAR=0
adb forward tcp:9876 tcp:9876 2>nul
powershell -Command "$sw=[Diagnostics.Stopwatch]::StartNew(); & 'C:\Projects\shannon-prime-burnhard\build_beast\bin\sp-engine.exe' chat --model 'F:\Qwen3.6-35B-A3B-Q4_K_M.gguf' --beast 'F:\Qwen3.6-35B-A3B-Q4_K_M.gguf' --furnace-dispatch=on --n-experts-used 4 --n-predict 16 'Once upon a time in a faraway kingdom, there was a wise old wizard who' 2>&1; $sw.Stop(); Write-Host ('=== WALL_MS=' + $sw.ElapsedMilliseconds)"
