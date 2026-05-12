@echo off
REM Pure ggml-cuda path with 20-layer offload — no Beast, no Furnace, no
REM Shannon-Prime tweaks. This is the "what does standard llama.cpp do
REM on this hardware" reference.  Hierarchical KV cache stays on (default
REM for CUDA backend) so the path matches the LM Studio + shannon-prime-
REM llama reference setup as closely as we can without the modded llama.dll.
set SP_ENGINE_BACKEND=cuda
powershell -Command "$sw=[Diagnostics.Stopwatch]::StartNew(); & 'C:\Projects\shannon-prime-burnhard\build_beast\bin\sp-engine.exe' chat --model 'F:\Qwen3.6-35B-A3B-Q4_K_M.gguf' -ngl 20 --n-experts-used 4 --n-predict 16 'Once upon a time in a faraway kingdom there was a wise old wizard who studied the stars and dreamed of magic' 2>&1; $sw.Stop(); Write-Host ('=== WALL_MS=' + $sw.ElapsedMilliseconds)"
