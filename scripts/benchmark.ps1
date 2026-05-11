$prompt = "The quick brown fox jumps over the lazy dog. Once upon a time in a faraway kingdom, there was a wise old wizard who possessed the secret of the universe, hidden in a shiny crystal orb."
$tokens = 50
$exe = "C:\Projects\shannon-prime-burnhard\build\bin\sp-engine.exe"
$model = "C:\Users\Knack\Downloads\Qwen2.5-Coder-3B-Q3_K_M.gguf"
$env:SP_ENGINE_BACKEND="cuda"
$sw = [System.Diagnostics.Stopwatch]::StartNew()
& $exe chat --model $model --n-gpu-layers 99 --n-predict $tokens $prompt *>&1
$sw.Stop()
$ms = $sw.ElapsedMilliseconds
$sec = $ms / 1000
$tok_sec = if ($sec -gt 0) { $tokens / $sec } else { 0 }
$lat_ms = if ($tokens -gt 0) { $ms / $tokens } else { 0 }

Write-Host "`n=== BENCHMARK RESULTS ==="
Write-Host "Tokens Generated: $tokens"
Write-Host "Total Time (incl. init & prefill): $ms ms"
Write-Host "Average Latency: $lat_ms ms/tok"
Write-Host "Speed: $tok_sec tok/sec"
