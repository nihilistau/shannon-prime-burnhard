# BurnHard-only bench: skip CPU/CUDA baselines (already known), only measure
# the modes that exercise the BurnHard stack — Furnace, Full Pulse (sidecar),
# CRT split. Model is expected to be Optane-mounted.
#
# Usage:
#   powershell -File scripts\bench_burnhard_only.ps1 -Model "F:\<model>.gguf" -Predict 16

param(
    [string] $Model   = "F:\Qwen3.6-35B-A3B-Q4_K_M.gguf",
    [int]    $Predict = 16,
    [string] $Prompt  = "Question: What is the capital of France? Answer: The capital of France is",
    [string] $Bin     = "C:\Projects\shannon-prime-burnhard\build_beast\bin\sp-engine.exe"
)

$Beast = $Model
$stamp   = Get-Date -Format "yyyyMMdd-HHmmss"
$outdir  = "C:\Projects\shannon-prime-burnhard\bench\burnhard_$stamp"
New-Item -ItemType Directory -Force -Path $outdir | Out-Null
Write-Host "[bench] writing logs to $outdir"
Write-Host "[bench] model      = $Model"
Write-Host "[bench] predict    = $Predict"

function Run-Mode {
    param([string]$tag, [hashtable]$envv, [string[]]$extra)
    Write-Host ""
    Write-Host "=== $tag ==="
    foreach ($k in $envv.Keys) { Set-Item -Path "Env:$k" -Value $envv[$k] }
    $log = Join-Path $outdir "$tag.log"
    $argz = @("chat","--n-predict",$Predict,"--model",$Model) + $extra + @($Prompt)
    $sw = [System.Diagnostics.Stopwatch]::StartNew()
    & $Bin $argz 2>&1 | Tee-Object -FilePath $log | Out-Null
    $sw.Stop()
    $elapsed = $sw.Elapsed.TotalSeconds
    Write-Host ("  wall_time={0:F2}s" -f $elapsed)

    $kv_pos       = $null
    $reservoir_ms = $null
    $sidecar      = "n/a"
    $gpus         = 0
    $crt          = "off"
    $furnace      = "off"
    Get-Content $log | ForEach-Object {
        if ($_ -match "Total boot:\s+([\d.]+)\s+ms") { $reservoir_ms = [double]$Matches[1] }
        if ($_ -match "kv_pos=(\d+)") { $kv_pos = [int]$Matches[1] }
        if ($_ -match "Furnace ARMED via --beast.*GPUs=(\d+)\s+sidecar=(\w+)") {
            $furnace = "ON"; $gpus = [int]$Matches[1]; $sidecar = $Matches[2]
        }
        if ($_ -match "CRT multi-GPU enabled") { $crt = "ON" }
    }
    $tps = if ($kv_pos -and $kv_pos -gt 0) { [Math]::Round($kv_pos / $elapsed, 2) } else { $null }
    [PSCustomObject]@{
        Mode=$tag; WallSec=[Math]::Round($elapsed,2); KvPos=$kv_pos; TokPerSec=$tps;
        GPUs=$gpus; Furnace=$furnace; CRT=$crt; Sidecar=$sidecar; ReservoirMs=$reservoir_ms;
    }
}

$results = @()
$results += Run-Mode "furnace_solo" @{ "SP_BEAST_ENABLE_SIDECAR"=""; "SP_ENGINE_BACKEND"="cuda" } @("--beast",$Beast)
$results += Run-Mode "full_pulse"   @{ "SP_BEAST_ENABLE_SIDECAR"="1"; "SP_ENGINE_BACKEND"="cuda" } @("--beast",$Beast)
$results += Run-Mode "crt_split"    @{ "SP_BEAST_ENABLE_SIDECAR"=""; "SP_ENGINE_BACKEND"="cuda" } @("--beast",$Beast,"--crt-split","--n-gpus","2")

Write-Host ""
Write-Host "=== SUMMARY (BurnHard-only) ==="
$results | Format-Table Mode,WallSec,KvPos,TokPerSec,GPUs,Furnace,CRT,Sidecar,ReservoirMs -AutoSize
$results | ConvertTo-Json -Depth 4 | Out-File (Join-Path $outdir "summary.json")
Write-Host "[bench] summary written to $(Join-Path $outdir summary.json)"
