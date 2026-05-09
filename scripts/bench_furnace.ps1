# Shannon-Prime BurnHard — Furnace stack benchmark
#
# Runs the chat verb across the four operating modes so we can compare
# tok/s, latency, and which subsystems engaged. Stores raw logs under
# bench/furnace_<timestamp>/ and prints a summary table at the end.
#
# Usage:  powershell -File scripts\bench_furnace.ps1 [-Model <gguf>] [-Predict 16]

param(
    [string] $Model   = "D:\Files\Models\New folder\Qwen3-0.6B-Q4_K_M.gguf",
    [string] $Beast   = $null,
    [int]    $Predict = 16,
    [string] $Prompt  = "Question: What is the capital of France? Answer: The capital of France is",
    [string] $Bin     = "C:\Projects\shannon-prime-burnhard\build_beast\bin\sp-engine.exe"
)

if (-not $Beast) { $Beast = $Model }

$stamp   = Get-Date -Format "yyyyMMdd-HHmmss"
$outdir  = "C:\Projects\shannon-prime-burnhard\bench\furnace_$stamp"
New-Item -ItemType Directory -Force -Path $outdir | Out-Null
Write-Host "[bench] writing logs to $outdir"

function Run-Mode {
    param(
        [string] $tag,
        [hashtable] $envv,
        [string[]] $extra
    )
    Write-Host ""
    Write-Host "=== $tag ==="
    foreach ($k in $envv.Keys) {
        Set-Item -Path "Env:$k" -Value $envv[$k]
    }
    $log = Join-Path $outdir "$tag.log"
    $args = @("chat","--n-predict",$Predict,"--model",$Model) + $extra + @($Prompt)
    $sw = [System.Diagnostics.Stopwatch]::StartNew()
    & $Bin $args 2>&1 | Tee-Object -FilePath $log | Out-Null
    $sw.Stop()
    $elapsed = $sw.Elapsed.TotalSeconds
    Write-Host ("  wall_time={0:F2}s  log={1}" -f $elapsed, (Split-Path $log -Leaf))

    # Parse interesting lines.
    $first_token_us = $null
    $tok_per_sec    = $null
    $kv_pos         = $null
    $reservoir_ms   = $null
    $sidecar_state  = "n/a"
    $gpus           = 0
    $crt            = "off"
    $furnace        = "off"
    Get-Content $log | ForEach-Object {
        if ($_ -match "Total boot:\s+([\d.]+)\s+ms") { $reservoir_ms = [double]$Matches[1] }
        if ($_ -match "kv_pos=(\d+)") { $kv_pos = [int]$Matches[1] }
        if ($_ -match "Furnace ARMED via --beast.*GPUs=(\d+)\s+sidecar=(\w+)") {
            $furnace = "ON"; $gpus = [int]$Matches[1]; $sidecar_state = $Matches[2]
        }
        if ($_ -match "CRT multi-GPU enabled") { $crt = "ON" }
        if ($_ -match "decode\(\) first call.*past_n=(\d+)") { $first_token_us = $sw.Elapsed.TotalMilliseconds * 1000 }
        if ($_ -match "GPU\[(\d+)\]") { if (([int]$Matches[1] + 1) -gt $gpus) { $gpus = ([int]$Matches[1] + 1) } }
    }
    if ($kv_pos -and $kv_pos -gt 0) {
        $tok_per_sec = [Math]::Round($kv_pos / $elapsed, 2)
    }

    [PSCustomObject]@{
        Mode         = $tag
        WallSec      = [Math]::Round($elapsed, 2)
        ReservoirMs  = $reservoir_ms
        KvPos        = $kv_pos
        TokPerSec    = $tok_per_sec
        GPUs         = $gpus
        Furnace      = $furnace
        CRT          = $crt
        Sidecar      = $sidecar_state
        Log          = (Split-Path $log -Leaf)
    }
}

$results = @()

# 1. BASELINE — no Beast, no sidecar, CPU-only backend.
$results += Run-Mode "baseline_cpu" `
    @{ "SP_BEAST_ENABLE_SIDECAR" = ""; "SP_ENGINE_BACKEND" = "cpu" } @()

# 2. DESKTOP_SOLO — CUDA backend, no Beast, no sidecar.
$results += Run-Mode "desktop_cuda" `
    @{ "SP_BEAST_ENABLE_SIDECAR" = ""; "SP_ENGINE_BACKEND" = "cuda" } @()

# 3. FURNACE without sidecar — Beast Canyon orchestrator armed,
#    sidecar disabled at boot (DESKTOP_SOLO mode for the Beast).
$results += Run-Mode "furnace_solo" `
    @{ "SP_BEAST_ENABLE_SIDECAR" = ""; "SP_ENGINE_BACKEND" = "cuda" } `
    @("--beast", $Beast)

# 4. FULL_PULSE — Beast + sidecar engaged. Requires phone listener
#    on tcp:9876 (set up before invoking).
$results += Run-Mode "full_pulse" `
    @{ "SP_BEAST_ENABLE_SIDECAR" = "1"; "SP_ENGINE_BACKEND" = "cuda" } `
    @("--beast", $Beast)

# 5. CRT split — Beast + crt-split (needs n_gpus >= 2).
$results += Run-Mode "crt_split" `
    @{ "SP_BEAST_ENABLE_SIDECAR" = ""; "SP_ENGINE_BACKEND" = "cuda" } `
    @("--beast", $Beast, "--crt-split", "--n-gpus", "2")

Write-Host ""
Write-Host "=== SUMMARY ==="
$results | Format-Table Mode, WallSec, KvPos, TokPerSec, GPUs, Furnace, CRT, Sidecar, ReservoirMs -AutoSize

# Persist as JSON next to the logs.
$results | ConvertTo-Json -Depth 4 | Out-File (Join-Path $outdir "summary.json")
Write-Host "[bench] summary written to $(Join-Path $outdir summary.json)"
