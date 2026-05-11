<#
.SYNOPSIS
Compiles and runs Shannon-Prime BurnHard targeting the Qwen3.6-35B model.
Hardware target: Beast Canyon (Optane + NVMe), RTX 2060, Intel UHD L0 SVM, S22 Sidecar.
#>

$ErrorActionPreference = "Stop"

Write-Host "[Build] Compiling BurnHard Engine..." -ForegroundColor Cyan
# Force MSVC compilers to prevent MinGW crossover and align with Beast Canyon flags
cmake -S . -B build -G "Ninja" -DCMAKE_BUILD_TYPE=Release -DCMAKE_C_COMPILER=cl -DCMAKE_CXX_COMPILER=cl -DSP_WITH_CUDA=ON -DSP_WITH_BEAST=ON -DSP_BEAST_CUDA=ON -DSP_BEAST_LEVEL_ZERO=ON
cmake --build build --config Release

Write-Host "[Run] Initializing Beast Canyon Orchestrator..." -ForegroundColor Green

# Set the runtime environment to reflect Beast Canyon + S22 limits
$env:SHANNON_PRIME_ENABLED=1
$env:SHANNON_PRIME_BACKEND="beast_canyon"
$env:SHANNON_PRIME_DRAFT_BACKEND="hexagon" # S22 Ultra Sidecar
$env:SHANNON_PRIME_DRAFT_PRESET="aggressive"

# Target your specific Qwen 3.6 35B Model
$ModelPath = "D:\Files\Models\lmstudio-community\Qwen3.6-35B-A3B-GGUF"

# We pass -md adb://s22_ultra to invoke the speculative draft model over the USB bridge
.\build\bin\sp-engine.exe generate `
    -m "$ModelPath\Qwen3.6-35B-A3B.gguf" `
    -md "adb://s22_ultra" `
    -c 4096 `
    -t 8