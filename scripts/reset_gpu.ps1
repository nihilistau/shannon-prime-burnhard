# Shannon-Prime BurnHard — GPU device reset utility.
#
# When the NVIDIA driver gets wedged (chat hangs after [sp-mode] line,
# nvidia-smi blocks, or sp-engine.exe stalls in CUDA backend init), this
# disables-and-re-enables the GPU PnP device, dropping the wedged CUDA
# context without a full Windows reboot. Takes ~10 seconds.
#
# Requires Administrator (pnputil /restart-device needs elevation).
# If launched non-elevated, self-elevates via Start-Process -Verb RunAs.
#
# Usage:
#   powershell -ExecutionPolicy Bypass -File scripts\reset_gpu.ps1
#   powershell -ExecutionPolicy Bypass -File scripts\reset_gpu.ps1 -DeviceMatch "UHD 750"

param(
    [string] $DeviceMatch = "RTX 2060"
)

# Self-elevate if not running as admin.
$current = [Security.Principal.WindowsPrincipal] [Security.Principal.WindowsIdentity]::GetCurrent()
if (-not $current.IsInRole([Security.Principal.WindowsBuiltInRole]::Administrator)) {
    Write-Host "[reset_gpu] not running elevated — relaunching as Administrator"
    Start-Process powershell.exe -Verb RunAs -ArgumentList @(
        "-ExecutionPolicy", "Bypass",
        "-File", $PSCommandPath,
        "-DeviceMatch", $DeviceMatch
    )
    exit 0
}

Write-Host "[reset_gpu] looking for display device matching '$DeviceMatch'..."
$gpu = Get-PnpDevice -Class Display -ErrorAction SilentlyContinue |
    Where-Object { $_.FriendlyName -match $DeviceMatch }

if (-not $gpu) {
    Write-Host "[reset_gpu] no matching device found." -ForegroundColor Red
    Write-Host "  available display devices:"
    Get-PnpDevice -Class Display | Select-Object FriendlyName, Status |
        Format-Table -AutoSize | Out-Host
    exit 1
}

Write-Host "[reset_gpu] target: $($gpu.FriendlyName)  ($($gpu.InstanceId))"
Write-Host "[reset_gpu] current status: $($gpu.Status)"

Write-Host "[reset_gpu] issuing pnputil /restart-device ..."
$result = & pnputil /restart-device "$($gpu.InstanceId)" 2>&1
$result | ForEach-Object { Write-Host "  $_" }

Start-Sleep -Seconds 3
$gpuAfter = Get-PnpDevice -InstanceId $gpu.InstanceId
Write-Host "[reset_gpu] post-reset status: $($gpuAfter.Status)"

Write-Host "[reset_gpu] verifying CUDA visibility ..."
$smi = & nvidia-smi --query-gpu=name,memory.free --format=csv,noheader 2>&1
Write-Host "  $smi"

if ($LASTEXITCODE -eq 0) {
    Write-Host "[reset_gpu] CUDA driver responsive — chat path should now work" -ForegroundColor Green
    exit 0
} else {
    Write-Host "[reset_gpu] nvidia-smi still wedged — recommend full reboot" -ForegroundColor Yellow
    exit 2
}
