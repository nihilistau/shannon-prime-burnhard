<#
.SYNOPSIS
  Build a Qualcomm Hexagon SDK Halide standalone-device example on Windows.

.DESCRIPTION
  Drives the Windows-native build for any example under
    <SDK>/Halide/Examples/standalone/device/apps/
  using qaic.exe + cl.exe + hexagon-clang + Android NDK clang directly,
  mirroring Qualcomm's test-<TEST>.cmd pattern.

  Bypasses the example's Linux Makefile, which doesn't work on Windows:
   - $(HALIDE_ROOT)/lib/libHalide.a doesn't exist on Windows installs
     (it's Halide.lib, an MSVC import library).
   - $(shell pwd) returns C:\... with backslashes that mix with the
     literal forward-slash path components in Makefile.common.

  Same script works for any example following the standard layout:
    <dir>/rpc/<TEST>.idl
    <dir>/halide/<TEST>_generator.cpp
    <dir>/dsp/<TEST>_run.cpp  +  (optional) print.cpp  +  other .cpp
    <dir>/host/main.cpp

  When Qualcomm ships new examples, point this script at them — no
  per-example wrappers to maintain.

.PARAMETER ExampleDir
  Path to the Halide example directory. Must live under the SDK examples
  tree (the script uses ../../utils relative paths the SDK provides).

.PARAMETER Q6Version
  DSP arch version. Default 66.

.PARAMETER HlosTarget
  Host-side build target. "android" (default) cross-compiles aarch64 with
  the NDK; "skip" omits the host build entirely (DSP skel + sim only).

.PARAMETER Clean
  Remove bin/ before building.

.PARAMETER Run
  After build, adb push and run the resulting main-<TEST>.out on a
  connected device. Requires ADB on PATH.

.EXAMPLE
  # First, source SDK env + vcvars64 in the same shell:
  cmd /c "C:\Qualcomm\Hexagon_SDK\5.5.6.0\setup_sdk_env.cmd && `
          ""C:\Program Files (x86)\Microsoft Visual Studio\2019\BuildTools\VC\Auxiliary\Build\vcvars64.bat"" && `
          powershell -File .\build-example.ps1 -ExampleDir C:\Qualcomm\HALIDE_Tools\2.4.07\Halide\Examples\standalone\device\apps\sp_dma_raw"

.EXAMPLE
  .\build-example.ps1 -ExampleDir <path> -Clean

.EXAMPLE
  # DSP skel only (no host cross-compile):
  .\build-example.ps1 -ExampleDir <path> -HlosTarget skip
#>

[CmdletBinding()]
param(
    [Parameter(Mandatory=$true, Position=0)]
    [string]$ExampleDir,

    [int]$Q6Version = 66,

    [ValidateSet("android","skip")]
    [string]$HlosTarget = "android",

    [switch]$Clean,
    [switch]$Run
)

$ErrorActionPreference = "Stop"

function Step($n, $msg) {
    Write-Host ""
    Write-Host "[$n] $msg" -ForegroundColor Cyan
}

function Fail($msg) {
    Write-Host ""
    Write-Host "[ERROR] $msg" -ForegroundColor Red
    exit 1
}

function Invoke-Tool {
    param([string]$Exe, [string[]]$ArgList, [string]$Label)
    Write-Host "  > $Exe $($ArgList -join ' ')" -ForegroundColor DarkGray
    & $Exe @ArgList
    if ($LASTEXITCODE -ne 0) { Fail "$Label failed (exit $LASTEXITCODE)" }
}

# ---- 1. Validate environment --------------------------------------------------

if (-not $env:HEXAGON_SDK_ROOT) {
    Fail "HEXAGON_SDK_ROOT not set. Run setup_sdk_env.cmd in the same shell first."
}
if (-not $env:HALIDE_ROOT) {
    Fail "HALIDE_ROOT not set. Run setup_sdk_env.cmd first (it sets HALIDE_ROOT)."
}
if ($HlosTarget -eq "android" -and -not $env:ANDROID_ARM64_TOOLCHAIN) {
    # Try to fall back to the SDK-bundled NDK
    $candidate = "$env:HEXAGON_SDK_ROOT\tools\android-ndk-r25c\toolchains\llvm\prebuilt\windows-x86_64"
    if (Test-Path "$candidate\bin\aarch64-linux-android21-clang.cmd") {
        $env:ANDROID_ARM64_TOOLCHAIN = $candidate
        Write-Host "[info] ANDROID_ARM64_TOOLCHAIN auto-set to $candidate" -ForegroundColor Yellow
    } else {
        Fail "ANDROID_ARM64_TOOLCHAIN not set and SDK-bundled NDK not found."
    }
}
if (-not (Get-Command cl.exe -ErrorAction SilentlyContinue)) {
    Fail "cl.exe not on PATH. Run from a 'Developer Command Prompt for VS' shell, or invoke vcvars64.bat first."
}

# ---- 2. Validate example directory --------------------------------------------

if (-not (Test-Path $ExampleDir)) { Fail "ExampleDir not found: $ExampleDir" }
$ExampleDir = (Resolve-Path $ExampleDir).Path
$TestName = Split-Path $ExampleDir -Leaf

$IdlPath = Join-Path $ExampleDir "rpc\$TestName.idl"
if (-not (Test-Path $IdlPath)) {
    # Some examples name the IDL differently; pick whatever .idl is there.
    $idl = Get-ChildItem -Path (Join-Path $ExampleDir "rpc") -Filter "*.idl" -ErrorAction SilentlyContinue | Select-Object -First 1
    if (-not $idl) { Fail "No rpc/*.idl found in $ExampleDir" }
    $IdlPath = $idl.FullName
    $TestName = $idl.BaseName
    Write-Host "[info] Using IDL $($idl.Name); TestName=$TestName" -ForegroundColor Yellow
}

$GenSrc = Join-Path $ExampleDir "halide\$($TestName)_generator.cpp"
if (-not (Test-Path $GenSrc)) { Fail "Generator source not found: $GenSrc" }

$DspRunSrc = Join-Path $ExampleDir "dsp\$($TestName)_run.cpp"
if (-not (Test-Path $DspRunSrc)) { Fail "DSP run source not found: $DspRunSrc" }

if ($HlosTarget -eq "android") {
    $HostMain = Join-Path $ExampleDir "host\main.cpp"
    if (-not (Test-Path $HostMain)) { Fail "Host main not found: $HostMain" }
}

# ---- 3. Detect DMA usage + check addon ----------------------------------------

$NeedsDma = $false
$mkPath = Join-Path $ExampleDir "Makefile"
if (Test-Path $mkPath) {
    $NeedsDma = (Select-String -Path $mkPath -Pattern "hexagon_dma" -Quiet)
}
if (-not $NeedsDma) {
    # Fallback: search dsp sources for dmaWrapper include
    $dmaIncluders = Get-ChildItem -Path (Join-Path $ExampleDir "dsp") -Filter "*.cpp" -ErrorAction SilentlyContinue |
        Select-String -Pattern "dmaWrapper|dma_def\.h|ubwcdma" -Quiet
    if ($dmaIncluders) { $NeedsDma = $true }
}

$UbwcLibDir = "$env:HEXAGON_SDK_ROOT\addons\compute\libs\ubwcdma\fw\v$Q6Version"
$UbwcLib    = Join-Path $UbwcLibDir "ubwcdma_dynlib.so"
$DmaInc     = Join-Path $UbwcLibDir "inc"

# Fallbacks when the Compute add-on isn't installed:
#   (1) A `compat/` subdir in the example provides shim headers (e.g. our
#       Halide-mini-vendored dma_def.h / dma_types.h / dmaWrapper.h).
#   (2) `ubwcdma_dynlib.so` staged anywhere under C:\Qualcomm or pulled
#       from the device via adb. Symbols are SDK-defined and stable across
#       vendor builds; runtime resolution on the phone goes via
#       /vendor/dsp/cdsp/ubwcdma_dynlib.so regardless of which build we link.
$LocalCompatDir = Join-Path $ExampleDir "compat"
$HasLocalCompat = Test-Path (Join-Path $LocalCompatDir "dmaWrapper.h")
if ($NeedsDma -and -not (Test-Path $UbwcLib)) {
    $candidates = @(
        "C:\Qualcomm\dsp\cdsp\ubwcdma_dynlib.so",
        "C:\Qualcomm\ubwcdma_dynlib.so"
    )
    $UbwcLib = $candidates | Where-Object { Test-Path $_ } | Select-Object -First 1
    if ($UbwcLib -and $HasLocalCompat) {
        Write-Host "[info] Compute add-on not installed; using staged lib + local compat shim:" -ForegroundColor Yellow
        Write-Host "         lib:    $UbwcLib"
        Write-Host "         compat: $LocalCompatDir"
        $DmaInc = $LocalCompatDir
    } elseif ($UbwcLib) {
        Fail @"
Found staged ubwcdma_dynlib.so at $UbwcLib, but no compat/ shim headers
in the example. Drop dma_def.h / dma_types.h / dmaWrapper.h shims into:
  $LocalCompatDir
(See backends/halide/probe/sp_dma_raw/compat/ for a working set vendored
from the Halide project's mini_hexagon_dma.h.)
"@
    } else {
        Fail @"
This example uses Hexagon DMA (hexagon_dma feature / dmaWrapper.h include),
but the Compute add-on isn't installed and no staged fallback was found.
Either:
  (a) Install via Qualcomm Software Center -> Hexagon SDK 5.5 -> Compute add-on
  (b) Stage ubwcdma_dynlib.so at C:\Qualcomm\dsp\cdsp\ubwcdma_dynlib.so
      (e.g. by adb-pulling /vendor/dsp/cdsp/ubwcdma_dynlib.so) AND drop
      compat shim headers into $LocalCompatDir
"@
    }
}

# ---- 4. Set up bin tree + working dir -----------------------------------------

Push-Location $ExampleDir
try {

if ($Clean -and (Test-Path "bin")) {
    Remove-Item -Recurse -Force bin
}
New-Item -ItemType Directory -Force -Path "bin","bin\src","bin\remote","bin\host" | Out-Null
New-Item -ItemType Directory -Force -Path "..\..\utils\bin" | Out-Null

Write-Host ""
Write-Host "============================================================"
Write-Host " Building $TestName (Q6=v$Q6Version, DMA=$NeedsDma, HLOS=$HlosTarget)"
Write-Host "============================================================"

# ---- 5. qaic: IDL -> stub/skel/h ---------------------------------------------

Step "1/8" "qaic: $TestName.idl -> bin\src\"
$qaic = "$env:HEXAGON_SDK_ROOT\ipc\fastrpc\qaic\WinNT\qaic.exe"
if (-not (Test-Path $qaic)) { Fail "qaic.exe not found: $qaic" }
Invoke-Tool $qaic @(
    "-I","$env:HEXAGON_SDK_ROOT\incs\stddef",
    $IdlPath,
    "-o","bin\src"
) "qaic"

# ---- 6. Build Halide generator with cl.exe ------------------------------------

Step "2/8" "cl.exe: build Halide generator (host MSVC)"
Invoke-Tool "cl.exe" @(
    "/EHsc","/nologo",
    "-DLOG2VLEN=7",
    "/I","$env:HALIDE_ROOT\include",
    $GenSrc,
    "$env:HALIDE_ROOT\tools\GenGen.cpp",
    "/link",
    "/libpath:$env:HALIDE_ROOT\lib",
    "Halide.lib",
    "/out:bin\$($TestName)_generator.exe"
) "cl.exe (Halide generator)"

# ---- 7. Run generator -> Hexagon .o + .h --------------------------------------

Step "3/8" "Run generator: emit Hexagon .o, .h, .s, bitcode"
# Generator was just linked against Halide.lib (MSVC import lib); to RUN
# it we need Halide.dll on PATH. The DLL ships at $HALIDE_ROOT\bin.
$env:PATH = "$env:HALIDE_ROOT\bin;$env:PATH"
# Also prepend the Hexagon toolchain bin so hexagon-clang(++).exe resolves
# in steps 4-6. setup_sdk_env.cmd doesn't put these on PATH by default.
$HexToolsVer = if ($env:HEXAGON_TOOLS_VER) { $env:HEXAGON_TOOLS_VER } else { "8.7.06" }
$HexBin = "$env:HEXAGON_SDK_ROOT\tools\HEXAGON_Tools\$HexToolsVer\Tools\bin"
if (-not (Test-Path "$HexBin\hexagon-clang.exe")) {
    Fail "hexagon-clang.exe not found under $HexBin (set HEXAGON_TOOLS_VER if your install differs)"
}
$env:PATH = "$HexBin;$env:PATH"
$genTarget = if ($NeedsDma) {
    "hexagon-32-qurt-hexagon_dma-hvx_128"
} else {
    "hexagon-32-qurt-hvx_128"
}
Invoke-Tool ".\bin\$($TestName)_generator.exe" @(
    "-g","$($TestName)_halide",
    "-e","o,h,assembly,bitcode",
    "-o","bin",
    "target=$genTarget"
) "generator run"

# ---- 8. hexagon-clang: build DSP skel + impl + print ------------------------

$HvxCFlags = @("-mv$Q6Version","-mhvx","-mhvx-length=128B","-O0","-g","-fPIC")
$HvxCxxFlags = @("--std=c++11") + $HvxCFlags
$HvxIncs = @(
    "-I","$env:HEXAGON_SDK_ROOT\incs",
    "-I","$env:HEXAGON_SDK_ROOT\incs\stddef",
    "-I","bin\src",
    "-I","$env:HEXAGON_SDK_ROOT\ipc\fastrpc\remote\ship\android",
    "-I","$env:HEXAGON_SDK_ROOT\rtos\qurt\computev$Q6Version\include",
    "-I","$env:HEXAGON_SDK_ROOT\rtos\qurt\computev$Q6Version\include\qurt",
    "-I","..\..\utils\dsp\include",
    "-I","$env:HALIDE_ROOT\include",
    "-I","bin"
)
if ($NeedsDma) { $HvxIncs += @("-I",$DmaInc) }

Step "4/8" "hexagon-clang: $($TestName)_skel.o"
Invoke-Tool "hexagon-clang.exe" ($HvxCFlags + $HvxIncs + @(
    "-c","bin\src\$($TestName)_skel.c",
    "-o","bin\$($TestName)_skel.o"
)) "hexagon-clang skel.c"

# Compile every .cpp in dsp/ — keeps the script working with examples that
# split logic across multiple DSP TUs.
$DspObjs = @()
foreach ($cpp in (Get-ChildItem -Path "dsp" -Filter "*.cpp")) {
    $obj = "bin\$($cpp.BaseName).o"
    Step "5/8" "hexagon-clang++: $($cpp.Name) -> $($cpp.BaseName).o"
    Invoke-Tool "hexagon-clang++.exe" ($HvxCxxFlags + $HvxIncs + @(
        "-c", $cpp.FullName,
        "-o", $obj
    )) "hexagon-clang++ $($cpp.Name)"
    $DspObjs += $obj
}

Step "6/8" "hexagon-clang: link lib$($TestName)_skel.so"
$linkArgs = @("-mv$Q6Version","-mG0lib","-G0","-fpic","-shared","-lc",
              "-Wl,--start-group",
              "bin\$($TestName)_skel.o",
              "bin\$($TestName)_halide.o") +
            $DspObjs +
            @("-Wl,--end-group")
if ($NeedsDma) { $linkArgs += $UbwcLib }
$linkArgs += @("-o","bin\remote\lib$($TestName)_skel.so")
Invoke-Tool "hexagon-clang.exe" $linkArgs "skel link"

if ($HlosTarget -eq "skip") {
    Write-Host ""
    Write-Host "[done] DSP skel built (HLOS skipped):" -ForegroundColor Green
    Write-Host "  bin\remote\lib$($TestName)_skel.so"
    return
}

# ---- 9. Android aarch64 host + stub + main -----------------------------------

$ArmCC  = "$env:ANDROID_ARM64_TOOLCHAIN\bin\aarch64-linux-android21-clang.cmd"
$ArmCxx = "$env:ANDROID_ARM64_TOOLCHAIN\bin\aarch64-linux-android21-clang++.cmd"
if (-not (Test-Path $ArmCC))  { Fail "NDK clang not found: $ArmCC" }
if (-not (Test-Path $ArmCxx)) { Fail "NDK clang++ not found: $ArmCxx" }
$ArmCFlags = @("-fsigned-char","-O3")
$ArmCxxFlags = @("-std=c++11") + $ArmCFlags
$ArmIncs = @(
    "-I","$env:HEXAGON_SDK_ROOT\incs",
    "-I","$env:HEXAGON_SDK_ROOT\incs\stddef",
    "-I","bin\src",
    "-I","..\..\utils\host\include",
    "-I","..\..\utils\ion",
    "-I","$env:HEXAGON_SDK_ROOT\ipc\fastrpc\rpcmem\inc",
    "-I","$env:HEXAGON_SDK_ROOT\ipc\fastrpc\remote\ship\android"
)

Step "7/8" "ndk clang++: main-$TestName.o + ion_allocation.o"
Invoke-Tool $ArmCxx ($ArmCxxFlags + $ArmIncs + @(
    "-I","$env:HALIDE_ROOT\Examples\include",
    "host\main.cpp",
    "-c","-o","bin\main-$TestName.o"
)) "ndk clang++ main.cpp"

Invoke-Tool $ArmCxx ($ArmCxxFlags + $ArmIncs + @(
    "..\..\utils\ion\ion_allocation.cpp",
    "-c","-o","..\..\utils\bin\ion_allocation.o"
)) "ndk clang++ ion_allocation.cpp"

Step "7/8" "ndk clang: lib$($TestName)_stub.so"
Invoke-Tool $ArmCC ($ArmCFlags + $ArmIncs + @(
    "bin\src\$($TestName)_stub.c",
    "-llog","-fPIE","-pie","-lcdsprpc",
    "-L","$env:HEXAGON_SDK_ROOT\ipc\fastrpc\remote\ship\android_aarch64",
    "-Wl,-soname,lib$($TestName)_stub.so",
    "-shared",
    "-o","bin\host\lib$($TestName)_stub.so"
)) "ndk stub link"

Step "8/8" "ndk clang++: link main-$TestName.out"
# rpcmem.a is the static archive that provides rpcmem_init/alloc/free/deinit.
# The upstream Halide DMA examples don't reference these (they use alloc_ion
# via /dev/ion), but on modern Android (Q+) /dev/ion is gone and dma_heap
# is SELinux-locked for shell-user apps, so the rpcmem path is the only
# one that actually works. Working S22U build (Hexagon IDE-generated) links
# this same archive for the same reason.
$RpcmemArchive = "$env:HEXAGON_SDK_ROOT\ipc\fastrpc\rpcmem\prebuilt\android_aarch64\rpcmem.a"
Invoke-Tool $ArmCxx ($ArmCxxFlags + $ArmIncs + @(
    "bin\main-$TestName.o",
    "-llog","-fPIE","-pie","-lcdsprpc",
    "-L","$env:HEXAGON_SDK_ROOT\ipc\fastrpc\remote\ship\android_aarch64",
    "..\..\utils\bin\ion_allocation.o",
    $RpcmemArchive,
    "-L","..\..\utils",
    "-l$($TestName)_stub",
    "-L","bin\host",
    "-o","bin\main-$TestName.out"
)) "ndk main link"

Write-Host ""
Write-Host "[done] Build artifacts:" -ForegroundColor Green
Write-Host "  bin\remote\lib$($TestName)_skel.so   (cDSP skel)"
Write-Host "  bin\host\lib$($TestName)_stub.so      (host FastRPC stub)"
Write-Host "  bin\main-$TestName.out                (Android aarch64 driver)"

# ---- 10. Optional adb push + run ---------------------------------------------

if ($Run) {
    Step "run" "adb push + run on device"
    if (-not (Get-Command adb -ErrorAction SilentlyContinue)) {
        Fail "adb not on PATH"
    }
    $devicePath = "/data/local/tmp/$TestName"
    & adb shell mkdir -p $devicePath
    & adb push "bin\main-$TestName.out"        $devicePath
    & adb push "bin\host\lib$($TestName)_stub.so"   $devicePath
    & adb push "bin\remote\lib$($TestName)_skel.so" $devicePath

    # freethedsp shim — bake into every device run. The shim itself is
    # a no-op pass-through unless SP_FREETHEDSP=1 is set in the caller's
    # env (which propagates through adb shell). Pushing libfreethedsp.so
    # always means we never have to remember to add it for backends that
    # need test-mode permissions (Halide DMA, CV-ISP MFNR, future
    # signed-PD-gated APIs). See backends/freethedsp/INTEGRATION.md.
    $FreeTheDspLib = "$PSScriptRoot\..\freethedsp\libfreethedsp.so"
    $LdPreload = ""
    if (Test-Path $FreeTheDspLib) {
        & adb push $FreeTheDspLib $devicePath | Out-Null
        $LdPreload = "LD_PRELOAD=./libfreethedsp.so "
        Write-Host "[info] freethedsp shim pushed (active when SP_FREETHEDSP=1)" -ForegroundColor Yellow
    } else {
        Write-Host "[info] freethedsp not built yet — skipping. Build with backends/freethedsp/build.cmd if any signed-PD-gated API is needed." -ForegroundColor DarkGray
    }
    $SpFlag = if ($env:SP_FREETHEDSP) { "SP_FREETHEDSP=$env:SP_FREETHEDSP " } else { "" }

    & adb shell "cd $devicePath; chmod 755 main-$TestName.out; LD_LIBRARY_PATH=. ADSP_LIBRARY_PATH=. $LdPreload$SpFlag./main-$TestName.out"
}

}
finally { Pop-Location }
