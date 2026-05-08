::==============================================================================
:: Mode D Stage 1 probe — Windows-native build script.
:: Copyright (C) 2026 Ray Daniels. All Rights Reserved. AGPLv3 / commercial.
::
:: Forked from the Hexagon SDK 5.5.6.0 sample
:: test-dma_raw_blur_rw_async.cmd. The upstream is the canonical
:: Qualcomm-blessed Windows build flow for these Halide DMA examples;
:: trying to drive the Linux Makefile.rules through gow's mingw32-make
:: hits unfixable issues (libHalide.a vs Halide.lib, $(shell pwd)
:: backslash mixing). This script uses cl.exe + qaic.exe + hexagon-clang
:: + Android NDK clang directly, mirroring the upstream pattern.
::
:: PREREQUISITES on the host:
::   1. Hexagon SDK env sourced (call setup_sdk_env.cmd first):
::        - %HEXAGON_SDK_ROOT%     points at the SDK
::        - %HALIDE_ROOT%          points at HALIDE_Tools\<ver>\Halide
::        - %ANDROID_ARM64_TOOLCHAIN% points at the NDK aarch64 toolchain dir
::          (e.g. <SDK>\tools\android-ndk-r25c\toolchains\llvm\prebuilt\
::          windows-x86_64)
::   2. Visual Studio C++ Build Tools 2019+: cl.exe must be on PATH
::      (run "Developer Command Prompt for VS 2019" or vcvars64.bat).
::   3. Hexagon SDK Compute add-on installed:
::        %HEXAGON_SDK_ROOT%\addons\compute\libs\ubwcdma\fw\v66\
::      Without this, link of lib%TEST%_skel.so fails — install via
::      Qualcomm Software Center -> Hexagon SDK -> Compute add-on.
::==============================================================================

@echo off
@set DSPTYPE=cdsp

if not defined Q6_VERSION (
  set Q6_VERSION=66
) else (
  if "%Q6_VERSION%" LSS 66 (set Q6_VERSION=66)
)

@set UBWC_LIB_DIR=%HEXAGON_SDK_ROOT%\addons\compute\libs\ubwcdma\fw\v%Q6_VERSION%
@set DMA_INC=%UBWC_LIB_DIR%\inc

if not exist "%UBWC_LIB_DIR%\ubwcdma_dynlib.so" (
  echo.
  echo [ERROR] Compute add-on missing: %UBWC_LIB_DIR%\ubwcdma_dynlib.so not found.
  echo         Install via Qualcomm Software Center -> Hexagon SDK -> Compute add-on.
  exit /b 1
)
if not defined HALIDE_ROOT (
  echo [ERROR] HALIDE_ROOT not set. Source setup_sdk_env.cmd first.
  exit /b 1
)
if not defined ANDROID_ARM64_TOOLCHAIN (
  echo [ERROR] ANDROID_ARM64_TOOLCHAIN not set. Source setup_sdk_env.cmd first.
  exit /b 1
)
where cl.exe >nul 2>&1
if errorlevel 1 (
  echo [ERROR] cl.exe not on PATH. Run from a "Developer Command Prompt for VS"
  echo         shell, or invoke vcvars64.bat first.
  exit /b 1
)

@set TEST=sp_dma_raw

@echo Cleaning bin\
del /q ..\..\utils\bin >nul 2>&1
mkdir ..\..\utils\bin  >nul 2>&1
del /q bin\src         >nul 2>&1
del /q bin\remote      >nul 2>&1
del /q bin\host        >nul 2>&1
del /q bin             >nul 2>&1
rmdir bin\src          >nul 2>&1
rmdir bin\remote       >nul 2>&1
rmdir bin\host         >nul 2>&1
rmdir bin              >nul 2>&1
mkdir bin              >nul 2>&1
mkdir bin\src          >nul 2>&1
mkdir bin\remote       >nul 2>&1
mkdir bin\host         >nul 2>&1

@echo [1/8] Generating _stub, _skel, .h from rpc\%TEST%.idl
%HEXAGON_SDK_ROOT%\ipc\fastrpc\qaic\WinNT\qaic.exe -I %HEXAGON_SDK_ROOT%\incs\stddef rpc\%TEST%.idl -o bin\src
if errorlevel 1 goto :fail

@echo [2/8] Build Halide generator (cl.exe ^+ Halide.lib)
cl.exe /EHsc -DLOG2VLEN=7 /I %HALIDE_ROOT%\include halide\%TEST%_generator.cpp %HALIDE_ROOT%\tools\GenGen.cpp /link /libpath:%HALIDE_ROOT%\lib Halide.lib /out:bin\sp_dma_halide_generator.exe /nologo
if errorlevel 1 goto :fail

@echo [3/8] Run generator -^> Hexagon .o + .h
bin\sp_dma_halide_generator.exe -g %TEST%_halide -e o,h,assembly,bitcode -o bin target=hexagon-32-qurt-hexagon_dma-hvx_128
if errorlevel 1 goto :fail

set HVX_CFLAGS=-mv%Q6_VERSION% -mhvx -mhvx-length=128B -O0 -g -fPIC
set HVX_CXXFLAGS=--std=c++11 %HVX_CFLAGS%
set HVX_INCLUDES=-I %HEXAGON_SDK_ROOT%\incs -I %HEXAGON_SDK_ROOT%\incs\stddef -I bin\src -I %HEXAGON_SDK_ROOT%\ipc\fastrpc\remote\ship\android -I %HEXAGON_SDK_ROOT%\rtos\qurt\computev%Q6_VERSION%\include -I %HEXAGON_SDK_ROOT%\rtos\qurt\computev%Q6_VERSION%\include\qurt -I ..\..\utils\dsp\include -I %HALIDE_ROOT%\include -I bin -I %DMA_INC%

@echo [4/8] Build %TEST%_skel.o
hexagon-clang.exe %HVX_CFLAGS% %HVX_INCLUDES% -c bin\src\%TEST%_skel.c -o bin\%TEST%_skel.o
if errorlevel 1 goto :fail

@echo [5/8] Build %TEST%_run.o
hexagon-clang++.exe %HVX_CXXFLAGS% %HVX_INCLUDES% -c dsp\%TEST%_run.cpp -o bin\%TEST%_run.o
if errorlevel 1 goto :fail

@echo [6/8] Build print.o
hexagon-clang++.exe %HVX_CXXFLAGS% %HVX_INCLUDES% -c dsp\print.cpp -o bin\print.o
if errorlevel 1 goto :fail

@echo [7/8] Link lib%TEST%_skel.so
hexagon-clang.exe -mv%Q6_VERSION% -mG0lib -G0 -fpic -shared -lc -Wl,--start-group bin\%TEST%_skel.o bin\%TEST%_halide.o bin\%TEST%_run.o bin\print.o -Wl,--end-group %UBWC_LIB_DIR%\ubwcdma_dynlib.so -o bin\remote\lib%TEST%_skel.so
if errorlevel 1 goto :fail

set ANDROID_ARM64_CC=%ANDROID_ARM64_TOOLCHAIN%\bin\aarch64-linux-android21-clang
set ANDROID_ARM64_CXX=%ANDROID_ARM64_TOOLCHAIN%\bin\aarch64-linux-android21-clang++
set ARM_CFLAGS=-fsigned-char -O3
set ARM_CXXFLAGS=-std=c++11 %ARM_CFLAGS%
set ARM_INCLUDES=-I %HEXAGON_SDK_ROOT%\incs -I %HEXAGON_SDK_ROOT%\incs\stddef -I bin\src -I ..\..\utils\host\include -I ..\..\utils\ion -I %HEXAGON_SDK_ROOT%\ipc\fastrpc\rpcmem\inc -I %HEXAGON_SDK_ROOT%\ipc\fastrpc\remote\ship\android

@echo [8/8] Build host main + stub + final binary
call %ANDROID_ARM64_CXX% host\main.cpp %ARM_CXXFLAGS% %ARM_INCLUDES% -I %HALIDE_ROOT%\Examples\include -c -o bin\main-%TEST%.o
if errorlevel 1 goto :fail

call %ANDROID_ARM64_CXX% ..\..\utils\ion\ion_allocation.cpp %ARM_CXXFLAGS% %ARM_INCLUDES% -c -o ..\..\utils\bin\ion_allocation.o
if errorlevel 1 goto :fail

call %ANDROID_ARM64_CC% %ARM_CFLAGS% %ARM_INCLUDES% bin\src\%TEST%_stub.c -llog -fPIE -pie -l%DSPTYPE%rpc -L %HEXAGON_SDK_ROOT%\ipc\fastrpc\remote\ship\android_aarch64 -Wl,-soname,lib%TEST%_stub.so -shared -o bin\host\lib%TEST%_stub.so
if errorlevel 1 goto :fail

call %ANDROID_ARM64_CXX% %ARM_CXXFLAGS% %ARM_INCLUDES% bin\main-%TEST%.o -llog -fPIE -pie -l%DSPTYPE%rpc -L %HEXAGON_SDK_ROOT%\ipc\fastrpc\remote\ship\android_aarch64 ..\..\utils\bin\ion_allocation.o -L ..\..\utils -l%TEST%_stub -L bin\host -o bin\main-%TEST%.out
if errorlevel 1 goto :fail

@echo.
@echo [done] Build artifacts:
@echo   bin\remote\lib%TEST%_skel.so   ^(cDSP skel^)
@echo   bin\host\lib%TEST%_stub.so      ^(host FastRPC stub^)
@echo   bin\main-%TEST%.out              ^(Android aarch64 host driver^)
@echo.
@echo Run-on-device hint:
@echo   adb push bin\main-%TEST%.out bin\host\lib%TEST%_stub.so bin\remote\lib%TEST%_skel.so /data/local/tmp/sp_dma_raw_probe/
@echo   adb shell "cd /data/local/tmp/sp_dma_raw_probe; LD_LIBRARY_PATH=. ADSP_LIBRARY_PATH=. ./main-%TEST%.out 1000"
exit /b 0

:fail
echo.
echo [ERROR] Build step failed. See above for the failing command.
exit /b 1
