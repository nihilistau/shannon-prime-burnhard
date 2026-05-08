@echo off
:: Shannon-Prime / freethedsp — Windows cross-compile script.
::
:: Builds discover.so (Phase D.1, dump-only) and libfreethedsp.so
:: (Phase D.3, patching). Both are aarch64 Android shared libraries
:: loaded via LD_PRELOAD on the S22 Ultra.
::
:: Pre-reqs (set by setup_sdk_env.cmd):
::   ANDROID_ARM64_TOOLCHAIN points at the NDK aarch64 toolchain.

setlocal
if not defined ANDROID_ARM64_TOOLCHAIN (
    set ANDROID_ARM64_TOOLCHAIN=C:\Qualcomm\Hexagon_SDK\5.5.6.0\tools\android-ndk-r25c\toolchains\llvm\prebuilt\windows-x86_64
)
set CC=%ANDROID_ARM64_TOOLCHAIN%\bin\aarch64-linux-android21-clang.cmd
if not exist "%CC%" (
    echo [ERROR] NDK clang not at %CC%
    echo Set ANDROID_ARM64_TOOLCHAIN to your NDK toolchain dir.
    exit /b 1
)

set CFLAGS=-shared -fPIC -O2 -Wall -fvisibility=default

set "TARGET=%~1"
if "%TARGET%"=="" set "TARGET=both"

if /i "%TARGET%"=="discover" goto :do_discover
if /i "%TARGET%"=="patch" goto :do_patch
if /i "%TARGET%"=="both" goto :do_both
if /i "%TARGET%"=="all" goto :do_both
if /i "%TARGET%"=="clean" goto :do_clean

echo Usage: build.cmd [discover^|patch^|both^|clean]
exit /b 1

:do_discover
echo [build] discover.so
"%CC%" %CFLAGS% discover.c -o discover.so -ldl || exit /b 1
echo [done]  %CD%\discover.so
exit /b 0

:do_patch
echo [build] libfreethedsp.so
"%CC%" %CFLAGS% freethedsp_s22u.c -o libfreethedsp.so -ldl || exit /b 1
echo [done]  %CD%\libfreethedsp.so
exit /b 0

:do_both
call :do_discover || exit /b 1
call :do_patch    || exit /b 1
exit /b 0

:do_clean
del /q discover.so libfreethedsp.so 2>nul
echo [clean]
exit /b 0
