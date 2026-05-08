@echo off
REM Shannon-Prime VHT2 - Hexagon DSP scaffold build wrapper.
REM Forked from the wrapper that ships with the C:\Qualcomm\Hexagon_IDE\S22U
REM reference project. Defaults DSP_ARCH=v69 (Snapdragon 8 Gen 1, S22 Ultra).
REM Override via env vars: set DSP_ARCH=v75 ^&^& build.cmd

setlocal

if "%HEXAGON_SDK_ROOT%"=="" set "HEXAGON_SDK_ROOT=C:\Qualcomm\Hexagon_SDK\5.5.6.0"

set "DSP_ARCH=v69"
set "BUILD_TYPE=Debug"
set "HLOS_ARCH=64"

if "%SDK_SETUP_ENV%"=="" (
    echo [build] Sourcing Hexagon SDK env from %HEXAGON_SDK_ROOT%
    REM setup_sdk_env.cmd's qaic-install step has a hardcoded Ubuntu18 path
    REM in its Makefile that the Windows branch should sidestep (via
    REM ifeq ^(^$^(OS^),Windows_NT^)) but gow's bash doesn't propagate OS,
    REM so make falls into the Linux branch and dies. Side effect: bin/
    REM gets cleaned but never repopulated. We continue past the failure
    REM and stage bin/ ourselves below.
    call "%HEXAGON_SDK_ROOT%\setup_sdk_env.cmd"
    if errorlevel 1 (
        if not exist "%HEXAGON_SDK_ROOT%\ipc\fastrpc\qaic\WinNT\qaic.exe" (
            echo [build] FATAL: setup_sdk_env failed and no WinNT qaic.exe found
            exit /b 1
        )
        echo [build] setup_sdk_env reported errors ^(known qaic-Ubuntu18 bug^);
        echo [build] continuing - we'll re-stage bin/ below.
    )
)

REM Re-stage qaic in bin/ after setup_sdk_env's failed install wiped it.
REM The CMake build_idl macro at hexagon_fun.cmake:287 hardcodes the path
REM ${HEXAGON_SDK_ROOT}/ipc/fastrpc/qaic/bin/qaic (no extension) so we need
REM both bin/qaic.exe (for direct PATH lookup) and bin/qaic (for the macro).
REM Plus the sibling .html/.txt — qaic.exe loads them at startup.
set "QAIC_BIN=%HEXAGON_SDK_ROOT%\ipc\fastrpc\qaic\bin"
set "QAIC_WIN=%HEXAGON_SDK_ROOT%\ipc\fastrpc\qaic\WinNT"
if not exist "%QAIC_BIN%\qaic.exe" (
    echo [build] Staging qaic in bin/ ^(setup_sdk_env wiped it^)
    if not exist "%QAIC_BIN%" mkdir "%QAIC_BIN%"
    copy /Y "%QAIC_WIN%\qaic.exe" "%QAIC_BIN%\qaic.exe" >nul
    copy /Y "%QAIC_WIN%\qaic.exe" "%QAIC_BIN%\qaic" >nul
    copy /Y "%QAIC_WIN%\IdlReference.html" "%QAIC_BIN%\" >nul
    copy /Y "%QAIC_WIN%\IdlReference.txt" "%QAIC_BIN%\" >nul
    copy /Y "%QAIC_WIN%\ReleaseNotes.html" "%QAIC_BIN%\" >nul
    copy /Y "%QAIC_WIN%\ReleaseNotes.txt" "%QAIC_BIN%\" >nul
    copy /Y "%QAIC_WIN%\UsersGuide.html" "%QAIC_BIN%\" >nul
    copy /Y "%QAIC_WIN%\UsersGuide.txt" "%QAIC_BIN%\" >nul
)

REM Make sure qaic is on PATH for any rule that does PATH lookup.
where qaic.exe >nul 2>&1
if errorlevel 1 set "PATH=%QAIC_BIN%;%PATH%"

set "TARGET=%~1"
if "%TARGET%"=="" set "TARGET=both"

if /i "%TARGET%"=="-h"      goto :usage
if /i "%TARGET%"=="--help"  goto :usage
if /i "%TARGET%"=="help"    goto :usage
if /i "%TARGET%"=="dsp"     goto :do_dsp
if /i "%TARGET%"=="hexagon" goto :do_dsp
if /i "%TARGET%"=="android" goto :do_android
if /i "%TARGET%"=="hlos"    goto :do_android
if /i "%TARGET%"=="sim"     goto :do_sim
if /i "%TARGET%"=="clean"   goto :do_clean
if /i "%TARGET%"=="both"    goto :do_both
if /i "%TARGET%"=="all"     goto :do_both

echo [build] Unknown target: %TARGET%
goto :usage

:usage
echo.
echo Usage: build.cmd [target]
echo.
echo Targets:
echo   both ^(default^)  - DSP skel + Android HLOS exe/stub
echo   dsp / hexagon   - DSP skel only ^(libsp_hex_skel.so^)
echo   android / hlos  - Android aarch64 exe + stub only
echo   sim             - DSP build + run on Hexagon simulator
echo   clean           - Clean both build trees
echo.
echo Config: DSP_ARCH=%DSP_ARCH% BUILD=%BUILD_TYPE% HLOS_ARCH=%HLOS_ARCH%
echo Override via env vars: set DSP_ARCH=v75 ^&^& build.cmd
exit /b 1

:do_both
call :do_dsp     || exit /b 1
call :do_android || exit /b 1
exit /b 0

:do_dsp
build_cmake hexagon DSP_ARCH=%DSP_ARCH% BUILD=%BUILD_TYPE% -gMake
exit /b %errorlevel%

:do_android
build_cmake android HLOS_ARCH=%HLOS_ARCH% BUILD=%BUILD_TYPE% -gMake
exit /b %errorlevel%

:do_sim
build_cmake hexagonsim DSP_ARCH=%DSP_ARCH% BUILD=%BUILD_TYPE% -gMake
exit /b %errorlevel%

:do_clean
build_cmake hexagon_clean DSP_ARCH=%DSP_ARCH% BUILD=%BUILD_TYPE%
build_cmake android_clean HLOS_ARCH=%HLOS_ARCH% BUILD=%BUILD_TYPE%
exit /b 0
