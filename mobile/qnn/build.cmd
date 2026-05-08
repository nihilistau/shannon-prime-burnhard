@echo off
REM Phase 2.3 - cross-compile sp_qnn shim + test driver for aarch64
REM Android, against QAIRT 2.45.40.260406 headers + dlopen of runtime.

setlocal
if not defined ANDROID_ARM64_TOOLCHAIN (
    set ANDROID_ARM64_TOOLCHAIN=C:\Qualcomm\Hexagon_SDK\5.5.6.0\tools\android-ndk-r25c\toolchains\llvm\prebuilt\windows-x86_64
)
if not defined QAIRT_ROOT (
    set QAIRT_ROOT=C:\Qualcomm\AIStack\QAIRT\2.45.40.260406
)
set CC=%ANDROID_ARM64_TOOLCHAIN%\bin\aarch64-linux-android21-clang.cmd
if not exist "%CC%" ( echo [ERROR] NDK clang not at %CC% & exit /b 1 )
if not exist "%QAIRT_ROOT%\include\QNN\QnnInterface.h" (
    echo [ERROR] QAIRT headers not at %QAIRT_ROOT%\include\QNN\
    exit /b 1
)

set CFLAGS=-std=c11 -O2 -Wall -fPIC -fvisibility=default
set INCS=-I "%QAIRT_ROOT%\include\QNN" -I "%QAIRT_ROOT%\include"

echo [build] libsp_qnn.so
"%CC%" -shared %CFLAGS% %INCS% sp_qnn.c -o libsp_qnn.so -ldl -llog
if errorlevel 1 ( echo [error] libsp_qnn.so build failed & exit /b 1 )

echo [build] test_sp_qnn
"%CC%" %CFLAGS% -fPIE -pie %INCS% test_sp_qnn.c sp_qnn.c -o test_sp_qnn -ldl -llog
if errorlevel 1 ( echo [error] test_sp_qnn build failed & exit /b 1 )

echo [done]
dir /b libsp_qnn.so test_sp_qnn 2>nul
exit /b 0
