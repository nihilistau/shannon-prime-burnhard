@echo off
:: Thin cmd.exe wrapper around build-example.ps1 — invokes PowerShell with
:: ExecutionPolicy Bypass so users don't need to touch their system policy.
::
:: Usage:
::   build-example.cmd <ExampleDir> [-Q6Version 66] [-Clean] [-HlosTarget skip] [-Run]
::
:: Example:
::   build-example.cmd C:\Qualcomm\HALIDE_Tools\2.4.07\Halide\Examples\standalone\device\apps\sp_dma_raw -Clean
::
:: Prerequisites in the calling shell:
::   1. Hexagon SDK env: call %HEXAGON_SDK_ROOT%\setup_sdk_env.cmd
::   2. MSVC env:        call "...\VC\Auxiliary\Build\vcvars64.bat"
:: Or just run from a "Developer Command Prompt for VS" shell with the SDK
:: env sourced after.

setlocal
set "SCRIPT_DIR=%~dp0"
powershell.exe -NoProfile -ExecutionPolicy Bypass -File "%SCRIPT_DIR%build-example.ps1" %*
exit /b %errorlevel%
