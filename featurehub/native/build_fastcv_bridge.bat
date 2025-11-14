@echo off
setlocal

rem IMPORTANT:
rem Run this script from:
rem   "x64 Native Tools Command Prompt for VS 2019"
rem NOT PowerShell or normal cmd.

echo Building FastCV bridge DLL...

set SDK_DIR=%~dp0fastcv_sdk

cl /LD fastcv_bridge.c ^
   /I "%SDK_DIR%\inc" ^
   "%SDK_DIR%\lib\libfastcv.lib" ^
   /Fe:fastcv_bridge.dll

echo -----------------------------------------
echo Build complete: fastcv_bridge.dll
echo -----------------------------------------

endlocal
