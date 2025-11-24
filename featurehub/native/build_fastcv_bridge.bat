@echo off
echo Building FastCV bridge DLL...

rem Use dynamic CRT (/MD) to play nicer with prebuilt libs
set CL=/MD

cl /LD fastcv_bridge.c ^
   /I fastcv_sdk\inc ^
   fastcv_sdk\lib\libfastcv.lib ^
   msvcrt.lib ^
   /link /OUT:fastcv_bridge.dll

echo -----------------------------------------
echo Build complete: fastcv_bridge.dll
echo -----------------------------------------
