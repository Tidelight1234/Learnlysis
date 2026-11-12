@echo off
setlocal
cd /d "%~dp0"
powershell.exe -NoProfile -STA -ExecutionPolicy Bypass -File "%~dp0publish_v5.ps1"
set "PUBLISH_EXIT=%ERRORLEVEL%"
if not "%PUBLISH_EXIT%"=="0" (
    echo.
    echo The upload window could not start or exited with an error.
    echo Check the PowerShell error above, then press any key to close.
    pause >nul
)
endlocal
exit /b %PUBLISH_EXIT%
