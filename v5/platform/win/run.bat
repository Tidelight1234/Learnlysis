@echo off
setlocal
rem Windows launcher: always prefer the isolated PySide6 environment.
cd /d "%~dp0..\..\src"

set "ENV_PY=%~dp0..\..\..\.venv\Scripts\python.exe"
if exist "%ENV_PY%" goto run_env

echo PySide6 environment is missing.
echo Create it from the PySide6_version folder and install requirements.txt.
pause
exit /b 1

:run_env
"%ENV_PY%" -X utf8 main.py
set "LEARNLYSIS_EXIT=%ERRORLEVEL%"
if not "%LEARNLYSIS_EXIT%"=="0" (
  echo.
  echo Learnlysis failed to start. Exit code: %LEARNLYSIS_EXIT%
  pause
)
exit /b %LEARNLYSIS_EXIT%

