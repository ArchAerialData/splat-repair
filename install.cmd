@echo off
rem Installs Splat Repair into this folder. Nothing is written outside it.
rem
rem   install.cmd         editor with Fast Fill
rem   install.cmd deep    also Deep Fill (NVIDIA GPU, about 3 GB of downloads)
rem
rem Safe to run again: it only adds what is missing and rebuilds the app.
setlocal
cd /d "%~dp0"

where node >nul 2>nul
if errorlevel 1 (
  echo Node.js is missing. Install version 22 or newer from https://nodejs.org and run this again.
  goto fail
)

set "PY="
for %%V in (3.12 3.11 3.13 3.10) do if not defined PY (
  py -%%V -c "import sys" >nul 2>nul && set "PY=py -%%V"
)
if not defined PY (
  python -c "import sys; raise SystemExit(0 if (3,10) <= sys.version_info[:2] <= (3,13) else 1)" >nul 2>nul && set "PY=python"
)
if not defined PY (
  echo Python 3.10 to 3.13 is missing. Install it from https://www.python.org and run this again.
  goto fail
)

echo.
echo [1/3] Engine (Python)
if not exist "core\.venv\Scripts\python.exe" (
  %PY% -m venv "core\.venv"
  if errorlevel 1 goto fail
)
"core\.venv\Scripts\python.exe" -m pip install --quiet --upgrade pip
"core\.venv\Scripts\python.exe" -m pip install --quiet -r "core\requirements.txt"
if errorlevel 1 goto fail

echo.
echo [2/3] App (Electron)
pushd app
call npm install
if errorlevel 1 (popd & goto fail)
rem The Electron runtime is downloaded on first use; do it now so start.cmd finds it.
call npx electron --version
if errorlevel 1 (popd & goto fail)
call npm run build
if errorlevel 1 (popd & goto fail)
popd

echo.
echo [3/3] Deep Fill
if /i "%~1"=="deep" (
  "core\.venv\Scripts\python.exe" "core\setup_deep.py"
  if errorlevel 1 goto fail
) else (
  echo Skipped. Run "install.cmd deep" later if you have an NVIDIA GPU and want Deep Fill.
)

echo.
echo Done. Start Splat Repair with app\start.cmd
echo For a desktop shortcut: powershell -ExecutionPolicy Bypass -File app\create-shortcut.ps1
exit /b 0

:fail
echo.
echo The installation did not finish. The lines above show what went wrong.
exit /b 1
