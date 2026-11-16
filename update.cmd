@echo off
rem Updates Splat Repair to the newest version and rebuilds it.
rem Works for a copy made with "git clone". If you downloaded a ZIP, download
rem the new ZIP instead and run install.cmd in it.
setlocal
cd /d "%~dp0"

where git >nul 2>nul
if errorlevel 1 (
  echo Git is missing. Install it from https://git-scm.com or download the new ZIP from GitHub.
  exit /b 1
)
if not exist ".git" (
  echo This folder is not a git clone. Download the new ZIP from GitHub and run install.cmd in it.
  exit /b 1
)

git pull --ff-only
if errorlevel 1 (
  echo.
  echo Could not update. If you changed files yourself, commit or undo those changes first.
  exit /b 1
)

call "%~dp0install.cmd" %*
