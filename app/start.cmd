@echo off
rem Starts Splat Repair. Builds first if nothing is built yet, or if a build
rem input is newer than the last build: a file under src\, electron\ or
rem public\, or package.json, package-lock.json, vite.config.ts, build.mjs,
rem tsconfig.json. Otherwise the shortcut would keep starting the old bundle
rem after a change.
rem This window closes again; the app keeps running on its own.
cd /d "%~dp0"

if not exist "dist\electron\main.mjs" goto build
if not exist "dist\electron\preload.cjs" goto build
if not exist "dist\renderer\index.html" goto build
rem Without node there is no way to check or build: start the existing build.
where node >nul 2>nul
if errorlevel 1 goto run
rem Exits with 1 if a source is newer than the oldest of the three build outputs.
rem Sources that do not exist (public\) do not count.
node -e "const fs=require('fs'),p=require('path');const m=f=>fs.statSync(f).mtimeMs;const t=Math.min(m('dist/electron/main.mjs'),m('dist/electron/preload.cjs'),m('dist/renderer/index.html'));const newer=f=>fs.statSync(f).isDirectory()?fs.readdirSync(f).some(e=>newer(p.join(f,e))):m(f)>t;process.exit(['src','electron','public','package.json','package-lock.json','vite.config.ts','build.mjs','tsconfig.json'].some(f=>fs.existsSync(f)&&newer(f))?1:0)"
if errorlevel 1 goto build
goto run

:build
echo Building Splat Repair ...
call npm run build
if errorlevel 1 (
  echo.
  echo Build failed. This window stays open.
  pause
  exit /b 1
)

:run
start "" "%~dp0node_modules\electron\dist\electron.exe" "%~dp0." %*
exit /b 0
