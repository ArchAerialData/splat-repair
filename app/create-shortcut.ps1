# Build branded Desktop and Start Menu shortcuts suitable for pinning.
# No administrator rights or registry modifications are required.
$ErrorActionPreference = 'Stop'
$appDirectory = Split-Path -Parent $MyInvocation.MyCommand.Definition
$runtime = Join-Path $appDirectory 'node_modules\electron\dist\electron.exe'
if (-not (Test-Path -LiteralPath $runtime)) { throw 'Run install.cmd first.' }
Push-Location $appDirectory
try {
    & npm.cmd run build
    if ($LASTEXITCODE -ne 0) { throw 'Build failed; shortcuts were not changed.' }
    $scriptPath = Join-Path $appDirectory 'create-shortcut.cjs'
    $programsPath = [Environment]::GetFolderPath('Programs')
    $installer = Start-Process -FilePath $runtime -ArgumentList @("`"$scriptPath`"", '--programs', "`"$programsPath`"") -WindowStyle Hidden -Wait -PassThru
    if ($installer.ExitCode -ne 0) { throw 'Shortcut creation failed.' }
} finally { Pop-Location }
Write-Host 'Right-click Splat Repair in Start or the running taskbar icon and choose Pin to taskbar.'
