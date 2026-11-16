# Puts a "Splat Repair" shortcut on the desktop.
# To undo it, delete the shortcut. Nothing else is installed and nothing
# is written to the registry.
$ErrorActionPreference = 'Stop'

$app = Split-Path -Parent $MyInvocation.MyCommand.Definition
$target = Join-Path $app 'start.cmd'
$icon = Join-Path $app 'assets\icon.ico'
foreach ($p in @($target, $icon)) {
  if (-not (Test-Path $p)) { throw "missing: $p" }
}

$link = Join-Path ([Environment]::GetFolderPath('Desktop')) 'Splat Repair.lnk'
$s = (New-Object -ComObject WScript.Shell).CreateShortcut($link)
$s.TargetPath = $target
$s.WorkingDirectory = $app
$s.IconLocation = "$icon,0"
$s.Description = 'Repair, clean up and export Gaussian splats'
$s.WindowStyle = 7          # minimized: start.cmd hands over to Electron right away
$s.Save()

Write-Host "Shortcut created on the desktop: $link"
