# Editor baseline audit — 2026-11-19

Base: upstream `8487b80`. Scope: the local editor changes developed before
classification. This is a code review and regression audit, not a claim that
all scan types, hardware, or reconstruction quality have been validated.

## Reviewed changes

| Topic | Behavior and limits |
|---|---|
| Theme | Charcoal panels, blue controls and selection tint; neutral scan backdrop retained. |
| Navigation | Metashape preset: left pan in Orbit, middle orbit, right/wheel zoom. Downward middle drag raises the camera. Other presets retained. |
| Orientation | Unrotated ENU PLY sidecars suggest +Z; saved override wins, Shift+U resets it. Unknown frames are not guessed from geometry. |
| Open dialog | Last successful open folder survives restart. Invalid preferences and unavailable folders fall back. |
| Inspection | Bright yellow/opacity boost and X-ray centre markers are display-only. At most 20,000 sampled markers; no membership changes. |
| Through brush | Screen-space strokes select centres at every depth, preserve add/subtract, and exclude deleted/cropped splats. Not footprint intersection. |
| Surface haze | Experimental CPU classifier selects faint off-surface candidates for review. No automatic deletion or image sharpening. |

## Findings fixed

- The Surface haze engine command was absent from Electron's production allowlist.
  Direct engine tests alone did not expose this; the production IPC smoke now does.
- Surface haze copied unnecessary full radiance/SH data. It now analyses only
  position, opacity and scale arrays with stable document IDs and an alive mask.
- Deleted splats cannot become haze reference geometry. Empty/stale scoped
  requests, malformed payloads and invalid thresholds are rejected explicitly.
- The inspection overlay updated DOM and redrew at rest. It now invalidates on
  selection/transform/camera/viewport changes and skips unchanged frames.
- Through-brush previews previously rasterised the entire stroke repeatedly.
  Raster coverage is now accumulated incrementally within each stroke.
- Open-folder writes are serialized to avoid overlapping temporary-file writes.
  Corrupt/null preferences and unavailable paths have regression coverage.
- Orientation detection is restricted to files actually identified as PLY.
- Installer-only lockfile churn was removed; no dependency upgrade is included.

## Validation

All 19 engine test scripts exited successfully. The modified orientation,
surface-haze and server tests were rerun after their final changes. Some tests
report conditional skips: GPU/PyTorch-dependent cases, Deep Fill without
LaMa/OpenCV, and an optional hut-scan fixture. Those capabilities are not certified
by this run. Local individual logs are in `.cache/audit/` (not published).

The TypeScript check, production build, navigation, open-folder, inspection and
Through-brush smoke tests passed. A hidden Electron smoke using an isolated
profile and a synthetic 2,122-splat fixture exercised the actual production
application on WebGPU: load, ENU orientation, Through selection, inspection
shader modes, X-ray canvas pixels, deselection, and Surface haze IPC (441
candidates). No customer scans or screenshots are included in the fork.

The build emits its existing large-bundle advisory. Classification is not part
of this baseline. Class hiding, resource unloading, tiled imports, and streaming
remain separate features. No performance claim for massive whole-site editing
is made from the small runtime fixture.

## Reproduce on Windows

From the repository root in PowerShell:

```powershell
Get-ChildItem core/tests/test_*.py | ForEach-Object {
  & core/.venv/Scripts/python.exe $_.FullName
  if ($LASTEXITCODE -ne 0) { throw "Failed: $($_.Name)" }
}
cd app
npm run typecheck
npm run build
node smoke/navigation.mjs
node smoke/open-folder.mjs
node smoke/selection-inspection.mjs
node smoke/through-brush.mjs
npx electron smoke/editor-baseline.cjs
```

Check every exit code. Electron smoke runs against the built bundle, so rebuild
after source changes. Review surface-haze candidates from several real views
before deletion; passing synthetic tests does not establish classification
accuracy on an arbitrary customer scan.
