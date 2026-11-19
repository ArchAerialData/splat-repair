# Windows launcher

The `codex/windows-launcher` branch makes the source checkout launchable from
Desktop, Start and taskbar shortcuts with the Splat Repair icon. It does not
package a standalone application: the repository, local Electron runtime and
installed engine environment must remain available.

## Setup and updates

Run this from the repository root after installation:

```powershell
powershell -ExecutionPolicy Bypass -File app\create-shortcut.ps1
```

The script builds the app, then creates or replaces **Splat Repair.lnk** in the
current user's Desktop and Start Menu Programs folders. Pin Splat Repair from
Start or its running taskbar icon. Remove an old generic pin before pinning the
new shortcut. Existing pinned copies do not update when another shortcut is
replaced.

Shortcuts start the built app directly, without a terminal window. After code
changes, run `app\start.cmd` to rebuild changed inputs, or `npm run build` from
`app`. Moving the checkout changes the absolute launch and icon paths: rerun
shortcut setup and unpin/re-pin afterward. To remove the app, remove its pins
and shortcuts along with the checkout.

## Implementation

- `app/windows-identity.json` is the shared source of the display name and
  `com.archaerial.splatrepair` application ID. The Electron process, window and
  shortcuts use the same ID so Windows can group and relaunch them correctly.
- The shortcuts target the installed `electron.exe` with the quoted absolute
  `app` directory as its argument and working directory. The window also sets
  the full relaunch command and display name before it is shown.
- The existing multi-resolution `app/assets/icon.ico` supplies the shortcut
  and taskbar icon; the application executable is not copied or renamed.
- Shortcut creation reads each link back and verifies its target, arguments,
  working directory, icon, icon index and application ID. A mismatch fails setup.
- `start.cmd` includes `windows-identity.json` in its build freshness check.

This follows Microsoft's [application ID guidance](https://learn.microsoft.com/en-us/windows/win32/shell/appids)
and Electron's [shortcut details](https://www.electronjs.org/docs/latest/api/structures/shortcut-details)
and [window relaunch metadata](https://www.electronjs.org/docs/latest/api/browser-window#winsetappdetailsoptions-windows).

## Validation record — 2026-11-19

- TypeScript check passed: `npx tsc --noEmit -p .` from `app`.
- Production build passed: `npm run build` from `app`.
- Desktop and Start Menu shortcuts were created successfully and passed the
  field-by-field read-back verification described above.
- Editor smoke test passed: `npx electron smoke/editor-baseline.cjs` exercised
  the built app with WebGPU and a 2,122-splat fixture.
- Launching the actual Desktop shortcut produced a visible **Splat Repair**
  main window using the expected local Electron executable.
- The displayed taskbar icon and pin/close/relaunch sequence have not been
  visually verified. UI capture timed out; shortcut metadata and successful
  launch alone do not establish those results.
