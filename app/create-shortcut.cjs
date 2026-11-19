// Run through create-shortcut.ps1, using the local Electron runtime (no window).
const { app, shell, nativeImage } = require('electron');
const path = require('node:path');
const fs = require('node:fs');
const identity = require('./windows-identity.json');

app.whenReady().then(() => {
  if (process.platform !== 'win32') throw new Error('Taskbar shortcuts require Windows.');
  const programsIndex = process.argv.indexOf('--programs');
  const programs = programsIndex >= 0 ? process.argv[programsIndex + 1] : undefined;
  if (!programs || !path.isAbsolute(programs)) throw new Error('A Start Menu Programs folder is required.');
  const icon = path.join(__dirname, 'assets', 'icon.ico');
  for (const file of [icon, path.join(__dirname, 'dist', 'electron', 'main.mjs'),
    path.join(__dirname, 'dist', 'electron', 'preload.cjs'), path.join(__dirname, 'dist', 'renderer', 'index.html')]) {
    if (!fs.existsSync(file)) throw new Error(`Missing ${file}; run install.cmd first.`);
  }
  if (nativeImage.createFromPath(path.join(__dirname, 'assets', 'icon-256.png')).isEmpty()) {
    throw new Error('The application icon could not be decoded.');
  }
  const options = {
    target: process.execPath, args: `"${__dirname}"`, cwd: __dirname,
    description: 'Repair, clean up and export Gaussian splats',
    icon, iconIndex: 0, appUserModelId: identity.appId
  };
  for (const directory of [app.getPath('desktop'), programs]) {
    fs.mkdirSync(directory, { recursive: true });
    const link = path.join(directory, `${identity.displayName}.lnk`);
    if (!shell.writeShortcutLink(link, 'create', options)) throw new Error(`Could not write ${link}`);
    const saved = shell.readShortcutLink(link);
    for (const field of ['target', 'args', 'cwd', 'icon', 'iconIndex', 'appUserModelId']) {
      if (saved[field] !== options[field]) throw new Error(`Shortcut verification failed: ${field} in ${link}`);
    }
    console.log(`Verified shortcut: ${link}`);
  }
  app.quit();
}).catch(error => { console.error(error.message); app.exit(1); });
