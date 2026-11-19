// Run after npm run build: npx electron smoke/editor-baseline.cjs
// Exercises production IPC and rendering with synthetic data and an isolated profile.
const { app, protocol, dialog } = require('electron');
const fs = require('node:fs/promises');
const fsSync = require('node:fs');
const os = require('node:os');
const path = require('node:path');
const { pathToFileURL } = require('node:url');
const assert = require('node:assert/strict');

const wait = ms => new Promise(resolve => setTimeout(resolve, ms));
const timeout = setTimeout(() => { console.error('Editor baseline timed out'); app.exit(1); }, 90000);
const errors = [];
const temp = fsSync.mkdtempSync(path.join(os.tmpdir(), 'splat-baseline-'));
fsSync.mkdirSync(path.join(temp, 'profile'));
app.setPath('userData', path.join(temp, 'profile'));
app.setPath('sessionData', path.join(temp, 'profile'));
// Async ESM import can run after Electron emits ready. Register the same
// production scheme synchronously, then let main's registration be a no-op.
protocol.registerSchemesAsPrivileged([{ scheme: 'app', privileges: {
  standard: true, secure: true, supportFetchAPI: true, stream: true
} }]);
protocol.registerSchemesAsPrivileged = () => {};

(async () => {
  const scan = path.join(temp, 'fixture.ply');
  const samples = [];
  for (let x = -20; x <= 20; x++) for (let y = -20; y <= 20; y++) samples.push([x / 10, y / 10, 0, .95]);
  for (let x = -10; x <= 10; x++) for (let y = -10; y <= 10; y++) samples.push([x / 10, y / 10, .3, .1]);
  const columns = ['x', 'y', 'z', 'f_dc_0', 'f_dc_1', 'f_dc_2', 'opacity', 'scale_0', 'scale_1', 'scale_2', 'rot_0', 'rot_1', 'rot_2', 'rot_3'];
  const header = `ply\nformat binary_little_endian 1.0\nelement vertex ${samples.length}\n`
    + columns.map(n => `property float ${n}\n`).join('') + 'end_header\n';
  const binary = Buffer.alloc(samples.length * columns.length * 4);
  samples.forEach(([x, y, z, alpha], i) => {
    const values = [x, y, z, .2, .2, .2, Math.log(alpha / (1 - alpha)), Math.log(.07), Math.log(.07), Math.log(.02), 1, 0, 0, 0];
    values.forEach((v, j) => binary.writeFloatLE(v, (i * columns.length + j) * 4));
  });
  await fs.writeFile(scan, Buffer.concat([Buffer.from(header), binary]));
  await fs.writeFile(path.join(temp, 'geo_desc.json'), JSON.stringify({ cs_type: 'LOCAL_ENU_CS', rvec: [0, 0, 0], scale: 1 }));
  dialog.showOpenDialog = async () => ({ canceled: false, filePaths: [scan] });
  let window;
  app.on('browser-window-created', (_event, w) => {
    window = w;
    // Hidden windows can return an old compositor frame indefinitely. Keep
    // this test window composited, but invisible, unfocused and noninteractive.
    w.setOpacity(0);
    w.setIgnoreMouseEvents(true);
    w.setSkipTaskbar(true);
    w.show = () => w.showInactive();
    w.webContents.setBackgroundThrottling(false);
    w.webContents.on('console-message', e => {
      if (e.level === 'error' || e.level === 3) errors.push(e.message);
    });
    w.webContents.on('render-process-gone', (_e, details) => errors.push(`Renderer crashed: ${details.reason}`));
  });
  await import(pathToFileURL(path.resolve('dist/electron/main.mjs')).href);
  const until = async (condition, label) => {
    for (let i = 0; i < 200; i++) {
      if (await condition()) return;
      await wait(100);
    }
    throw new Error(`Timed out: ${label}`);
  };
  await until(() => window && !window.webContents.isLoading(), 'window');
  const run = script => window.webContents.executeJavaScript(script);
  await until(() => run(`!!window.__test && /ready/.test(document.getElementById('status-msg').textContent)`), 'engine ready');
  await run(`document.getElementById('btn-open').click()`);
  await until(() => run(`window.__test.stats().total === ${samples.length}`), 'fixture opened');
  assert.equal(await run(`window.__test.upAxis()`), '+z');
  await run(`document.querySelector('[data-reach="through"]').click()`);
  const selected = await run(`window.__test.paintStroke([[.05, .5], [.95, .5]], 300)`);
  assert.ok(selected > 0);
  for (const mode of ['normal', 'bright', 'xray']) {
    await run(`(() => { const e = document.getElementById('selection-display'); e.value = '${mode}'; e.dispatchEvent(new Event('change')); })()`);
    // Hidden native windows can composite at 1 Hz despite unthrottled JS.
    // Let the changed DOM and GPU work-buffer reach the captured frame.
    await wait(1200);
    const screenshot = await window.webContents.capturePage();
    assert.ok(!screenshot.isEmpty(), `${mode} screenshot`);
    if (process.env.SPLAT_BASELINE_SHOTS) {
      await fs.mkdir(process.env.SPLAT_BASELINE_SHOTS, { recursive: true });
      await fs.writeFile(path.join(process.env.SPLAT_BASELINE_SHOTS, `baseline-${mode}.png`), screenshot.toPNG());
    }
    if (mode === 'xray') {
      const pixels = await run(`(() => { const c = document.getElementById('selection-inspection');
        return c.getContext('2d').getImageData(0, 0, c.width, c.height).data.filter((_, i) => i % 4 === 3).some(a => a > 0); })()`);
      assert.ok(pixels, 'X-ray has visible selected-centre markers');
    }
  }
  await run(`document.getElementById('surface-noise-scope').value = 'all'; document.getElementById('btn-surface-noise').click()`);
  await until(() => run(`/surface haze candidates selected/.test(window.__test.stats().status)`), 'production surface-noise IPC');
  assert.ok((await run(`window.__test.stats().selected`)) > 0, 'Haze detector selects the synthetic layer');
  await run(`window.__test.clearSelection()`);
  await until(() => run(`(() => { const c = document.getElementById('selection-inspection');
    return !c.getContext('2d').getImageData(0, 0, c.width, c.height).data.some(v => v !== 0); })()`), 'deselect clears overlay');
  assert.deepEqual(errors, [], 'No renderer or shader errors');
  console.log('Editor baseline passed:', JSON.stringify(await run(`window.__test.stats()`)));
  clearTimeout(timeout);
  // The isolated profile is left in the OS temp directory for failure diagnosis.
  app.quit();
})().catch(error => { console.error(error); console.error(errors); clearTimeout(timeout); app.exit(1); });
