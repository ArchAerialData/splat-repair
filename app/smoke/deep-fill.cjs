// Run from app after npm run build: npx electron smoke/deep-fill.cjs
// Requires a working CUDA/gsplat/LaMa installation. Uses only a generated
// fixture and an isolated profile; it never opens or changes a user's scan.
const { app, BrowserWindow, protocol, dialog, ipcMain } = require('electron');
const fs = require('node:fs/promises');
const fsSync = require('node:fs');
const os = require('node:os');
const path = require('node:path');
const { pathToFileURL } = require('node:url');
const assert = require('node:assert/strict');

const wait = ms => new Promise(resolve => setTimeout(resolve, ms));
const temp = fsSync.mkdtempSync(path.join(os.tmpdir(), 'splat-deep-smoke-'));
const errors = [];
const replies = [];
let window;
let finishing = false;
function finish(code) {
  if (finishing) return;
  finishing = true;
  clearTimeout(timeout);
  // Destroy only this test process's windows. window-all-closed executes the
  // production engine cleanup before exit, including when the test fails.
  for (const w of BrowserWindow.getAllWindows()) w.destroy();
  app.exit(code);
}
const timeout = setTimeout(() => {
  console.error('Deep Fill smoke timed out; artifacts:', temp);
  finish(1);
}, 330000);

fsSync.mkdirSync(path.join(temp, 'profile'));
app.setPath('userData', path.join(temp, 'profile'));
app.setPath('sessionData', path.join(temp, 'profile'));
protocol.registerSchemesAsPrivileged([{ scheme: 'app', privileges: {
  standard: true, secure: true, supportFetchAPI: true, stream: true
} }]);
protocol.registerSchemesAsPrivileged = () => {};

// Observe replies without bypassing the production IPC handler, sidecar,
// renderer selection, or repair code. A capability flag alone is insufficient:
// the real repair reply below must identify LaMa and a completed GPU fit.
const register = ipcMain.handle.bind(ipcMain);
ipcMain.handle = (channel, handler) => register(channel, async (...args) => {
  const reply = await handler(...args);
  if (channel === 'engine:call') {
    replies.push({ command: args[1], header: reply.header });
  }
  return reply;
});

(async () => {
  const scan = path.join(temp, 'fixture.ply');
  const samples = [];
  // A flat, textured surface with a small red blemish at its centre. The
  // surrounding surface supplies geometry and appearance for the repair.
  for (let x = -40; x <= 40; x++) for (let y = -40; y <= 40; y++) {
    const px = x * .05, py = y * .05;
    const tone = .5 + .12 * Math.sin(px * 5) * Math.cos(py * 5);
    const rgb = Math.hypot(px, py) < .2 ? [.8, .15, .15] : [tone, tone * .9, tone * .8];
    samples.push([px, py, ...rgb]);
  }
  const columns = ['x', 'y', 'z', 'f_dc_0', 'f_dc_1', 'f_dc_2', 'opacity',
    'scale_0', 'scale_1', 'scale_2', 'rot_0', 'rot_1', 'rot_2', 'rot_3'];
  const header = `ply\nformat binary_little_endian 1.0\nelement vertex ${samples.length}\n`
    + columns.map(n => `property float ${n}\n`).join('') + 'end_header\n';
  const binary = Buffer.alloc(samples.length * columns.length * 4);
  samples.forEach(([x, y, r, g, b], i) => {
    const values = [x, y, 0, (r - .5) / .28209479177387814,
      (g - .5) / .28209479177387814, (b - .5) / .28209479177387814,
      Math.log(.95 / .05), Math.log(.045), Math.log(.045), Math.log(.01), 1, 0, 0, 0];
    values.forEach((v, j) => binary.writeFloatLE(v, (i * columns.length + j) * 4));
  });
  await fs.writeFile(scan, Buffer.concat([Buffer.from(header), binary]));
  dialog.showOpenDialog = async () => ({ canceled: false, filePaths: [scan] });
  app.on('browser-window-created', (_event, w) => {
    window = w;
    w.setOpacity(0);
    w.setIgnoreMouseEvents(true);
    w.setSkipTaskbar(true);
    w.show = () => w.showInactive();
    w.webContents.setBackgroundThrottling(false);
    w.webContents.on('console-message', e => {
      if (e.level === 'error' || e.level === 3) errors.push(e.message);
    });
    w.webContents.on('render-process-gone', (_event, detail) => {
      errors.push(`Renderer crashed: ${detail.reason}`);
    });
  });
  await import(pathToFileURL(path.resolve(__dirname, '../dist/electron/main.mjs')).href);
  const run = script => window.webContents.executeJavaScript(script);
  const until = async (condition, label, budget = 30000) => {
    const end = Date.now() + budget;
    while (Date.now() < end) {
      if (await condition()) return;
      await wait(100);
    }
    throw new Error(`Timed out: ${label}`);
  };
  await until(() => window && !window.webContents.isLoading(), 'window');
  await until(() => run(`!!window.__test && /ready/.test(document.getElementById('status-msg').textContent)`), 'engine ready');
  await until(() => run(`!document.getElementById('depth-deep').disabled
    && !document.querySelector('#deep-fill option[value="lama"]').disabled`), 'GPU and LaMa controls enabled', 180000);
  const caps = await run('window.__test.caps()');
  assert.equal(caps.gpu, true, JSON.stringify(caps));
  assert.equal(caps.viewFill, true, JSON.stringify(caps));
  assert.equal(caps.deep, true, JSON.stringify(caps));
  assert.equal(caps.inpainter, 'LaMa');
  console.log('Deep Fill capabilities:', JSON.stringify(caps));

  await run(`document.getElementById('btn-open').click()`);
  await until(() => run(`window.__test.stats().total === ${samples.length}`), 'fixture opened');
  await run(`window.__test.upAxis('+y'); window.__test.aim([0, 0, 0], [0, 0, -1], 5)`);
  await run('window.__test.settle()');
  await run('window.__test.objectSelection(false)');
  await run(`document.querySelector('[data-reach="through"]').click()`);
  const selected = await run(`(() => {
    const pose = window.__test.aim([0, 0, 0], [0, 0, -1], 5);
    return window.__test.paintWorld([[0, 0, 0]], .3 / pose.worldPerPixel).selected;
  })()`);
  assert.ok(selected >= 20 && selected < samples.length / 5, `Expected a local surface patch, selected ${selected}`);
  assert.equal(await run(`window.__test.setDepth('deep')`), 'deep');
  assert.equal(await run(`window.__test.setFill('lama')`), 'lama');
  console.log('Repairing synthetic patch:', selected, 'selected splats');
  const result = await run('window.__test.repair()');
  const repair = replies.filter(r => r.command === 'deep_repair').at(-1)?.header;
  assert.ok(repair?.ok, JSON.stringify(repair ?? result));
  assert.ok(repair.addedCount > 0, 'Repair must produce new splats, not only delete');
  assert.equal(repair.info.fill_source, 'LaMa', JSON.stringify(repair.info));
  assert.ok(repair.info.fill_seconds > 0, 'LaMa fill timing is recorded');
  assert.equal(typeof repair.info.deep, 'boolean');
  assert.ok(Number.isFinite(repair.info.fit), 'The GPU fit completed and returned an image error');
  assert.ok(!repair.info.fill_fallback, 'No alternative inpainter was used');
  assert.ok(!(await run(`document.getElementById('status-msg').classList.contains('err')`)), result.status);
  assert.equal(result.generated, repair.addedCount);
  assert.deepEqual(errors, [], 'No renderer or shader errors');
  // The quality guard may correctly keep the geometric repair. This verifies
  // execution and integration, not a promise that LaMa always looks better.
  if (!repair.info.deep) assert.ok(repair.info.rejected, 'Rejected refinement explains why');
  if (process.env.SPLAT_DEEP_SHOTS) {
    await fs.mkdir(process.env.SPLAT_DEEP_SHOTS, { recursive: true });
    await wait(1200);
    await fs.writeFile(path.join(process.env.SPLAT_DEEP_SHOTS, 'deep-fill.png'),
      (await window.webContents.capturePage()).toPNG());
  }
  const undo = await run('window.__test.undo()');
  assert.equal(undo.alive, samples.length, 'Undo restores the entire original fixture');
  assert.equal(undo.generated, 0, 'Undo removes generated repair splats');
  const report = { capabilities: caps, selected, repair: repair.info,
    added: repair.addedCount, seconds: repair.seconds, status: result.status, undo };
  await fs.writeFile(path.join(temp, 'report.json'), JSON.stringify(report, null, 2));
  console.log('Deep Fill smoke passed:', JSON.stringify(report));
  console.log('Artifacts:', temp);
  finish(0);
})().catch(error => {
  console.error(error);
  console.error('Artifacts:', temp);
  console.error('Renderer errors:', errors);
  console.error('Last engine replies:', JSON.stringify(replies.slice(-3)));
  finish(1);
});
