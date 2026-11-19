// Run from app/: node smoke/open-folder.mjs
import { build } from 'esbuild';
import assert from 'node:assert/strict';
import { mkdtemp, mkdir, writeFile, rm } from 'node:fs/promises';
import os from 'node:os';
import path from 'node:path';
const result = await build({entryPoints:['electron/open-folder.ts'], bundle:true,
  platform:'node', format:'esm', write:false});
const { OpenFolderPreference } = await import('data:text/javascript;base64,'
  + Buffer.from(result.outputFiles[0].text).toString('base64'));
const root = await mkdtemp(path.join(os.tmpdir(), 'splat-folder-test-'));
try {
  const settings = path.join(root, 'prefs', 'open-folder.json');
  const make = () => new OpenFolderPreference(() => settings);
  const first = path.join(root, 'first'), second = path.join(root, 'second');
  await mkdir(first); await mkdir(second);
  assert.equal(await make().get(), undefined);
  const pref = make();
  await Promise.all([pref.remember(path.join(first, 'scan.ply')), pref.remember(path.join(second, 'scan.ply'))]);
  assert.equal(await pref.get(), second);
  assert.equal(await make().get(), second, 'Last open survives a restart and concurrent writes');
  await rm(second, {recursive:true});
  assert.equal(await make().get(), undefined, 'Missing folder falls back');
  for (const data of ['null', '{broken', '{"folder":23}', '{"folder":"relative"}']) {
    await writeFile(settings, data);
    assert.equal(await make().get(), undefined);
  }
  console.log('Open folder preference checks passed');
} finally { await rm(root, {recursive:true, force:true}); }
