// Run from app/: node smoke/navigation.mjs. No GPU or window required.
import { build } from 'esbuild';
import assert from 'node:assert/strict';

const result = await build({ entryPoints: ['src/engine/viewer.ts'], bundle: true,
  platform: 'node', format: 'esm', write: false, logLevel: 'silent' });
const { Viewer } = await import('data:text/javascript;base64,'
  + Buffer.from(result.outputFiles[0].text).toString('base64'));
const viewer = Object.create(Viewer.prototype);
const event = button => ({ button, shiftKey: false, ctrlKey: false,
  metaKey: false, altKey: false });
viewer.preset = 'metashape';
viewer.leftNavigates = true;
for (const [button, gesture] of [[0, 'pan'], [1, 'orbit'], [2, 'zoom']]) {
  assert.equal(viewer.gestureFor(event(button)), gesture);
}
viewer.leftNavigates = false;
assert.equal(viewer.gestureFor(event(0)), null, 'Left drag belongs to editing tools');
assert.equal(viewer.gestureFor(event(1)), 'orbit');
assert.equal(viewer.gestureFor(event(2)), 'zoom');
assert.equal(viewer.gestureFor({ ...event(0), altKey: true }), 'orbit');
viewer.leftNavigates = true;
viewer.preset = 'blender';
assert.equal(viewer.gestureFor(event(0)), 'orbit');
assert.equal(viewer.gestureFor({ ...event(1), shiftKey: true }), 'pan');
viewer.preset = 'supersplat';
assert.equal(viewer.gestureFor(event(2)), 'pan');
assert.equal(viewer.gestureFor(event(1)), 'zoom');
console.log('Navigation preset checks passed');
