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

// Drive the actual pointer handlers: the preset mapping test alone cannot
// catch an accidental vertical inversion in the movement code.
class TestCanvas extends EventTarget {
  getBoundingClientRect() { return { left: 0, top: 0 }; }
  setPointerCapture() {}
  releasePointerCapture() {}
}
const pointer = (type, button, y) => Object.assign(new Event(type), {
  button, clientX: 10, clientY: y, pointerId: 1,
  shiftKey: false, ctrlKey: false, metaKey: false, altKey: false
});
for (const [preset, button, expected] of [['metashape', 1, -10], ['blender', 1, 10], ['metashape', 0, null]]) {
  globalThis.window = new EventTarget();
  const v = new Viewer();
  v.preset = preset;
  v.leftNavigates = true;
  const c = new TestCanvas();
  let delta = null;
  v.turn = (_x, y) => { delta = y; };
  v.beginGesture = () => {};
  v.leaveAutoOrtho = () => {};
  if (button === 0) { v.worldPerPixelPerUnit = () => 0; v.camera = { right: { clone: () => ({ mulScalar: () => ({}) }) }, up: { clone: () => ({ mulScalar: () => ({}) }) } }; v.target = { add() { return this; } }; v.applyCamera = () => {}; }
  v.attachControls(c);
  c.dispatchEvent(pointer('pointerdown', button, 0));
  c.dispatchEvent(pointer('pointermove', button, 10));
  assert.equal(delta, expected, `${preset}, button ${button}: vertical orbit delta`);
}
console.log('Navigation preset checks passed');
