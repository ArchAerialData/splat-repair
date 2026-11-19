// Run from app/: node smoke/selection-inspection.mjs
import { build } from 'esbuild';
import assert from 'node:assert/strict';
const result = await build({ entryPoints: ['src/ui/selection-inspection.ts'],
  bundle: true, platform: 'node', format: 'esm', write: false });
const { selectionMarkers, MARKER_LIMIT } = await import('data:text/javascript;base64,'
  + Buffer.from(result.outputFiles[0].text).toString('base64'));
const identity = new Float32Array([1, 0, 0, 0, 0, 1, 0, 0, 0, 0, 1, 0, 0, 0, 0, 1]);
const centres = new Float32Array([0, 0, 0, .5, .5, 0, 2, 0, 0, 0, 0, 2, NaN, 0, 0]);
const original = centres.slice();
const indices = new Uint32Array([0, 1, 2, 3, 4]);
assert.deepEqual(selectionMarkers(centres, indices, new Uint8Array([1, 1, 1, 1, 1]), identity, 100, 100),
  [50, 50, 75, 25], 'Only finite, on-screen, depth-clipped centres get markers');
assert.deepEqual(selectionMarkers(centres, indices, new Uint8Array([0, 1, 1, 1, 1]), identity, 100, 100),
  [75, 25], 'Deleted/cropped splats have no markers');
const behind = identity.slice(); behind[15] = -1;
assert.deepEqual(selectionMarkers(centres, indices, new Uint8Array(5).fill(1), behind, 100, 100), []);
const translated = identity.slice(); translated[12] = .5;
assert.deepEqual(selectionMarkers(centres, new Uint32Array([0]), new Uint8Array([1]), translated, 100, 100), [75, 50]);
const n = MARKER_LIMIT * 3 + 1;
assert.ok(selectionMarkers(new Float32Array(n * 3), Uint32Array.from({ length: n }, (_, i) => i),
  new Uint8Array(n).fill(1), identity, 100, 100).length <= MARKER_LIMIT * 2);
assert.deepEqual(centres, original, 'Inspection never changes document coordinates');
console.log('Selection inspection checks passed');
