import assert from 'node:assert/strict';
import { build } from 'esbuild';
const result = await build({ stdin: { contents: `
export { throughBrushMask, ThroughBrushRaster } from './src/engine/through-brush';
export { selectInRegion } from './src/engine/selection';`, resolveDir: process.cwd() },
  bundle: true, platform: 'node', format: 'esm', write: false });
const { throughBrushMask, ThroughBrushRaster, selectInRegion } = await import('data:text/javascript;base64,'
  + Buffer.from(result.outputFiles[0].text).toString('base64'));
const mask = throughBrushMask([{ x: 10, y: 30, r: 4 }, { x: 80, y: 30, r: 4 }], 100, 60);
assert.equal(mask[30 * 100 + 45], 1, 'Fast strokes have no gaps between samples');
assert.equal(mask[36 * 100 + 45], 0, 'Outside the brush is untouched');
const projection = { sx: new Float32Array([45, 45, 45, 45, 45, 45]),
  sy: new Float32Array([30, 30, 30, 40, 30, 30]),
  dz: new Float32Array([1, 100, 10000, 1, 1, -1]), onScreen: new Uint8Array([1, 1, 1, 1, 1, 0]) };
const hits = selectInRegion(projection, mask, 100, 60, { depthMode: 'through',
  tolPx: 0, perPixelPerUnit: 1, alive: new Uint8Array([1, 1, 1, 1, 0, 1]) });
assert.deepEqual([...hits], [0, 1, 2], 'All depths included; outside, deleted/cropped and behind-camera points excluded');
assert.ok(throughBrushMask([{ x: 10, y: 10, r: 3 }], 20, 20)[210], 'Single click selects');
assert.equal(throughBrushMask([], 20, 20).some(Boolean), false);
assert.equal(throughBrushMask([{ x: -10, y: -10, r: 1 }], 20, 20).some(Boolean), false);
const raster = new ThroughBrushRaster();
const samples = [];
for (let i = 0; i < 30; i++) {
  samples.push({ x: i * 3, y: 20 + Math.sin(i) * 10, r: 2 + i % 5 });
  assert.deepEqual(raster.update(samples, 100, 60), throughBrushMask(samples, 100, 60), 'Incremental raster matches complete stroke');
}
assert.deepEqual(raster.update(samples, 200, 80), throughBrushMask(samples, 200, 80), 'Resize rerasterises');
assert.deepEqual(raster.update([], 200, 80), new Uint8Array(16000), 'Next stroke clears old pixels');
console.log('Through brush checks passed');
