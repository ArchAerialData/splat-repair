// Bundles the Electron main + preload. The renderer goes through Vite.
import { build } from 'esbuild';
const dev = process.argv.includes('--dev');
await build({
  entryPoints: ['electron/main.ts'],
  outfile: 'dist/electron/main.mjs',
  bundle: true, platform: 'node', format: 'esm', target: 'node20',
  external: ['electron'], sourcemap: dev, logLevel: 'info'
});
await build({
  entryPoints: ['electron/preload.ts'],
  outfile: 'dist/electron/preload.cjs',
  bundle: true, platform: 'node', format: 'cjs', target: 'node20',
  external: ['electron'], sourcemap: dev, logLevel: 'info'
});
