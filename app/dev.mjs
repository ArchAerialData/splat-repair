// Vite dev server + Electron pointed at it, so the renderer hot-reloads.
import { createServer } from 'vite';
import { spawn } from 'node:child_process';
import { build } from 'esbuild';

await build({ entryPoints: ['electron/main.ts'], outfile: 'dist/electron/main.mjs',
  bundle: true, platform: 'node', format: 'esm', target: 'node20',
  external: ['electron'], logLevel: 'error' });
await build({ entryPoints: ['electron/preload.ts'], outfile: 'dist/electron/preload.cjs',
  bundle: true, platform: 'node', format: 'cjs', target: 'node20',
  external: ['electron'], logLevel: 'error' });

const server = await createServer();
await server.listen();
const url = server.resolvedUrls.local[0];
console.log('renderer on ' + url);
const electron = (await import('electron')).default;
const child = spawn(electron, ['.'], {
  stdio: 'inherit', env: { ...process.env, VITE_DEV_SERVER_URL: url }
});
child.on('exit', async () => { await server.close(); process.exit(0); });
