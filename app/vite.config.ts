import { defineConfig } from 'vite';
import fs from 'node:fs';
import path from 'node:path';

/**
 * In development the renderer is served over http, so opened splats must come
 * from the same origin too. Production does this in the Electron main process
 * against an explicit allow-list; the dev server is bound to localhost and
 * serves any .ply, which is a development-only convenience.
 */
function serveLocalSplats() {
  return {
    name: 'serve-local-splats',
    configureServer(server: any) {
      server.middlewares.use('/_file/', (req: any, res: any, next: any) => {
        const rel = decodeURIComponent((req.url ?? '').split('?')[0]).replace(/^\//, '');
        const file = path.normalize(rel);
        if (!/\.(ply|sog|splat)$/i.test(file) || !fs.existsSync(file)) return next();
        res.setHeader('Content-Type', 'application/octet-stream');
        res.setHeader('Content-Length', fs.statSync(file).size);
        fs.createReadStream(file).pipe(res);
      });
    }
  };
}

export default defineConfig({
  plugins: [serveLocalSplats()],
  root: 'src',
  publicDir: '../public',
  base: './',
  build: {
    outDir: '../dist/renderer',
    emptyOutDir: true,
    target: 'chrome120',
    // 130 MB splats are read from disk at runtime, never bundled
    assetsInlineLimit: 0
  },
  server: { port: 5181, strictPort: true }
});
