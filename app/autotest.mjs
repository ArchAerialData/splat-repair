/**
 * Drives the built app end to end with no human: open a splat, orbit to the
 * sole, paint over the stand, repair, screenshot each step. The strokes and
 * camera angles were measured on a shoe scan; on other files some checks
 * will not find what they aim at.
 *
 * Run:  node autotest.mjs <splat.ply> [outDir]
 */
import { spawn } from 'node:child_process';
import path from 'node:path';
import { fileURLToPath } from 'node:url';

const __dirname = path.dirname(fileURLToPath(import.meta.url));
const ply = process.argv[2];
if (!ply) {
  console.error('Usage: node autotest.mjs <splat.ply> [outDir]\n\n'
    + '  <splat.ply>  splat file to run the test on (required)\n'
    + '  [outDir]     folder for screenshots and log.txt (default: ./shots)');
  process.exit(2);
}
const out = process.argv[3] ?? path.join(__dirname, 'shots');

const electron = (await import('electron')).default;
const child = spawn(electron, ['.'], {
  cwd: __dirname,
  stdio: 'inherit',
  env: { ...process.env, SPLAT_AUTOTEST: path.resolve(ply), SPLAT_AUTOTEST_OUT: path.resolve(out) }
});
child.on('exit', (code) => process.exit(code ?? 0));
