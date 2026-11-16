/**
 * The viewport's PLY copy of an opened .splat or .spz, and its removal.
 *
 * The engine writes that copy into a folder of its own in the temp directory
 * (core/server.py, _view_mirror: "splatcore-view-…", one folder per engine)
 * and removes the folder when it ends. On quit it does not end, it is killed
 * (sidecar.ts, stop), and the folder stayed behind: a whole PLY, until an
 * engine a day later swept it up. So the main process removes it.
 *
 * Only that folder, ever: directly in the temp directory, named with the
 * engine's prefix, a real directory and no link. Nothing is removed
 * recursively either: the files in it, then the folder if that emptied it.
 */
import fs from 'node:fs';
import os from 'node:os';
import path from 'node:path';

/** core/server.py, MIRROR_PREFIX. */
export const MIRROR_PREFIX = 'splatcore-view-';

/**
 * Where the engine's temp directory can be. Python asks TMPDIR, TEMP and TMP
 * in that order, node only the last two: all of them, so the two sides
 * cannot disagree about which one it is.
 */
export function tempDirs(): string[] {
  const all = [os.tmpdir(), process.env.TMPDIR, process.env.TEMP, process.env.TMP];
  return [...new Set(all.filter((d): d is string => !!d && path.isAbsolute(d)))];
}

/** As the file system has it: short names and links resolved, where it exists. */
function real(p: string): string {
  try { return fs.realpathSync.native(p); } catch { return path.resolve(p); }
}

function samePath(a: string, b: string): boolean {
  const x = path.normalize(real(a)), y = path.normalize(real(b));
  return process.platform === 'win32' ? x.toLowerCase() === y.toLowerCase() : x === y;
}

/**
 * The engine's mirror folder that `viewPath` lies in, or null: for the
 * opened PLY itself (most opens) and for any path that is not exactly
 * <temp directory>/splatcore-view-…/<file>.
 */
export function mirrorFolderOf(viewPath: unknown, temp: string[] = tempDirs()): string | null {
  if (typeof viewPath !== 'string' || !path.isAbsolute(viewPath)) return null;
  const folder = path.dirname(path.normalize(viewPath));
  if (!path.basename(folder).startsWith(MIRROR_PREFIX)) return null;
  const parent = path.dirname(folder);
  return temp.some(t => samePath(parent, t)) ? folder : null;
}

/**
 * Remove a mirror folder with the files in it. Checked again here, whatever
 * the caller holds. True when the folder is gone afterwards.
 */
export function removeMirrorFolder(folder: string | null, temp: string[] = tempDirs()): boolean {
  if (!folder) return false;
  const f = path.normalize(folder);
  if (mirrorFolderOf(path.join(f, 'x.ply'), temp) !== f) return false;
  try {
    const st = fs.lstatSync(f);
    if (!st.isDirectory() || st.isSymbolicLink()) return false;
    for (const e of fs.readdirSync(f, { withFileTypes: true })) {
      if (e.isFile()) unlinkHeld(path.join(f, e.name));
    }
    fs.rmdirSync(f);              // fails on anything left in it, and then it stays
  } catch { /* not there any more, or not empty */ }
  return !fs.existsSync(f);
}

/**
 * Delete one file. A copy the engine was still writing when it was killed is
 * let go a moment later, so a file that is held gets three more tries, a
 * third of a second in all; after that it is left for the engine's own sweep
 * (core/server.py, _sweep_mirrors). The waiting blocks: the app is quitting.
 */
function unlinkHeld(file: string): boolean {
  for (let k = 0; ; k++) {
    try {
      fs.unlinkSync(file);
      return true;
    } catch (e: any) {
      if (e?.code === 'ENOENT') return true;
      if (k >= 3 || !['EBUSY', 'EPERM', 'EACCES'].includes(e?.code)) return false;
      Atomics.wait(new Int32Array(new SharedArrayBuffer(4)), 0, 0, 55 * (k + 1));
    }
  }
}
