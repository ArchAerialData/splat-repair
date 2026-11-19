import path from 'node:path';
import { mkdir, readFile, writeFile, rename, stat } from 'node:fs/promises';

/** Best-effort picker preference. Serialized writes avoid temp-file races. */
export class OpenFolderPreference {
  private folder: string | undefined;
  private pending: Promise<void> = Promise.resolve();
  constructor(private settingsFile: () => string) {}

  remember(file: string): Promise<void> {
    const folder = path.dirname(path.resolve(file));
    this.folder = folder;
    this.pending = this.pending.then(async () => {
      const settings = this.settingsFile();
      await mkdir(path.dirname(settings), { recursive: true });
      await writeFile(settings + '.tmp', JSON.stringify({ folder }), 'utf8');
      await rename(settings + '.tmp', settings);
    }).catch(error => { console.warn('Could not save the Open dialog folder:', error); });
    return this.pending;
  }

  async get(): Promise<string | undefined> {
    try {
      if (!this.folder) {
        const saved = JSON.parse(await readFile(this.settingsFile(), 'utf8'));
        if (typeof saved?.folder === 'string' && path.isAbsolute(saved.folder)) this.folder = saved.folder;
      }
      if (this.folder && (await stat(this.folder)).isDirectory()) return this.folder;
    } catch { /* Invalid preferences or an unavailable drive use the system default. */ }
    return undefined;
  }
}
