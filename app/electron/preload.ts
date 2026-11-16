/**
 * The only surface the renderer gets. Named commands in, structured data out --
 * no filesystem, no process, no arbitrary IPC channel.
 */
import { contextBridge, ipcRenderer, webUtils } from 'electron';

export interface EngineReply {
  header: Record<string, any>;
  payload: ArrayBuffer | null;
}

const api = {
  engine: {
    call(cmd: string, args: Record<string, any> = {},
         payload?: ArrayBuffer): Promise<EngineReply> {
      return ipcRenderer.invoke('engine:call', cmd, args, payload ?? null);
    },
    onLog(fn: (s: string) => void) {
      ipcRenderer.on('engine:log', (_e, s: string) => fn(s));
    },
    onExit(fn: (code: number) => void) {
      ipcRenderer.on('engine:exit', (_e, c: number) => fn(c));
    },
    /** Progress and other out-of-band notices from a long operation. */
    onEvent(fn: (e: Record<string, any>) => void) {
      ipcRenderer.on('engine:event', (_e, v: Record<string, any>) => fn(v));
    }
  },
  openSplatDialog: (): Promise<string | null> => ipcRenderer.invoke('dialog:openSplat'),
  /**
   * Edits that were never exported are about to go (another file is opened):
   * a native "Verwerfen / Abbrechen". Resolves true when they may go.
   */
  confirmDiscard: (): Promise<boolean> => ipcRenderer.invoke('dialog:confirmDiscard'),
  /**
   * The main process asks, before the window closes, whether the page holds
   * edits that were never exported. `fn` answers for the document as it is.
   */
  onUnexportedQuery(fn: () => boolean) {
    ipcRenderer.on('doc:ask-unexported', () => {
      let v = false;
      try { v = !!fn(); } catch { /* no answer is "nothing to lose" */ }
      ipcRenderer.send('doc:unexported', v);
    });
  },
  /**
   * The path on disk behind a file dropped onto the window ('' for one that
   * is not on disk). Electron removed File.path; this is its replacement,
   * and it only answers for File objects the OS handed to the page.
   */
  pathForFile: (f: File): string => {
    try { return webUtils.getPathForFile(f); } catch { return ''; }
  },
  /**
   * Where the viewport streams an opened splat from. Relative on purpose: it
   * resolves against whatever origin the page was served from -- the app://
   * bundle in production, the Vite dev server in development -- so the request
   * is always same-origin.
   */
  splatUrl: (p: string) =>
    `/_file/${p.replace(/\\/g, '/').split('/').map(encodeURIComponent).join('/')}`,
  saveSplatDialog: (suggested: string): Promise<string | null> =>
    ipcRenderer.invoke('dialog:saveSplat', suggested),
  /**
   * Close the window the way its X does. `window.close()` in the page closes
   * it at once, past the main process's question about unexported edits.
   */
  closeWindow: (): void => ipcRenderer.send('window:close')
};

contextBridge.exposeInMainWorld('host', api);
export type HostApi = typeof api;
