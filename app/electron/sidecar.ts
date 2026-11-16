/**
 * Supervises the Python engine process and speaks its framed stdio protocol.
 *
 * Framing, both directions:
 *   [u32 headerLen][u32 payloadLen][header JSON utf-8][payload bytes]
 *
 * No socket, no port, no HTTP. The pipe is private to this process pair, so
 * there is no Local Network Access prompt, no firewall dialog, and no way for
 * another page on the machine to drive the engine.
 */
import { spawn, ChildProcessWithoutNullStreams } from 'node:child_process';
import { existsSync } from 'node:fs';
import path from 'node:path';
import { EventEmitter } from 'node:events';

export interface Reply {
  header: Record<string, any>;
  payload: Buffer;
}

type Pending = {
  resolve: (r: Reply) => void;
  reject: (e: Error) => void;
};

export class Sidecar extends EventEmitter {
  private proc: ChildProcessWithoutNullStreams | null = null;
  /**
   * What has arrived and is not a whole frame yet: the pipe's chunks as they
   * came, and how many bytes they hold. They used to be joined on every chunk,
   * which copies everything received so far each time -- a 128 MB reply in
   * 64 kB chunks cost 16 s of this process. Joined once now, when the frame
   * is complete.
   */
  private chunks: Buffer[] = [];
  private buffered = 0;
  /** Bytes the frame at the front needs in all; 0 until its 8-byte head is here. */
  private need = 0;
  private pending = new Map<number, Pending>();
  private nextId = 1;
  private ready: Promise<void> | null = null;
  private stderrTail: string[] = [];
  /** The interpreter's own pid, from its "ready" frame (see `stop`). */
  private enginePid = 0;

  constructor(private coreDir: string) {
    super();
  }

  private pythonPath(): string {
    // Prefer the project venv; fall back to whatever python is on PATH so a
    // fresh checkout still runs before anyone has made the venv.
    const venv = path.join(this.coreDir, '.venv', 'Scripts', 'python.exe');
    if (existsSync(venv)) return venv;
    const posix = path.join(this.coreDir, '.venv', 'bin', 'python');
    if (existsSync(posix)) return posix;
    return process.platform === 'win32' ? 'python' : 'python3';
  }

  start(): Promise<void> {
    if (this.ready) return this.ready;
    const py = this.pythonPath();
    const script = path.join(this.coreDir, 'server.py');
    this.proc = spawn(py, ['-u', script], {
      cwd: this.coreDir,
      stdio: ['pipe', 'pipe', 'pipe'],
      windowsHide: true
    });

    this.ready = new Promise<void>((resolve, reject) => {
      let started = false;
      const onReady = () => { started = true; resolve(); };
      this.once('ready', onReady);
      // `on`, not `once`: a second 'error' without a listener would take the
      // main process down. After the first one the reject is a no-op.
      this.proc!.on('error', (e: NodeJS.ErrnoException) => {
        // "spawn python ENOENT" told nobody what to do about it.
        reject(new Error(e.code === 'ENOENT'
          ? `Python was not found (${py}). The engine needs Python 3, `
            + 'either in core\\.venv or on the PATH.'
          : `The engine could not be started: ${e.message}`));
      });
      this.proc!.once('exit', (code) => {
        this.off('ready', onReady);
        if (!started) reject(new Error(this.startFailure(code)));
        this.failPending('The engine quit before it could answer.');
        this.emit('exit', code);
      });
    });
    // A write to an engine that has gone ends in EPIPE on this stream. Without
    // a listener that is an uncaught exception in the main process.
    this.proc.stdin.on('error', (e: Error) => {
      this.emit('log', `stdin: ${e.message}\n`);
      this.failPending('The engine no longer accepts input. Restart the app.');
    });

    this.proc.stdout.on('data', (chunk: Buffer) => this.onData(chunk));
    this.proc.stderr.on('data', (chunk: Buffer) => {
      const s = chunk.toString('utf-8');
      this.stderrTail.push(s);
      if (this.stderrTail.length > 40) this.stderrTail.shift();
      this.emit('log', s);
    });
    return this.ready;
  }

  /** The engine died before its "ready" frame: say why in a sentence, not as a traceback. */
  private startFailure(code: number | null): string {
    const tail = this.stderrTail.join('');
    const missing = tail.match(/No module named '([^']+)'/);
    if (missing) {
      return `The engine does not start: the Python module "${missing[1]}" is missing. `
        + "Install it in the engine's Python environment (core\\.venv).";
    }
    const lines = tail.split(/\r?\n/).map(l => l.trim()).filter(Boolean);
    const last = lines[lines.length - 1];
    return `The engine stopped during startup (code ${code}).`
      + (last ? ` Last message: ${last}` : '');
  }

  /** Every call still waiting gets an answer: the engine will not send one. */
  private failPending(why: string) {
    for (const [, p] of this.pending) p.reject(new Error(why));
    this.pending.clear();
  }

  private onData(chunk: Buffer) {
    this.chunks.push(chunk);
    this.buffered += chunk.length;
    for (;;) {
      if (!this.need) {
        if (this.buffered < 8) return;
        // the 8-byte head may itself be split across chunks
        if (this.chunks[0].length < 8) this.chunks = [Buffer.concat(this.chunks, this.buffered)];
        const head = this.chunks[0];
        this.need = 8 + head.readUInt32LE(0) + head.readUInt32LE(4);
      }
      if (this.buffered < this.need) return;
      // The whole frame is here: join what it arrived in, once.
      const joined = this.chunks.length > 1;
      const buf = joined ? Buffer.concat(this.chunks, this.buffered) : this.chunks[0];
      const hl = buf.readUInt32LE(0);
      const total = this.need;
      const text = hl ? buf.subarray(8, 8 + hl).toString('utf-8') : '';
      let header: Record<string, any> | null = null;
      try {
        header = text ? JSON.parse(text) : {};
      } catch (e) {
        this.emit('log', `bad frame header: ${e}\n`);
      }
      // A single chunk belongs to the stream, so the payload is copied out of
      // it; a joined buffer is ours alone and the payload can stay a view.
      const payload = joined ? buf.subarray(8 + hl, total)
                             : Buffer.from(buf.subarray(8 + hl, total));
      // What follows is the start of the next frame, at most the rest of the
      // last chunk. Copied out of a joined buffer, or those few bytes would
      // keep the whole reply alive until the next one is complete.
      const rest = buf.subarray(total);
      this.chunks = rest.length ? [joined ? Buffer.from(rest) : rest] : [];
      this.buffered = rest.length;
      this.need = 0;

      if (!header) {
        // The header could not be read (Python writes NaN, which JSON does not
        // know). It used to be dropped: the call it answered never resolved and
        // the busy overlay stayed for good. The engine answers one command at a
        // time, in order, so the reply belongs to the oldest call still open.
        // A lost progress report is not worth failing the call it reports on.
        if (/^\s*\{\s*"event"\s*:/.test(text)) continue;
        const oldest = this.pending.entries().next();
        if (!oldest.done) {
          this.pending.delete(oldest.value[0]);
          oldest.value[1].reject(new Error(
            'The engine sent a reply that could not be read. '
            + 'If something looks wrong afterwards, open the file again.'));
        }
        continue;
      }

      // Event frames share the request id they belong to, so they must be
      // routed before the pending lookup -- otherwise a progress report would
      // resolve the call it is reporting on, mid-operation.
      if (header.event) {
        if (header.event === 'ready' && typeof header.pid === 'number') this.enginePid = header.pid;
        this.emit(header.event === 'ready' ? 'ready' : 'event', header);
        continue;
      }
      const p = this.pending.get(header.id);
      if (p) {
        this.pending.delete(header.id);
        p.resolve({ header, payload });
      }
    }
  }

  async call(cmd: string, args: Record<string, any> = {},
             payload?: Uint8Array): Promise<Reply> {
    await this.start();
    if (!this.proc || this.proc.exitCode !== null) {
      throw new Error('The engine is not running. Restart the app.');
    }
    const id = this.nextId++;
    const header = Buffer.from(JSON.stringify({ ...args, cmd, id }), 'utf-8');
    const body = payload ? Buffer.from(payload.buffer, payload.byteOffset, payload.byteLength)
                         : Buffer.alloc(0);
    const head = Buffer.alloc(8);
    head.writeUInt32LE(header.length, 0);
    head.writeUInt32LE(body.length, 4);

    return new Promise<Reply>((resolve, reject) => {
      this.pending.set(id, { resolve, reject });
      this.proc!.stdin.write(head);
      this.proc!.stdin.write(header);
      if (body.length) this.proc!.stdin.write(body);
    });
  }

  stop() {
    if (!this.proc) return;
    const p = this.proc;
    this.proc = null;
    try {
      p.stdin.end();
    } catch { /* already gone */ }
    if (p.exitCode !== null) return;
    // Now, not after a grace period: stop() runs while the app quits, and a
    // timer set here died with the app before it fired -- an engine in the
    // middle of a deep repair then lived on with its memory and the GPU's.
    try { p.kill('SIGKILL'); } catch { /* gone */ }
    // The venv's python.exe on Windows is a launcher that starts the real
    // interpreter as its child. The engine names its own pid when it is ready,
    // so that one is ended as well rather than left to notice the closed pipe.
    if (this.enginePid && this.enginePid !== p.pid) {
      try { process.kill(this.enginePid, 'SIGKILL'); } catch { /* gone */ }
    }
  }
}
