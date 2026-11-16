/**
 * What the stamp will put down, shown before the engine has put it down.
 *
 * Two layers over the viewport, both copies of the rendered image:
 *
 * - **Peek:** while the brush only hovers, the brush disc shows what lies
 *   under the crosshair, half transparent and with the stamp's soft edge --
 *   Photoshop's clipped clone overlay. Copied from the frame just rendered,
 *   so it is exactly what the camera sees at the source.
 * - **Live:** while painting, each dab copies the source's pixels to the
 *   brush, the way a 2-D stamp would. The engine stamps the whole stroke on
 *   release (about 70 ms a dab on the hut, plus building the stroke's search
 *   tree -- too slow to keep up with a drag), and the provisional picture
 *   stays until its result is on screen.
 *
 * Both are taken from the WebGL/WebGPU canvas in the frame's `frameend`,
 * the one moment its drawing buffer is guaranteed to hold the frame.
 * The pixels come from a snapshot taken when the stroke starts, so a dab
 * never copies a dab: the engine does not either, it copies the scan.
 */

type Pt = { x: number; y: number };

interface FrameSource { on(ev: 'frameend', fn: () => void): unknown }

interface Dab { at: Pt; src: Pt; r: number; feather: number }

export class ClonePreview {
  private peek: HTMLCanvasElement;
  private live: HTMLCanvasElement;
  private snap: HTMLCanvasElement | null = null;
  private dabCanvas = document.createElement('canvas');
  private pending: Dab[] = [];
  private painting = false;
  private peekAt: { at: Pt; src: Pt; r: number; feather: number } | null = null;
  private clearIn = -1;
  private liveDirty = false;
  /** Dabs drawn in the current stroke, for checks. */
  dabsDrawn = 0;

  constructor(private gl: HTMLCanvasElement, host: HTMLElement, before: Element | null,
              frames: FrameSource) {
    this.live = document.createElement('canvas');
    this.live.id = 'clone-live';
    this.peek = document.createElement('canvas');
    this.peek.id = 'clone-peek';
    host.insertBefore(this.live, before);
    host.insertBefore(this.peek, before);
    frames.on('frameend', () => this.onFrame());
  }

  /** Device pixels per CSS pixel of the rendered canvas. */
  private get scale() {
    const w = this.gl.clientWidth || 1;
    return this.gl.width / w;
  }

  /**
   * Show the source inside the brush at `at`, or hide it (`null`).
   * `feather` is the soft edge as a fraction of the radius, as the engine
   * gets it.
   */
  hover(at: Pt | null, src: Pt | null, r: number, feather: number) {
    this.peekAt = at && src && !this.painting ? { at, src, r, feather } : null;
    if (!this.peekAt) this.peek.classList.remove('on');
  }

  /** A stroke starts: snapshot the next frame, then draw dabs as they come. */
  begin() {
    this.painting = true;
    this.peekAt = null;
    this.peek.classList.remove('on');
    this.snap = null;
    this.pending = [];
    this.dabsDrawn = 0;
    this.clearIn = -1;
    this.sizeLive();
    const ctx = this.live.getContext('2d')!;
    ctx.clearRect(0, 0, this.live.width, this.live.height);
    this.live.classList.add('on');
  }

  /** One dab: the pixels around `src` copied to `at`, both in CSS px. */
  dab(at: Pt, src: Pt, r: number, feather: number) {
    if (!this.painting) return;
    this.pending.push({ at, src, r, feather });
    if (this.snap) this.flush();
  }

  /** The stroke has been sent; keep the picture until the result shows. */
  end() { this.painting = false; }

  /** For checks: is the source shown in the brush, and the stroke drawn. */
  state() {
    return { peek: this.peek.classList.contains('on'),
             live: this.live.classList.contains('on'), dabs: this.dabsDrawn,
             coverage: this.coverage() };
  }

  /** The engine's result is in: drop the provisional picture after it has
   *  been rendered (two frames, so the new splats are on screen first). */
  done() {
    this.painting = false;
    this.pending = [];
    this.clearIn = 2;
  }

  /** Drop everything now (nothing was stamped, or the tool changed). */
  cancel() {
    this.painting = false;
    this.pending = [];
    this.snap = null;
    this.clearIn = -1;
    this.live.classList.remove('on');
    this.peek.classList.remove('on');
    this.peekAt = null;
  }

  private sizeLive() {
    if (this.live.width !== this.gl.width || this.live.height !== this.gl.height) {
      this.live.width = this.gl.width;
      this.live.height = this.gl.height;
    }
  }

  private onFrame() {
    if (this.clearIn >= 0 && this.clearIn-- === 0) {
      this.live.classList.remove('on');
      this.snap = null;
    }
    // also after release: a short stroke can end before a frame was drawn
    if ((this.painting || this.pending.length) && !this.snap) {
      const c = document.createElement('canvas');
      c.width = this.gl.width;
      c.height = this.gl.height;
      c.getContext('2d')!.drawImage(this.gl, 0, 0);
      this.snap = c;
      this.flush();
    }
    if (this.peekAt) this.drawPeek(this.peekAt);
  }

  /** The disc of `img` around `src`, softened, into `ctx` around `at`. */
  private stamp(ctx: CanvasRenderingContext2D, img: CanvasImageSource, d: Dab, k: number) {
    const R = Math.max(2, d.r * k);
    const size = Math.ceil(2 * R);
    const dc = this.dabCanvas;
    if (dc.width !== size || dc.height !== size) { dc.width = size; dc.height = size; }
    const x = dc.getContext('2d')!;
    x.globalCompositeOperation = 'copy';
    x.drawImage(img, d.src.x * k - R, d.src.y * k - R, size, size, 0, 0, size, size);
    x.globalCompositeOperation = 'destination-in';
    const g = x.createRadialGradient(R, R, 0, R, R, R);
    g.addColorStop(0, 'rgba(0,0,0,1)');
    g.addColorStop(Math.max(0, Math.min(0.999, 1 - d.feather)), 'rgba(0,0,0,1)');
    g.addColorStop(1, 'rgba(0,0,0,0)');
    x.fillStyle = g;
    x.fillRect(0, 0, size, size);
    ctx.drawImage(dc, d.at.x * k - R, d.at.y * k - R);
  }

  private flush() {
    if (!this.snap || !this.pending.length) return;
    this.sizeLive();
    const ctx = this.live.getContext('2d')!;
    const k = this.scale;
    for (const d of this.pending) {
      this.stamp(ctx, this.snap, d, k);
      this.dabsDrawn++;
    }
    this.pending = [];
    this.liveDirty = true;
  }

  private drawPeek(p: { at: Pt; src: Pt; r: number; feather: number }) {
    const k = this.scale;
    const R = Math.max(2, p.r * k);
    const size = Math.ceil(2 * R);
    if (this.peek.width !== size || this.peek.height !== size) {
      this.peek.width = size;
      this.peek.height = size;
    }
    const ctx = this.peek.getContext('2d')!;
    ctx.clearRect(0, 0, size, size);
    // the disc is drawn at its own origin; the element is moved to `at`
    this.stamp(ctx, this.gl, { at: { x: R / k, y: R / k }, src: p.src, r: p.r,
                               feather: p.feather }, k);
    this.peek.style.setProperty('left', `${p.at.x}px`);
    this.peek.style.setProperty('top', `${p.at.y}px`);
    this.peek.style.setProperty('width', `${size / k}px`);
    this.peek.style.setProperty('height', `${size / k}px`);
    this.peek.classList.add('on');
  }

  /** For checks: how much of the live layer is painted, 0..1. */
  coverage(): number {
    if (!this.live.classList.contains('on') || !this.liveDirty) return 0;
    const ctx = this.live.getContext('2d')!;
    const { width: w, height: h } = this.live;
    if (!w || !h) return 0;
    const px = ctx.getImageData(0, 0, w, h).data;
    let n = 0;
    for (let i = 3; i < px.length; i += 16) if (px[i] > 8) n++;
    return n / (px.length / 16);
  }
}
