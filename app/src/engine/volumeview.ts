/**
 * The painted volume, drawn as its outline over the viewport.
 *
 * The orange tint shows which splats are selected. Over a hole there are no
 * splats to tint, so the tint cannot show that the hole is part of the
 * selection -- which is the point of painting it. The outline can: it is the
 * silhouette of the volume itself, from wherever the camera is.
 *
 * Drawn on a 2-D layer of its own: the boundary points of the volume
 * (volume.ts, `volumeShell`) are projected into a grid of 2 px cells, the
 * covered cells closed, and the edge of the covered area traced (marching
 * squares) and stroked twice -- a dark line under a light one, so it reads
 * on a white sole and on a black one. A faint wash fills it, enough to see
 * the hole is inside. Redrawn only when the camera, the viewport or the
 * volume change.
 *
 * What the scan hides is drawn quieter. A 2-D layer has no depth test, so a
 * volume painted on the sole showed through the whole shoe when seen from
 * above, as bright as the part in view. Each shell point is now tested
 * against the depth of the nearest opaque splats in its screen cell (a
 * strided sample of the centres, like the navigation's picking); an edge
 * with no visible shell point beside it is drawn dashed and faint.
 *
 * While a stroke is being painted the outline follows it (`preview`). The
 * shell is rebuilt with a smaller point budget and at most every few frames,
 * spaced by what the last rebuild cost, so the brush stays smooth.
 */
import { volumeShell, type Shell, type Volume } from './volume';
import type { NavData } from './navpick';

const CELL = 2;                 // CSS px per grid cell
const LINE = 'rgba(96, 165, 250, 0.95)';     // blue, matching the interface accent
const HALO = 'rgba(16, 12, 10, 0.62)';
const LINE_HIDDEN = 'rgba(96, 165, 250, 0.55)';
const HALO_HIDDEN = 'rgba(16, 12, 10, 0.35)';
const WASH = [96, 165, 250, 26];             // rgba, alpha out of 255
const WASH_HIDDEN = 10;
/** Dash of the hidden part: on and off blocks of this many CSS px. */
const DASH = 4;

/** Depth cells for the occlusion test, CSS px. */
const DEPTH_CELL = 6;
/** Centres sampled for it at most; more are strided. */
const DEPTH_SAMPLES = 160_000;
/** An edge counts as seen with a visible shell point this many cells away. */
const SEEN_REACH = 3;

/** Shell points while a stroke is live: a quarter of the final budget. */
const LIVE_BUDGET = 35_000;
/** Live rebuilds at least this far apart, ms; more when they cost more. */
const LIVE_MIN_MS = 90;
const LIVE_MAX_MS = 400;

type LiveSource = Volume | null | (() => Volume | null);

interface DepthMap { key: string; surf: Float32Array; cw: number; ch: number; }

export class VolumeOutline {
  private canvas: HTMLCanvasElement;
  private shell: Shell | null = null;
  private volume: Volume | null = null;
  private lastKey = '';
  private wash: HTMLCanvasElement = document.createElement('canvas');
  /** A live stroke's volume, waiting for its turn (see `preview`). */
  private pending: LiveSource | undefined = undefined;
  private liveShown = false;
  private lastLive = -Infinity;
  private liveGap = LIVE_MIN_MS;
  private depth: DepthMap | null = null;
  private object: Shell | null = null;
  visible = true;

  /**
   * `occluders`: what can hide the volume -- the scan's centres, which of
   * them are on screen, their opacity. Without it nothing is dimmed.
   */
  constructor(private stage: HTMLCanvasElement, after?: HTMLElement | null,
              private occluders?: () => NavData | null) {
    const c = document.createElement('canvas');
    c.id = 'volume-outline';
    c.setAttribute('aria-hidden', 'true');
    for (const [k, v] of [['position', 'absolute'], ['inset', '0'], ['width', '100%'],
                          ['height', '100%'], ['pointer-events', 'none']]) {
      c.style.setProperty(k, v);
    }
    const parent = stage.parentElement ?? document.body;
    parent.insertBefore(c, after?.nextSibling ?? stage.nextSibling);
    this.canvas = c;
  }

  /** Show this volume (null: nothing). Rebuilds the boundary points. */
  set(v: Volume | null) {
    this.pending = undefined;
    if (v === this.volume && !this.liveShown) return;
    this.liveShown = false;
    if (v !== this.volume) this.object = null;     // the object belonged to the old one
    this.volume = v;
    this.shell = withObject(v ? volumeShell(v) : null, this.object);
    this.depth = null;          // the splats may have changed along with it
    this.lastKey = '';
  }

  /**
   * The object the engine grew the stroke to ("Detect object"): its splat
   * centres, drawn inside the same outline as the painted volume, so what
   * the repair will take shows before anything changes -- its sides, its
   * faint edge, its drop shadow. `spacing`: a world size per point (the
   * splats' own), so neighbours close up. Null clears it; a new volume
   * clears it too.
   */
  setObject(pts: Float32Array | null, spacing = 0) {
    this.object = pts && pts.length ? { pts, spacing: new Float32Array(pts.length / 3).fill(spacing),
                                        count: (pts.length / 3) | 0 } : null;
    this.shell = withObject(this.volume ? volumeShell(this.volume) : null, this.object);
    this.liveShown = false;
    this.lastKey = '';
  }

  /** How many object points the outline holds (tests). */
  get objectCount() { return this.object?.count ?? 0; }

  /**
   * The volume of a stroke still being painted. Taken up on a later frame,
   * not at once: a function is only called when its turn comes, so the
   * caller can hand over the work instead of doing it every mouse move.
   * `set` with the stroke's result ends it.
   */
  preview(v: LiveSource) { this.pending = v; }

  get pointCount() { return this.shell?.count ?? 0; }

  /** Call once per frame; draws only when something changed. */
  frame(vp: ArrayLike<number>) {
    if (this.pending !== undefined) {
      const now = performance.now();
      if (now - this.lastLive >= this.liveGap) {
        const src = this.pending;
        this.pending = undefined;
        const v = typeof src === 'function' ? src() : src;
        const t0 = performance.now();
        this.shell = v ? volumeShell(v, LIVE_BUDGET) : null;
        this.liveShown = true;
        this.lastKey = '';
        this.lastLive = now;
        this.liveBuilds++;
        this.liveMs = performance.now() - t0;
      }
    }
    const r = this.stage.getBoundingClientRect();
    const dpr = Math.min(window.devicePixelRatio, 2);
    let key = `${r.width}x${r.height}@${dpr}:${this.visible}:${this.shell?.count ?? 0}`;
    for (let i = 0; i < 16; i++) key += ',' + (+vp[i]).toPrecision(7);
    if (key === this.lastKey) return;
    this.lastKey = key;
    this.draw(vp, r.width, r.height, dpr);
    if (this.liveShown) {
      // the next live rebuild waits three times what this one cost, all in
      this.liveGap = Math.min(LIVE_MAX_MS, Math.max(LIVE_MIN_MS, 3 * (this.liveMs + this.stats.ms)));
    }
  }

  /** Force a redraw on the next frame. */
  invalidate() { this.lastKey = ''; this.depth = null; }

  /** Live rebuilds since the app started, and what the last one cost (ms). */
  liveBuilds = 0;
  liveMs = 0;

  /**
   * The last draw, for tests: covered cells, edge segments, how many of
   * those are hidden behind the scan, and the time it took.
   */
  stats = { cells: 0, segments: 0, hidden: 0, ms: 0, live: false };

  /**
   * Camera depth of the nearest opaque splat per screen cell, from a strided
   * sample. Kept while the camera stands still (a live stroke redraws often
   * from the same view).
   */
  private depthMap(vp: ArrayLike<number>, W: number, H: number): DepthMap | null {
    const data = this.occluders?.();
    if (!data) return null;
    const c = data.centres();
    const n = (c.length / 3) | 0;
    if (!n) return null;
    let key = `${W}x${H}:${n}`;
    for (let i = 0; i < 16; i++) key += ',' + (+vp[i]).toPrecision(7);
    if (this.depth?.key === key) return this.depth;
    const vis = data.visible(), alpha = data.alpha(), minA = data.minAlpha();
    const cw = Math.max(1, Math.ceil(W / DEPTH_CELL)), ch = Math.max(1, Math.ceil(H / DEPTH_CELL));
    const raw = new Float32Array(cw * ch).fill(Infinity);
    const step = Math.max(1, Math.ceil(n / DEPTH_SAMPLES));
    const m0 = vp[0], m1 = vp[1], m3 = vp[3], m4 = vp[4], m5 = vp[5], m7 = vp[7];
    const m8 = vp[8], m9 = vp[9], m11 = vp[11], m12 = vp[12], m13 = vp[13], m15 = vp[15];
    const hw = W * 0.5, hh = H * 0.5;
    for (let i = 0; i < n; i += step) {
      if (vis && !vis[i]) continue;
      if (alpha && alpha[i] < minA) continue;
      const x = c[i * 3], y = c[i * 3 + 1], z = c[i * 3 + 2];
      const w = m3 * x + m7 * y + m11 * z + m15;
      if (w <= 1e-6) continue;
      const px = (m0 * x + m4 * y + m8 * z + m12) / w * hw + hw;
      const py = hh - (m1 * x + m5 * y + m9 * z + m13) / w * hh;
      const gx = (px / DEPTH_CELL) | 0, gy = (py / DEPTH_CELL) | 0;
      if (px < 0 || py < 0 || gx >= cw || gy >= ch) continue;
      const k = gy * cw + gx;
      if (w < raw[k]) raw[k] = w;
    }
    // A strided sample leaves cells empty inside a surface; the nearest of
    // the 3x3 around fills them (selection.ts, depthGrid, does the same).
    const surf = new Float32Array(raw.length);
    for (let y = 0; y < ch; y++) {
      for (let x = 0; x < cw; x++) {
        let m = raw[y * cw + x];
        for (let oy = -1; oy <= 1; oy++) {
          const uy = y + oy;
          if (uy < 0 || uy >= ch) continue;
          for (let ox = -1; ox <= 1; ox++) {
            const ux = x + ox;
            if (ux < 0 || ux >= cw) continue;
            const v = raw[uy * cw + ux];
            if (v < m) m = v;
          }
        }
        surf[y * cw + x] = m;
      }
    }
    this.depth = { key, surf, cw, ch };
    return this.depth;
  }

  private draw(vp: ArrayLike<number>, W: number, H: number, dpr: number) {
    const t0 = performance.now();
    const c = this.canvas;
    const bw = Math.max(1, Math.round(W * dpr)), bh = Math.max(1, Math.round(H * dpr));
    if (c.width !== bw || c.height !== bh) { c.width = bw; c.height = bh; }
    const ctx = c.getContext('2d')!;
    ctx.setTransform(1, 0, 0, 1, 0, 0);
    ctx.clearRect(0, 0, bw, bh);
    const s = this.shell;
    this.stats = { cells: 0, segments: 0, hidden: 0, ms: 0, live: this.liveShown };
    if (!s || !s.count || !this.visible) return;

    const dm = this.depthMap(vp, W, H);
    const gw = Math.ceil(W / CELL) + 2, gh = Math.ceil(H / CELL) + 2;   // a cell of margin
    let occ: Uint8Array<ArrayBufferLike> = new Uint8Array(gw * gh);
    // the same, from the shell points nothing stands in front of
    const vis: Uint8Array<ArrayBufferLike> = dm ? new Uint8Array(gw * gh) : occ;
    const m0 = vp[0], m1 = vp[1], m3 = vp[3], m4 = vp[4], m5 = vp[5], m7 = vp[7];
    const m8 = vp[8], m9 = vp[9], m11 = vp[11], m12 = vp[12], m13 = vp[13], m15 = vp[15];
    const hw = W * 0.5, hh = H * 0.5;
    const pxPerUnit = Math.hypot(m1, m5, m9) * hh;      // pixels per world unit at depth 1
    const P = s.pts, S = s.spacing;
    for (let i = 0; i < s.count; i++) {
      const x = P[i * 3], y = P[i * 3 + 1], z = P[i * 3 + 2];
      const w = m3 * x + m7 * y + m11 * z + m15;
      if (w <= 1e-6) continue;
      const px = (m0 * x + m4 * y + m8 * z + m12) / w * hw + hw;
      const py = hh - (m1 * x + m5 * y + m9 * z + m13) / w * hh;
      const gx = Math.floor(px / CELL) + 1, gy = Math.floor(py / CELL) + 1;
      // footprint: half the sample spacing, rounded up, so neighbours meet
      // (rounded down, samples 3 px apart left a lattice of 2 px holes)
      const rc = Math.min(10, Math.ceil(0.5 * S[i] * pxPerUnit / w / CELL - 0.15));
      if (gx + rc < 0 || gy + rc < 0 || gx - rc >= gw || gy - rc >= gh) continue;
      // Hidden: an opaque surface clearly in front. The volume sits on the
      // surface it was painted on, so its rim is level with it; the margin
      // (2 % of the depth, a few sample spacings) keeps that rim in view.
      let shown = true;
      if (dm) {
        const dx = (px / DEPTH_CELL) | 0, dy = (py / DEPTH_CELL) | 0;
        if (px >= 0 && py >= 0 && dx < dm.cw && dy < dm.ch) {
          const d = dm.surf[dy * dm.cw + dx];
          if (d < w - Math.max(0.02 * w, 3 * S[i])) shown = false;
        }
      }
      if (rc <= 0) {
        if (gx >= 0 && gy >= 0 && gx < gw && gy < gh) {
          occ[gy * gw + gx] = 1;
          if (shown) vis[gy * gw + gx] = 1;
        }
        continue;
      }
      const r2 = rc * rc + rc;
      for (let dy = -rc; dy <= rc; dy++) {
        const yy = gy + dy;
        if (yy < 0 || yy >= gh) continue;
        for (let dx = -rc; dx <= rc; dx++) {
          const xx = gx + dx;
          if (xx < 0 || xx >= gw || dx * dx + dy * dy > r2) continue;
          occ[yy * gw + xx] = 1;
          if (shown) vis[yy * gw + xx] = 1;
        }
      }
    }
    // Two steps: where two parts of a stroke sit at different depths their
    // silhouettes left a sliver of one or two cells between them, and the
    // edge was traced twice along it.
    occ = close(occ, gw, gh, 2);
    // the wash: visible cells grown by one, enough to close the speckle
    const washVis = dm ? dilate(vis, gw, gh, 1) : occ;

    // the wash, at grid resolution, scaled up smoothly
    const wc = this.wash;
    if (wc.width !== gw || wc.height !== gh) { wc.width = gw; wc.height = gh; }
    const wctx = wc.getContext('2d')!;
    const img = wctx.createImageData(gw, gh);
    let cells = 0;
    for (let k = 0; k < occ.length; k++) {
      if (!occ[k]) continue;
      cells++;
      img.data[k * 4] = WASH[0]; img.data[k * 4 + 1] = WASH[1];
      img.data[k * 4 + 2] = WASH[2]; img.data[k * 4 + 3] = washVis[k] ? WASH[3] : WASH_HIDDEN;
    }
    wctx.putImageData(img, 0, 0);
    ctx.setTransform(dpr, 0, 0, dpr, 0, 0);
    ctx.imageSmoothingEnabled = true;
    ctx.drawImage(wc, -CELL, -CELL, gw * CELL, gh * CELL);

    // The edge: marching squares over cell centres, on the covered cells
    // blurred 3x3 and cut at one half. On the bare 0/1 grid every crossing
    // sat at an edge midpoint, and a gently sloping edge came out as a
    // staircase of long runs that read as two lines.
    const f = new Float32Array(gw * gh);
    for (let y = 0; y < gh; y++) {
      for (let x = 0; x < gw; x++) {
        let s = 0;
        for (let dy = -1; dy <= 1; dy++) {
          const yy = y + dy;
          if (yy < 0 || yy >= gh) continue;
          for (let dx = -1; dx <= 1; dx++) {
            const xx = x + dx;
            if (xx >= 0 && xx < gw) s += occ[yy * gw + xx];
          }
        }
        f[y * gw + x] = s / 9;
      }
    }
    const path = new Path2D(), hiddenPath = new Path2D();
    let segs = 0, hidden = 0;
    const X = (gx: number) => (gx - 1 + 0.5) * CELL, Y = (gy: number) => (gy - 1 + 0.5) * CELL;
    const cut = (p: number, q: number) => {
      const d = q - p;
      return Math.abs(d) < 1e-6 ? 0.5 : Math.min(1, Math.max(0, (0.5 - p) / d));
    };
    for (let y = 0; y + 1 < gh; y++) {
      for (let x = 0; x + 1 < gw; x++) {
        const fa = f[y * gw + x], fb = f[y * gw + x + 1];
        const fc = f[(y + 1) * gw + x + 1], fd = f[(y + 1) * gw + x];
        const a = fa >= 0.5 ? 1 : 0, b = fb >= 0.5 ? 1 : 0;
        const cc = fc >= 0.5 ? 1 : 0, d = fd >= 0.5 ? 1 : 0;
        const code = a | (b << 1) | (cc << 2) | (d << 3);
        if (code === 0 || code === 15) continue;
        // seen: a visible shell point close by, looked up only on the edge
        // (growing the whole grid for it cost more than the rest of the draw)
        let seen = 1;
        if (dm) {
          seen = 0;
          const y0 = Math.max(0, y - SEEN_REACH), y1 = Math.min(gh - 1, y + 1 + SEEN_REACH);
          const x0 = Math.max(0, x - SEEN_REACH), x1 = Math.min(gw - 1, x + 1 + SEEN_REACH);
          for (let yy = y0; yy <= y1 && !seen; yy++) {
            for (let xx = x0; xx <= x1; xx++) if (vis[yy * gw + xx]) { seen = 1; break; }
          }
        }
        // dashes: every other block of DASH px along any direction
        const dash = ((Math.floor(X(x) / DASH) + Math.floor(Y(y) / DASH)) & 1) === 0;
        const into = seen ? path : hiddenPath;
        if (!seen) hidden++;
        // crossings on the top, right, bottom and left edges
        const tx = X(x) + CELL * cut(fa, fb), ty = Y(y);
        const rx = X(x + 1), ry = Y(y) + CELL * cut(fb, fc);
        const bx = X(x) + CELL * cut(fd, fc), by = Y(y + 1);
        const lx = X(x), ly = Y(y) + CELL * cut(fa, fd);
        const seg = (x0: number, y0: number, x1: number, y1: number) => {
          segs++;
          if (!seen && !dash) return;
          into.moveTo(x0, y0); into.lineTo(x1, y1);
        };
        switch (code) {
          case 1: case 14: seg(lx, ly, tx, ty); break;
          case 2: case 13: seg(tx, ty, rx, ry); break;
          case 3: case 12: seg(lx, ly, rx, ry); break;
          case 4: case 11: seg(rx, ry, bx, by); break;
          case 6: case 9: seg(tx, ty, bx, by); break;
          case 7: case 8: seg(lx, ly, bx, by); break;
          case 5: seg(lx, ly, tx, ty); seg(rx, ry, bx, by); break;
          case 10: seg(tx, ty, rx, ry); seg(lx, ly, bx, by); break;
        }
      }
    }
    ctx.lineJoin = 'round';
    ctx.lineCap = 'butt';
    ctx.strokeStyle = HALO_HIDDEN;
    ctx.lineWidth = 2.5;
    ctx.stroke(hiddenPath);
    ctx.strokeStyle = LINE_HIDDEN;
    ctx.lineWidth = 1;
    ctx.stroke(hiddenPath);
    ctx.lineCap = 'round';
    ctx.strokeStyle = HALO;
    ctx.lineWidth = 3;
    ctx.stroke(path);
    ctx.strokeStyle = LINE;
    ctx.lineWidth = 1.25;
    ctx.stroke(path);
    this.stats = { cells, segments: segs, hidden, live: this.liveShown,
                   ms: +(performance.now() - t0).toFixed(2) };
  }
}

/** The volume's boundary points and the object's centres, as one shell. */
function withObject(s: Shell | null, o: Shell | null): Shell | null {
  if (!o || !o.count) return s;
  if (!s || !s.count) return o;
  const pts = new Float32Array((s.count + o.count) * 3);
  pts.set(s.pts.subarray(0, s.count * 3));
  pts.set(o.pts.subarray(0, o.count * 3), s.count * 3);
  const sp = new Float32Array(s.count + o.count);
  sp.set(s.spacing.subarray(0, s.count));
  sp.set(o.spacing.subarray(0, o.count), s.count);
  return { pts, spacing: sp, count: s.count + o.count };
}

/** Grow the covered cells by `steps` cells (3x3 cross each step). */
function dilate(m: Uint8Array<ArrayBufferLike>, w: number, h: number,
                steps: number): Uint8Array<ArrayBufferLike> {
  let cur = m;
  for (let s = 0; s < steps; s++) {
    const dil = new Uint8Array(cur.length);
    for (let y = 0; y < h; y++) {
      for (let x = 0; x < w; x++) {
        const k = y * w + x;
        dil[k] = cur[k] | (x > 0 ? cur[k - 1] : 0) | (x + 1 < w ? cur[k + 1] : 0)
          | (y > 0 ? cur[k - w] : 0) | (y + 1 < h ? cur[k + w] : 0);
      }
    }
    cur = dil;
  }
  return cur;
}

/** Morphological closing with a 3x3 cross: bridges the gaps between samples. */
function close(m: Uint8Array<ArrayBufferLike>, w: number, h: number,
               steps = 1): Uint8Array<ArrayBufferLike> {
  let cur = dilate(m, w, h, steps);
  for (let s = 0; s < steps; s++) {
    const out = new Uint8Array(cur.length);
    for (let y = 0; y < h; y++) {
      for (let x = 0; x < w; x++) {
        const k = y * w + x;
        out[k] = cur[k] & (x > 0 ? cur[k - 1] : 1) & (x + 1 < w ? cur[k + 1] : 1)
          & (y > 0 ? cur[k - w] : 1) & (y + 1 < h ? cur[k + w] : 1);
      }
    }
    cur = out;
  }
  return cur;
}
