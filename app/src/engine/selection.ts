/**
 * Brush selection: a 3-D capsule swept along the stroke.
 *
 * The obvious implementation -- take everything under the painted pixels,
 * within a depth window of the nearest surface -- breaks at silhouettes. Paint
 * across the edge of a shoe and some of the brush misses the sole entirely,
 * finds the far wall of the shoe several centimetres behind it, and takes that
 * too. No depth window fixes it, because the surface the window is measured
 * from is already the wrong surface.
 *
 * So the stroke is turned into geometry instead. Each sample is unprojected
 * onto the surface it touches, giving a world-space point; consecutive points
 * become capsule segments with a world radius; selection is "within R of the
 * capsule". Painting past an edge then selects nothing there rather than
 * reaching through, and the same stroke selects the same splats regardless of
 * where the camera happens to be.
 */

export interface StrokeSample { x: number; y: number; r: number; }

export interface ProjectResult {
  sx: Float32Array<ArrayBufferLike>;      // screen x, pixels
  sy: Float32Array<ArrayBufferLike>;      // screen y, pixels
  dz: Float32Array<ArrayBufferLike>;      // camera-space depth (+forward)
  onScreen: Uint8Array<ArrayBufferLike>;
}

type F32 = Float32Array<ArrayBufferLike>;
type U8 = Uint8Array<ArrayBufferLike>;
type U32 = Uint32Array<ArrayBufferLike>;

const CELL = 6;          // depth-probe cell size in pixels

/** Project splat centres with a column-major view-projection matrix. */
export function project(
  centres: F32, m: F32 | number[], width: number, height: number,
  out?: ProjectResult
): ProjectResult {
  const n = centres.length / 3;
  const r: ProjectResult = out && out.sx.length === n ? out : {
    sx: new Float32Array(n), sy: new Float32Array(n),
    dz: new Float32Array(n), onScreen: new Uint8Array(n)
  };
  // Column-major (PlayCanvas Mat4): element [col*4 + row].
  const m0 = m[0], m1 = m[1], m3 = m[3];
  const m4 = m[4], m5 = m[5], m7 = m[7];
  const m8 = m[8], m9 = m[9], m11 = m[11];
  const m12 = m[12], m13 = m[13], m15 = m[15];
  const hw = width * 0.5, hh = height * 0.5;
  for (let i = 0, j = 0; i < n; i++, j += 3) {
    const x = centres[j], y = centres[j + 1], z = centres[j + 2];
    const cw = m3 * x + m7 * y + m11 * z + m15;
    if (cw <= 1e-6) { r.onScreen[i] = 0; r.dz[i] = -1; continue; }
    const cx = m0 * x + m4 * y + m8 * z + m12;
    const cy = m1 * x + m5 * y + m9 * z + m13;
    const inv = 1 / cw;
    const px = (cx * inv) * hw + hw;
    const py = hh - (cy * inv) * hh;
    r.sx[i] = px; r.sy[i] = py; r.dz[i] = cw;
    r.onScreen[i] = (px >= -64 && px <= width + 64 && py >= -64 && py <= height + 64) ? 1 : 0;
  }
  return r;
}

export interface SurfaceOptions {
  alive?: U8;
  alpha?: F32;
  minAlpha?: number;
}

/**
 * Depth of the nearest reasonably opaque splat per screen cell.
 *
 * Only opaque splats may define the surface: removal artefacts leave
 * translucent haze in front of the real thing, and anchoring on the literal
 * nearest splat would measure the fog instead of the surface under it.
 */
export interface DepthGrid { surf: Float32Array; cw: number; ch: number; }

export function depthGrid(p: ProjectResult, width: number, height: number,
                          opts: SurfaceOptions): DepthGrid {
  const cw = Math.max(1, Math.ceil(width / CELL));
  const ch = Math.max(1, Math.ceil(height / CELL));
  const surf = new Float32Array(cw * ch).fill(Infinity);
  const alive = opts.alive, alpha = opts.alpha;
  const minAlpha = opts.minAlpha ?? 0.25;
  const n = p.sx.length;
  for (let i = 0; i < n; i++) {
    if (!p.onScreen[i]) continue;
    if (alive && !alive[i]) continue;
    if (alpha && alpha[i] < minAlpha) continue;
    const d = p.dz[i];
    if (d <= 0) continue;
    const cx = (p.sx[i] / CELL) | 0, cy = (p.sy[i] / CELL) | 0;
    if (cx < 0 || cx >= cw || cy < 0 || cy >= ch) continue;
    const c = cy * cw + cx;
    if (d < surf[c]) surf[c] = d;
  }

  // One splat writes one cell, at its centre -- its footprint is ignored. So
  // a cell that no front-surface centre happens to land in keeps Infinity,
  // and the depth gate then falls through to whatever is behind: an interior
  // splat, or the far wall. Taking the nearest of each cell's 3x3 neighbours
  // fills those holes with the surface that surrounds them. A few thousand
  // cells, so the cost does not register.
  const spread = new Float32Array(surf);
  for (let y = 0; y < ch; y++) {
    for (let x = 0; x < cw; x++) {
      let m = surf[y * cw + x];
      for (let oy = -1; oy <= 1; oy++) {
        const uy = y + oy;
        if (uy < 0 || uy >= ch) continue;
        for (let ox = -1; ox <= 1; ox++) {
          const ux = x + ox;
          if (ux < 0 || ux >= cw) continue;
          const v = surf[uy * cw + ux];
          if (v < m) m = v;
        }
      }
      spread[y * cw + x] = m;
    }
  }
  return { surf: spread, cw, ch };
}

/**
 * The visible surface under a lasso or box: per cell, the depth at which
 * what lies in front has covered it to `coverage` (0.5), as the renderer
 * blends it.
 *
 * `depthGrid` takes the nearest opaque splat *centre* per cell and then the
 * nearest of the 3x3 cells around it. Good enough to put a brush on a
 * surface, wrong for a window around it: on a surface seen at a slant the
 * nearest of nine cells lies well in front of the splats in the middle one,
 * so they fell out of a 9 px window (the side of the shoe came out as a
 * scatter of specks, 1,774 splats), and a cell no front centre landed in saw
 * the far side. This splats every splat's own footprint (its covariance
 * projected along the camera's right and up axes) over the cells it covers,
 * sorts each cell by depth and blends front to back: a large splat covers
 * the cells around its centre, haze in front covers little, and the cell's
 * depth is the surface's at that cell, not its neighbours'. Only cells
 * inside `box` (the outline's bounds) are worked out; the rest stay empty.
 */
export interface FootprintOptions {
  alive?: U8;
  alpha: F32;
  /** Covariance per splat: [xx, xy, xz, yy] and [yz, zz] (document.ts). */
  covA: F32;
  covB: F32;
  /** Camera axes in world space; screen y runs against `up`. */
  right: [number, number, number];
  up: [number, number, number];
  /** World units per pixel at one unit of depth. */
  perPixelPerUnit: number;
  /** Viewport pixels [x0, y0, x1, y1] whose cells are wanted. */
  box: [number, number, number, number];
  /** Opacity at which a cell counts as covered. */
  coverage?: number;
}

/** A footprint reaches at most this many cells from its centre. */
const FOOT_MAX_CELLS = 10;
/** Below this opacity a splat is haze and does not count at all. */
const FOOT_MIN_ALPHA = 0.02;

export function surfaceGrid(p: ProjectResult, width: number, height: number,
                            o: FootprintOptions): DepthGrid & { entries: number } {
  const cw = Math.max(1, Math.ceil(width / CELL));
  const ch = Math.max(1, Math.ceil(height / CELL));
  const surf = new Float32Array(cw * ch).fill(Infinity);
  const gx0 = Math.max(0, Math.floor(o.box[0] / CELL)), gy0 = Math.max(0, Math.floor(o.box[1] / CELL));
  const gx1 = Math.min(cw - 1, Math.floor(o.box[2] / CELL)), gy1 = Math.min(ch - 1, Math.floor(o.box[3] / CELL));
  if (gx1 < gx0 || gy1 < gy0) return { surf, cw, ch, entries: 0 };
  const gw = gx1 - gx0 + 1, gh = gy1 - gy0 + 1;
  const n = p.sx.length;
  const { alive, alpha, covA, covB } = o;
  const covOk = covA.length >= n * 4 && covB.length >= n * 2;
  const [rx, ry, rz] = o.right, [ux, uy, uz] = o.up;
  const ppu = o.perPixelPerUnit;
  const cellVar = CELL * CELL / 12;          // a cell's own spread, as a box
  const margin = FOOT_MAX_CELLS * CELL;

  let cap = 1 << 18;
  let eCell = new Int32Array(cap), eDep = new Float32Array(cap), eW = new Float32Array(cap);
  let m = 0;
  for (let i = 0; i < n; i++) {
    if (!p.onScreen[i]) continue;
    if (alive && !alive[i]) continue;
    const a0 = alpha[i];
    if (!(a0 >= FOOT_MIN_ALPHA)) continue;
    const d = p.dz[i];
    if (d <= 0) continue;
    const sx = p.sx[i], sy = p.sy[i];
    if (sx < o.box[0] - margin || sx > o.box[2] + margin
        || sy < o.box[1] - margin || sy > o.box[3] + margin) continue;
    // the footprint in pixels: r'Sr, r'Su, u'Su over (ppu d)^2, plus the
    // renderer's own 0.3 px^2 low-pass
    let sxx = 0.3, syy = 0.3, sxy = 0;
    if (covOk) {
      const A = i * 4, B = i * 2;
      const xx = covA[A], xy = covA[A + 1], xz = covA[A + 2], yy = covA[A + 3];
      const yz = covB[B], zz = covB[B + 1];
      const r0 = xx * rx + xy * ry + xz * rz, r1 = xy * rx + yy * ry + yz * rz, r2 = xz * rx + yz * ry + zz * rz;
      const u0 = xx * ux + xy * uy + xz * uz, u1 = xy * ux + yy * uy + yz * uz, u2 = xz * ux + yz * uy + zz * uz;
      const k = 1 / ((ppu * d) * (ppu * d));
      const vrr = (rx * r0 + ry * r1 + rz * r2) * k;
      const vuu = (ux * u0 + uy * u1 + uz * u2) * k;
      const vru = (ux * r0 + uy * r1 + uz * r2) * k;
      if (isFinite(vrr) && isFinite(vuu) && isFinite(vru)) {
        sxx += Math.max(vrr, 0); syy += Math.max(vuu, 0); sxy = -vru;   // screen y runs down
      }
    }
    let det = sxx * syy - sxy * sxy;
    if (!(det > 1e-6)) { sxx = syy = 0.3; sxy = 0; det = 0.09; }
    // A splat smaller than a cell covers only part of it.
    const share = Math.min(1, 2 * Math.PI * Math.sqrt(det) / (CELL * CELL));
    // evaluated at cell centres, with the cell's spread added
    const exx = sxx + cellVar, eyy = syy + cellVar;
    const edet = exx * eyy - sxy * sxy;
    const rad = 3 * Math.sqrt(Math.max(exx, eyy));
    const rc = Math.min(FOOT_MAX_CELLS, Math.ceil(rad / CELL));
    const cx0 = (sx / CELL) | 0, cy0 = (sy / CELL) | 0;
    const ya = Math.max(gy0, cy0 - rc), yb = Math.min(gy1, cy0 + rc);
    const xa = Math.max(gx0, cx0 - rc), xb = Math.min(gx1, cx0 + rc);
    const peak = a0 * share;
    for (let cy = ya; cy <= yb; cy++) {
      const dy = cy * CELL + CELL * 0.5 - sy;
      for (let cx = xa; cx <= xb; cx++) {
        const dx = cx * CELL + CELL * 0.5 - sx;
        const q = (eyy * dx * dx - 2 * sxy * dx * dy + exx * dy * dy) / edet;
        if (q > 9) continue;
        const w = peak * Math.exp(-0.5 * q);
        if (w < 0.004) continue;
        if (m === cap) {
          cap *= 2;
          const c2 = new Int32Array(cap); c2.set(eCell); eCell = c2;
          const d2 = new Float32Array(cap); d2.set(eDep); eDep = d2;
          const w2 = new Float32Array(cap); w2.set(eW); eW = w2;
        }
        eCell[m] = (cy - gy0) * gw + (cx - gx0);
        eDep[m] = d;
        eW[m] = w > 0.99 ? 0.99 : w;
        m++;
      }
    }
  }

  // bucket by cell, then blend each cell front to back
  const start = new Int32Array(gw * gh + 1);
  for (let k = 0; k < m; k++) start[eCell[k] + 1]++;
  for (let c = 0; c < gw * gh; c++) start[c + 1] += start[c];
  const fill = start.slice(0, gw * gh);
  const order = new Int32Array(m);
  for (let k = 0; k < m; k++) order[fill[eCell[k]]++] = k;
  const want = o.coverage ?? 0.5;
  let idx = new Int32Array(64);
  for (let c = 0; c < gw * gh; c++) {
    const s = start[c], e = start[c + 1], len = e - s;
    if (!len) continue;
    let T = 1;
    for (let k = s; k < e; k++) T *= 1 - eW[order[k]];
    const total = 1 - T;
    if (total < 0.15) continue;               // nothing much there: a hole or background
    if (idx.length < len) idx = new Int32Array(len * 2);
    for (let k = 0; k < len; k++) idx[k] = order[s + k];
    const view = idx.subarray(0, len);
    if (len <= 24) {                          // the usual case: insertion sort
      for (let a = 1; a < len; a++) {
        const v = view[a], dv = eDep[v];
        let b = a - 1;
        while (b >= 0 && eDep[view[b]] > dv) { view[b + 1] = view[b]; b--; }
        view[b + 1] = v;
      }
    } else {
      view.sort((a, b) => eDep[a] - eDep[b]);
    }
    // a faint cell (a thin surface, sparse splats) is still a surface: its
    // own share of what covers it counts, up to `want`
    const thr = Math.min(want, 0.7 * total);
    let t = 1;
    for (let k = 0; k < len; k++) {
      t *= 1 - eW[view[k]];
      if (1 - t >= thr) { surf[(gy0 + ((c / gw) | 0)) * cw + gx0 + (c % gw)] = eDep[view[k]]; break; }
    }
  }
  return { surf, cw, ch, entries: m };
}

export interface Unprojector {
  (x: number, y: number, depth: number): [number, number, number];
  }

export interface CapsulePoint { x: number; y: number; z: number; r: number; }

/**
 * Turn a screen stroke into world-space capsule points.
 *
 * A sample that lands on nothing (painted past the object) contributes no
 * point, which is exactly right: there is no surface there to repair. One in
 * a hole between two parts of the same surface does (see `strokeDepths`).
 */
export function strokeToCapsules(
  grid: DepthGrid, stroke: StrokeSample[],
  unproject: Unprojector, worldPerPixelAt: (depth: number) => number
): CapsulePoint[] {
  const depths = strokeDepths(grid, stroke);
  const out: CapsulePoint[] = [];
  for (let k = 0; k < stroke.length; k++) {
    const best = depths[k];
    if (!isFinite(best)) continue;
    const s = stroke[k];
    const [wx, wy, wz] = unproject(s.x, s.y, best);
    out.push({ x: wx, y: wy, z: wz, r: s.r * worldPerPixelAt(best) });
  }
  return out;
}

/**
 * How far two surfaces on either side of a gap may differ in depth and still
 * count as one surface with a hole in it, as a ratio.
 */
const BRIDGE_RATIO = 1.3;
/** The longest gap bridged, in brush radii along the stroke. */
const BRIDGE_RADII = 8;
/** How far along the stroke a sample is compared with, to see a hole. */
const HOLE_RADII = 4;
/** Off only to measure what looking for holes changes (autotest). */
export const strokeHoles = { on: true };

/**
 * The surface depth under each stroke sample.
 *
 * A sample that finds nothing in its disc used to be dropped. Past the edge
 * of the object that is right -- there is nothing there to repair -- but in
 * the middle of a stroke it is a hole in the surface: exactly the spot the
 * repair is for, and the selection had a gap there. So a run of empty
 * samples between two that found the same surface (depths within 30 %) gets
 * the depth interpolated along the stroke; the ends of a stroke that run off
 * the object still find nothing.
 */
export function strokeDepths(grid: DepthGrid, stroke: StrokeSample[]): Float64Array {
  const { surf, cw, ch } = grid;
  const out = new Float64Array(stroke.length).fill(Infinity);
  for (let k = 0; k < stroke.length; k++) {
    const s = stroke[k];
    // Nearest surface within this sample's disc, so the capsule sits on the
    // surface the user is pointing at rather than at an arbitrary depth.
    let best = Infinity;
    const x0 = Math.max(0, Math.floor((s.x - s.r) / CELL));
    const x1 = Math.min(cw - 1, Math.floor((s.x + s.r) / CELL));
    const y0 = Math.max(0, Math.floor((s.y - s.r) / CELL));
    const y1 = Math.min(ch - 1, Math.floor((s.y + s.r) / CELL));
    const r2 = s.r * s.r;
    for (let cy = y0; cy <= y1; cy++) {
      const py = cy * CELL + CELL * 0.5;
      for (let cx = x0; cx <= x1; cx++) {
        const px = cx * CELL + CELL * 0.5;
        const dx = px - s.x, dy = py - s.y;
        if (dx * dx + dy * dy > r2) continue;
        const v = surf[cy * cw + cx];
        if (v < best) best = v;
      }
    }
    out[k] = best;
  }
  // A dab whose whole disc lies over a hole finds the far wall through it,
  // and anchored there it selected that wall -- which the repair then took
  // away. So a sample much deeper (BRIDGE_RATIO) than the stroke's own
  // surface within HOLE_RADII brush radii along it is looking through a
  // hole: it is dropped here and gets the stroke's surface depth below.
  // A stroke that runs on from an object onto the ground behind it keeps
  // the ground beyond that window.
  const n = stroke.length;
  const arc = new Float64Array(n);
  for (let k = 1; k < n; k++) {
    arc[k] = arc[k - 1] + Math.hypot(stroke[k].x - stroke[k - 1].x, stroke[k].y - stroke[k - 1].y);
  }
  const through = new Uint8Array(n);
  for (let k = 0; k < n && strokeHoles.on; k++) {
    if (!isFinite(out[k])) continue;
    const reach = HOLE_RADII * Math.max(stroke[k].r, 1);
    let near = Infinity;
    for (let j = k - 1; j >= 0 && arc[k] - arc[j] <= reach; j--) near = Math.min(near, out[j]);
    for (let j = k + 1; j < n && arc[j] - arc[k] <= reach; j++) near = Math.min(near, out[j]);
    if (out[k] > BRIDGE_RATIO * near) through[k] = 1;
  }
  for (let k = 0; k < n; k++) if (through[k]) out[k] = Infinity;
  // bridge holes: runs of misses between two hits on one surface
  let last = -1;
  for (let k = 0; k < stroke.length; k++) {
    if (!isFinite(out[k])) continue;
    if (last >= 0 && k - last > 1) {
      const d0 = out[last], d1 = out[k];
      // by length along the stroke, so uneven sampling does not skew it
      let total = 0;
      for (let j = last + 1; j <= k; j++) {
        total += Math.hypot(stroke[j].x - stroke[j - 1].x, stroke[j].y - stroke[j - 1].y);
      }
      const rad = Math.max(stroke[last].r, stroke[k].r, 1);
      if (Math.max(d0, d1) <= BRIDGE_RATIO * Math.min(d0, d1)
          && total <= BRIDGE_RADII * rad) {
        let run = 0;
        for (let j = last + 1; j < k; j++) {
          run += Math.hypot(stroke[j].x - stroke[j - 1].x, stroke[j].y - stroke[j - 1].y);
          const t = total > 0 ? run / total : (j - last) / (k - last);
          out[j] = d0 + (d1 - d0) * t;
        }
      }
    }
    last = k;
  }
  // Looked through a hole and not bridged (at an end of the stroke, or the
  // two sides too different): the depth of the nearest sample that found
  // the surface, the stroke's running surface.
  for (let k = 0; k < n; k++) {
    if (!through[k] || isFinite(out[k])) continue;
    let best = Infinity, dist = Infinity;
    for (let j = 0; j < n; j++) {
      if (through[j] || !isFinite(out[j])) continue;
      const d = Math.abs(arc[j] - arc[k]);
      if (d < dist) { dist = d; best = out[j]; }
    }
    out[k] = best;
  }
  return out;
}

export interface SelectOptions {
  depth: number;        // how far behind the surface to reach, as a fraction of brush radius
  alive?: U8;
  /** Per-splat covariance, see `SizeReach`. Without it every splat is a point. */
  size?: SizeReach;
}

/**
 * How much a splat's own extent widens the depth window.
 *
 * The window is measured from a splat's centre, but a Gaussian is not a point.
 * A large blurry splat whose centre sits a little behind the window can still
 * cover the surface in front of it -- on a scan's stand these are exactly the
 * smear that stays behind after a repair (brush_final: 68 splats, median sigma
 * 0.64, alpha 0.58, 1-3 units behind the surface). So a splat counts as near
 * the surface when its extent *along the view ray* reaches the window:
 *
 *     |along| - min(k * sigma_ray, cap * r)  <=  window
 *
 * sigma_ray, not the largest axis: a flat splat lying on the far wall of a
 * sole is wide but thin along the ray, and must not reach through.
 */
export interface SizeReach {
  covA: F32;            // per splat (xx, xy, xz, yy)
  covB: F32;            // per splat (yz, zz)
  k: number;            // sigmas a splat reaches
  cap: number;          // most it may reach, in brush radii
}

/**
 * Shared defaults, so the CPU and GPU selectors cannot drift apart.
 *
 * Picked by deleting the autotest's stand stroke and looking at what stayed:
 * on brush_final k = 2 left part of the brown smear, k = 3 cleared it to the
 * insole, k = 4 changed nothing more. On the main shoe the same stroke takes
 * 15,480 -> 16,513 splats (+6.7 %), and the interior behind it stays intact.
 *
 * Off for now (0 = the old selection, exactly): with the smear selected, the
 * repair after it let the insole show through the fill where the old one left
 * a solid brown smear -- the fill is not yet opaque enough over an exposed
 * interior. Turn on once it is (reference/app_fix/cmp_brush_before_after.png).
 */
export const SIZE_K = 0;
export const SIZE_CAP = 2;

/**
 * Standard deviation of splat `i` along a unit direction.
 * Float32 inputs, the same ones the GPU reads, so both sides agree.
 */
export function sigmaAlong(covA: F32, covB: F32, i: number,
                           vx: number, vy: number, vz: number): number {
  const a = i * 4, b = i * 2;
  const v = covA[a] * vx * vx + covA[a + 3] * vy * vy + covB[b + 1] * vz * vz
    + 2 * (covA[a + 1] * vx * vy + covA[a + 2] * vx * vz + covB[b] * vy * vz);
  return v > 0 ? Math.sqrt(v) : 0;
}

/**
 * A stroke as capsule rows, the form the volume keeps and the engine gets:
 * `[ax, ay, az, bx, by, bz, r, depth]` per segment between consecutive
 * samples (the last sample is a segment of length zero), `r` the larger of
 * the two radii, `depth` the half-extent along the view ray, `r * squash`.
 */
export const ROW = 8;
export function segmentRows(caps: CapsulePoint[], squash: number): Float64Array {
  const s = Math.max(squash, 1e-3);
  const out = new Float64Array(caps.length * ROW);
  for (let k = 0; k < caps.length; k++) {
    const a = caps[k];
    const b = k + 1 < caps.length ? caps[k + 1] : a;
    const r = Math.max(a.r, b.r);
    out.set([a.x, a.y, a.z, b.x, b.y, b.z, r, r * s], k * ROW);
  }
  return out;
}

/**
 * Splats inside the swept capsule.
 *
 * The capsule is squashed along the view direction by `depth`, so the brush
 * takes a thin shell of surface rather than a ball: painting a 6 cm circle on a
 * shoe sole should not also take 6 cm of what is behind it.
 */
export function selectInCapsules(
  centres: F32, caps: CapsulePoint[], viewDir: [number, number, number],
  opts: SelectOptions
): U32 {
  if (!caps.length) return new Uint32Array(0);
  return selectInRows(centres, segmentRows(caps, opts.depth), viewDir, opts);
}

/**
 * Splats inside capsule rows (see `segmentRows`), all painted from `viewDir`.
 *
 * The shape: the segment swept by an ellipsoid with semi-axes (r, r, depth),
 * the short one along the view. Measured in a frame where the view axis is
 * stretched by r/depth -- the point *and* the segment. The segment used to
 * stay unstretched, so where one dab sat deeper than the next the far end
 * of the capsule fell outside its own capsule. core/splatcore/volume.py
 * holds the same test, and gpuselect.ts runs it term for term.
 */
export function selectInRows(
  centres: F32, rows: Float64Array | Float32Array | number[],
  viewDir: [number, number, number], opts: Omit<SelectOptions, 'depth'>
): U32 {
  const m = (rows.length / ROW) | 0;
  if (!m) return new Uint32Array(0);
  const alive = opts.alive;
  const n = centres.length / 3;
  const [vx, vy, vz] = viewDir;
  const size = opts.size && opts.size.covA.length >= n * 4 && opts.size.covB.length >= n * 2
    ? opts.size : null;
  // k = 0 means points. It has to skip the product, not just multiply by 0:
  // a splat whose covariance overflows float32 has sigma = Infinity, and
  // 0 * Infinity is NaN, which no comparison passes -- the splat dropped out
  // of the selection altogether.
  const sizeK = size && size.k > 0 ? size.k : 0, sizeCap = size ? size.cap : 0;
  const out: number[] = [];

  // Per row: stretch, stretched segment, its length, and a box to reject
  // most splats with six compares. The box is padded by the full reach along
  // the ray: padded by r alone, at a depth above 2 (window wider than r) the
  // CPU quietly cut off what the GPU, which has no box, still took.
  const P = 16;
  const pre = new Float64Array(m * P);
  let minX = Infinity, minY = Infinity, minZ = Infinity;
  let maxX = -Infinity, maxY = -Infinity, maxZ = -Infinity;
  for (let k = 0; k < m; k++) {
    const o = k * ROW, q = k * P;
    const ax = rows[o], ay = rows[o + 1], az = rows[o + 2];
    const bx = rows[o + 3], by = rows[o + 4], bz = rows[o + 5];
    const r = rows[o + 6], depth = Math.max(rows[o + 7], 1e-9);
    const s = r / depth;
    const ux = bx - ax, uy = by - ay, uz = bz - az;
    const ua = (ux * vx + uy * vy + uz * vz) * (s - 1);
    const tx = ux + vx * ua, ty = uy + vy * ua, tz = uz + vz * ua;
    pre[q] = s; pre[q + 1] = tx; pre[q + 2] = ty; pre[q + 3] = tz;
    pre[q + 4] = tx * tx + ty * ty + tz * tz;
    pre[q + 5] = r;
    const pad = Math.max(r, depth + sizeCap * r);
    pre[q + 6] = Math.min(ax, bx) - pad; pre[q + 7] = Math.min(ay, by) - pad;
    pre[q + 8] = Math.min(az, bz) - pad; pre[q + 9] = Math.max(ax, bx) + pad;
    pre[q + 10] = Math.max(ay, by) + pad; pre[q + 11] = Math.max(az, bz) + pad;
    if (pre[q + 6] < minX) minX = pre[q + 6];
    if (pre[q + 7] < minY) minY = pre[q + 7];
    if (pre[q + 8] < minZ) minZ = pre[q + 8];
    if (pre[q + 9] > maxX) maxX = pre[q + 9];
    if (pre[q + 10] > maxY) maxY = pre[q + 10];
    if (pre[q + 11] > maxZ) maxZ = pre[q + 11];
  }

  for (let i = 0, j = 0; i < n; i++, j += 3) {
    if (alive && !alive[i]) continue;
    const x = centres[j], y = centres[j + 1], z = centres[j + 2];
    if (x < minX || x > maxX || y < minY || y > maxY || z < minZ || z > maxZ) continue;
    let reach = -1;
    for (let k = 0; k < m; k++) {
      const q = k * P;
      if (x < pre[q + 6] || x > pre[q + 9] || y < pre[q + 7] || y > pre[q + 10]
          || z < pre[q + 8] || z > pre[q + 11]) continue;
      if (reach < 0) {
        reach = size && sizeK > 0 ? sizeK * sigmaAlong(size.covA, size.covB, i, vx, vy, vz) : 0;
      }
      const o = k * ROW;
      const s = pre[q], r = pre[q + 5];
      // A splat's own extent along the ray is taken off its distance first
      // (see SizeReach), then the view axis is stretched.
      const wx = x - rows[o], wy = y - rows[o + 1], wz = z - rows[o + 2];
      const along = wx * vx + wy * vy + wz * vz;
      const give = Math.min(reach, sizeCap * r);
      const shrunk = along > 0 ? Math.max(along - give, 0) : Math.min(along + give, 0);
      const extra = shrunk * s - along;
      const px = wx + vx * extra, py = wy + vy * extra, pz = wz + vz * extra;
      const tx = pre[q + 1], ty = pre[q + 2], tz = pre[q + 3], uu = pre[q + 4];
      let t = uu > 1e-12 ? (px * tx + py * ty + pz * tz) / uu : 0;
      t = t < 0 ? 0 : t > 1 ? 1 : t;
      const dx = px - tx * t, dy = py - ty * t, dz = pz - tz * t;
      if (dx * dx + dy * dy + dz * dz <= r * r) {
        out.push(i);
        break;
      }
    }
  }
  return Uint32Array.from(out);
}

/* ------------------------------------------------------- region selection */

export type SelectMode = 'replace' | 'add' | 'subtract';
export type DepthMode = 'surface' | 'through';

/**
 * Scanline-fill a screen polygon into a full-resolution mask.
 *
 * Testing every splat against every edge would be O(splats x edges); filling
 * once and then looking each splat up is O(area + splats), which at 550k splats
 * and a 200-point lasso is the difference between a pause and no pause.
 */
export function fillPolygon(points: { x: number; y: number }[],
                            width: number, height: number): Uint8Array {
  const mask = new Uint8Array(width * height);
  const n = points.length;
  if (n < 3) return mask;
  let minY = Infinity, maxY = -Infinity;
  for (const p of points) {
    if (p.y < minY) minY = p.y;
    if (p.y > maxY) maxY = p.y;
  }
  const y0 = Math.max(0, Math.floor(minY));
  const y1 = Math.min(height - 1, Math.ceil(maxY));
  const xs: number[] = [];
  for (let y = y0; y <= y1; y++) {
    xs.length = 0;
    const sy = y + 0.5;
    for (let i = 0, j = n - 1; i < n; j = i++) {
      const a = points[j], b = points[i];
      if ((a.y <= sy && b.y > sy) || (b.y <= sy && a.y > sy)) {
        xs.push(a.x + (sy - a.y) / (b.y - a.y) * (b.x - a.x));
      }
    }
    if (!xs.length) continue;
    xs.sort((p, q) => p - q);
    const row = y * width;
    for (let k = 0; k + 1 < xs.length; k += 2) {
      const xa = Math.max(0, Math.ceil(xs[k] - 0.5));
      const xb = Math.min(width - 1, Math.floor(xs[k + 1] - 0.5));
      for (let x = xa; x <= xb; x++) mask[row + x] = 1;
    }
  }
  return mask;
}

export interface RegionOptions {
  depthMode: DepthMode;
  /**
   * How far behind the visible surface still counts as "the surface", in
   * viewport pixels at that surface's own distance.
   *
   * Pixels, not world units. It used to be a fraction of the whole scene's
   * bounding radius, which has nothing to do with how far away the thing
   * under the cursor is: on the shoe that worked out at a third of a unit
   * behind the front face, which is most of the way through the sole, so a
   * lasso set to "surface only" took the inside with it. The brush has
   * always scaled its reach this way; the lasso and box now agree with it.
   */
  tolPx: number;
  /** World units per pixel at one unit of depth. Scales linearly. */
  perPixelPerUnit: number;
  frontScale?: number;
  alive?: U8;
  grid?: DepthGrid;
  /**
   * Splat extent along the view, as for the brush. `cap` is in units of the
   * window here, since a region has no radius. Needs `viewDir`.
   */
  size?: SizeReach;
  viewDir?: [number, number, number];
}

/**
 * Splats whose centre falls inside a screen mask.
 *
 * In ``surface`` mode only splats near the visible surface count, so drawing
 * around a shoe takes the side facing you. In ``through`` mode everything
 * behind it comes too, which is what you want when the point is to remove an
 * object rather than to touch its surface.
 */
export function selectInRegion(p: ProjectResult, mask: Uint8Array,
                               width: number, height: number,
                               opts: RegionOptions): U32 {
  const out: number[] = [];
  const alive = opts.alive;
  const grid = opts.grid;
  const cw = grid ? grid.cw : 0;
  const n = p.sx.length;
  const size = opts.size && opts.viewDir
    && opts.size.covA.length >= n * 4 && opts.size.covB.length >= n * 2 ? opts.size : null;
  const [vx, vy, vz] = opts.viewDir ?? [0, 0, 1];
  for (let i = 0; i < n; i++) {
    if (!p.onScreen[i]) continue;
    if (alive && !alive[i]) continue;
    const x = p.sx[i] | 0, y = p.sy[i] | 0;
    if (x < 0 || x >= width || y < 0 || y >= height) continue;
    if (!mask[y * width + x]) continue;
    if (opts.depthMode === 'surface' && grid) {
      const cx = (p.sx[i] / CELL) | 0, cy = (p.sy[i] / CELL) | 0;
      if (cx < 0 || cx >= cw || cy < 0 || cy >= grid.ch) continue;
      const d0 = grid.surf[cy * cw + cx];
      if (!isFinite(d0)) continue;
      const d = p.dz[i];
      const back = Math.max(opts.tolPx * opts.perPixelPerUnit * d0, 1e-6);
      // k = 0 skips the product (0 * an overflowed sigma is NaN, see above)
      const give = size && size.k > 0
        ? Math.min(size.k * sigmaAlong(size.covA, size.covB, i, vx, vy, vz), size.cap * back)
        : 0;
      if (d < d0 - back * (opts.frontScale ?? 1.2) - give || d > d0 + back + give) continue;
    }
    out.push(i);
  }
  return Uint32Array.from(out);
}

/** The depth cells a region keeps: its polygon's cells, holes filled. */
export interface RegionGrid {
  grid: DepthGrid;                       // full-size copy, holes filled
  crop: [number, number, number, number]; // gx0, gy0, gw, gh (cells)
}

/**
 * The surface under a lasso or box, with its holes closed.
 *
 * A cell inside the outline where no opaque splat lands is either background
 * the outline happened to include, or a hole in the surface -- the thing the
 * repair is for. The difference is whether it is enclosed: empty cells that
 * reach the edge of the outline through other empty cells are background
 * and stay empty; the rest are holes and get the depth of the surface around
 * them (repeated averaging of filled neighbours, from the rim inwards). Then
 * the volume, its outline and the engine's masks cover the hole even where
 * there is not one splat in it.
 */
export function fillRegionHoles(grid: DepthGrid, mask: Uint8Array, width: number,
                                height: number, poly: { x: number; y: number }[],
                                fill = true): RegionGrid {
  const { cw, ch } = grid;
  const surf = new Float32Array(grid.surf);
  let px0 = Infinity, py0 = Infinity, px1 = -Infinity, py1 = -Infinity;
  for (const p of poly) {
    if (p.x < px0) px0 = p.x; if (p.x > px1) px1 = p.x;
    if (p.y < py0) py0 = p.y; if (p.y > py1) py1 = p.y;
  }
  const gx0 = Math.max(0, Math.min(cw - 1, Math.floor(Math.max(0, px0 - 1) / CELL)));
  const gy0 = Math.max(0, Math.min(ch - 1, Math.floor(Math.max(0, py0 - 1) / CELL)));
  const gx1 = Math.max(gx0, Math.min(cw - 1, Math.floor(Math.min(width - 1, px1 + 1) / CELL)));
  const gy1 = Math.max(gy0, Math.min(ch - 1, Math.floor(Math.min(height - 1, py1 + 1) / CELL)));
  const gw = gx1 - gx0 + 1, gh = gy1 - gy0 + 1;
  // which cells the outline touches at all
  const inPoly = new Uint8Array(gw * gh);
  for (let y = gy0 * CELL; y < Math.min(height, (gy1 + 1) * CELL); y++) {
    const row = y * width, cy = ((y / CELL) | 0) - gy0;
    for (let x = gx0 * CELL; x < Math.min(width, (gx1 + 1) * CELL); x++) {
      if (mask[row + x]) inPoly[cy * gw + (((x / CELL) | 0) - gx0)] = 1;
    }
  }
  // empty cells connected to the outside are background
  const state = new Uint8Array(gw * gh);       // 1 empty, 2 background, 0 surface/out
  const queue: number[] = [];
  for (let c = 0; c < gw * gh; c++) {
    if (!inPoly[c]) continue;
    const gc = (gy0 + ((c / gw) | 0)) * cw + gx0 + (c % gw);
    if (!isFinite(surf[gc])) state[c] = 1;
  }
  // A cell that sees the far wall through a hole has a depth -- the wall's --
  // and selecting around it took the wall. So a cell much deeper than the
  // nearest surface within HOLE_CELLS of it counts as empty too; whether it
  // is a hole or background beyond an edge is decided below, like the rest.
  const HOLE_CELLS = 8;
  const deep = new Uint8Array(gw * gh);
  for (let c = 0; c < gw * gh; c++) {
    if (!inPoly[c] || state[c]) continue;
    const cx = c % gw, cy = (c / gw) | 0;
    const d = surf[(gy0 + cy) * cw + gx0 + cx];
    let near = Infinity;
    for (let y = Math.max(0, cy - HOLE_CELLS); y <= Math.min(gh - 1, cy + HOLE_CELLS); y++) {
      for (let x = Math.max(0, cx - HOLE_CELLS); x <= Math.min(gw - 1, cx + HOLE_CELLS); x++) {
        if (!inPoly[y * gw + x]) continue;
        const v = surf[(gy0 + y) * cw + gx0 + x];
        if (v < near) near = v;
      }
    }
    if (d > BRIDGE_RATIO * near) { state[c] = 1; deep[c] = 1; }
  }
  const touchesOut = (cx: number, cy: number) => {
    for (const [dx, dy] of [[1, 0], [-1, 0], [0, 1], [0, -1]]) {
      const ux = cx + dx, uy = cy + dy;
      if (ux < 0 || uy < 0 || ux >= gw || uy >= gh || !inPoly[uy * gw + ux]) return true;
    }
    return false;
  };
  for (let c = 0; c < gw * gh; c++) {
    if (state[c] === 1 && touchesOut(c % gw, (c / gw) | 0)) { state[c] = 2; queue.push(c); }
  }
  while (queue.length) {
    const c = queue.pop()!;
    const cx = c % gw, cy = (c / gw) | 0;
    for (const [dx, dy] of [[1, 0], [-1, 0], [0, 1], [0, -1]]) {
      const ux = cx + dx, uy = cy + dy;
      if (ux < 0 || uy < 0 || ux >= gw || uy >= gh) continue;
      const u = uy * gw + ux;
      if (state[u] === 1) { state[u] = 2; queue.push(u); }
    }
  }
  // holes: fill from the rim inwards
  let todo: number[] = [];
  if (fill) {
    for (let c = 0; c < gw * gh; c++) {
      if (state[c] !== 1) continue;
      // the far wall seen through the hole gives way to the surface around it
      if (deep[c]) surf[(gy0 + ((c / gw) | 0)) * cw + gx0 + (c % gw)] = Infinity;
      todo.push(c);
    }
  }
  while (todo.length) {
    const next: number[] = [];
    const fills: [number, number][] = [];
    for (const c of todo) {
      const cx = c % gw, cy = (c / gw) | 0;
      let sum = 0, cnt = 0;
      for (let dy = -1; dy <= 1; dy++) {
        for (let dx = -1; dx <= 1; dx++) {
          const ux = cx + dx, uy = cy + dy;
          if ((!dx && !dy) || ux < 0 || uy < 0 || ux >= gw || uy >= gh) continue;
          const v = surf[(gy0 + uy) * cw + gx0 + ux];
          if (isFinite(v)) { sum += v; cnt++; }
        }
      }
      if (cnt) fills.push([c, sum / cnt]); else next.push(c);
    }
    if (!fills.length) break;          // nothing to grow from (cannot happen)
    for (const [c, v] of fills) surf[(gy0 + ((c / gw) | 0)) * cw + gx0 + (c % gw)] = v;
    todo = next;
  }
  return { grid: { surf, cw, ch }, crop: [gx0, gy0, gw, gh] };
}

/** Combine a new hit set into an existing selection mask, in place. */
export function applyMode(current: U8, hits: U32, mode: SelectMode) {
  if (mode === 'replace') current.fill(0);
  const v = mode === 'subtract' ? 0 : 1;
  for (let k = 0; k < hits.length; k++) current[hits[k]] = v;
}

/** Indices of a boolean selection mask. */
export function maskToIndices(mask: U8): U32 {
  let n = 0;
  for (let i = 0; i < mask.length; i++) n += mask[i];
  const out = new Uint32Array(n);
  let w = 0;
  for (let i = 0; i < mask.length; i++) if (mask[i]) out[w++] = i;
  return out;
}
