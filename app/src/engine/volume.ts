/**
 * The selection as the volume the user painted.
 *
 * A list of splat indices says nothing about a hole that has no splats in
 * it, and that is the part of a repair that most needs saying. So every
 * brush stroke, lasso and box is also kept as world-space geometry, and the
 * engine gets that geometry with every command that uses the selection: it
 * can project it into any camera for its masks, and fill where nothing was.
 *
 * - A brush stroke is a chain of capsules: the segment between two dabs,
 *   swept by an ellipsoid with semi-axes (r, r, depth), the short one along
 *   the view it was painted from (selection.ts, `selectInRows`).
 * - A lasso or box is its outline extruded over the depth window around the
 *   surface it was drawn on, in the camera it was drawn in, with the holes
 *   inside the outline closed (`fillRegionHoles`); "through" takes every depth.
 * - Ops are kept in order. A point is inside if the newest op containing it
 *   adds; un-painting and painting again over a spot ends selected.
 *
 * The selected splats are exactly the ones inside: each op's hits are the
 * same functions the selection itself runs. `exact` is false only when an op
 * was added on top of a selection that was not painted (debris, invert).
 *
 * Wire format and the reference implementation: core/splatcore/volume.py.
 */
import { project, selectInRows, selectInRegion, fillPolygon, ROW,
         type DepthGrid, type SizeReach } from './selection';

export type V3 = [number, number, number];
type Pt = { x: number; y: number };

export interface BrushOp {
  kind: 'brush'; add: boolean; seq: number; view: V3;
  /** Capsule rows, `ROW` numbers each (selection.ts, `segmentRows`). */
  rows: Float64Array;
}

export interface RegionOp {
  kind: 'region'; add: boolean; seq: number; view: V3;
  /** View-projection, column-major, and the viewport it maps to (CSS px). */
  vp: Float32Array; width: number; height: number;
  polygon: Pt[];
  through: boolean;
  /** Depth cells kept: [gx0, gy0, gw, gh] of a cw x ch grid of 6 px cells. */
  crop: [number, number, number, number]; cw: number; ch: number;
  depth: Float32Array;              // gw * gh, Infinity where no surface
  /** Window behind the surface per unit of depth; in front, x frontScale. */
  backPerDepth: number; frontScale: number;
  near: number; far: number;
}

export type VolumeOp = BrushOp | RegionOp;

export interface Volume {
  readonly ops: readonly VolumeOp[];
  readonly exact: boolean;
  /** Next op's sequence number. */
  readonly seq: number;
}

export const EMPTY_VOLUME: Volume = Object.freeze({ ops: [], exact: true, seq: 1 });

/**
 * How much deeper un-painting reaches than painting. With the same window,
 * a subtract anchored a hair off the surface the add was anchored on left a
 * sliver of the add behind it: no splats in it, but the masks and the
 * outline kept the spot. Twice the window takes the whole of it -- the
 * tool's own window, or the painted volume's where that is deeper
 * (`paintedReach`).
 */
export const SUBTRACT_REACH = 2;

type OpBody = Omit<BrushOp, 'seq'> | Omit<RegionOp, 'seq'>;

/**
 * The volume after one more op. `replace` starts again; `prevEmpty` says the
 * selection the op lands on was empty, so a volume added onto it is exact.
 */
export function withOp(base: Volume | null, op: OpBody, mode: 'replace' | 'add' | 'subtract',
                       prevEmpty: boolean): Volume {
  if (mode === 'replace') {
    return { ops: [{ ...op, add: true, seq: 1 } as VolumeOp], exact: true, seq: 2 };
  }
  const b = base ?? { ops: [], exact: prevEmpty, seq: 1 };
  return {
    ops: [...b.ops, { ...op, add: mode !== 'subtract', seq: b.seq } as VolumeOp],
    exact: base ? b.exact : prevEmpty,
    seq: b.seq + 1
  };
}

export function hasAdds(v: Volume | null): boolean {
  return !!v && v.ops.some(o => o.add);
}

/**
 * How far the painted volume reaches from the surface it was painted on,
 * measured along `view`, in world units: the largest half-extent of an add
 * op along that direction.
 *
 * For un-painting, which has to take all of what was painted there and not
 * its own tool's window of it. A brush reaches half its radius times the
 * depth setting, a lasso or box six pixels times it: a box drawn in mode
 * "Subtract" over a brush stroke, or a small brush over a large one's, took the
 * splats at the surface and left the stroke's volume standing in front of
 * and behind its own thin window. The outline did not move at all, and what
 * the stroke had selected deeper down stayed selected under it.
 *
 * A stroke seen from another side counts with its extent along this view
 * (an ellipsoid r, r, depth, the short axis along its own). An outline
 * taken "through" has no depth of its own and does not count.
 */
export function paintedReach(v: Volume | null, view: V3): number {
  let best = 0;
  if (!v) return 0;
  for (const op of v.ops) {
    if (!op.add) continue;
    if (op.kind === 'brush') {
      const c = op.view[0] * view[0] + op.view[1] * view[1] + op.view[2] * view[2];
      const c2 = Math.min(1, c * c);
      for (let o = 0; o + ROW <= op.rows.length; o += ROW) {
        const r = op.rows[o + 6], d = op.rows[o + 7];
        const e = Math.sqrt(r * r * (1 - c2) + d * d * c2);
        if (e > best) best = e;
      }
    } else if (!op.through) {
      // a slab about its surface: the window at its farthest cell, the wider side
      let far = 0;
      for (let k = 0; k < op.depth.length; k++) if (isFinite(op.depth[k]) && op.depth[k] > far) far = op.depth[k];
      const e = op.backPerDepth * far * Math.max(1, op.frontScale);
      if (e > best) best = e;
    }
  }
  return best;
}

/* ------------------------------------------------------------------ wire */

const r6 = (x: number) => Math.round(x * 1e6) / 1e6;

/**
 * The header value the engine gets: see core/splatcore/volume.py.
 *
 * `cut` (the default, what every command sends): the add rows of a brush
 * stroke follow what later ops took away again (`cutRows`). Without it the
 * rows are as painted, and the engine's membership test selects exactly what
 * the selection holds -- the form its reference test checks dumps against.
 */
export function volumeToWire(v: Volume | null, cut = true): Record<string, any> | null {
  if (!v || !hasAdds(v)) return null;
  if (cut && wireCache.has(v)) return wireCache.get(v) ?? null;
  const capsules: number[][] = [], subtract: number[][] = [];
  const regions: Record<string, any>[] = [];
  let view: V3 = [0, 0, 1];
  let exact = v.exact;
  for (let i = 0; i < v.ops.length; i++) {
    const op = v.ops[i];
    view = op.view;
    if (op.kind === 'brush') {
      let rows: ArrayLike<number> = op.rows;
      if (cut && op.add && v.ops.some((o, j) => j > i && !o.add)) {
        const c = cutRows(v, op);
        rows = c.rows;
        if (c.cut) exact = false;       // near a cut the rows are a raster, not the stroke
      }
      const n = (rows.length / ROW) | 0;
      for (let k = 0; k < n; k++) {
        const row = Array.from({ length: ROW }, (_, j) => r6(rows[k * ROW + j]));
        (op.add ? capsules : subtract).push([...row, op.seq, ...op.view.map(r6)]);
      }
    } else {
      regions.push({
        mode: op.add ? 'add' : 'subtract', seq: op.seq, view: op.view.map(r6),
        vp: Array.from(op.vp), size: [op.width, op.height],
        polygon: op.polygon.map(p => [r6(p.x), r6(p.y)]),
        through: op.through, cell: 6, grid: op.crop,
        depth: Array.from(op.depth, d => isFinite(d) ? r6(d) : null),
        back_per_depth: op.backPerDepth, front_scale: op.frontScale,
        near: r6(op.near), far: r6(op.far)
      });
    }
  }
  // every stroke taken away again and no outline added: nothing is painted
  const wire = capsules.length || regions.some(g => g.mode === 'add')
    ? { version: 1, view: view.map(r6), capsules, subtract, regions, exact } : null;
  if (cut) wireCache.set(v, wire);
  return wire;
}

/** Volumes never change, so the rows cut for one are cut once. */
const wireCache = new WeakMap<Volume, Record<string, any> | null>();

/* ------------------------------------------------- strokes with cuts in them */

/** A row is looked at in stretches this long, in brush radii. */
const CUT_STRETCH = 0.25;
/** Where a cut runs through a stroke: raster spacing in brush radii, a dab's radius in spacings. */
const CUT_CELL = 0.25;
const CUT_DAB = 0.7;
/** Most dabs one stroke becomes; past that its raster is made coarser. */
const CUT_MAX_DABS = 400;

/** Two unit vectors across a view direction. */
function across(v: V3): [V3, V3] {
  const a: V3 = Math.abs(v[0]) < 0.9 ? [1, 0, 0] : [0, 1, 0];
  const d = a[0] * v[0] + a[1] * v[1] + a[2] * v[2];
  let e1: V3 = [a[0] - v[0] * d, a[1] - v[1] * d, a[2] - v[2] * d];
  const l = Math.hypot(e1[0], e1[1], e1[2]) || 1;
  e1 = [e1[0] / l, e1[1] / l, e1[2] / l];
  return [e1, [v[1] * e1[2] - v[2] * e1[1], v[2] * e1[0] - v[0] * e1[2], v[0] * e1[1] - v[1] * e1[0]]];
}

/**
 * The add rows of a brush stroke after what later ops took away again.
 *
 * Membership is not the reason: the engine works that out from all rows in
 * order, newest first, and a subtract row does its work there. But its
 * picture fill (core/splatcore/imfill.py) takes the stroke's first footprint
 * from the *add* rows alone, and every pixel in that footprint that changes
 * when the selection leaves counts as hole. Sent as painted, a stroke of
 * which half had been un-painted was filled over its whole length again. On
 * the shoe's stand: 16,900 new splats in the half no longer selected with
 * the rows as painted, 2,200 (the fill's own rim) with no volume at all, and
 * 2,500 with the rows cut as here.
 *
 * So the stroke is looked at stretch by stretch, by the same membership the
 * selection uses (`evaluateVolume`):
 *   - a stretch whose footprint is all still inside stays a row as painted,
 *   - one with nothing left inside is left out,
 *   - one a cut runs through becomes a raster of small flat dabs over the
 *     part that is left. Its rim follows the cut to a fraction of the brush;
 *     trimming rows at their axis alone left the brush's radius hanging over
 *     the cut, and the fill went that far into it.
 * The subtract rows are sent all the same, so the order (paint, un-paint,
 * paint again) holds on the engine's side as it does here.
 */
function cutRows(v: Volume, op: BrushOp): { rows: number[]; cut: boolean } {
  const R = op.rows, m = (R.length / ROW) | 0;
  const view = op.view, [e1, e2] = across(view);
  const dot = (x: number, y: number, z: number, e: V3) => x * e[0] + y * e[1] + z * e[2];
  // offsets of the samples across a stretch: its middle and three rings, no
  // two samples further apart than a third of the brush (with two rings a
  // spot taken out with a brush a quarter the size fell between them, and
  // the stretch around it counted as whole)
  const ring: [number, number][] = [[0, 0]];
  for (const [f, k] of [[0.33, 6], [0.66, 12], [0.95, 18]]) {
    for (let j = 0; j < k; j++) ring.push([f * Math.cos(j / k * 2 * Math.PI), f * Math.sin(j / k * 2 * Math.PI)]);
  }
  const S = ring.length;

  // 1. stretches, and where each is sampled
  const st: { row: number; i: number; n: number }[] = [];
  const pts: number[] = [];
  let rMin = Infinity;
  for (let k = 0; k < m; k++) {
    const o = k * ROW, r = R[o + 6];
    if (r < rMin) rMin = r;
    const ux = R[o + 3] - R[o], uy = R[o + 4] - R[o + 1], uz = R[o + 5] - R[o + 2];
    const n = Math.max(1, Math.ceil(Math.hypot(ux, uy, uz) / (CUT_STRETCH * r)));
    for (let i = 0; i < n; i++) {
      const t = (i + 0.5) / n;
      const cx = R[o] + ux * t, cy = R[o + 1] + uy * t, cz = R[o + 2] + uz * t;
      for (const [a, b] of ring) {
        pts.push(cx + (e1[0] * a + e2[0] * b) * r, cy + (e1[1] * a + e2[1] * b) * r,
                 cz + (e1[2] * a + e2[2] * b) * r);
      }
      st.push({ row: k, i, n });
    }
  }
  const inside = evaluateVolume(v, Float32Array.from(pts), pts.length / 3);
  const state = st.map((_, q) => {
    let c = 0;
    for (let j = 0; j < S; j++) c += inside[q * S + j];
    return c === S ? 1 : c === 0 ? 0 : 2;          // whole, gone, a cut runs through
  });
  if (state.every(s => s === 1)) return { rows: Array.from(R), cut: false };

  // 2. whole stretches: runs of them, as rows
  const rows: number[] = [];
  const whole: number[][] = [];                    // the same, flat across the view: x0, y0, x1, y1, r
  const end = (k: number, t: number): V3 => {
    const o = k * ROW;
    return [R[o] + (R[o + 3] - R[o]) * t, R[o + 1] + (R[o + 4] - R[o + 1]) * t, R[o + 2] + (R[o + 5] - R[o + 2]) * t];
  };
  for (let q = 0; q < st.length; q++) {
    if (state[q] !== 1) continue;
    let e = q;
    while (e + 1 < st.length && st[e + 1].row === st[q].row && state[e + 1] === 1) e++;
    const k = st[q].row, o = k * ROW;
    const a = end(k, st[q].i / st[q].n), b = end(k, (st[e].i + 1) / st[e].n);
    rows.push(a[0], a[1], a[2], b[0], b[1], b[2], R[o + 6], R[o + 7]);
    whole.push([dot(a[0], a[1], a[2], e1), dot(a[0], a[1], a[2], e2),
                dot(b[0], b[1], b[2], e1), dot(b[0], b[1], b[2], e2), R[o + 6]]);
    q = e;
  }

  // 3. stretches with a cut through them: a raster across the view, shared
  //    by all of them, so neighbours do not lay their dabs twice
  const toSeg = (x: number, y: number, x0: number, y0: number, x1: number, y1: number) => {
    const ux = x1 - x0, uy = y1 - y0, uu = ux * ux + uy * uy;
    let t = uu > 1e-18 ? ((x - x0) * ux + (y - y0) * uy) / uu : 0;
    t = t < 0 ? 0 : t > 1 ? 1 : t;
    return { t, l: Math.hypot(x - x0 - ux * t, y - y0 - uy * t) };
  };
  let s = CUT_CELL * rMin;
  for (let pass = 0; pass < 3; pass++) {
    const rho = CUT_DAB * s, dy = s * Math.sqrt(3) / 2;
    const nodes = new Map<string, { p: V3; l: number; depth: number }>();
    for (let q = 0; q < st.length; q++) {
      if (state[q] !== 2) continue;
      const k = st[q].row, o = k * ROW, r = R[o + 6];
      const a = end(k, st[q].i / st[q].n), b = end(k, (st[q].i + 1) / st[q].n);
      const x0 = dot(a[0], a[1], a[2], e1), y0 = dot(a[0], a[1], a[2], e2);
      const x1 = dot(b[0], b[1], b[2], e1), y1 = dot(b[0], b[1], b[2], e2);
      const z0 = dot(a[0], a[1], a[2], view), z1 = dot(b[0], b[1], b[2], view);
      for (let j = Math.floor((Math.min(y0, y1) - r) / dy); j <= Math.ceil((Math.max(y0, y1) + r) / dy); j++) {
        const shift = (j & 1) * 0.5;
        for (let i = Math.floor((Math.min(x0, x1) - r) / s - shift); i <= Math.ceil((Math.max(x0, x1) + r) / s - shift); i++) {
          const x = (i + shift) * s, y = j * dy;
          const { t, l } = toSeg(x, y, x0, y0, x1, y1);
          if (l > r) continue;
          const key = i + ',' + j;
          const had = nodes.get(key);
          if (had && had.l <= l) continue;
          const z = z0 + (z1 - z0) * t;
          nodes.set(key, {
            p: [e1[0] * x + e2[0] * y + view[0] * z, e1[1] * x + e2[1] * y + view[1] * z,
                e1[2] * x + e2[2] * y + view[2] * z],
            l,
            // as deep as the stroke is there, and never under a third of its depth
            depth: R[o + 7] * Math.sqrt(Math.max(1 - (l / r) * (l / r), 0.11))
          });
        }
      }
    }
    // a whole stretch next door already covers some of them
    const list = [...nodes.values()].filter(nd => {
      const x = dot(nd.p[0], nd.p[1], nd.p[2], e1), y = dot(nd.p[0], nd.p[1], nd.p[2], e2);
      return !whole.some(w => toSeg(x, y, w[0], w[1], w[2], w[3]).l <= w[4] - rho);
    });
    // A dab stays if its middle is inside and most of its rim is: one that
    // straddles a cut or the stroke's own edge would reach its radius past
    // it, and what it covers there counts as painted. Short of the edge by
    // part of a dab is the better side to err on -- the splats there are in
    // the selection anyway.
    const RIM = 6;
    const flat = new Float32Array(list.length * (RIM + 1) * 3);
    list.forEach((nd, q) => {
      flat.set(nd.p, q * (RIM + 1) * 3);
      for (let j = 0; j < RIM; j++) {
        const a = 0.9 * rho * Math.cos(j / RIM * 2 * Math.PI), b = 0.9 * rho * Math.sin(j / RIM * 2 * Math.PI);
        flat.set([nd.p[0] + e1[0] * a + e2[0] * b, nd.p[1] + e1[1] * a + e2[1] * b,
                  nd.p[2] + e1[2] * a + e2[2] * b], (q * (RIM + 1) + 1 + j) * 3);
      }
    });
    const keep = evaluateVolume(v, flat, list.length * (RIM + 1));
    const dabs = list.filter((_, q) => {
      const o = q * (RIM + 1);
      if (!keep[o]) return false;
      let rim = 0;
      for (let j = 1; j <= RIM; j++) rim += keep[o + j];
      return rim >= 4;
    });
    if (dabs.length > CUT_MAX_DABS && pass < 2) {
      s *= Math.sqrt(dabs.length / CUT_MAX_DABS) * 1.05;
      continue;
    }
    for (const nd of dabs) rows.push(nd.p[0], nd.p[1], nd.p[2], nd.p[0], nd.p[1], nd.p[2], rho, nd.depth);
    break;
  }
  return { rows, cut: true };
}

/* ------------------------------------------------------------ membership */

/** The full-size depth grid a region was selected with. */
export function regionGrid(op: RegionOp): DepthGrid {
  const surf = new Float32Array(op.cw * op.ch).fill(Infinity);
  const [gx0, gy0, gw, gh] = op.crop;
  for (let y = 0; y < gh; y++) {
    for (let x = 0; x < gw; x++) surf[(gy0 + y) * op.cw + gx0 + x] = op.depth[y * gw + x];
  }
  return { surf, cw: op.cw, ch: op.ch };
}

const maskCache = new WeakMap<RegionOp, Uint8Array>();
function regionMask(op: RegionOp): Uint8Array {
  let m = maskCache.get(op);
  if (!m) {
    m = fillPolygon(op.polygon, Math.max(1, op.width | 0), Math.max(1, op.height | 0));
    maskCache.set(op, m);
  }
  return m;
}

/** The splats an op contains -- the same functions the selection ran. */
export function opHits(op: VolumeOp, centres: Float32Array<ArrayBufferLike>,
                       alive?: Uint8Array<ArrayBufferLike>,
                       size?: OpSize): Uint32Array<ArrayBufferLike> {
  if (op.kind === 'brush') return selectInRows(centres, op.rows, op.view, { alive, size: size?.brush });
  const w = Math.max(1, op.width | 0), h = Math.max(1, op.height | 0);
  const p = project(centres, op.vp, op.width, op.height);
  return selectInRegion(p, regionMask(op), w, h, {
    depthMode: op.through ? 'through' : 'surface', tolPx: 1,
    perPixelPerUnit: op.backPerDepth, frontScale: op.frontScale,
    alive, grid: regionGrid(op),
    size: size?.region, viewDir: size?.region ? op.view : undefined
  });
}

/**
 * The splat extents a selection was made with (selection.ts, `SizeReach`):
 * with size-aware selection a stroke also takes splats whose extent reaches
 * its window, so the volume has to be evaluated with the same reach to stand
 * for the same splats. Brush and region may cap it differently.
 */
export interface OpSize { brush?: SizeReach; region?: SizeReach }

/** The selection mask a volume stands for, worked out from scratch. */
export function evaluateVolume(v: Volume, centres: Float32Array<ArrayBufferLike>, n: number,
                               alive?: Uint8Array<ArrayBufferLike>,
                               size?: OpSize): Uint8Array<ArrayBuffer> {
  const mask = new Uint8Array(n);
  for (const op of v.ops) {
    const hits = opHits(op, centres, alive, size);
    const val = op.add ? 1 : 0;
    for (let k = 0; k < hits.length; k++) if (hits[k] < n) mask[hits[k]] = val;
  }
  return mask;
}

/* --------------------------------------------------------------- geometry */

/** Pixel + camera depth back to a world point, for any stored view-projection. */
export function unprojectVP(vp: ArrayLike<number>, W: number, H: number,
                            px: number, py: number, w: number): V3 {
  const hw = W * 0.5, hh = H * 0.5;
  const nx = (px - hw) / hw, ny = (hh - py) / hh;
  // rows of the matrix (column-major storage: row r is vp[r], vp[4+r], ...)
  const row = (r: number) => [vp[r], vp[4 + r], vp[8 + r], vp[12 + r]];
  const m0 = row(0), m1 = row(1), m3 = row(3);
  const A = [
    [m0[0] - nx * m3[0], m0[1] - nx * m3[1], m0[2] - nx * m3[2]],
    [m1[0] - ny * m3[0], m1[1] - ny * m3[1], m1[2] - ny * m3[2]],
    [m3[0], m3[1], m3[2]]
  ];
  const b = [-(m0[3] - nx * m3[3]), -(m1[3] - ny * m3[3]), w - m3[3]];
  const det = (M: number[][]) =>
    M[0][0] * (M[1][1] * M[2][2] - M[1][2] * M[2][1])
    - M[0][1] * (M[1][0] * M[2][2] - M[1][2] * M[2][0])
    + M[0][2] * (M[1][0] * M[2][1] - M[1][1] * M[2][0]);
  const D = det(A) || 1e-30;
  const col = (i: number) => A.map((r, k) => r.map((v, j) => (j === i ? b[k] : v)));
  return [det(col(0)) / D, det(col(1)) / D, det(col(2)) / D];
}

/** World units per pixel at one unit of depth, read off a view-projection. */
export function perPixelPerUnitOf(vp: ArrayLike<number>, H: number): number {
  const p11 = Math.hypot(vp[1], vp[5], vp[9]);
  return 2 / Math.max(p11 * H, 1e-9);
}

/**
 * Points on the boundary of the volume, for drawing its outline from any
 * view. Each point carries the spacing it was sampled at, so the drawer
 * knows how large a footprint to give it at the current zoom.
 *
 * Capsules are sampled in their stretched frame, where they are plain
 * capsules, and squashed back; points that fall inside the next or previous
 * capsule of the same stroke are dropped, which leaves the outside of the
 * swept tube instead of a thousand overlapping balls. Regions give their
 * front and back faces and, along the outline, the walls between them.
 * With any un-painting, only points that are inside the final volume stay
 * -- the cut faces of a subtract included.
 */
export interface Shell { pts: Float32Array; spacing: Float32Array; count: number; }

export function volumeShell(v: Volume | null, budget = 140000): Shell {
  const pts: number[] = [], sp: number[] = [];
  if (!v || !hasAdds(v)) return { pts: new Float32Array(0), spacing: new Float32Array(0), count: 0 };
  const anySub = v.ops.some(o => !o.add);

  // spacing: from the smallest radius, relaxed until the estimate fits
  let area = 0, rMin = Infinity;
  for (const op of v.ops) {
    if (op.kind !== 'brush') continue;
    const n = (op.rows.length / ROW) | 0;
    for (let k = 0; k < n; k++) {
      const o = k * ROW;
      const r = op.rows[o + 6];
      const L = Math.hypot(op.rows[o + 3] - op.rows[o], op.rows[o + 4] - op.rows[o + 1],
                           op.rows[o + 5] - op.rows[o + 2]);
      area += 2 * Math.PI * r * L + (k === 0 || k === n - 1 ? 2 * Math.PI * r * r : 0);
      rMin = Math.min(rMin, r);
    }
  }
  const hBrush = isFinite(rMin) ? Math.max(rMin / 7, Math.sqrt(area / (budget * 0.7))) : 0;

  for (const op of v.ops) {
    const scale = op.add ? 0.985 : 1.015;   // just inside an add, just outside a cut
    if (op.kind === 'brush') brushShell(op, hBrush, scale, pts, sp);
    else regionShell(op, scale, pts, sp, budget);
  }
  const all = Float32Array.from(pts), allSp = Float32Array.from(sp);
  if (!anySub) return { pts: all, spacing: allSp, count: allSp.length };
  // the same membership the selection uses, over the shell points
  const keep = evaluateVolume(v, all, allSp.length);
  let m = 0;
  for (let i = 0; i < keep.length; i++) m += keep[i];
  const out = new Float32Array(m * 3), outSp = new Float32Array(m);
  for (let i = 0, w = 0; i < keep.length; i++) {
    if (!keep[i]) continue;
    out[w * 3] = all[i * 3]; out[w * 3 + 1] = all[i * 3 + 1]; out[w * 3 + 2] = all[i * 3 + 2];
    outSp[w++] = allSp[i];
  }
  return { pts: out, spacing: outSp, count: m };
}

function brushShell(op: BrushOp, h: number, scale: number, pts: number[], sp: number[]) {
  const R = op.rows, n = (R.length / ROW) | 0;
  const [vx, vy, vz] = op.view;
  // stretched frame per row: T(y) = y + v (y.v)(s - 1), s = r / depth
  const insideRow = (k: number, x: number, y: number, z: number) => {
    if (k < 0 || k >= n) return false;
    const o = k * ROW;
    const r = R[o + 6], s = r / Math.max(R[o + 7], 1e-9);
    const wx = x - R[o], wy = y - R[o + 1], wz = z - R[o + 2];
    const wa = (wx * vx + wy * vy + wz * vz) * (s - 1);
    const px = wx + vx * wa, py = wy + vy * wa, pz = wz + vz * wa;
    const ux = R[o + 3] - R[o], uy = R[o + 4] - R[o + 1], uz = R[o + 5] - R[o + 2];
    const ua = (ux * vx + uy * vy + uz * vz) * (s - 1);
    const tx = ux + vx * ua, ty = uy + vy * ua, tz = uz + vz * ua;
    const uu = tx * tx + ty * ty + tz * tz;
    let t = uu > 1e-12 ? (px * tx + py * ty + pz * tz) / uu : 0;
    t = t < 0 ? 0 : t > 1 ? 1 : t;
    const dx = px - tx * t, dy = py - ty * t, dz = pz - tz * t;
    return dx * dx + dy * dy + dz * dz < r * r * 0.97;
  };
  for (let k = 0; k < n; k++) {
    const o = k * ROW;
    const ax = R[o], ay = R[o + 1], az = R[o + 2];
    const r = R[o + 6] * scale, s = R[o + 6] / Math.max(R[o + 7], 1e-9);
    const inv = 1 / s - 1;               // squash back along the view
    const ux = R[o + 3] - ax, uy = R[o + 4] - ay, uz = R[o + 5] - az;
    const ua = (ux * vx + uy * vy + uz * vz) * (s - 1);
    const tx = ux + vx * ua, ty = uy + vy * ua, tz = uz + vz * ua;
    const L = Math.hypot(tx, ty, tz);
    // an axis and two normals in the stretched frame
    let ex = 0, ey = 0, ez = 1;
    if (L > 1e-9) { ex = tx / L; ey = ty / L; ez = tz / L; }
    else { ex = vx; ey = vy; ez = vz; }
    let n1x = ey * 0 - ez * 1, n1y = ez * 0 - ex * 0, n1z = ex * 1 - ey * 0;   // e x (0,1,0)
    if (Math.hypot(n1x, n1y, n1z) < 1e-6) { n1x = 0; n1y = ez; n1z = -ey; }  // e x (1,0,0)
    const nl = Math.hypot(n1x, n1y, n1z); n1x /= nl; n1y /= nl; n1z /= nl;
    const n2x = ey * n1z - ez * n1y, n2y = ez * n1x - ex * n1z, n2z = ex * n1y - ey * n1x;
    const emit = (yx: number, yy: number, yz: number) => {
      const ya = (yx * vx + yy * vy + yz * vz) * inv;
      const x = ax + yx + vx * ya, y = ay + yy + vy * ya, z = az + yz + vz * ya;
      if (insideRow(k - 1, x, y, z) || insideRow(k + 1, x, y, z)) return;
      pts.push(x, y, z);
      sp.push(h);
    };
    // the cylinder
    const around = Math.max(8, Math.ceil(2 * Math.PI * r / h));
    const along = L > 1e-9 ? Math.max(1, Math.ceil(L / h)) : 0;
    for (let i = 0; i <= along && along; i++) {
      const t = i / along;
      for (let j = 0; j < around; j++) {
        const ang = (j + (i & 1) * 0.5) / around * 2 * Math.PI;
        const c = Math.cos(ang) * r, sn = Math.sin(ang) * r;
        emit(tx * t + n1x * c + n2x * sn, ty * t + n1y * c + n2y * sn,
             tz * t + n1z * c + n2z * sn);
      }
    }
    // a ball at the start (the end's ball is the next row's start), and at the
    // very end of the stroke
    const balls = k === n - 1 ? [0, 1] : [0];
    const nb = Math.max(24, Math.ceil(4 * Math.PI * r * r / (h * h)));
    for (const end of balls) {
      const cx = tx * end, cy = ty * end, cz = tz * end;
      for (let i = 0; i < nb; i++) {
        // Fibonacci sphere
        const yv = 1 - 2 * (i + 0.5) / nb;
        const rad = Math.sqrt(Math.max(0, 1 - yv * yv));
        const th = i * 2.399963229728653;
        const px = Math.cos(th) * rad, pz = Math.sin(th) * rad;
        emit(cx + (ex * yv + n1x * px + n2x * pz) * r,
             cy + (ey * yv + n1y * px + n2y * pz) * r,
             cz + (ez * yv + n1z * px + n2z * pz) * r);
      }
    }
  }
}

function regionShell(op: RegionOp, scale: number, pts: number[], sp: number[], budget: number) {
  const W = op.width, H = op.height;
  const mask = regionMask(op);
  const mw = Math.max(1, W | 0), mh = Math.max(1, H | 0);
  const [gx0, gy0, gw, gh] = op.crop;
  const ppu = perPixelPerUnitOf(op.vp, H);
  // sample every `step` pixels inside the outline; coarser for big regions
  let area = 0;
  for (let i = 0; i < mask.length; i++) area += mask[i];
  const step = Math.max(3, Math.ceil(Math.sqrt(area / Math.max(budget * 0.15, 1))));
  const cols = Math.ceil(gw * 6 / step), rows = Math.ceil(gh * 6 / step);
  const range = new Float64Array(cols * rows * 2).fill(NaN);
  for (let j = 0; j < rows; j++) {
    for (let i = 0; i < cols; i++) {
      const px = gx0 * 6 + (i + 0.5) * step, py = gy0 * 6 + (j + 0.5) * step;
      const ix = px | 0, iy = py | 0;
      if (ix < 0 || iy < 0 || ix >= mw || iy >= mh || !mask[iy * mw + ix]) continue;
      let lo: number, hi: number;
      if (op.through) { lo = op.near; hi = op.far; }
      else {
        const cx = ((px / 6) | 0) - gx0, cy = ((py / 6) | 0) - gy0;
        if (cx < 0 || cy < 0 || cx >= gw || cy >= gh) continue;
        const d0 = op.depth[cy * gw + cx];
        if (!isFinite(d0)) continue;
        const back = Math.max(op.backPerDepth * d0, 1e-6);
        lo = d0 - back * op.frontScale; hi = d0 + back;
      }
      const mid = 0.5 * (lo + hi), half = 0.5 * (hi - lo) * scale;
      range[(j * cols + i) * 2] = Math.max(mid - half, 1e-4);
      range[(j * cols + i) * 2 + 1] = mid + half;
    }
  }
  const has = (i: number, j: number) => i >= 0 && j >= 0 && i < cols && j < rows
    && !isNaN(range[(j * cols + i) * 2]);
  for (let j = 0; j < rows; j++) {
    for (let i = 0; i < cols; i++) {
      if (!has(i, j)) continue;
      const lo = range[(j * cols + i) * 2], hi = range[(j * cols + i) * 2 + 1];
      const px = gx0 * 6 + (i + 0.5) * step, py = gy0 * 6 + (j + 0.5) * step;
      const edge = !has(i - 1, j) || !has(i + 1, j) || !has(i, j - 1) || !has(i, j + 1);
      const hw = step * ppu * hi;
      const layers = edge ? Math.min(24, Math.max(1, Math.ceil((hi - lo) / Math.max(hw, 1e-9)))) : 1;
      for (let l = 0; l <= layers; l++) {
        if (!edge && l > 0 && l < layers) continue;
        const w = lo + (hi - lo) * (l / layers);
        const p = unprojectVP(op.vp, W, H, px, py, w);
        pts.push(p[0], p[1], p[2]);
        sp.push(step * ppu * w);
      }
    }
  }
}
