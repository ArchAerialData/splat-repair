/**
 * The viewport's mirror of the document the engine process holds.
 *
 * Both sides index splats the same way: the loaded file occupies [0, nBase),
 * repairs append after it, and indices never move. Editing flips `alive` flags
 * and appends; it never renumbers. That is what lets a repair cross the pipe as
 * a few thousand integers instead of megabytes of geometry.
 *
 * Rebuilding the renderable set copies the surviving splats into fresh property
 * arrays. At 550k that is a few hundred milliseconds -- fine for a discrete
 * "Repair", and the obvious thing to replace with a per-splat instance stream
 * once the brush needs to preview edits live.
 */
import * as pc from 'playcanvas';

/** Typed arrays here may be views on any buffer kind; don't over-constrain. */
type F32 = Float32Array<ArrayBufferLike>;
type U8 = Uint8Array<ArrayBufferLike>;

export interface PropertyArrays {
  names: string[];
  storage: F32[];
}

/** Pull the per-property arrays out of a loaded GSplatData. */
export function propsOf(gsplatData: any): PropertyArrays {
  const el = gsplatData.elements.find((e: any) => e.name === 'vertex');
  const names: string[] = [];
  const storage: F32[] = [];
  for (const p of el.properties) {
    if (!p.storage) continue;
    names.push(p.name);
    storage.push(p.storage as Float32Array);
  }
  return { names, storage };
}

export class SplatDocument {
  nBase = 0;
  /** Per-property arrays for base + generated, parallel to `names`. */
  names: string[] = [];
  base: F32[] = [];
  generated: F32[] = [];
  nGenerated = 0;
  alive: U8 = new Uint8Array(0);
  centres: F32 = new Float32Array(0);
  /** Activated opacity per splat, so the brush can tell surface from haze. */
  alpha: F32 = new Float32Array(0);
  /**
   * World covariance per splat, (xx, xy, xz, yy) and (yz, zz), so a depth
   * test can ask how far a splat reaches along the view ray. Float32 because
   * the GPU selector reads exactly these numbers.
   */
  covA: F32 = new Float32Array(0);
  covB: F32 = new Float32Array(0);
  /**
   * How opaque a splat must be to define the surface the brush lands on.
   * 0.25 suits a typical scan, but some are built from thin translucent
   * layers -- the hut's median opacity is 0.14, and 72 % of it sits below
   * 0.25, so the brush found no surface on much of it. Lowered to the
   * scan's median there, never below 0.1, never above 0.25.
   */
  surfaceAlpha = 0.25;

  load(gsplatData: any) {
    const { names, storage } = propsOf(gsplatData);
    this.names = names;
    this.base = storage;
    this.nBase = gsplatData.numSplats;
    this.generated = names.map(() => new Float32Array(0));
    this.nGenerated = 0;
    this.alive = new Uint8Array(this.nBase).fill(1);
    this.centres = this.collectCentres();
    this.alpha = this.collectAlpha();
    this.covA = new Float32Array(this.nBase * 4);
    this.covB = new Float32Array(this.nBase * 2);
    this.fillCovariance(0, this.nBase, 0);
    this.surfaceAlpha = surfaceAlphaOf(this.alpha);
  }

  get nTotal() { return this.nBase + this.nGenerated; }
  get nAlive() {
    let k = 0;
    for (let i = 0; i < this.alive.length; i++) k += this.alive[i];
    return k;
  }

  private col(name: string): { base: F32; gen: F32 } | null {
    const i = this.names.indexOf(name);
    if (i < 0) return null;
    return { base: this.base[i], gen: this.generated[i] };
  }

  /** xyz of every splat (alive or not), in document index order. */
  private collectCentres(): F32 {
    const n = this.nTotal;
    const out = new Float32Array(n * 3);
    const axes = ['x', 'y', 'z'];
    for (let a = 0; a < 3; a++) {
      const c = this.col(axes[a]);
      if (!c) continue;
      for (let i = 0; i < this.nBase; i++) out[i * 3 + a] = c.base[i];
      for (let i = 0; i < this.nGenerated; i++) out[(this.nBase + i) * 3 + a] = c.gen[i];
    }
    return out;
  }

  /** Opacity is stored as a logit; the brush wants it in 0..1. */
  private collectAlpha(): F32 {
    const n = this.nTotal;
    const out = new Float32Array(n);
    const c = this.col('opacity');
    if (!c) { out.fill(1); return out; }
    for (let i = 0; i < this.nBase; i++) out[i] = 1 / (1 + Math.exp(-c.base[i]));
    for (let i = 0; i < this.nGenerated; i++) {
      out[this.nBase + i] = 1 / (1 + Math.exp(-c.gen[i]));
    }
    return out;
  }

  /**
   * Covariance R S S^T R^T of splats [from, to) of one block (base when
   * `genOffset` is 0 and from < nBase, otherwise generated), written at the
   * same document index. Scales are stored as logs, rotation as (w, x, y, z).
   */
  private fillCovariance(from: number, to: number, genOffset: number) {
    const gen = genOffset > 0;
    const col = (name: string) => {
      const c = this.col(name);
      return c ? (gen ? c.gen : c.base) : null;
    };
    const s0 = col('scale_0'), s1 = col('scale_1'), s2 = col('scale_2');
    const r0 = col('rot_0'), r1 = col('rot_1'), r2 = col('rot_2'), r3 = col('rot_3');
    const A = this.covA, B = this.covB;
    if (!s0 || !s1 || !s2) return;          // no sizes: every splat stays a point
    for (let i = from; i < to; i++) {
      const sx = Math.exp(s0[i]), sy = Math.exp(s1[i]), sz = Math.exp(s2[i]);
      let w = r0 ? r0[i] : 1, x = r1 ? r1[i] : 0, y = r2 ? r2[i] : 0, z = r3 ? r3[i] : 0;
      const l = Math.hypot(w, x, y, z) || 1;
      w /= l; x /= l; y /= l; z /= l;
      // columns of R scaled by the axis lengths
      const m00 = (1 - 2 * (y * y + z * z)) * sx, m01 = 2 * (x * y - w * z) * sy, m02 = 2 * (x * z + w * y) * sz;
      const m10 = 2 * (x * y + w * z) * sx, m11 = (1 - 2 * (x * x + z * z)) * sy, m12 = 2 * (y * z - w * x) * sz;
      const m20 = 2 * (x * z - w * y) * sx, m21 = 2 * (y * z + w * x) * sy, m22 = (1 - 2 * (x * x + y * y)) * sz;
      const d = i + genOffset;
      A[d * 4] = m00 * m00 + m01 * m01 + m02 * m02;
      A[d * 4 + 1] = m00 * m10 + m01 * m11 + m02 * m12;
      A[d * 4 + 2] = m00 * m20 + m01 * m21 + m02 * m22;
      A[d * 4 + 3] = m10 * m10 + m11 * m11 + m12 * m12;
      B[d * 2] = m10 * m20 + m11 * m21 + m12 * m22;
      B[d * 2 + 1] = m20 * m20 + m21 * m21 + m22 * m22;
    }
  }

  setDead(indices: Uint32Array<ArrayBufferLike> | number[]) {
    for (let k = 0; k < indices.length; k++) this.alive[indices[k]] = 0;
  }

  setAlive(indices: Uint32Array<ArrayBufferLike> | number[]) {
    for (let k = 0; k < indices.length; k++) this.alive[indices[k]] = 1;
  }

  /**
   * Fold a repair's new splats in.
   * `block` is interleaved float32, `layout` describes its fields.
   */
  appendGenerated(block: F32, layout: any) {
    const count: number = layout.count;
    const stride: number = layout.stride;
    const fields: [string, number, number][] = layout.fields;
    const shCoeffs: number = layout.shCoeffs;

    const named = new Map<string, number>();
    for (const [name, off, len] of fields) {
      if (name === 'xyz') { named.set('x', off); named.set('y', off + 1); named.set('z', off + 2); }
      else if (name === 'f_dc') for (let i = 0; i < 3; i++) named.set(`f_dc_${i}`, off + i);
      else if (name === 'opacity') named.set('opacity', off);
      else if (name === 'scale') for (let i = 0; i < 3; i++) named.set(`scale_${i}`, off + i);
      else if (name === 'rot') for (let i = 0; i < 4; i++) named.set(`rot_${i}`, off + i);
      else if (name === 'f_rest') for (let i = 0; i < shCoeffs * 3; i++) named.set(`f_rest_${i}`, off + i);
      else for (let i = 0; i < len; i++) named.set(name, off + i);
    }

    const next: F32[] = [];
    for (let p = 0; p < this.names.length; p++) {
      const name = this.names[p];
      const old = this.generated[p];
      const arr = new Float32Array(this.nGenerated + count);
      arr.set(old, 0);
      const src = named.get(name);
      if (src !== undefined) {
        for (let i = 0; i < count; i++) arr[this.nGenerated + i] = block[i * stride + src];
      }
      next[p] = arr;
    }
    this.generated = next;
    const first = this.nGenerated;
    this.nGenerated += count;
    this.alive = growUint8(this.alive, count, 1);
    this.centres = this.collectCentres();
    this.alpha = this.collectAlpha();
    // Only the new tail needs computing; what was there is copied over.
    const covA = new Float32Array(this.nTotal * 4), covB = new Float32Array(this.nTotal * 2);
    covA.set(this.covA.subarray(0, Math.min(this.covA.length, covA.length)));
    covB.set(this.covB.subarray(0, Math.min(this.covB.length, covB.length)));
    this.covA = covA; this.covB = covB;
    this.fillCovariance(first, this.nGenerated, this.nBase);
  }

  /** Drop the last `count` generated splats (undo of a repair). */
  truncateGenerated(count: number) {
    if (count <= 0) return;
    const keep = Math.max(0, this.nGenerated - count);
    this.generated = this.generated.map(a => a.slice(0, keep));
    this.nGenerated = keep;
    this.alive = this.alive.slice(0, this.nTotal);
    this.centres = this.collectCentres();
    this.alpha = this.collectAlpha();
    this.covA = this.covA.slice(0, this.nTotal * 4);
    this.covB = this.covB.slice(0, this.nTotal * 2);
  }

  /**
   * A GSplatData over the generated splats only.
   *
   * The loaded file is never rebuilt -- its 130 MB sit on the GPU untouched for
   * the whole session, and deleting from it is a flag in a texture. Only this
   * much smaller set has to be re-uploaded, and only when a repair adds to it.
   */
  buildGenerated(): any | null {
    if (this.nGenerated === 0) return null;
    const n = this.nGenerated;
    const properties = this.names.map((name, p) => ({
      type: 'float', name, storage: this.generated[p].slice(0, n), byteSize: 4
    }));
    return new (pc as any).GSplatData(
      [{ name: 'vertex', count: n, properties }], []);
  }

  /** Deleted flags for the loaded file. */
  baseDeleted(): Uint32Array<ArrayBufferLike> {
    const st = new Uint32Array(this.nBase);
    for (let i = 0; i < this.nBase; i++) st[i] = this.alive[i] ? 0 : 1;
    return st;
  }

  /** Deleted flags for splats a repair generated. */
  genDeleted(): Uint32Array<ArrayBufferLike> {
    const st = new Uint32Array(this.nGenerated);
    for (let i = 0; i < this.nGenerated; i++) st[i] = this.alive[this.nBase + i] ? 0 : 1;
    return st;
  }

  /** Selected flags for the loaded file, from a whole-document mask. */
  baseSelectedMask(mask: Uint8Array<ArrayBufferLike>): Uint32Array<ArrayBufferLike> {
    const st = new Uint32Array(this.nBase);
    const n = Math.min(this.nBase, mask.length);
    for (let i = 0; i < n; i++) st[i] = mask[i] && this.alive[i] ? 1 : 0;
    return st;
  }

  /** The same for generated splats. */
  genSelectedMask(mask: Uint8Array<ArrayBufferLike>): Uint32Array<ArrayBufferLike> {
    const st = new Uint32Array(this.nGenerated);
    for (let i = 0; i < this.nGenerated; i++) {
      const g = this.nBase + i;
      st[i] = g < mask.length && mask[g] && this.alive[g] ? 1 : 0;
    }
    return st;
  }

  baseSelected(sel?: Uint32Array<ArrayBufferLike>): Uint32Array<ArrayBufferLike> {
    const st = new Uint32Array(this.nBase);
    if (sel) {
      for (let k = 0; k < sel.length; k++) {
        const i = sel[k];
        if (i < this.nBase && this.alive[i]) st[i] = 1;
      }
    }
    return st;
  }

  genSelected(sel?: Uint32Array<ArrayBufferLike>): Uint32Array<ArrayBufferLike> {
    const st = new Uint32Array(this.nGenerated);
    if (sel) {
      for (let k = 0; k < sel.length; k++) {
        const i = sel[k] - this.nBase;
        if (i >= 0 && i < this.nGenerated && this.alive[sel[k]]) st[i] = 1;
      }
    }
    return st;
  }
}

/** Median opacity, from a strided sample, clamped to [0.1, 0.25]. */
function surfaceAlphaOf(alpha: F32): number {
  const step = Math.max(1, Math.floor(alpha.length / 100000));
  const s: number[] = [];
  for (let i = 0; i < alpha.length; i += step) s.push(alpha[i]);
  if (!s.length) return 0.25;
  s.sort((p, q) => p - q);
  return Math.min(0.25, Math.max(0.1, s[s.length >> 1]));
}

function growUint8(a: U8, by: number, fill: number) {
  const out = new Uint8Array(a.length + by);
  out.set(a, 0);
  if (fill) out.fill(fill, a.length);
  return out;
}
