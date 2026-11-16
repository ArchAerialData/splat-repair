/**
 * The view cube in the corner of the viewport.
 *
 * A face looks straight at that side (and goes orthographic, the way
 * Blender's numpad does), an edge or a corner looks from in between, and the
 * coloured axis ends look along the file's own X, Y and Z. Dragging the cube
 * turns the view about its centre.
 *
 * The cube is laid out in the viewer's up frame, so "Top" is whatever the
 * up-axis toggle says is up; the axis ends are the file's axes, which is what
 * the crop sliders and the engine work in.
 */
import * as pc from 'playcanvas';
import type { Viewer } from '../engine/viewer';

type V3 = [number, number, number];

interface Face { n: V3; u: V3; v: V3; name: string; }

/** Up frame: +x right, +y up, +z front (the side the default view looks at). */
const FACES: Face[] = [
  { n: [0, 0, 1], u: [1, 0, 0], v: [0, 1, 0], name: 'Front' },
  { n: [0, 0, -1], u: [-1, 0, 0], v: [0, 1, 0], name: 'Back' },
  { n: [1, 0, 0], u: [0, 0, -1], v: [0, 1, 0], name: 'Right' },
  { n: [-1, 0, 0], u: [0, 0, 1], v: [0, 1, 0], name: 'Left' },
  { n: [0, 1, 0], u: [1, 0, 0], v: [0, 0, -1], name: 'Top' },
  { n: [0, -1, 0], u: [1, 0, 0], v: [0, 0, 1], name: 'Bottom' }
];

const AXES: Array<{ w: V3; name: string; colour: string }> = [
  { w: [1, 0, 0], name: 'X', colour: '#d0544d' },
  { w: [0, 1, 0], name: 'Y', colour: '#9dbb9a' },
  { w: [0, 0, 1], name: 'Z', colour: '#7fb3be' }
];

/** How far out the axis ends sit, in cube half-sizes. */
const AXIS_LEN = 1.7;

/** Words for a view direction in the up frame, e.g. "Front right top". */
export function viewName(d: V3): string {
  const parts: string[] = [];
  if (d[2] > 0.3) parts.push('front'); else if (d[2] < -0.3) parts.push('back');
  if (d[0] > 0.3) parts.push('right'); else if (d[0] < -0.3) parts.push('left');
  if (d[1] > 0.3) parts.push('top'); else if (d[1] < -0.3) parts.push('bottom');
  const s = parts.join(' ');
  return s ? s[0].toUpperCase() + s.slice(1) : 'View';
}

interface Hit { dir: V3; label: string; poly?: Array<[number, number]>; axis?: number; neg?: boolean; }
interface AxisEnd { a: typeof AXES[number]; i: number; neg: boolean; d: V3; p: [number, number, number]; shown: boolean; }

export class ViewCube {
  private ctx: CanvasRenderingContext2D;
  private size = 0;
  private scale = 0;
  private hover: Hit | null = null;
  private lastKey = '';
  private drag: { x: number; y: number; moved: boolean; id: number } | null = null;
  /** Called after a click snapped the view, with the direction's name. */
  onSnap: ((name: string, ortho: boolean) => void) | null = null;

  constructor(private canvas: HTMLCanvasElement, private viewer: Viewer) {
    this.ctx = canvas.getContext('2d')!;
    viewer.addFrameListener(() => this.draw());
    canvas.addEventListener('pointermove', (e) => this.onMove(e));
    canvas.addEventListener('pointerleave', () => { if (!this.drag) { this.hover = null; this.draw(true); } });
    canvas.addEventListener('pointerdown', (e) => {
      if (e.button !== 0) return;
      this.drag = { x: e.clientX, y: e.clientY, moved: false, id: e.pointerId };
      try { canvas.setPointerCapture(e.pointerId); } catch { /* synthetic pointer */ }
      e.preventDefault();
    });
    const end = (e: PointerEvent) => {
      const d = this.drag;
      this.drag = null;
      try { canvas.releasePointerCapture(e.pointerId); } catch { /* released */ }
      if (!d || d.moved) return;
      const h = this.hitAt(e.clientX, e.clientY);
      if (h) this.snap(h);
    };
    canvas.addEventListener('pointerup', end);
    canvas.addEventListener('pointercancel', () => { this.drag = null; });
    canvas.addEventListener('contextmenu', (e) => e.preventDefault());
    document.fonts?.ready.then(() => this.draw(true));
  }

  /* -------------------------------------------------------------- math */

  /** Up-frame vector to cube-canvas pixels and depth (towards the eye > 0). */
  private toScreen(v: V3, rot: pc.Quat): [number, number, number] {
    const c = rot.transformVector(new pc.Vec3(v[0], v[1], v[2]));
    const h = this.size / 2;
    return [h + c.x * this.scale, h - c.y * this.scale, c.z];
  }

  /** The camera's inverse turn: up frame to camera space. */
  private camRot(): pc.Quat {
    return this.viewer.localRotation().invert();
  }

  /** A file axis in the up frame. */
  private axisLocal(w: V3): V3 {
    const inv = this.viewer.upRotation().invert();
    const v = inv.transformVector(new pc.Vec3(w[0], w[1], w[2]));
    return [v.x, v.y, v.z];
  }

  /**
   * The six axis ends. An end that points at or away from the eye lands on
   * the cube's face and would cover its name; it is left out -- that face
   * looks along the same axis anyway.
   */
  private axisEnds(rot: pc.Quat): AxisEnd[] {
    const o = this.toScreen([0, 0, 0], rot);
    return AXES.flatMap((a, i) => [false, true].map(neg => {
      const l = this.axisLocal(a.w);
      const d: V3 = neg ? [-l[0], -l[1], -l[2]] : l;
      const p = this.toScreen([d[0] * AXIS_LEN, d[1] * AXIS_LEN, d[2] * AXIS_LEN], rot);
      const shown = Math.hypot(p[0] - o[0], p[1] - o[1]) > this.scale * 1.15;
      return { a, i, neg, d, p, shown };
    }));
  }

  private faceRegion(f: Face, a0: number, a1: number, b0: number, b1: number,
                     rot: pc.Quat): Array<[number, number]> {
    const pt = (a: number, b: number): [number, number] => {
      const p = this.toScreen([f.n[0] + a * f.u[0] + b * f.v[0],
                               f.n[1] + a * f.u[1] + b * f.v[1],
                               f.n[2] + a * f.u[2] + b * f.v[2]], rot);
      return [p[0], p[1]];
    };
    return [pt(a0, b0), pt(a1, b0), pt(a1, b1), pt(a0, b1)];
  }

  private hitAt(clientX: number, clientY: number): Hit | null {
    const r = this.canvas.getBoundingClientRect();
    const x = clientX - r.left, y = clientY - r.top;
    const rot = this.camRot();
    // axis ends first: they sit outside the cube and on top of it
    let best: Hit | null = null, bestZ = -Infinity;
    for (const e of this.axisEnds(rot)) {
      if (!e.shown) continue;
      if (Math.hypot(e.p[0] - x, e.p[1] - y) <= 9 && e.p[2] > bestZ) {
        bestZ = e.p[2];
        best = { dir: e.d, label: (e.neg ? '−' : '+') + e.a.name, axis: e.i, neg: e.neg };
      }
    }
    if (best) return best;
    for (const f of FACES) {
      const nz = this.toScreen(f.n, rot)[2] - this.toScreen([0, 0, 0], rot)[2];
      if (nz <= 1e-3) continue;
      // solve screen = c + a*U + b*V for (a, b)
      const c = this.toScreen(f.n, rot);
      const pu = this.toScreen([f.n[0] + f.u[0], f.n[1] + f.u[1], f.n[2] + f.u[2]], rot);
      const pv = this.toScreen([f.n[0] + f.v[0], f.n[1] + f.v[1], f.n[2] + f.v[2]], rot);
      const ux = pu[0] - c[0], uy = pu[1] - c[1], vx = pv[0] - c[0], vy = pv[1] - c[1];
      const det = ux * vy - uy * vx;
      if (Math.abs(det) < 1e-6) continue;
      const qx = x - c[0], qy = y - c[1];
      const a = (qx * vy - qy * vx) / det, b = (ux * qy - uy * qx) / det;
      if (Math.abs(a) > 1 || Math.abs(b) > 1) continue;
      const ia = a < -1 / 3 ? -1 : a > 1 / 3 ? 1 : 0;
      const ib = b < -1 / 3 ? -1 : b > 1 / 3 ? 1 : 0;
      const seg = (i: number): [number, number] => i < 0 ? [-1, -1 / 3] : i > 0 ? [1 / 3, 1] : [-1 / 3, 1 / 3];
      const [a0, a1] = seg(ia), [b0, b1] = seg(ib);
      const dir: V3 = [f.n[0] + ia * f.u[0] + ib * f.v[0], f.n[1] + ia * f.u[1] + ib * f.v[1],
                       f.n[2] + ia * f.u[2] + ib * f.v[2]];
      return { dir, label: viewName(dir), poly: this.faceRegion(f, a0, a1, b0, b1, rot) };
    }
    return null;
  }

  private snap(h: Hit) {
    const ortho = h.axis !== undefined || h.dir.filter(v => Math.abs(v) > 1e-6).length === 1;
    this.viewer.snapTo(h.dir, ortho);
    this.onSnap?.(h.axis !== undefined ? `Along ${h.label}` : h.label, ortho);
  }

  /** Where a point of the cube (in the up frame) is on the page, for tests. */
  screenPointOf(dir: V3): { x: number; y: number } {
    const p = this.toScreen(dir, this.camRot());
    const r = this.canvas.getBoundingClientRect();
    return { x: r.left + p[0], y: r.top + p[1] };
  }

  /* ------------------------------------------------------------- input */

  private onMove(e: PointerEvent) {
    if (this.drag) {
      const dx = e.clientX - this.drag.x, dy = e.clientY - this.drag.y;
      if (!this.drag.moved && Math.hypot(dx, dy) < 3) return;
      this.drag.moved = true;
      this.drag.x = e.clientX; this.drag.y = e.clientY;
      this.viewer.orbitDrag(dx, dy);
      return;
    }
    const h = this.hitAt(e.clientX, e.clientY);
    const key = h ? h.label + (h.axis ?? '') : '';
    const old = this.hover ? this.hover.label + (this.hover.axis ?? '') : '';
    this.hover = h;
    this.canvas.title = h ? (h.axis !== undefined ? `Look along ${h.label}` : `${h.label} view`)
      + ' · drag to rotate' : 'Click: pick a view · drag: rotate';
    if (key !== old) this.draw(true);
  }

  /* -------------------------------------------------------------- draw */

  draw(force = false) {
    const r = this.canvas.getBoundingClientRect();
    if (!r.width) return;
    const dpr = Math.min(window.devicePixelRatio || 1, 2);
    const q = this.viewer.localRotation();
    const key = `${q.x.toFixed(5)},${q.y.toFixed(5)},${q.z.toFixed(5)},${q.w.toFixed(5)},${this.viewer.upAxis},${r.width},${dpr}`;
    if (!force && key === this.lastKey) return;
    this.lastKey = key;
    const w = Math.round(r.width * dpr), hgt = Math.round(r.height * dpr);
    if (this.canvas.width !== w || this.canvas.height !== hgt) {
      this.canvas.width = w; this.canvas.height = hgt;
    }
    this.size = r.width;
    this.scale = r.width * 0.25;
    const ctx = this.ctx;
    ctx.setTransform(dpr, 0, 0, dpr, 0, 0);
    ctx.clearRect(0, 0, r.width, r.height);
    const rot = this.camRot();
    const o = this.toScreen([0, 0, 0], rot);
    const centreZ = o[2];
    const ends = this.axisEnds(rot);

    const drawAxis = (it: AxisEnd) => {
      if (!it.shown) return;
      ctx.strokeStyle = it.a.colour;
      ctx.globalAlpha = it.neg ? 0.45 : 0.9;
      ctx.lineWidth = 1.5;
      // only the part outside the cube, so no line runs across a face's name
      const t0 = 1.05 / AXIS_LEN;
      ctx.beginPath();
      ctx.moveTo(o[0] + (it.p[0] - o[0]) * t0, o[1] + (it.p[1] - o[1]) * t0);
      ctx.lineTo(it.p[0], it.p[1]); ctx.stroke();
      ctx.globalAlpha = 1;
      const hot = this.hover?.axis === it.i && this.hover?.neg === it.neg;
      ctx.beginPath();
      ctx.arc(it.p[0], it.p[1], it.neg ? 5 : 7.5, 0, Math.PI * 2);
      if (it.neg) {
        ctx.fillStyle = hot ? it.a.colour : '#100c0a';
        ctx.fill();
        ctx.strokeStyle = it.a.colour; ctx.stroke();
      } else {
        ctx.fillStyle = it.a.colour; ctx.fill();
        if (hot) { ctx.strokeStyle = '#f1e9dc'; ctx.lineWidth = 1.5; ctx.stroke(); }
        ctx.fillStyle = '#100c0a';
        ctx.font = '700 9.5px Prompt, Ubuntu, sans-serif';
        ctx.textAlign = 'center'; ctx.textBaseline = 'middle';
        ctx.fillText(it.a.name, it.p[0], it.p[1] + 0.5);
      }
    };
    for (const it of ends) if (it.p[2] < centreZ) drawAxis(it);

    // faces facing the eye (a convex cube: they never overlap)
    for (const f of FACES) {
      const c = this.toScreen(f.n, rot);
      const facing = c[2] - centreZ;
      if (facing <= 1e-3) continue;
      const poly = this.faceRegion(f, -1, 1, -1, 1, rot);
      ctx.beginPath();
      poly.forEach((p, k) => k ? ctx.lineTo(p[0], p[1]) : ctx.moveTo(p[0], p[1]));
      ctx.closePath();
      // lit from the eye: faces turned away a little darker
      const l = 0.55 + 0.45 * facing;
      ctx.fillStyle = `rgba(${Math.round(37 * l + 8)}, ${Math.round(28 * l + 6)}, ${Math.round(23 * l + 5)}, 0.96)`;
      ctx.fill();
      ctx.strokeStyle = '#4a3b31'; ctx.lineWidth = 1; ctx.stroke();
      if (facing > 0.3) {
        // The name lies in the face, turned and squeezed with it: written
        // flat on the screen it ran past a face seen at an angle and under
        // the axis end beside it ("Righ" with the X over the t).
        const pu = this.toScreen([f.n[0] + f.u[0], f.n[1] + f.u[1], f.n[2] + f.u[2]], rot);
        const pv = this.toScreen([f.n[0] + f.v[0], f.n[1] + f.v[1], f.n[2] + f.v[2]], rot);
        const k = 1 / this.scale;
        ctx.save();
        ctx.setTransform(dpr * (pu[0] - c[0]) * k, dpr * (pu[1] - c[1]) * k,
                         -dpr * (pv[0] - c[0]) * k, -dpr * (pv[1] - c[1]) * k,
                         dpr * c[0], dpr * c[1]);
        ctx.globalAlpha = Math.min(1, (facing - 0.3) / 0.4);
        ctx.fillStyle = '#e9dfd0';
        ctx.font = '500 11px Prompt, Ubuntu, sans-serif';
        ctx.textAlign = 'center'; ctx.textBaseline = 'middle';
        ctx.fillText(f.name, 0, 0, this.scale * 1.8);
        ctx.restore();
      }
    }
    if (this.hover?.poly) {
      ctx.beginPath();
      this.hover.poly.forEach((p, k) => k ? ctx.lineTo(p[0], p[1]) : ctx.moveTo(p[0], p[1]));
      ctx.closePath();
      ctx.fillStyle = 'rgba(229, 162, 98, 0.42)';
      ctx.fill();
    }
    for (const it of ends) if (it.p[2] >= centreZ) drawAxis(it);
  }
}
