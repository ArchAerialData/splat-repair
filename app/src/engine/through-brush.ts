import type { StrokeSample } from './selection';

/** Rasterise a continuous screen-space stroke, independent of surface depth.
 * Interpolated radii preserve coverage when brush size changes mid-stroke.
 */
export function throughBrushMask(stroke: StrokeSample[], width: number, height: number): Uint8Array {
  const mask = new Uint8Array(width * height);
  appendStroke(mask, stroke, width, height, 0);
  return mask;
}

/** Reuse the raster during a drag instead of painting the entire stroke again. */
export class ThroughBrushRaster {
  private stroke: StrokeSample[] | null = null;
  private count = 0;
  private width = 0;
  private height = 0;
  private mask = new Uint8Array(0);

  update(stroke: StrokeSample[], width: number, height: number): Uint8Array {
    if (this.stroke !== stroke || this.width !== width || this.height !== height || stroke.length < this.count) {
      this.mask = new Uint8Array(width * height);
      this.count = 0;
      this.stroke = stroke;
      this.width = width;
      this.height = height;
    }
    appendStroke(this.mask, stroke, width, height, this.count);
    this.count = stroke.length;
    return this.mask;
  }
}

function appendStroke(mask: Uint8Array, stroke: StrokeSample[], width: number, height: number, first: number) {
  for (let i = first; i < stroke.length; i++) {
    const a = stroke[Math.max(0, i - 1)], b = stroke[i];
    const dx = b.x - a.x, dy = b.y - a.y, length2 = dx * dx + dy * dy;
    const radius = Math.max(a.r, b.r);
    const x0 = Math.max(0, Math.floor(Math.min(a.x, b.x) - radius));
    const x1 = Math.min(width - 1, Math.ceil(Math.max(a.x, b.x) + radius));
    const y0 = Math.max(0, Math.floor(Math.min(a.y, b.y) - radius));
    const y1 = Math.min(height - 1, Math.ceil(Math.max(a.y, b.y) + radius));
    for (let y = y0; y <= y1; y++) for (let x = x0; x <= x1; x++) {
      const t = length2 > 0 ? Math.max(0, Math.min(1, ((x - a.x) * dx + (y - a.y) * dy) / length2)) : 0;
      const r = a.r + (b.r - a.r) * t;
      const u = x - a.x - t * dx, v = y - a.y - t * dy;
      if (u * u + v * v <= r * r) mask[y * width + x] = 1;
    }
  }
}
