/**
 * What the navigation needs to know about the scene: how far away the surface
 * under the cursor is, and where the bulk of the splats sits.
 *
 * Orbiting around the point under the cursor, zooming towards it and panning
 * so the grabbed point sticks to the hand all need one depth per gesture or
 * wheel notch. The brush's depth map would answer that, but it is rebuilt
 * from every splat after each camera move -- 15 to 50 ms, per wheel notch.
 * Here a strided sample of at most a quarter of a million centres is tested
 * against a small window round the cursor: about a millisecond, and plenty
 * to find a surface a hand can aim at.
 *
 * Framing uses percentiles instead of the bounding box. A handful of floaters
 * far out set the box on most real scans (the castle's box is several times
 * the castle), so "frame all" framed empty space with the scene a dot in it.
 */

export interface NavData {
  /** Splat centres, xyz per splat, in file coordinates. */
  centres(): Float32Array<ArrayBufferLike>;
  /** 1 = on screen (alive and not cropped away); null = all. */
  visible(): Uint8Array<ArrayBufferLike> | null;
  /** Opacity per splat, or null. */
  alpha(): Float32Array<ArrayBufferLike> | null;
  /** Opacity a splat needs to count as surface. */
  minAlpha(): number;
}

/** Camera basis in world space, plus what turns depth into pixels. */
export interface PickCamera {
  eye: [number, number, number];
  right: [number, number, number];
  up: [number, number, number];
  fwd: [number, number, number];
  /** tan(fov / 2), vertical. */
  tanHalf: number;
  width: number;
  height: number;
}

const MAX_SAMPLES = 250_000;
let scratch = new Float32Array(4096);

/**
 * Camera-space depth of the surface under a viewport pixel, or null when the
 * cursor is over empty space.
 *
 * The 5th percentile of the opaque centres that land in the window, not the
 * nearest one: the window also catches the far side of the object and the
 * odd floater in front, and the nearest single splat is usually one of those.
 */
export function pickDepth(data: NavData, cam: PickCamera, x: number, y: number): number | null {
  const c = data.centres();
  const n = (c.length / 3) | 0;
  if (!n) return null;
  const vis = data.visible();
  const alpha = data.alpha();
  const minA = data.minAlpha();
  const step = Math.max(1, Math.floor(n / MAX_SAMPLES));
  const [ex, ey, ez] = cam.eye;
  const [rx, ry, rz] = cam.right, [ux, uy, uz] = cam.up, [fx, fy, fz] = cam.fwd;
  const aspect = cam.width / Math.max(cam.height, 1);
  // pixel -> normalised device coordinates at unit depth
  const nx = (2 * x / Math.max(cam.width, 1) - 1) * cam.tanHalf * aspect;
  const ny = (1 - 2 * y / Math.max(cam.height, 1)) * cam.tanHalf;
  const pxPerUnit = cam.height / (2 * cam.tanHalf);   // pixels per unit at depth 1

  // Widen the window until enough splats land in it: a tight window on a
  // dense surface, a wider one on a sparse scan or a thin edge.
  for (const radiusPx of [5, 14, 36]) {
    const tol = radiusPx / pxPerUnit;      // at unit depth
    let hits = 0;
    for (let i = 0; i < n; i += step) {
      if (vis && !vis[i]) continue;
      if (alpha && alpha[i] < minA) continue;
      const j = i * 3;
      const dx = c[j] - ex, dy = c[j + 1] - ey, dz = c[j + 2] - ez;
      const d = dx * fx + dy * fy + dz * fz;
      if (d <= 1e-6) continue;
      const sx = (dx * rx + dy * ry + dz * rz) / d - nx;
      if (sx > tol || sx < -tol) continue;
      const sy = (dx * ux + dy * uy + dz * uz) / d - ny;
      if (sy > tol || sy < -tol) continue;
      if (hits >= scratch.length) {
        const grown = new Float32Array(scratch.length * 2);
        grown.set(scratch);
        scratch = grown;
      }
      scratch[hits++] = d;
    }
    if (hits >= 3 || (radiusPx === 36 && hits > 0)) {
      const a = scratch.subarray(0, hits).sort();
      return a[Math.min(hits - 1, Math.max(1, Math.floor(hits * 0.05)))];
    }
  }
  return null;
}

export interface Bounds { centre: [number, number, number]; radius: number; count: number; }

/**
 * Where most of a set of splats sits: the box between the low and high
 * percentile on each axis, as a centre and the radius of its sphere.
 *
 * `mask` limits it to some splats (the selection); `visible` drops deleted and
 * cropped ones. Small sets use their full extent -- with twenty splats the
 * percentiles would throw away the ones the user meant.
 */
export function robustBounds(centres: Float32Array<ArrayBufferLike>,
                             visible: Uint8Array<ArrayBufferLike> | null,
                             mask: Uint8Array<ArrayBufferLike> | null = null,
                             lo = 0.02, hi = 0.98): Bounds | null {
  const n = (centres.length / 3) | 0;
  let count = 0;
  for (let i = 0; i < n; i++) {
    if (visible && !visible[i]) continue;
    if (mask && !mask[i]) continue;
    count++;
  }
  if (!count) return null;
  const step = Math.max(1, Math.floor(count / 200_000));
  const xs: number[] = [], ys: number[] = [], zs: number[] = [];
  for (let i = 0, k = 0; i < n; i++) {
    if (visible && !visible[i]) continue;
    if (mask && !mask[i]) continue;
    if (k++ % step) continue;
    xs.push(centres[i * 3]); ys.push(centres[i * 3 + 1]); zs.push(centres[i * 3 + 2]);
  }
  const small = xs.length < 200;
  const range = (a: number[]): [number, number] => {
    a.sort((p, q) => p - q);
    if (small) return [a[0], a[a.length - 1]];
    return [a[Math.floor(lo * (a.length - 1))], a[Math.ceil(hi * (a.length - 1))]];
  };
  const [x0, x1] = range(xs), [y0, y1] = range(ys), [z0, z1] = range(zs);
  const radius = Math.max(0.5 * Math.hypot(x1 - x0, y1 - y0, z1 - z0), 1e-3);
  return { centre: [(x0 + x1) / 2, (y0 + y1) / 2, (z0 + z1) / 2], radius, count };
}
