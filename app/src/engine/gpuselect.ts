/**
 * The capsule test, run on the GPU while painting.
 *
 * On the CPU the test costs about 9 ms at 550k splats -- inside a frame, but
 * most of one, and it grows linearly: the same stroke on a 5M-splat scene would
 * cost 80 ms and the brush would stop following the cursor. The test is
 * embarrassingly parallel, so it belongs on the card.
 *
 * This writes the `editSel` stream directly, which the splat shader already
 * reads to tint the selection. Nothing comes back across the bus: the preview
 * never needs indices on the CPU, and the authoritative selection is computed
 * once when the stroke ends, where 9 ms does not matter.
 */
import * as pc from 'playcanvas';
import { SIZE_K, SIZE_CAP, type CapsulePoint } from './selection';

// One row of RGBA32F, two texels per capsule. 192 was not a GPU limit, it
// was a guess, and at 144 Hz a stroke passed it in a second and a third.
const MAX_SEGMENTS = 1024;

const PROCESS_WGSL = `
uniform uCapCount: u32;
uniform uViewDir: vec3f;
uniform uInvSquash: f32;
uniform uMode: u32;        // 0 replace, 1 add, 2 subtract
uniform uSizeK: f32;       // sigmas a splat reaches along the ray, 0 = points
uniform uSizeCap: f32;     // most it may reach, in brush radii
var uCaps: texture_2d<f32>;

fn process() {
    let c = getCenter();
    let base = loadEditBase().r;
    var hit: u32 = 0u;
    if (loadEditDel().r == 0u) {
        // The splat's own extent along the ray (selection.ts, SizeReach).
        let vd = uniform.uViewDir;
        let ca = bitcast<vec4f>(loadEditCovA());
        let cb = bitcast<vec4f>(loadEditCovB());
        let var_ = ca.x * vd.x * vd.x + ca.w * vd.y * vd.y + cb.y * vd.z * vd.z
            + 2.0 * (ca.y * vd.x * vd.y + ca.z * vd.x * vd.z + cb.x * vd.y * vd.z);
        // k = 0 is points: skip the product, 0 * an overflowed sigma is NaN
        var reach: f32 = 0.0;
        if (uniform.uSizeK > 0.0) {
            reach = uniform.uSizeK * sqrt(max(var_, 0.0));
        }
        for (var k: u32 = 0u; k < uniform.uCapCount; k = k + 1u) {
            let a = textureLoad(uCaps, vec2i(i32(k * 2u), 0), 0);
            let b = textureLoad(uCaps, vec2i(i32(k * 2u + 1u), 0), 0);
            let r = max(a.w, b.w);
            // Flatten towards the camera so a wide brush takes a shell of
            // surface rather than a ball of everything behind it.
            let along = dot(c - a.xyz, vd);
            let give = min(reach, uniform.uSizeCap * r);
            var shrunk: f32 = min(along + give, 0.0);
            if (along > 0.0) {
                shrunk = max(along - give, 0.0);
            }
            // the point and the segment both stretched along the view
            // (selection.ts, selectInRows)
            let p = c + vd * (shrunk * uniform.uInvSquash - along);
            let u = b.xyz - a.xyz;
            let v = u + vd * (dot(u, vd) * (uniform.uInvSquash - 1.0));
            let w = p - a.xyz;
            let vv = dot(v, v);
            var t: f32 = 0.0;
            if (vv > 1e-12) {
                t = clamp(dot(w, v) / vv, 0.0, 1.0);
            }
            let d = w - v * t;
            if (dot(d, d) <= r * r) {
                hit = 1u;
                break;
            }
        }
    }
    var result: u32 = hit;
    if (uniform.uMode == 1u) {
        result = select(hit, 1u, base != 0u);
    } else if (uniform.uMode == 2u) {
        result = select(base, 0u, hit != 0u);
    }
    writeEditSel(vec4u(result, 0u, 0u, 0u));
}
`;

const PROCESS_GLSL = `
uniform uint uCapCount;
uniform vec3 uViewDir;
uniform float uInvSquash;
uniform uint uMode;
uniform float uSizeK;
uniform float uSizeCap;
uniform highp sampler2D uCaps;

void process() {
    vec3 c = getCenter();
    uint base = loadEditBase().r;
    uint hit = 0u;
    if (loadEditDel().r == 0u) {
        vec3 vd = uViewDir;
        vec4 ca = uintBitsToFloat(loadEditCovA());
        vec4 cb = uintBitsToFloat(loadEditCovB());
        float var_ = ca.x * vd.x * vd.x + ca.w * vd.y * vd.y + cb.y * vd.z * vd.z
            + 2.0 * (ca.y * vd.x * vd.y + ca.z * vd.x * vd.z + cb.x * vd.y * vd.z);
        float reach = uSizeK > 0.0 ? uSizeK * sqrt(max(var_, 0.0)) : 0.0;
        for (uint k = 0u; k < uCapCount; k++) {
            vec4 a = texelFetch(uCaps, ivec2(int(k * 2u), 0), 0);
            vec4 b = texelFetch(uCaps, ivec2(int(k * 2u + 1u), 0), 0);
            float r = max(a.w, b.w);
            float along = dot(c - a.xyz, vd);
            float give = min(reach, uSizeCap * r);
            float shrunk = along > 0.0 ? max(along - give, 0.0) : min(along + give, 0.0);
            vec3 p = c + vd * (shrunk * uInvSquash - along);
            vec3 u = b.xyz - a.xyz;
            vec3 v = u + vd * (dot(u, vd) * (uInvSquash - 1.0));
            vec3 w = p - a.xyz;
            float vv = dot(v, v);
            float t = vv > 1e-12 ? clamp(dot(w, v) / vv, 0.0, 1.0) : 0.0;
            vec3 d = w - v * t;
            if (dot(d, d) <= r * r) { hit = 1u; break; }
        }
    }
    uint result = hit;
    if (uMode == 1u) { result = (base != 0u) ? 1u : hit; }
    else if (uMode == 2u) { result = (hit != 0u) ? 0u : base; }
    writeEditSel(uvec4(result, 0u, 0u, 0u));
}
`;

export class GpuSelector {
  private proc: any = null;
  private caps: pc.Texture | null = null;
  private data = new Float32Array(MAX_SEGMENTS * 2 * 4);
  ready = false;
  reason = '';

  constructor(device: pc.GraphicsDevice, private entity: pc.Entity) {
    try {
      const P = pc as any;
      this.caps = new pc.Texture(device, {
        name: 'brushCapsules',
        width: MAX_SEGMENTS * 2,
        height: 1,
        format: P.PIXELFORMAT_RGBA32F,
        mipmaps: false,
        minFilter: P.FILTER_NEAREST,
        magFilter: P.FILTER_NEAREST,
        addressU: P.ADDRESS_CLAMP_TO_EDGE,
        addressV: P.ADDRESS_CLAMP_TO_EDGE
      });
      this.proc = new P.GSplatProcessor(
        device,
        { component: (entity as any).gsplat },
        { component: (entity as any).gsplat, streams: ['editSel'] },
        { processWGSL: PROCESS_WGSL, processGLSL: PROCESS_GLSL }
      );
      this.ready = true;
    } catch (e: any) {
      this.reason = String(e?.message ?? e);
      this.ready = false;
    }
  }

  /** Run the test. Returns false if it could not, so the caller can fall back. */
  run(capsules: CapsulePoint[], viewDir: [number, number, number],
      depth: number, mode: 'replace' | 'add' | 'subtract' = 'replace',
      size: { k: number; cap: number } = { k: SIZE_K, cap: SIZE_CAP }): boolean {
    if (!this.ready || !this.proc || !this.caps) return false;
    try {
      // Segments are consecutive pairs; a single sample becomes a degenerate
      // segment, which the distance test already handles. No samples means
      // clear the selection, which is a dispatch with zero segments.
      // A long stroke used to be truncated to its first MAX_SEGMENTS
      // capsules -- and this still reported success, so the preview quietly
      // stopped following the cursor while the stroke kept growing. The tail
      // is exactly where the pointer is, so that is the part that must never
      // be dropped. Thin the whole stroke evenly instead, last sample kept.
      let caps = capsules;
      if (caps.length > MAX_SEGMENTS) {
        const step = (caps.length - 1) / (MAX_SEGMENTS - 1);
        caps = Array.from({ length: MAX_SEGMENTS },
                          (_, k) => capsules[Math.round(k * step)]);
      }
      const n = caps.length;
      const d = this.data;
      d.fill(0);
      for (let k = 0; k < n; k++) {
        const a = caps[k];
        const b = caps[Math.min(k + 1, n - 1)];
        d[k * 8 + 0] = a.x; d[k * 8 + 1] = a.y; d[k * 8 + 2] = a.z; d[k * 8 + 3] = a.r;
        d[k * 8 + 4] = b.x; d[k * 8 + 5] = b.y; d[k * 8 + 6] = b.z; d[k * 8 + 7] = b.r;
      }
      const lvl = this.caps.lock() as Float32Array;
      lvl.set(d.subarray(0, Math.min(d.length, lvl.length)));
      this.caps.unlock();

      this.proc.setParameter('uCaps', this.caps);
      this.proc.setParameter('uCapCount', n);
      this.proc.setParameter('uViewDir', viewDir);
      this.proc.setParameter('uInvSquash', 1.0 / Math.max(depth, 1e-3));
      this.proc.setParameter('uMode', mode === 'add' ? 1 : mode === 'subtract' ? 2 : 0);
      this.proc.setParameter('uSizeK', size.k);
      this.proc.setParameter('uSizeCap', size.cap);
      this.proc.process();
      (this.entity as any).gsplat?._placement?.markDirty?.();
      return true;
    } catch (e: any) {
      this.reason = String(e?.message ?? e);
      this.ready = false;
      return false;
    }
  }

  /** Clear the selection without a stroke. */
  clear(): boolean {
    return this.run([], [0, 0, 1], 1);
  }

  destroy() {
    try { this.proc?.destroy?.(); } catch { /* already gone */ }
    try { this.caps?.destroy?.(); } catch { /* already gone */ }
    this.proc = null;
    this.caps = null;
    this.ready = false;
  }
}
