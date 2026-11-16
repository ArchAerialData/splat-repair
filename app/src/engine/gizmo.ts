/**
 * Transform handles for the selection.
 *
 * The gizmo drags a proxy entity; the splats never move while you drag. The
 * delta between where the proxy started and where it is now goes into four
 * shader uniforms, and the splat shader applies it to anything flagged as
 * selected. Letting go sends that one delta to the engine, which does the real
 * transform -- including the spherical harmonics, which the preview skips.
 *
 * Doing it this way means dragging costs four uniforms a frame instead of
 * rewriting half a million positions, and abandoning a drag costs nothing.
 */
import * as pc from 'playcanvas';
import {
  TranslateGizmo, RotateGizmo, ScaleGizmo
} from 'playcanvas/build/playcanvas/src/extras/index.js';

export type GizmoMode = 'translate' | 'rotate' | 'scale';

export interface GizmoDelta {
  quat: [number, number, number, number];   // (x, y, z, w)
  translation: [number, number, number];
  pivot: [number, number, number];
  scale: number;
}

const IDENTITY: GizmoDelta = {
  quat: [0, 0, 0, 1], translation: [0, 0, 0], pivot: [0, 0, 0], scale: 1
};

export class SelectionGizmo {
  private layer: pc.Layer;
  private proxy: pc.Entity;
  private current: any = null;
  private mode: GizmoMode = 'translate';
  private start = { pos: new pc.Vec3(), rot: new pc.Quat(), scale: new pc.Vec3(1, 1, 1) };
  private pivot: [number, number, number] = [0, 0, 0];
  private dragging = false;

  onPreview: ((d: GizmoDelta) => void) | null = null;
  onCommit: ((d: GizmoDelta) => void) | null = null;

  /**
   * `upRotation` is the viewer's up frame. The handles are laid out in it, so
   * the green one points up on screen on a y-down scan too; the delta sent to
   * the engine is still in file coordinates, since it is taken from the
   * proxy's world transform.
   */
  constructor(private app: pc.AppBase, private camera: pc.Entity,
              private upRotation: () => pc.Quat = () => new pc.Quat()) {
    this.layer = new pc.Layer({ name: 'gizmo', clearDepthBuffer: true });
    app.scene.layers.push(this.layer);
    this.proxy = new pc.Entity('gizmo-proxy');
    app.root.addChild(this.proxy);
  }

  get active() { return this.current !== null; }
  get isDragging() { return this.dragging; }

  /** Show handles of the given kind at a pivot, or hide them with null. */
  show(mode: GizmoMode | null, pivot?: [number, number, number]) {
    if (this.current) {
      this.current.detach();
      this.current.destroy();
      this.current = null;
    }
    if (!mode) return;
    this.mode = mode;
    if (pivot) this.pivot = pivot;
    this.proxy.setPosition(this.pivot[0], this.pivot[1], this.pivot[2]);
    this.proxy.setLocalScale(1, 1, 1);
    this.proxy.setRotation(this.upRotation());

    const cam = this.camera.camera as pc.CameraComponent;
    const G = mode === 'translate' ? TranslateGizmo
      : mode === 'rotate' ? RotateGizmo : ScaleGizmo;
    const g = new (G as any)(cam, this.layer);
    g.coordSpace = 'local';
    g.attach([this.proxy]);
    this.current = g;

    g.on('transform:start', () => {
      this.dragging = true;
      this.start.pos.copy(this.proxy.getPosition());
      this.start.rot.copy(this.proxy.getRotation());
      this.start.scale.copy(this.proxy.getLocalScale());
    });
    g.on('transform:move', () => this.onPreview?.(this.delta()));
    g.on('transform:end', () => {
      this.dragging = false;
      const d = this.delta();
      const moved = d.translation.some(v => Math.abs(v) > 1e-7)
        || Math.abs(d.quat[3] - 1) > 1e-7 || Math.abs(d.scale - 1) > 1e-6;
      // the proxy goes back to neutral; the splats now carry the change
      this.show(this.mode, this.pivot);
      // After `show`, so the app has the last word on the handles: a drag
      // that ends with nothing selected any more takes them away, and they
      // used to come straight back.
      if (moved) this.onCommit?.(d);
      else this.onPreview?.(IDENTITY);
    });
  }

  /** Where the handles sit. Call when the selection changes. */
  setPivot(pivot: [number, number, number]) {
    this.pivot = pivot;
    if (this.current) this.show(this.mode, pivot);
  }

  private delta(): GizmoDelta {
    const p = this.proxy.getPosition();
    const r = this.proxy.getRotation();
    const s = this.proxy.getLocalScale();
    // Rotation relative to where the drag began.
    const inv = new pc.Quat().copy(this.start.rot).invert();
    const dq = new pc.Quat().mul2(r, inv);
    const ds = this.start.scale.x !== 0 ? s.x / this.start.scale.x : 1;
    return {
      quat: [dq.x, dq.y, dq.z, dq.w],
      translation: [p.x - this.start.pos.x, p.y - this.start.pos.y,
                    p.z - this.start.pos.z],
      pivot: [this.start.pos.x, this.start.pos.y, this.start.pos.z],
      scale: ds
    };
  }

  destroy() {
    this.show(null);
    this.proxy.destroy();
    const i = this.app.scene.layers.layerList.indexOf(this.layer);
    if (i >= 0) this.app.scene.layers.layerList.splice(i, 1);
  }
}

/** The engine wants (w, x, y, z); the shader and PlayCanvas use (x, y, z, w). */
export function toEngineQuat(q: [number, number, number, number]):
    [number, number, number, number] {
  return [q[3], q[0], q[1], q[2]];
}
