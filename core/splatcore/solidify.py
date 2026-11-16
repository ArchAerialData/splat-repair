"""The solidify brush: make a surface that shows through as solid as it should be.

Under-trained patches of a scan come out half transparent: the background or
the far wall shows through a surface that is, in reality, solid. Raising the
opacity of one splat does little -- on this shoe the median splat is 0.57
opaque and a column of surface holds eighteen of them, so whether a surface
is solid is a property of rays, not of splats. So the brush works on rays.

* **Measure what each view sees.** Render the painted patch from the user's
  view and from two views tilted 25 degrees off it; the coverage the patch
  should have is the coverage of the surface around it -- never more, so a
  mesh fabric is matched to the surrounding mesh, not sealed.
* **Raise transmittance to a power.** If a ray through the patch keeps a
  fraction ``T`` of the light and should keep ``T_t``, raising every splat's
  ``1 - alpha`` to the power ``log T_t / log T`` gets it there exactly when all
  of them are scaled alike. Every layer of the surface takes its share --
  which layer should carry the opacity is not in the file, so the even
  distribution is the honest one. Each splat takes the strongest exponent any
  view asks of it; one view alone left the tilted ones still transparent.
* **Keep what the user saw.** A half-transparent splat's colour was trained
  composited over whatever was behind it. Made opaque, it shows that colour
  pure, which can be lighter or darker than what was on screen. By default
  the colours are refitted so the patch, now solid, looks as it did -- but
  only where it was mostly solid already; where the background dominated,
  there was nothing of the surface's look to keep.
* **Report what cannot be raised.** A spot with no splats in it cannot be
  made opaque by changing opacities. That is a hole, and the repair brush is
  the tool for it; the status says so instead of inventing material here.
"""
from __future__ import annotations

from dataclasses import dataclass
from typing import Optional

import numpy as np
from scipy import ndimage

from .io import SplatSet
from .repair import RepairError, RepairResult
from .surface import surface_patch


@dataclass
class SolidParams:
    # "material" keeps the colours, "halten" refits them so the patch looks as
    # it did. Material is the default: what gets painted is a patch that
    # looks wrong, and holding its look keeps the wrong look -- on the shoe,
    # "halten" restored the opacity of a thinned label and kept it a blur,
    # "material" brought the lettering back.
    colour: str = "material"
    target: str = "umgebung"    # "umgebung": as solid as around it; "dicht": opaque
    tilt_deg: float = 25.0      # the two extra views
    iterations: int = 4
    gamma_cap: float = 16.0     # never raise transmittance to more than this power
    alpha_cap: float = 0.99
    resolution: int = 256
    colour_steps: int = 150
    seed: int = 0


def _cameras(centre, n_out, up, view, radius, W, tilt):
    from .gpu import look_at_camera
    d = n_out.copy()
    if view is not None:
        v = -np.asarray(view, float)
        v /= max(np.linalg.norm(v), 1e-12)
        # the user's view, if it looks at the front of this surface at all
        if float(v @ n_out) > 0.2:
            d = v
    side = np.cross(d, up)
    if np.linalg.norm(side) < 1e-6:
        side = np.cross(d, np.array([1.0, 0.0, 0.0]))
    side /= np.linalg.norm(side)
    fov = 30.0
    dist = radius / (np.tan(np.radians(fov) * 0.5) * 0.7)
    dirs = [d]
    for sgn in (1.0, -1.0):
        t = d * np.cos(np.radians(tilt)) + sgn * side * np.sin(np.radians(tilt))
        dirs.append(t / np.linalg.norm(t))
    return [look_at_camera(centre + e * dist, centre, up, W, W, fov) for e in dirs], fov


def _draw(sub, cams, use_gpu, deg, fov):
    """Colour and coverage of ``sub`` from each camera, as numpy arrays."""
    if use_gpu:
        import torch
        from . import gpu
        p = gpu.to_tensors(sub, torch.device("cuda"))
        with torch.no_grad():
            rgb, a, _ = gpu.render(p, cams, sh_degree=deg, antialiased=False,
                                   with_depth=False)
        return rgb.cpu().numpy().astype(np.float64), a[..., 0].cpu().numpy()
    from . import preview
    out_rgb, out_a = [], []
    for c in cams:
        R = c.view[:3, :3]
        eye = -R.T @ c.view[:3, 3]
        img, a = preview.render(sub, eye, eye + R[2], up=-R[1], W=c.width,
                                H=c.height, fov=fov, sh_degree=deg)
        out_rgb.append(img.astype(np.float64) / 255.0)
        out_a.append(np.asarray(a, np.float64))
    return np.stack(out_rgb), np.stack(out_a)


def _hold_colour(sub, raise_local, cam, target_rgb, weight, deg, steps):
    """Refit the raised splats' base colour so the reference view looks as before."""
    import torch
    from . import gpu
    dev = torch.device("cuda")
    p = gpu.to_tensors(sub, dev)
    idx = torch.tensor(raise_local, device=dev, dtype=torch.long)
    dc = p["colors"][idx, 0, :].clone().requires_grad_(True)
    dc0 = dc.detach().clone()
    tgt = torch.tensor(target_rgb, device=dev, dtype=torch.float32)
    w = torch.tensor(weight, device=dev, dtype=torch.float32)[..., None]
    opt = torch.optim.Adam([dc], lr=2.5e-3)
    for _ in range(max(steps, 1)):
        opt.zero_grad(set_to_none=True)
        c = p["colors"].clone()
        c[idx, 0, :] = dc
        out, _, _ = gpu.render({**p, "colors": c}, [cam], sh_degree=deg,
                               antialiased=False, with_depth=False)
        loss = ((out[0] - tgt).abs() * w).sum() / w.sum().clamp_min(1.0) \
            + 0.01 * (dc - dc0).pow(2).mean()
        loss.backward()
        opt.step()
    return dc.detach().cpu().numpy()


def solidify(s: SplatSet, selection, view=None,
             params: Optional[SolidParams] = None) -> RepairResult:
    """Make the painted surface as solid as its surroundings."""
    params = params or SolidParams()
    rng = np.random.default_rng(params.seed)
    sel = np.zeros(s.n, bool)
    sel[selection] = True
    P = surface_patch(s, sel, rng, view=None if view is None else np.asarray(view, float),
                      thinned=True)

    # What may change: everything painted at or under the face. Haze in
    # front of it stays as faint as it is -- raising it would turn
    # reconstruction fuzz into a crust. Not only the first few layers,
    # either: the shell here is some eighteen splats deep, and stopping at
    # ten left exactly the deep layers' pixels showing through.
    under = P.depth < 3.0 * P.spacing
    raise_g = P.cand[under & sel[P.cand]]
    if len(raise_g) < 10:
        raise RepairError("there is no surface in the stroke to solidify")

    xyz = s.xyz.astype(np.float64)
    centre = xyz[sel].mean(0)
    near = np.where(np.linalg.norm(xyz - centre, axis=1) < 4.0 * P.radius)[0]
    pos = np.full(s.n, -1, np.int64)
    pos[near] = np.arange(len(near))
    sub = s.take(near)
    r_loc = pos[raise_g]
    keep = r_loc >= 0
    raise_g, r_loc = raise_g[keep], r_loc[keep]
    b_loc = pos[np.where(sel)[0]]
    b_loc = b_loc[b_loc >= 0]

    from . import gpu
    use_gpu = gpu.available()[0]
    deg = int(s.sh_degree)
    W = int(params.resolution)
    n_out = P.basis[2] * P.outward
    cams, fov = _cameras(centre, n_out, P.basis[1], view, P.radius, W,
                         params.tilt_deg)

    from .deep import _project
    feet, rings, pix = [], [], []
    for c in cams:
        x, y, z = _project(xyz[near], c)
        ok = np.isfinite(x) & (z > 0)
        ix = np.where(ok, np.floor(np.nan_to_num(x)), -1).astype(np.int64)
        iy = np.where(ok, np.floor(np.nan_to_num(y)), -1).astype(np.int64)
        inside = (ix >= 0) & (ix < W) & (iy >= 0) & (iy < W)
        foot = np.zeros((W, W), bool)
        bb = b_loc[inside[b_loc]]
        foot[iy[bb], ix[bb]] = True
        foot = ndimage.binary_fill_holes(
            ndimage.binary_closing(foot, np.ones((3, 3)), iterations=2))
        ring = ndimage.binary_dilation(foot, np.ones((3, 3)), iterations=4) & ~foot
        feet.append(foot)
        rings.append(ring)
        pix.append((np.clip(ix, 0, W - 1), np.clip(iy, 0, W - 1), inside))

    rgb0, a0 = _draw(sub, cams, use_gpu, deg, fov)
    targets = []
    for k in range(len(cams)):
        ring_a = a0[k][rings[k]]
        t = float(np.median(ring_a)) if ring_a.size else 0.995
        targets.append(0.995 if params.target == "dicht" else min(t, 0.995))

    before = [float(a0[k][feet[k]].mean()) if feet[k].any() else 1.0
              for k in range(len(cams))]
    thin_px = max(float((a0[k][feet[k]] < targets[k] - 0.02).mean())
                  if feet[k].any() else 0.0 for k in range(len(cams)))
    if thin_px < 0.005:
        # Nothing shows through: say so instead of refitting colours that
        # were fine, which is all the brush would otherwise have done.
        raise RepairError("the spot is already as solid as its surroundings "
                          "(opacity %s)" % ("%.3f" % min(before)))

    work = sub.copy()
    alpha_in = work.alpha[r_loc].copy()
    for _ in range(max(params.iterations, 1)):
        _, a = _draw(work, cams, use_gpu, deg, fov)
        gam = np.ones(len(r_loc))
        for k in range(len(cams)):
            T_now = np.clip(1.0 - a[k], 1e-4, 1.0)
            g = np.log(max(1.0 - targets[k], 1e-4)) / np.log(np.minimum(T_now, 1.0 - 1e-6))
            g = np.where(feet[k], np.clip(g, 1.0, params.gamma_cap), 1.0)
            g = ndimage.uniform_filter(ndimage.median_filter(g, 3), 3)
            ix, iy, inside = pix[k]
            here = inside[r_loc] & feet[k][iy[r_loc], ix[r_loc]]
            gam = np.where(here, np.maximum(gam, g[iy[r_loc], ix[r_loc]]), gam)
        al = work.alpha[r_loc].astype(np.float64)
        al = np.minimum(1.0 - (1.0 - np.clip(al, 0, 1 - 1e-6)) ** gam, params.alpha_cap)
        al = np.clip(al, 1e-4, 1.0 - 1e-6)
        work.opacity[r_loc] = np.log(al / (1.0 - al)).astype(np.float32)

    colour = params.colour
    if colour == "halten" and use_gpu:
        w = np.clip((a0[0] - 0.5) / 0.4, 0.0, 1.0) * feet[0]
        if w.sum() > 50:
            dc = _hold_colour(work, r_loc, cams[0], rgb0[0], w, deg,
                              params.colour_steps)
            work.f_dc[r_loc] = dc.astype(np.float32)
    elif colour == "halten":
        colour = "material (no GPU)"

    _, a1 = _draw(work, cams, use_gpu, deg, fov)
    after = [float(a1[k][feet[k]].mean()) if feet[k].any() else 1.0 for k in range(len(cams))]
    gaps = float((a1[0][feet[0]] < targets[0] - 0.05).mean()) if feet[0].any() else 0.0

    # only what actually changed goes into the edit
    moved = ((np.abs(work.alpha[r_loc] - alpha_in) > 1e-4)
             | (np.abs(work.f_dc[r_loc] - sub.f_dc[r_loc]).max(1) > 1e-5))
    if not moved.any():
        raise RepairError("nothing to solidify")
    raise_g, r_loc = raise_g[moved], r_loc[moved]
    added = work.take(r_loc)
    warn = []
    if min(targets) < 0.9:
        warn.append("the surroundings are translucent themselves (opacity %s): "
                    "matched, not sealed"
                    % ("%.2f" % min(targets)))
    if gaps > 0.03:
        warn.append("%d %% of the spot stays open: material is missing there; "
                    "fill it with the Repair Brush" % round(100 * gaps))
    info = {
        "mode": "solidify", "raised": int(len(raise_g)),
        "coverageBefore": [round(v, 3) for v in before],
        "coverageAfter": [round(v, 3) for v in after],
        "target": [round(v, 3) for v in targets],
        "gapShare": round(gaps, 3), "colour": colour,
        "backend": "gsplat" if use_gpu else "CPU",
        "warning": "; ".join(warn) or None,
    }
    return RepairResult(removed=raise_g, added=added, info=info)
