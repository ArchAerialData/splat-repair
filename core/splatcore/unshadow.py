"""The shadow brush: paint over a shadow baked into the scan, and lift it.

A scan's colours are what the camera saw, lighting included. A shadow cast
on a surface at capture time is therefore part of the surface's colour, and
it stays wherever the model is lit or viewed from. This brush takes it out.

The model is the simplest one that holds: a shadow multiplies the colour of
what it falls on by a gain per channel, the same gain everywhere it is fully
dark, ramping to one across a penumbra. Then the job is to find three
numbers and an edge.

* **The edge is found, not taken from the stroke.** The user paints
  generously; the shadow is somewhere inside. Each grid cell's log colour is
  compared with the lit tone of the surroundings continued across the stroke
  (a membrane, as the repair brush's tone field), and Otsu's threshold on
  that deficit splits shadow from lit. Measured on the shoe: 0.83 to 0.93 of
  the true shadow recovered (intersection over union).
* **The gain comes from both sides of that edge.** A strip just inside and a
  strip just outside are the same material under different light, so the
  ratio of their medians is the shadow and nothing else -- Guo, Dai and
  Hoiem's paired regions (2011), Shor and Lischinski's strips (2008). One
  number per channel for the whole shadow, pooled over hundreds of cells,
  which is what makes it immune to texture and to dark details inside the
  shadow. Solving a gain per cell instead -- the obvious thing -- measured
  three to six times worse and lifted a black logo inside the shadow to grey.
* **The view-dependent bands are scaled with the colour.** A change of
  illumination scales the light leaving a surface in every direction, so it
  scales every SH coefficient; leaving them alone would leave about a sixth of
  the shadow in the view-dependent part, and zeroing them would flatten the
  surface's sheen.

Work is in log space throughout, because a shadow is a product: the stored
colours are gamma-encoded, and a product stays a product under a power law.

Where the cells refuse -- soft shadows on grained surfaces, shadows wider
than the stroke, strokes of a hundred splats -- the stroke is looked at as a
picture along the painting view (`_unshadow_picture`), and every splat that
picture shows is fitted, not only the painted ones (`_picture_fit`).
"""
from __future__ import annotations

from dataclasses import dataclass
from typing import Optional

import numpy as np
from scipy import ndimage
from scipy.spatial import cKDTree

from .io import SplatSet, SH_C0
from .repair import RepairError, RepairResult, _accum, fit_reference_surface
from .surface import surface_patch


@dataclass
class ShadowParams:
    strength: float = 1.0       # 0..1, how much of the shadow to take out
    shading: bool = True        # lit reference minus a shading model in the surface normal
    min_step_grain: float = 1.0  # the shadow's step must exceed the surface's own grain (0 = off)
    edge: str = "detect"        # "detect": find it in the stroke; "stroke": the stroke is it
    penumbra: int = 2           # cells of soft edge either side
    strip: int = 5              # cells of each paired strip
    gap: int = 4                # strips start this far from the edge
    tone_smooth: float = 6.0    # lit-reference stiffness, in cells
    lift_cap: float = 6.0       # never brighten by more than this
    min_depth: float = 0.1      # log-brightness gap that counts as a shadow
    # how far the texture may differ across the edge. Asymmetric, from the
    # shoe: real shadows 0.82 to 1.52 (a dark logo inside raises it), a
    # change of material 0.38 to 0.62
    texture_tol: float = 1.6
    texture_low: float = 0.7
    min_coverage: float = 0.85  # stroke's surface share against its surroundings'
    # several materials under one shadow (see _labels)
    k_labels: int = 5           # material clusters tried; unsupported ones merge
    label_min_ring: int = 15    # cells a label needs just outside the stroke
    label_sep: float = 4.0      # clusters closer than this (in spreads) are one material
    chroma_smooth: int = 5      # cells the chroma/roughness features are pooled over
    window: float = 8.0         # along-edge radius of a local pairing window, cells
    texture_low_w: float = 0.5  # per-window texture ratio bounds (noisier than global)
    texture_tol_w: float = 2.0
    min_good_windows: float = 0.3  # share of windows that must pass the texture guard
    pen_max: int = 9            # widest penumbra half-width the profile fit may choose
    label_tol: float = 0.3      # materials' gains may differ by this much (log)
    reflex: bool = True         # also take out coloured reflections in the SH bands
    reflex_min: float = 3.0     # how far a reflection stands out, in the surroundings' noise
    reflex_mass: float = 60.0   # its summed excess over twice the noise, in cells
    reflex_chroma: float = 0.28  # share of its excess that is colour, not brightness
    # where the edge pairs two surfaces, take the shadow from a wide view
    view_fallback: bool = True
    view_context: float = 3.0   # the view's half width in stroke radii
    view_cap: float = 1.25      # deepest lift, in multiples of the core's gain
    faint: bool = True          # also lift the faint splats in the slab under the stroke
    off_surface: float = 25.0   # a core further off the surface than this (spacings) is no shadow
    # where the cells refuse, look at the stroke as a picture (_picture_shadow)
    picture: bool = True
    picture_ctx: float = 1.25   # the picture's half width in stroke lengths
    # the lifted splats' brightness above this percentile takes the lit
    # surface's spread (_lit_tail); 100 = off
    tail_from: float = 50.0
    seed: int = 0
    # "auto": find the shadow and take it out (everything above);
    # "lift": brighten what was painted by ``lift``, no detection (see _lift)
    mode: str = "auto"
    lift: float = 1.5           # "lift": the factor in the middle of the stroke
    feather: float = 0.35       # "lift": soft edge, as a share of the stroke's half width
    lift_chroma: float = 0.25   # "lift": the colour's distance from grey follows gain ** this


def _picture_frame(s, idx, n, feather=0.35, W=384, volume=None, ctx=1.0):
    """The stroke looked at as a picture along the painting direction.

    Rendered nearly orthographic, half the frame margin (``ctx`` widens
    it), only what lies at or behind the painted surface. Returns a dict
    with the camera, the splats in the frame (``near``, ``sub``), the
    stroke's mask (``occ``) and its soft rim (``mask``), or None when the
    picture shows nothing of the stroke.

    With the painted ``volume`` (the app sends it) the frame and the mask are
    what was painted. Without it they come from the selected splats, and on
    the hut those are the problem: 70 of the 94 sat in one clump at the end
    of a 0.6 m stroke, so the mask was a speck.
    """
    from . import gpu
    xyz = s.xyz.astype(np.float64)
    p = xyz[idx]
    a = s.alpha[idx]
    vpts = None
    if volume is not None and hasattr(volume, "surface_points"):
        try:
            vpts = np.asarray(volume.surface_points(), np.float64).reshape(-1, 3)
        except Exception:                        # noqa: BLE001 -- a malformed volume: splats only
            vpts = None
        if vpts is not None and len(vpts) < 8:
            vpts = None
    if vpts is not None:
        p, a = vpts, np.ones(len(vpts))
    centre = (p * np.clip(a, 1e-3, None)[:, None]).sum(0) / np.clip(a, 1e-3, None).sum()
    tmp = np.array([1.0, 0.0, 0.0]) if abs(n[0]) < 0.9 else np.array([0.0, 1.0, 0.0])
    e0 = np.cross(n, tmp)
    e0 /= np.linalg.norm(e0)
    e1 = np.cross(n, e0)
    d = p - centre
    # robust: one stray haze splat in the selection made the frame three
    # times too wide and the stroke a speck in it
    dv = d[a > 0.1] if (a > 0.1).sum() >= 8 else d
    span = lambda x: float(np.percentile(x, 98) - np.percentile(x, 2))   # noqa: E731
    ext = max(span(dv @ e0), span(dv @ e1), 1e-6)
    # nearly orthographic: far back, narrow; half the frame is margin, so a
    # large splat's weight outside the stroke is counted
    D = 20.0 * ext
    fov = 2.0 * np.degrees(np.arctan(ctx * ext / D))
    cam = gpu.look_at_camera(centre - n * D, centre, e1, W, W, fov)
    # Only what lies at or behind the painted surface: the virtual eye is far
    # back, and whatever stood between the user and the surface was not in
    # the user's picture of it either.
    depth = (xyz - centre) @ n
    front = float(np.percentile(depth[idx], 1))
    if vpts is not None:
        front = min(front, float(np.percentile((vpts - centre) @ n, 1)))
    front -= 0.25 * ext
    reach = 1.5 * ctx * ext
    near = np.where((depth >= front)
                    & (np.abs((xyz - centre) @ e0) < reach)
                    & (np.abs((xyz - centre) @ e1) < reach))[0]
    if len(near) < 8:
        return None
    sub = s.take(near)
    q = sub.xyz.astype(np.float64) @ cam.view[:3, :3].T + cam.view[:3, 3]
    uvw = q @ cam.K.T
    u = uvw[:, 0] / uvw[:, 2]
    v = uvw[:, 1] / uvw[:, 2]
    is_sel = np.zeros(len(near), bool)
    is_sel[np.isin(near, idx)] = True
    vis = is_sel & (sub.alpha > 0.1)
    if vis.sum() < 3:
        vis = is_sel
    occ = None
    if vpts is not None and hasattr(volume, "mask_in_view"):
        try:
            occ = np.asarray(volume.mask_in_view(cam, W, W), bool)
        except Exception:                        # noqa: BLE001
            occ = None
        if occ is not None and occ.sum() < 16:
            occ = None
    if occ is None:
        uu, vv = u[vis], v[vis]
        inside = (uu >= 0) & (uu < W) & (vv >= 0) & (vv < W)
        if inside.sum() < 3:
            return None
        occ = np.zeros((W, W), bool)
        occ[vv[inside].astype(int), uu[inside].astype(int)] = True
        pts = np.stack([uu[inside], vv[inside]], 1)
        nn = cKDTree(pts).query(pts, k=min(3, len(pts)))[0][:, -1]
        it = int(np.clip(np.ceil(np.median(nn)), 1, 40))
        occ = ndimage.binary_fill_holes(ndimage.binary_closing(
            np.pad(occ, it + 1), np.ones((3, 3)), iterations=it))[it + 1:-(it + 1), it + 1:-(it + 1)]
    dist = ndimage.distance_transform_edt(occ)
    half = float(dist.max())
    if half < 1.0:
        return None
    mask = _smoothstep(dist / max(feather * half, 1.5))
    return dict(cam=cam, near=near, sub=sub, u=u, v=v, is_sel=is_sel, occ=occ,
                mask=mask, half=half, W=W, centre=centre, normal=n, ext=ext)


# The fit renders the way the app shows the model: PlayCanvas draws every
# splat at its full opacity, however small ("classic" in gsplat). The fit
# used to look through gsplat's antialiased mode, which dims a splat smaller
# than a pixel by its lost area. There the gravel's tiny pebble splats
# hardly counted, the fit let them go to the map's full lift, and in the app
# -- where they count fully -- they lit up as white-green glitter. Finding
# the shadow stays on the antialiased picture: the classic one doubles the
# grain of the siding, and every threshold below was set on the smooth one.
_APP_AA = False


def _picture_base(F, depth=False):
    """The frame rendered as it is: ``(rgb, alpha[, depth])``, float numpy."""
    import torch
    from . import gpu
    tt = gpu.to_tensors(F["sub"], torch.device("cuda"))
    with torch.no_grad():
        img, a, dep = gpu.render(tt, [F["cam"]], sh_degree=0, with_depth=depth)
    out = (img[0].cpu().numpy().astype(np.float64), a[0, ..., 0].cpu().numpy().astype(np.float64))
    if depth:
        d = dep[0].cpu().numpy().astype(np.float64)
        out = out + (d[..., 0] if d.ndim == 3 else d,)
    return out


def _picture_fit(s, F, idx, lg_map, w_sel, cap, steps=40, init=None):
    """Fit every splat seen in the frame so the picture is ``base * exp(lg_map)``.

    ``lg_map`` is (W, W, C), C = 1 (one factor for all channels) or 3, the
    log lift asked for per pixel, already with the stroke's soft rim. Every
    splat seen in the frame gets a gain fitted to that picture (Adam on the
    log gain, only upwards, never past white): a small splat inside ends up
    with the full lift, a large one half outside about half, and the
    surroundings stay as they were. ``w_sel`` (len(idx), C) is what the
    painted splats get where the picture does not see them (behind the
    face, or in front of the frame's block); ``init`` (len(idx), C) their
    starting point. Returns ``(indices, log gains (n, C))`` over ``s``.
    """
    import torch
    from . import gpu
    near, sub, is_sel = F["near"], F["sub"], F["is_sel"]
    lg_map = np.asarray(lg_map, np.float64)
    C = lg_map.shape[-1]
    w_sel = np.asarray(w_sel, np.float64).reshape(len(idx), C)
    dev = torch.device("cuda")
    tt = gpu.to_tensors(sub, dev)
    c0 = tt["colors"].detach()
    # who is seen at all: a splat's weight anywhere in the frame
    probe = torch.zeros((sub.n, 1), device=dev, requires_grad=True)
    pp = dict(tt)
    pp["colors"] = torch.cat([c0[:, :1] + probe[:, :, None], c0[:, 1:]], 1)
    img, _alpha, _ = gpu.render(pp, [F["cam"]], sh_degree=0, with_depth=False,
                                antialiased=_APP_AA)
    g_all = torch.autograd.grad(img[0].sum(), probe)[0][:, 0]
    g_all = g_all.cpu().numpy().astype(np.float64)
    F["weight"] = g_all                          # what each splat adds to the picture (_lit_tail)
    seen = g_all > 1e-4 * max(float(g_all.max()), 1e-12)
    # Seen, but barely: a splat half behind the face, or nearly clear. Adam
    # normalises its step, so the faintest gradient walks such a splat to its
    # bound in forty steps -- dark splats four centimetres behind the band
    # under the balcony went x4 and lit up as spots wherever another view
    # sees past the face. They keep the map's lift at their centre.
    strong = seen & (g_all > 0.05 * float(np.median(g_all[seen]))) if seen.any() else seen
    base = img[0].detach()
    m = torch.tensor(lg_map, device=dev, dtype=img.dtype)
    target = torch.clamp(base * torch.exp(m), max=1.0)
    order = np.argsort(idx)
    sel_pos = np.where(is_sel)[0]
    own = np.zeros((len(near), C))
    own[sel_pos] = w_sel[order][np.searchsorted(idx[order], near[sel_pos])]
    # every splat starts at the lift asked for where its centre is; the fit
    # then corrects what its footprint makes of that
    x0 = _sample_map(F, lg_map, sub.xyz)
    if init is not None:
        x0[sel_pos] = np.asarray(init, np.float64).reshape(len(idx), C)[order][
            np.searchsorted(idx[order], near[sel_pos])]
    # No splat lifts past what the map asks for anywhere under its own
    # footprint. Fitted freely, faint and half-hidden splats were pushed far
    # past the map to make up the mean -- white sparkles in lifted sand.
    W = lg_map.shape[0]
    zc = np.maximum(sub.xyz.astype(np.float64) @ F["cam"].view[2, :3] + F["cam"].view[2, 3], 1e-9)
    rad = 3.0 * sub.sigma.max(1) * F["cam"].K[0, 0] / zc
    levels = np.array([0, 1, 2, 4, 8, 16, 32, 64, 128])
    lvl = np.clip(np.searchsorted(levels, rad), 0, len(levels) - 1)
    iu = np.clip(np.round(F["u"] - 0.5).astype(int), 0, W - 1)
    iv = np.clip(np.round(F["v"] - 0.5).astype(int), 0, W - 1)
    inside = (F["u"] >= 0) & (F["u"] < W) & (F["v"] >= 0) & (F["v"] < W)
    bound = np.zeros((len(near), C))
    for k, r in enumerate(levels):
        on = lvl == k
        if not on.any():
            continue
        for c in range(C):
            mx = ndimage.maximum_filter(lg_map[..., c], size=2 * int(r) + 1) if r else lg_map[..., c]
            bound[on, c] = mx[iv[on], iu[on]]
    bound = np.where(inside[:, None], bound, 0.0)
    # A dark splat may go further, up to the lit picture's colour around it:
    # the pole's shadow on the siding is drawn by half-transparent dark
    # splats, and held to the map's x1.4 they left it at 80 per cent. What
    # makes a spark is a splat ending brighter than its surroundings, not a
    # large gain, so that is the bound: the target's colour, low-passed, where
    # the splat sits. (The brightest under its footprint let the dark crease
    # along the shutter box under the balcony go to the band's lit colour --
    # a light line.)
    dc = 0.5 + SH_C0 * sub.f_dc.astype(np.float64)
    ldc = np.log(np.clip(dc, 1e-6, None))
    tgt_np = np.log(np.clip(np.clip(base.detach().cpu().numpy().astype(np.float64), 0, 1)
                            * np.exp(lg_map), 1.0 / 255, 1.0))
    tgt_np = np.stack([ndimage.gaussian_filter(tgt_np[..., c], 1.5) for c in range(3)], -1)
    cb = tgt_np[iv, iu] - ldc
    cb = cb.min(1, keepdims=True) if C == 1 else cb
    lifted_px = lg_map.max(-1) > 1e-3
    near_lift = ndimage.maximum_filter(lifted_px.astype(np.uint8), size=5)[iv, iu] > 0
    cb = np.where((inside & near_lift & strong)[:, None], np.maximum(cb, 0.0), 0.0)
    bound = np.maximum(bound, cb)
    bound = np.where(is_sel[:, None] & ~seen[:, None], cap, np.minimum(bound, cap))
    # and none past white: the fit has to know, or it leans on channels and
    # splats the clamp afterwards takes back (lifted sand came out grey).
    # By one factor for all three channels: clamped channel by channel, a
    # light pebble stopped in red at white while green went on to x1.95 --
    # the green in the glitter (`_white_clamp`).
    if C == 1:
        bound = np.minimum(bound, np.maximum(-ldc.max(1, keepdims=True), 0.0))
    x = torch.tensor(_white_clamp(np.clip(x0, 0.0, bound), ldc), device=dev,
                     dtype=img.dtype).requires_grad_(True)
    bound_t = torch.tensor(bound, device=dev, dtype=img.dtype)
    ldc_t = torch.tensor(ldc, device=dev, dtype=img.dtype)
    seen_t = torch.tensor(seen, device=dev, dtype=img.dtype)[:, None]
    opt = torch.optim.Adam([x], lr=0.05)
    rgb0 = c0[:, :1] * SH_C0 + 0.5
    strong_t = torch.tensor(strong, device=dev, dtype=img.dtype)[:, None]
    for _ in range(steps):
        g = torch.exp(x * seen_t)[:, None, :]
        cols = torch.cat([((rgb0 * g) - 0.5) / SH_C0, c0[:, 1:] * g], 1)
        pp["colors"] = cols
        out, _a, _ = gpu.render(pp, [F["cam"]], sh_degree=0, with_depth=False,
                                antialiased=_APP_AA)
        loss = ((out[0] - target) ** 2).mean() + 1e-6 * (x ** 2).mean()
        opt.zero_grad()
        loss.backward()
        x.grad.mul_(strong_t)
        opt.step()
        with torch.no_grad():
            y = torch.minimum(torch.clamp(x, min=0.0), bound_t)
            if C == 3:
                over = torch.clamp((ldc_t + y).max(1, keepdim=True)[0], min=0.0)
                y = torch.clamp(y - over, min=0.0)
            x.copy_(y)
    xs = x.detach().cpu().numpy().astype(np.float64)
    # painted splats the picture does not see (behind the face) keep their
    # own weight
    lg = np.where(seen[:, None], xs, np.where(is_sel[:, None], own, 0.0))
    lg = _neighbour_cap(sub.xyz, lg, ldc)
    keep = lg.max(1) > 1e-3
    # painted splats outside the rendered block (in front of it) as before
    out_sel = ~np.isin(idx, near)
    return (np.concatenate([near[keep], idx[out_sel]]),
            np.concatenate([lg[keep], w_sel[out_sel]]))


def _white_clamp(lg, ldc):
    """Log gains ``lg`` (n, C) lowered so no channel of ``exp(ldc)`` passes white.

    One shift for all channels of a splat, so its hue stays; ``ldc`` (n, 3) is
    the log base colour. With C = 1 the brightest channel decides.
    """
    lg = np.asarray(lg, np.float64)
    top = np.asarray(ldc, np.float64).max(1, keepdims=True)
    if lg.shape[1] == 1:
        return np.clip(np.minimum(lg, -top), 0.0, None)
    over = np.maximum((np.asarray(ldc, np.float64) + lg).max(1, keepdims=True), 0.0)
    return np.clip(lg - over, 0.0, None)


def _neighbour_cap(xyz, lg, ldc=None, k=8, margin=0.15):
    """No splat lifted far past the lifted splats around it.

    A shadow is light, and light is smooth across the splats of a surface:
    a splat asking for much more than its neighbours is noise in the fit --
    a faint splat half behind the face, a pebble the map caught at its peak
    -- and in the app it shows as a spark. Each lifted splat is held to the
    median of its ``k`` nearest lifted neighbours plus ``margin`` (log), by
    one factor over its channels -- unless it ends no brighter than they do
    (``ldc``, the log base colours, given): a dark splat that draws the
    shadow catches up with its surroundings, it does not outshine them.
    """
    lg = np.asarray(lg, np.float64)
    on = np.where(lg.max(1) > 1e-3)[0]
    if len(on) <= k:
        return lg
    p = np.asarray(xyz, np.float64)[on]
    mean = lg[on].mean(1)
    _, nb = cKDTree(p).query(p, k=k + 1)
    ref = np.median(mean[nb[:, 1:]], 1)
    cap = ref + margin
    if ldc is not None:
        lum = np.asarray(ldc, np.float64)[on].mean(1)
        after = lum + mean
        nb_after = np.median(after[nb[:, 1:]], 1)
        cap = np.maximum(cap, nb_after + margin - lum)
        # And a highlight rolls off: no splat ends more than x1.5 brighter
        # than the lifted splats around it, unless it already was. Lifted by
        # the full gain, the light pebbles in the hut's gravel -- twice as
        # bright as the sand between them -- came out white glitter.
        cap = np.minimum(cap, np.maximum(nb_after + 0.4 - lum, 0.0))
    over = mean > cap
    if over.any():
        out = lg.copy()
        sc = cap[over] / np.maximum(mean[over], 1e-9)
        out[on[over]] = lg[on[over]] * sc[:, None]
        return out
    return lg


def _wquant(x, w, p):
    """Weighted percentiles ``p`` (0..100) of ``x``."""
    o = np.argsort(x)
    x, cw = x[o], np.cumsum(np.asarray(w, np.float64)[o])
    cw /= cw[-1]
    return x[np.minimum(np.searchsorted(cw, np.asarray(p, np.float64) / 100.0), len(x) - 1)]


def _seen_lum(sub, n, tilt=30.0):
    """Each splat's brightest look (mean log colour, bands included) from
    along ``n`` and four directions ``tilt`` degrees around it.

    A light pebble's glint sits in its view-dependent bands, which a lift
    scales with the colour: measured on the base colour alone the pebbles
    lifted in the hut's gravel looked tame, in the views they sparkled.
    """
    from .shtools import sh_basis
    n = np.asarray(n, np.float64) / np.linalg.norm(n)
    dc = 0.5 + SH_C0 * sub.f_dc.astype(np.float64)
    rest = sub.f_rest.astype(np.float64)
    if rest.shape[1] == 0:
        return np.log(np.clip(dc, 1e-4, None)).mean(1)
    deg = int(round(np.sqrt(rest.shape[1] + 1))) - 1
    tmp = np.array([1.0, 0.0, 0.0]) if abs(n[0]) < 0.9 else np.array([0.0, 1.0, 0.0])
    e0 = np.cross(n, tmp)
    e0 /= np.linalg.norm(e0)
    e1 = np.cross(n, e0)
    t = np.radians(tilt)
    dirs = [n] + [np.cos(t) * n + np.sin(t) * e for e in (e0, -e0, e1, -e1)]
    best = None
    for d in dirs:
        bas = sh_basis(np.asarray(d, np.float64), deg)[1:]
        col = dc + np.einsum("k,nkc->nc", bas, rest)
        lum = np.log(np.clip(col, 1e-4, None)).mean(1)
        best = lum if best is None else np.maximum(best, lum)
    return best


def _lit_tail(F, L, ix, lg, ldc, tail_from=50.0):
    """The lifted splats' bright end held to the spread of the lit surface.

    A shadow lifted by one depth keeps the contrast the surface had in it.
    In the hut's gravel that contrast is high: under the sky light the light
    pebbles stand far above the gravel between them, and lifted with it they
    came out as single bright specks over the lit sand beside -- weighted by
    what they show, the lifted splats' 95th percentile lay 1.08 log over
    their median, the lit sand's 0.49. So the bright end is matched to the
    lit surface's own: from the ``tail_from`` percentile up, the lifted
    splats' brightness distribution (weighted by each splat's share of the
    picture) takes the lit reference's shape above that same percentile.
    Brightness is each splat's brightest look from around the painting view
    (`_seen_lum`), bands included. Never under a splat's own colour, by one
    factor over its channels. What the bright end loses comes back as one
    factor over all of them, so the mean stays the depth's; only the spread
    changes. On the hut's sand: the specks over the lit sand's 99th
    percentile 182 / 156 -> 77 / 63 (front, oblique), the median lift
    x1.51 -> x1.48.

    ``F["lit_ref"]`` is the lit surface's pixels near the stroke
    (`_picture_shadow`), ``F["weight"]`` each splat's weight in the picture
    (`_picture_fit`), ``L`` the lift map, ``ldc`` (len(ix), 3) the log base
    colours. Returns ``(lg, info or None)``.
    """
    ref, w = F.get("lit_ref"), F.get("weight")
    if ref is None or w is None or len(ix) == 0 or tail_from >= 95.0:
        return lg, None
    near, Wd = F["near"], F["W"]
    u, v = F["u"], F["v"]
    inside = (u >= 0) & (u < Wd) & (v >= 0) & (v < Wd)
    iu = np.clip(np.round(u - 0.5).astype(int), 0, Wd - 1)
    iv = np.clip(np.round(v - 0.5).astype(int), 0, Wd - 1)
    k = np.minimum(np.searchsorted(near, ix), len(near) - 1)
    ok = (near[k] == ix) & (lg.max(1) > 1e-3)
    ok &= w[k] > 0
    if ok.sum() < 50:
        return lg, None
    lifted_px = ndimage.binary_dilation(np.asarray(L).max(-1) > 1e-3, iterations=2)
    lifted_s = np.zeros(len(near), bool)
    lifted_s[k[ok]] = True
    sub = F["sub"]
    lsub = _seen_lum(sub, F["normal"])
    rs = inside & ref[iv, iu] & ~lifted_px[iv, iu] & ~lifted_s & (w > 0)
    if rs.sum() < 200:
        return lg, None
    P = np.unique(np.clip([tail_from, 60, 70, 80, 90, 95, 97, 99, 99.5, 99.9], tail_from, 99.9))
    b = lsub[k[ok]]
    a = b + lg[ok].mean(1)
    wa = w[k[ok]]
    qa = _wquant(a, wa, P)
    # the lit surface's shape, set at the lifted splats' own level
    ql = _wquant(lsub[rs], w[rs], P)
    ql = ql - ql[0] + qa[0]
    qm = np.maximum.accumulate(np.minimum(ql, qa))
    if float((qa - qm).max()) < 0.02:
        return lg, None
    T = np.interp(a, qa, qm)
    T = np.where(a > qa[-1], qm[-1] + (a - qa[-1]), T)
    T = np.where(a < qa[0], a, T)
    red = a - np.maximum(np.minimum(a, T), b)
    # and the light that went out of the bright end back over all of them:
    # held alone, the gravel sank back to x1.3 and looked shaded again
    back = float(np.log(np.sum(wa * np.exp(a)) / np.sum(wa * np.exp(a - red))))
    red = red - max(back, 0.0)
    out = lg.copy()
    out[ok] = np.maximum(lg[ok] - red[:, None], 0.0)
    out[ok] = _white_clamp(out[ok], ldc[ok])
    return out, {"held": int((red > 0.02).sum()),
                 "spread": round(float(qa[P == 95][0] - qa[0]), 3),
                 "litSpread": round(float(ql[P == 95][0] - ql[0]), 3)}


def _sample_map(F, mp, xyz):
    """``mp`` (W, W, C) at the projections of ``xyz``; 0 outside the frame."""
    cam, W = F["cam"], F["W"]
    q = np.asarray(xyz, np.float64) @ cam.view[:3, :3].T + cam.view[:3, 3]
    uvw = q @ cam.K.T
    z = np.where(np.abs(uvw[:, 2]) > 1e-9, uvw[:, 2], 1e-9)
    u, v = uvw[:, 0] / z - 0.5, uvw[:, 1] / z - 0.5
    out = np.stack([ndimage.map_coordinates(mp[..., c], [v, u], order=1, mode="constant", cval=0.0)
                    for c in range(mp.shape[-1])], 1)
    out[q[:, 2] <= 0] = 0.0
    return out


def _nconv(x, w, sigma):
    """Gaussian smoothing of ``x`` over the pixels ``w`` (normalised)."""
    return (ndimage.gaussian_filter(x * w, sigma)
            / np.maximum(ndimage.gaussian_filter(w, sigma), 1e-6))


def _lift_map(rgb, alpha, F, logf, most=1.5):
    """Where "Brighten" lifts, per pixel: a dodge, the stroke's middle by the factor.

    The picture's brightness ``B`` (log, smoothed over a fifth of the
    stroke's half width, so the grain keeps its own dark and light and the
    lift is one smooth field) goes through a power curve anchored at the
    stroke's median ``m`` and at white: lift ``logf * B / m``. The middle of
    the stroke gets the factor the slider says, darker parts more (at most
    ``most`` times the factor, in log), lighter parts less, white nothing --
    what a dodge does, and nothing is pushed into white.

    Two earlier maps gave the factor to the dark part only (x1.5 changed the
    median of a stroke on the shoe's sole by 0.4 per cent), then everything
    half of it and the dark part all (median x1.22, x1.25 with the soft rim:
    on the result sheet hard to see). Now the median pixel of that stroke
    comes up x1.42 (front view), its dark third x1.53, its light third x1.40.

    Returns the log lift per pixel (W, W, 1), without the soft rim, and the
    stroke's middle level (None without one).
    """
    occ = F["occ"]
    valid = alpha > 0.5
    lum = np.log(np.clip(rgb, 1.0 / 255, 1.0)).mean(-1)
    sg = float(np.clip(0.2 * F["half"], 1.5, 8.0))
    B = _nconv(lum, valid.astype(float), sg)
    inside = occ & valid
    if inside.sum() < 16:
        return np.full(rgb.shape[:2] + (1,), float(logf)), None
    m = min(float(np.median(B[inside])), -0.05)
    g = np.clip(np.minimum(B, 0.0) / m, 0.0, float(most))
    return (g * float(logf))[..., None], m


def _soft_rim(occ, half, inside=0.25, outside=0.75):
    """The stroke's soft edge, as a dodge brush's soft tip: the full lift from
    ``inside`` of the half width within the stroke, falling smoothly to none
    ``outside`` of it beyond the painted rim. Ramped inside only, the outer
    third of every stroke lay in the ramp and x1.5 read as x1.25; ramped
    over a third of the half width across the rim, the stroke's end showed
    as an edge round a pasted-on patch."""
    d = ndimage.distance_transform_edt(occ) - ndimage.distance_transform_edt(~occ)
    r = max((inside + outside) * half, 3.0)
    return _smoothstep((d + outside * half) / r)


def _picture_share(s, idx, n, w_sel, logf, cap, feather=0.35, W=384, steps=40,
                   volume=None, aware=True):
    """Every splat that makes the painted picture, with its share of the stroke.

    The brush selects splats by their centre. On the hut that misses the
    picture: a stroke over the shadow under the balcony selected 94 splats
    that make 9 per cent of what is seen there; the rest comes from splats
    up to 7 cm long (the stroke was 3.6 cm wide) whose centres lie beside the
    stroke or a hand's breadth behind it. Lifting the 94 changed a thin line.

    So the stroke is looked at as a picture (`_picture_frame`), the picture
    asked for is this one lifted under the mask, and every splat seen in
    the frame gets a gain fitted to that (`_picture_fit`). A first try
    divided each splat's weight under the mask by its weight in the frame;
    that lifted the rim's splats too little and the stroke came out at half
    the factor. With ``aware`` the lift is a dodge curve over the stroke
    (`_lift_map`); otherwise it is the factor everywhere.

    Returns ``(indices, log gains, info)`` over ``s``, or None when nothing
    is seen.
    """
    from . import gpu
    ok, _why = gpu.available()
    if not ok:
        return None
    F = _picture_frame(s, idx, n, feather, W, volume)
    if F is None:
        return None
    info = {}
    if aware:
        rgb, alpha = _picture_base(F)
        lmap, mid = _lift_map(rgb, alpha, F, logf)
        if mid is not None:
            info = {"midLevel": round(float(np.exp(mid)), 3), "curve": True}
    else:
        lmap = np.full((F["W"], F["W"], 1), float(logf))
    lmap = lmap * _soft_rim(F["occ"], F["half"])[..., None]
    # the painted splats the picture does not see: the map where they lie,
    # at most their own rim weight's share
    own = np.minimum(_sample_map(F, lmap, s.xyz[idx]), w_sel[:, None] * logf)
    ix, lg = _picture_fit(s, F, idx, lmap, own, cap, steps)
    return ix, lg[:, 0], info


def _picture_shadow(rgb, alpha, F, params):
    """The shadow in the stroke's picture: ``(G, info)``, G (W, W, 3) log gains <= 0.

    The cell method needs a shadow edge inside the stroke, deeper than the
    surface's own grain. The shadows baked into real scans mostly are not:
    soft, 0.1 to 0.3 log deep on grained tread and sand, and wider than the
    stroke. On the picture two references answer what the cells cannot.

    * **Inside the stroke: LaMa.** The stroke is painted out as the lit
      layout around it (the mask grown by a sixth of its half width, so the
      penumbra does not count as light). The log deficit of the picture
      against that, smoothed over a fifth of the half width, is the shadow --
      LaMa continues the grooves of the siding and the grain of the sand, so
      they do not count as shadow, which a plain level would. The pole's
      shadow on the hut's siding: gone, the planks stay.
    * **Wider than the stroke: the lit level of its material.** When the
      ring just outside is as dark as the stroke, LaMa continues the shadow.
      Then the material's lit level comes from the whole view: the pixels of
      the stroke's chroma (low-passed, a shadow shifts it little) outside
      the stroke, split by Otsu into shadowed and lit. Used only when the
      ring lies clearly under that level too -- a shadow running on past
      the stroke, as the hut's on the sand.

    Either way only a coherent region deeper than its threshold counts (0.08
    log or half the grain; the cells' own guard is the grain itself), no
    pixel lifts past ``view_cap`` times the shadow's median (a dark logo
    stays dark), and a pixel more than four times under the lit level is a
    gap, not a shadow -- left alone.
    """
    from .deep import load_inpainter
    occ = F["occ"]
    W = occ.shape[0]
    valid = alpha > 0.5
    ins = occ & valid
    if ins.sum() < 30:
        raise RepairError("the stroke hits hardly any surface in the image")
    half = F["half"]
    Lc = np.log(np.clip(rgb, 1.0 / 255, 1.0))
    lum_px = Lc.mean(-1)
    ss = float(np.clip(0.2 * half, 1.5, 6.0))
    vw = valid.astype(float)
    g = max(2, int(round(0.15 * half)))
    Mg = ndimage.binary_dilation(occ, iterations=g)
    ring = (ndimage.binary_dilation(Mg, iterations=max(3, int(0.5 * half))) & ~Mg & valid)
    hp = lum_px - _nconv(lum_px, vw, ss)
    grain = float(hp[ring].std()) if ring.sum() > 30 else float(hp[valid].std())
    t1 = max(0.08, 0.5 * grain)
    cap = -np.log(params.lift_cap)
    info = {"grain": round(grain, 3)}

    def coherent(d, t):
        """The region deeper than t, as the components big enough to be one."""
        core = ins & (d < -t)
        lb, nl = ndimage.label(core)
        if nl == 0:
            return core, 0.0
        sizes = np.bincount(lb.ravel())[1:]
        keep = 1 + np.where(sizes >= max(30, 0.02 * ins.sum()))[0]
        core = np.isin(lb, keep)
        return core, float(core.sum()) / float(ins.sum())

    # --- inside the stroke: LaMa's lit picture --------------------------------
    G1, share1, depth1, pattern = None, 0.0, 0.0, False
    # finer than the level below: LaMa keeps the texture, and a pole's
    # shadow six pixels wide smoothed over four lit up a halo beside it
    s1 = float(np.clip(0.06 * half, 1.0, 4.0))

    def lama_deficit(mask):
        """Log deficit (W, W, 3) of the picture against LaMa's lit fill of ``mask``,
        and that fill."""
        fill, _ = load_inpainter("auto")
        u8 = (np.clip(rgb, 0.0, 1.0) * 255.0 + 0.5).astype(np.uint8)
        lit = fill(u8, (mask * 255).astype(np.uint8)).astype(np.float64) / 255.0
        wm = (mask & valid).astype(float)
        # Ratio of local means, not mean of log ratios: in the middle of a
        # wide stroke LaMa's texture goes soft, and the log of a soft texture
        # lies above the mean log of a contrasty one -- a lit synthetic plane
        # read 0.11 log "shadow" over 6 per cent of the stroke.
        return np.stack([np.log(np.clip(_nconv(rgb[..., c], wm, s1), 1.0 / 255, None))
                         - np.log(np.clip(_nconv(lit[..., c], wm, s1), 1.0 / 255, None))
                         for c in range(3)], -1), lit

    def plateau(D, core):
        """The shadow's full depth per channel (<= 0): its deep end, not its median.

        A soft shadow is mostly penumbra. The pole's on the siding is 0.39
        log deep in the middle and eight pixels wide; its median over the
        region was 0.17, and capped at 1.25 times that the brush lifted it by
        x1.19 and left the line standing. The deep end (the 95th percentile)
        is the umbra -- the line's continuation below the stroke, fainter,
        pulled the 85th down to 0.27 -- and a dark logo inside covers too
        little to move it.
        """
        return np.percentile(D[core], 5, axis=0)

    def lama_lift(D, core_, lit):
        """The lift from a LaMa deficit ``D``: what the shadow darkens near ``core_``.

        Near the shadow every darkening counts from half the threshold on: the
        threshold itself, as a soft gate, left the pole's penumbra standing as
        two thin dark lines either side of the lifted middle. And no pixel
        goes past LaMa's lit picture around it (``lit``): a local mean lifts
        a pixel that was already bright beside a dark one over the top -- the
        fascia's lower edge came out a white line.
        """
        near_ = ndimage.binary_dilation(core_, iterations=int(np.ceil(2 * s1)))
        gate = _smoothstep(-D.mean(-1) / (0.5 * t1)) * near_
        Gx = np.minimum(D, 0.0) * gate[..., None]
        Gx = np.maximum(Gx, params.view_cap * plateau(D, core_)[None, None, :])
        top = np.stack([ndimage.maximum_filter(ndimage.gaussian_filter(lit[..., c], 1.0), 3)
                        for c in range(3)], -1) * 1.03
        room = np.minimum(Lc - np.log(np.clip(top, 1.0 / 255, None)), 0.0)
        return np.maximum(Gx, room)

    try:
        Dl, lit1 = lama_deficit(Mg)
        d = Dl.mean(-1)
        core1, share1 = coherent(d, t1)
        # LaMa's texture is not the surface's: on lit tread and on a lit
        # synthetic plane it was 0.1 log off over 5 to 14 per cent of the
        # stroke -- darker as often as brighter. A shadow is darker only.
        _, share_pos = coherent(-d, t1)
        info["brightShare"] = round(share_pos, 2)
        # And the region must stand clearly under the ring in the picture
        # itself, by a tenth: in the middle of a stroke 80 px wide LaMa drifts
        # lighter, and a lit synthetic plane read as a shadow 0.11 deep over
        # 6 per cent of it where the picture is 0.08 under its ring (the
        # pole's shadow on the siding: 0.13).
        lin = rgb.mean(-1)
        below = (float(np.log(max(lin[ring].mean(), 1e-6) / max(lin[core1].mean(), 1e-6)))
                 if core1.any() and ring.any() else 0.0)
        info["belowRing"] = round(below, 3)
        # dark in the picture, but LaMa paints it dark too (or as often
        # lighter as darker): it belongs to the surface's pattern or form
        pattern = below >= max(t1, 0.1) and (share1 < 0.04 or share1 < 2.0 * share_pos + 0.02)
        if share1 >= 0.04 and share1 >= 2.0 * share_pos + 0.02 and below >= max(t1, 0.1):
            depth1 = -float(np.median(d[core1]))
            # only the shadow's own region: noise elsewhere stays
            G1 = np.where(ins[..., None], lama_lift(Dl, core1, lit1), 0.0)
    except RepairError:
        raise
    except Exception:                            # noqa: BLE001 -- no inpainter: the lit level only
        G1 = None
    info.update(shadowShare=round(share1, 2), depthInside=round(depth1, 3))

    # --- wider than the stroke: the lit level of its material -------------------
    low = np.stack([_nconv(Lc[..., c], vw, ss) for c in range(3)], -1)
    lum = low.mean(-1)
    ch = low - lum[..., None]
    chs = np.stack([_nconv(ch[..., c], vw, 2 * ss) for c in range(3)], -1)
    c_in = np.median(chs[ins], 0)
    # Same material is near in chroma -- except along warm/cool: a shadow
    # lit by the sky is bluer than the sun-lit sand beside it (0.33 on the
    # hut's sand), an indoor one warmer. That axis gets its own, wider
    # tolerance; matched isotropically the lit sand was not "the same", the
    # grey shadow was lifted to a grey, and came out as fog.
    ax = np.array([-1.0, 0.0, 1.0]) / np.sqrt(2.0)
    dvec = chs - c_in
    along = dvec @ ax
    orth = np.linalg.norm(dvec - along[..., None] * ax, axis=-1)
    tau = max(0.05, 2.0 * float(np.median(orth[ins])))
    dch = np.maximum(orth, np.abs(along) * tau / 0.4)
    out = valid & ~Mg
    match = out & (dch < tau)
    G2, d_in, d_ring, gap_only, even = None, 0.0, 0.0, False, False
    no_lit = match.sum() < 200
    if not no_lit:
        v = lum[match]
        thr = _otsu(v)
        lo_, hi_ = v[v < thr], v[v >= thr]
        sep = ((hi_.mean() - lo_.mean()) / max(np.sqrt(0.5 * (lo_.var() + hi_.var())), 1e-9)
               if len(lo_) > 2 and len(hi_) > 2 else 0.0)
        two = sep > 2.0 and len(hi_) > 0.15 * len(v)
        litm = match & (lum >= thr) if two else match & (lum >= np.percentile(v, 40))
        # the lit surface near the stroke, for the spread of its brightness
        # (`_lit_tail`); a lit wall at the frame's edge is not the sand
        F["lit_ref"] = litm & (ndimage.distance_transform_edt(~occ) <= 3.0 * half)
        lit_c = np.median(low[litm], 0)
        lit_l = float(lit_c.mean())
        mine = ins & (dch < tau)
        ring_m = ring & (dch < tau)
        if mine.sum() >= 30 and ring_m.sum() >= 30:
            # the stroke's dark part (a stroke half over the light has its
            # median there) against the ring's middle
            d_in = float(np.percentile(lum[mine], 25)) - lit_l
            d_ring = float(np.median(lum[ring_m])) - lit_l
            info.update(depthMaterial=round(-d_in, 3), depthRing=round(-d_ring, 3))
            t2 = max(0.2, 2.0 * t1)
            # Shallow and just as deep all round is the light falling off
            # across the part, not a shadow: the app's test stroke on intact
            # sole of brush_final stood 0.43 under the heel's level with its
            # ring 0.42 under, and was lifted x1.6. A cast shadow either shows
            # its edge near the stroke (the ring lighter: the balcony's fascia
            # 0.49 / 0.16) or is deep (band under the balcony 0.87, sand 0.72).
            even = d_in <= -t2 and d_ring <= -0.15 and d_ring - d_in < 0.15 and d_in > -0.5
            if d_in <= -t2 and d_ring <= -0.15 and not even:
                dl = lum - lit_l
                gate = _smoothstep((-dl - 0.5 * t1) / t1) * (dch < tau)
                # a gap is not a shadow: four times under the lit level
                gate = gate * (1.0 - _smoothstep((-dl - 1.2) / 0.3))
                core2 = ins & (gate > 0.5)
                if core2.sum() >= max(30, 0.04 * ins.sum()):
                    # The shadow is light, not paint: one depth ``k`` for
                    # everything under it, and per pixel only how much of it
                    # falls there (the matte). Lifted pixel by pixel to the
                    # sand's lit level, the darker gravel in the hut's
                    # shadow went to sand brightness and its light pebbles
                    # to white. ``k`` is the same material's shadowed level
                    # just outside the stroke against its lit one -- the
                    # shadow runs on past the stroke, or this branch would
                    # not be taken -- else the stroke's own median; a pixel
                    # deeper than ``k`` (a darker material) gets ``k``.
                    k_in = np.minimum(np.median(low[core2], 0) - lit_c, 0.0)
                    k = k_in
                    shd = ring_m & (lum < lit_l - 2.0 * t1) & (lum > lit_l - 1.2)
                    if shd.sum() >= 30:
                        k_ring = np.minimum(np.median(low[shd], 0) - lit_c, 0.0)
                        info["depthShadowRing"] = round(-float(k_ring.mean()), 3)
                        if k_ring.mean() > k_in.mean():
                            k = k_ring
                    info["depthIn"] = round(-float(k_in.mean()), 3)
                    k = np.minimum(k, -0.02)
                    kl = float(-k.mean())
                    matte = np.clip(-dl / kl, 0.0, 1.0) * gate
                    lit_layer = matte[..., None] * k[None, None, :]
                    G2 = np.where(ins[..., None], lit_layer, 0.0)
                    info["depthLight"] = round(kl, 3)
                else:
                    gap_only = bool((dl[mine] < -1.2).mean() > 0.5)
    if G2 is not None:
        G, how = G2, "lit"
    elif G1 is not None:
        G, how = G1, "lama"
    elif gap_only:
        raise RepairError("the dark part is more than four times darker than the same surface "
                          "in the light: a gap or another part, not a shadow")
    elif even and G1 is None:
        raise RepairError("no shadow found in the stroke: the surface is just as dark all around, "
                          "more likely the light falls off across the whole part; paint past "
                          "the shadow edge")
    elif no_lit:
        raise RepairError("no shadow found in the stroke: nothing in it is darker than the surface "
                          "around it, and the same surface is nowhere brighter in the image")
    elif pattern:
        raise RepairError("no shadow found in the stroke: what is dark in it continues "
                          "the pattern or the shape around it")
    elif info.get("belowRing", 0.0) > 0.02:
        raise RepairError("no shadow found in the stroke: nothing in it is clearly "
                          "darker than the surface around it (step %s, needed %s)"
                          % (("%.2f" % info["belowRing"]),
                             ("%.2f" % max(t1, 0.1))))
    else:
        raise RepairError("no shadow found in the stroke: nothing in it is darker "
                          "than the surface around it")
    G = np.clip(G, cap, 0.0)
    core = ins & (G.mean(-1) < -0.5 * t1)
    if core.sum() < 16:
        raise RepairError("no shadow found in the stroke")

    # --- the whole shadow, not only the stroke ---------------------------------
    # Where the stroke ends inside the shadow its rim used to stay: the
    # band under the balcony kept a lower edge, the pole's line went on
    # below the stroke. As a content-aware selection does, the region grows
    # out of the stroke along what is the same shadow on the same material
    # (`_grow_shadow`), and all of it is lifted.
    gk = np.median(G[core], 0)
    reach = 3.0 * half
    low1 = np.stack([_nconv(Lc[..., c], vw, s1) for c in range(3)], -1)
    grown, tw = _grow_shadow(low1, valid, occ, core, gk, reach)
    region = ins.copy()
    if grown.sum() >= 20:
        region |= grown
        near_r = ndimage.binary_dilation(region, iterations=int(np.ceil(2 * s1))) & valid
        G = np.where(ins[..., None], G, 0.0)
        # The grown region painted out as a whole: LaMa then continues the lit
        # surface from beyond the shadow's end, not the shadow itself -- with
        # only the stroke masked it carried the band under the balcony in
        # from the ring and found nothing to lift. A region larger than the
        # stroke is too much for it to paint, and so is a broad one beside
        # more shadow (the hut's sand, where it painted the gravel dark
        # again): there the lit level of the material lifts the grown part.
        done = False
        thick = float(ndimage.distance_transform_edt(grown).max())
        thin = thick <= max(3.0, 0.75 * half)
        info["grownWidth"] = round(thick, 1)
        if grown.sum() <= 1.5 * ins.sum() and (how == "lama" or thin):
            try:
                D2, lit2 = lama_deficit(ndimage.binary_dilation(region, iterations=g))
                c2 = region & valid & (D2.mean(-1) < -t1)
                # and LaMa must find the shadow about as deep as it is: on
                # the sand, beside more shadow, it painted the region half
                # dark again
                deep_ok = (how == "lama" or (c2.sum() >= 16 and
                           -float(np.median(D2.mean(-1)[c2])) >= 0.6 * float(-gk.mean())))
                if c2.sum() >= max(16, 0.3 * core.sum()) and deep_ok:
                    G = np.where(near_r[..., None], lama_lift(D2, c2, lit2), 0.0)
                    how = how if how == "lama" else "lama+" + how
                    done = True
            except Exception:                    # noqa: BLE001 -- no inpainter
                pass
        if not done and how == "lit":
            G = np.where(near_r[..., None], lit_layer, 0.0)
        G = np.clip(G, cap, 0.0)
        info["grown"] = round(float(grown.sum()) / max(float(ins.sum()), 1.0), 2)
        info["grownCut"] = bool((grown & (tw < 0.05)).sum() >= 10)
    else:
        G = np.where(ins[..., None], G, 0.0)
    G = G * float(np.clip(params.strength, 0, 1))
    # The soft rim of the whole region; where the growth stopped at its reach
    # (the shadow runs on) it fades over half the reach instead of ending.
    dist = ndimage.distance_transform_edt(ndimage.binary_dilation(region, iterations=2) | occ)
    Wt = _smoothstep(dist / max(params.feather * half, 1.5))
    Wt = np.where(occ, Wt, Wt * tw)
    core = ins & (G.mean(-1) < -0.5 * t1)
    if core.sum() < 16:
        raise RepairError("no shadow found in the stroke")
    info.update(how=how, gain=[round(float(x), 3) for x in np.exp(np.median(G[core], 0))])
    return G, info, Wt


def _grow_shadow(low, valid, occ, core, gk, reach):
    """The shadow's continuation outside the stroke: ``(grown mask, fade)``.

    ``low`` (W, W, 3) is the picture's low-passed log colour, ``core`` the
    stroke's pixels deep in the shadow and ``gk`` (3,) its pooled log gain.
    A pixel continues the shadow when its colour lies on the line from the
    core's shadowed colour to the colour it has when lit -- more than half
    way towards the shadow, and near that line, which a different material
    is not -- and it is connected to the core. At most ``reach`` pixels from
    the stroke. ``fade`` (W, W) is 1 well inside the reach and falls to 0 at
    it, so a shadow running on past it is lifted out softly, not cut.
    """
    gk = np.asarray(gk, np.float64)
    nn = float(gk @ gk)
    fade = np.ones(occ.shape)
    if nn < 1e-4 or core.sum() < 16:
        return np.zeros_like(occ), fade
    mu_s = np.median(low[core], 0)
    mu_l = mu_s - gk
    rel = low - mu_l
    tt = (rel @ gk) / nn
    res = np.linalg.norm(rel - tt[..., None] * gk, axis=-1)
    tau = max(0.06, 2.5 * float(np.median(res[core])))
    dist = ndimage.distance_transform_edt(~occ)
    # past six tenths: the grey shutter box under the balcony stands half
    # way between the band's shadow and its lit wall, and was taken along
    # off the line by less than its tolerance, which grows with the depth:
    # the deeper a sky-lit shadow, the bluer (the pole's below the stroke)
    cand = valid & ~occ & (tt > 0.6) & (res < tau * np.maximum(tt, 1.0)) & (dist <= reach)
    # a thin shadow fades for a few pixels where the stroke ends (the pole's
    # did, between two planks): bridge that
    cand = ndimage.binary_closing(cand | core, iterations=2) & valid & ~occ & (dist <= reach)
    # joined to the core through the stroke's own shadowed pixels: the core
    # is only the deep part, and stopped short of the stroke's end
    lb, nl = ndimage.label(cand | core | (occ & valid & (tt > 0.6)))
    if nl == 0:
        return np.zeros_like(occ), fade
    hit = np.unique(lb[core])
    grown = cand & np.isin(lb, hit[hit > 0])
    fade = _smoothstep((reach - dist) / (0.5 * reach))
    return grown, fade


def _picture_of_field(s, idx, lift, n, params, volume, field, W=384):
    """A shadow field in 3-D (``field(xyz) -> log gain``) as a picture fit.

    The frame is rendered with its depth, every pixel's surface point asks
    the field for its gain, and the splats seen are fitted to that picture.
    ``lift`` (len(idx), 3) is what the cells gave the painted splats; they
    start there, and keep it where the picture does not see them. Returns
    ``(indices, log lifts (n, 3))`` or None.
    """
    from . import gpu
    ok, _why = gpu.available()
    if not ok:
        return None
    F = _picture_frame(s, idx, n, params.feather, W, volume)
    if F is None:
        return None
    _rgb, alpha, dep = _picture_base(F, depth=True)
    cam = F["cam"]
    vv, uu = np.mgrid[0:W, 0:W]
    good = (alpha > 0.5) & np.isfinite(dep) & (dep > 0)
    if good.sum() < 16:
        return None
    pc = (np.linalg.inv(cam.K) @ np.stack([uu[good] + 0.5, vv[good] + 0.5,
                                           np.ones(int(good.sum()))])) * dep[good]
    xyz = (cam.view[:3, :3].T @ (pc - cam.view[:3, 3:4])).T
    L = np.zeros((W, W, 3))
    L[good] = np.maximum(-np.asarray(field(xyz), np.float64), 0.0)
    # the cells' field holds inside the stroke; outside it is the cells' guess
    L *= ndimage.binary_dilation(F["occ"], iterations=2)[..., None]
    if L.max() < 1e-3:
        return None
    lift = np.maximum(np.asarray(lift, np.float64), 0.0)
    return _picture_fit(s, F, idx, L, lift, np.log(params.lift_cap), init=lift)


def _unshadow_picture(s, idx, n, params, volume=None, W=384):
    """The shadow found and taken out on the picture (see `_picture_shadow`)."""
    from . import gpu
    ok, why = gpu.available()
    if not ok:
        raise RuntimeError(why)
    idx = np.unique(np.asarray(idx, np.int64))
    F = _picture_frame(s, idx, n, params.feather, W, volume, ctx=params.picture_ctx)
    if F is None:
        raise RepairError("the spot is not visible from here")
    rgb, alpha = _picture_base(F)
    G, pinfo, Wt = _picture_shadow(rgb, alpha, F, params)
    L = -G * Wt[..., None]
    own = _sample_map(F, L, s.xyz[idx])
    ix, lg = _picture_fit(s, F, idx, L, own, np.log(params.lift_cap), init=own)
    rgb_s = 0.5 + SH_C0 * s.f_dc[ix].astype(np.float64)
    # no channel past white (a lit grain in the stroke was never in shadow),
    # by one factor over the channels so the hue stays
    lg = _white_clamp(lg, np.log(np.clip(rgb_s, 1e-6, None)))
    # Lifted by one depth, the stroke keeps the contrast it had in the shadow;
    # its bright end takes the lit surface's spread. LaMa's lit picture has
    # the lit texture already: there it held the pole's line half in shadow.
    tail = None
    if pinfo.get("how") == "lit":
        lg, tail = _lit_tail(F, L, ix, lg, np.log(np.clip(rgb_s, 1e-6, None)), params.tail_from)
    if tail:
        pinfo["litTail"] = tail
    touched = lg.max(1) > 1e-3
    if not touched.any():
        raise RepairError("no shadow found in the stroke")
    ix, lg, rgb_s = ix[touched], lg[touched], rgb_s[touched]
    gain = np.exp(lg)
    f_dc = (rgb_s * gain - 0.5) / SH_C0
    f_rest = s.f_rest[ix].astype(np.float64)
    if f_rest.shape[1]:
        f_rest = f_rest * gain[:, None, :]
    added = SplatSet(
        xyz=s.xyz[ix].copy(), f_dc=f_dc.astype(np.float32),
        f_rest=f_rest.astype(np.float32), opacity=s.opacity[ix].copy(),
        scale=s.scale[ix].copy(), rot=s.rot[ix].copy(),
        extra={k: v[ix].copy() for k, v in s.extra.items()})
    warn = []
    lift = float(1.0 / min(pinfo["gain"]))
    if lift > 5.0:
        warn.append("lift ×%s: the noise in the shadow becomes visible"
                    % ("%.1f" % lift))
    if pinfo.get("grownCut"):
        warn.append("the shadow reaches further than the brush follows it; there the "
                    "brightening fades out softly; keep painting over the rest")
    elif pinfo["how"] == "lit" and not pinfo.get("grown"):
        warn.append("the shadow reaches beyond the stroke; the brush raised it to the light "
                    "of the same surface, the edge of the stroke may stay visible")
    info = {"mode": "unshadow", "edge": "picture", "shadow": True, "shadowWhyNot": None,
            "reflex": False, **pinfo, "lifted": int(len(ix)), "painted": int(len(idx)),
            "picture": True, "strength": float(params.strength),
            "warning": "; ".join(warn) or None}
    return RepairResult(removed=ix, added=added, info=info)


def _lift(s, selection, params, view=None, volume=None):
    """Brighten the painted splats by a set factor: the brush that always acts.

    Detection refuses most shadows baked into real scans, and for good
    reasons it cannot get round: on the hut a stroke over the shadow under
    the balcony holds a hundred splats, too few to find an edge in; the shoe's
    lace and stand shadows are soft and 0.1 to 0.3 log deep on a surface
    whose own grain is as large, which is exactly what the grain guard was
    built to refuse (it stopped intact sole being lifted by 13 per cent).
    What is left is what an image editor's dodge does: the user says how
    much, the brush does that, visibly, every time.

    * Brightness more than colour, hue kept: the channels' mean rises by the
      gain, the colour's distance from grey by the gain to the power
      ``lift_chroma`` (0.25: the Lab chroma of the sole stays 17.5 -> 17.8;
      times the whole gain it went to 24 and the patch read as peach). The
      view-dependent bands likewise, their grey part and the rest.
    * A soft edge, found where the stroke is and not where the surface is: the
      painted splats are laid flat along the painting view, their footprint
      rasterised and closed (a sparse wall leaves gaps between splats), and
      the factor ramps up with the distance from the footprint's rim over
      ``feather`` of its half width. No seam where the stroke ends.
    * Nothing past white: a splat is lifted at most until its brightest
      channel reaches one.
    * With a GPU, a dodge curve (`_lift_map`): the stroke's middle by the
      factor, darker parts more, lighter parts less, white not at all; the
      soft edge from a quarter of the half width inside the stroke to three
      quarters outside it (`_soft_rim`), as a soft brush tip.
    """
    idx = np.unique(np.asarray(selection, np.int64))
    if len(idx) == 0:
        raise RepairError("nothing painted")
    f = float(params.lift)
    if not np.isfinite(f) or f <= 1.005:
        raise RepairError("brightening is set to zero; raise the slider")
    f = min(f, float(params.lift_cap))
    xyz = s.xyz[idx].astype(np.float64)
    alpha = s.alpha[idx]
    if view is not None and np.linalg.norm(view) > 0:
        n = np.asarray(view, np.float64) / np.linalg.norm(view)
    elif len(idx) >= 3:
        n = np.linalg.svd(xyz - xyz.mean(0), full_matrices=False)[2][2]
    else:
        n = np.array([0.0, 0.0, 1.0])
    tmp = np.array([1.0, 0.0, 0.0]) if abs(n[0]) < 0.9 else np.array([0.0, 1.0, 0.0])
    e0 = np.cross(n, tmp)
    e0 /= np.linalg.norm(e0)
    e1 = np.cross(n, e0)
    uv = (xyz - xyz.mean(0)) @ np.stack([e0, e1], 1)
    # The footprint from what can be seen: faint haze scatters past the stroke.
    vis = alpha > 0.1
    if vis.sum() < 3:
        vis = np.ones(len(idx), bool)
    w_full = np.ones(len(idx))
    feather = 0.0
    if len(idx) >= 8:
        q = uv[vis]
        k = min(2, len(q) - 1)
        dnn = cKDTree(q).query(q, k=k + 1)[0][:, k]
        ext = float(np.max(np.ptp(q, axis=0)))
        cell = max(float(np.median(dnn)), ext / 400.0, 1e-9)
        lo = q.min(0) - 3 * cell
        nu, nv = (np.ceil((q.max(0) - lo) / cell).astype(int) + 4)
        iu = np.clip(((uv[:, 0] - lo[0]) / cell).astype(int), 0, nu - 1)
        iv = np.clip(((uv[:, 1] - lo[1]) / cell).astype(int), 0, nv - 1)
        occ = np.zeros((nv, nu), bool)
        occ[iv[vis], iu[vis]] = True
        occ = ndimage.binary_fill_holes(ndimage.binary_closing(
            np.pad(occ, 3), np.ones((3, 3)), iterations=2))[3:-3, 3:-3]
        dist = ndimage.distance_transform_edt(occ)
        half = float(dist.max())
        if half > 1.0:
            feather = max(params.feather * half, 1.5)
            w_full = _smoothstep(dist[iv, iu] / feather)
            # a splat outside the closed footprint (a faint one in the gap)
            # takes the ramp's floor, not nothing
            w_full = np.maximum(w_full, 0.0)
            feather *= cell
    n_painted = len(idx)
    reach = None
    if view is not None and np.linalg.norm(view) > 0 and len(idx) >= 8:
        try:
            reach = _picture_share(s, idx, n, w_full, np.log(f),
                                   np.log(params.lift_cap), params.feather,
                                   volume=volume)
        except Exception:                        # noqa: BLE001 -- no GPU: the painted splats only
            reach = None
    pinfo = {}
    if reach is not None:
        idx, lg, pinfo = reach
    else:
        lg = np.log(f) * w_full
    rgb = 0.5 + SH_C0 * s.f_dc[idx].astype(np.float64)
    # Brightness first, colour after (see below): a splat reaches white when
    # its brightest channel does.
    kc = float(np.clip(params.lift_chroma, 0.0, 1.0))
    Y = rgb.mean(1)
    dev = rgb - Y[:, None]
    grey = Y > 1e-3
    want = lg.copy()

    def top(x):
        return Y * np.exp(x) + (dev * np.exp(kc * x)[:, None]).max(1)

    lo_, hi_ = np.zeros_like(lg), lg.copy()
    over = top(hi_) > 1.0
    for _ in range(24):                          # the largest lift still under white
        if not over.any():
            break
        m_ = 0.5 * (lo_ + hi_)
        ok_ = top(m_) <= 1.0
        lo_ = np.where(over & ok_, m_, lo_)
        hi_ = np.where(over & ~ok_, m_, hi_)
    lg = np.where(over, lo_, lg)
    lg = np.where(grey, lg, 0.0)
    clamped = int((want - lg > 1e-3).sum())
    touched = lg > 1e-3
    if not touched.any():
        raise RepairError("the spot is already white, it cannot get any brighter")
    idx, lg, Y, dev = idx[touched], lg[touched], Y[touched], dev[touched]
    gain = np.exp(lg)
    # A dodge lifts brightness more than colour. Times the gain in every
    # channel the sole's warm brown went peach at x1.5: the ratios stayed,
    # so the colour's distance from grey grew by the whole gain, and the
    # stand's red cast in the bands with it. Now the brightness (the
    # channels' mean) rises by the gain and the distance from grey by the
    # gain to the power ``lift_chroma``; in the bands likewise their grey
    # part and the rest.
    cg = np.exp(kc * lg)
    f_dc = (Y[:, None] * gain[:, None] + dev * cg[:, None] - 0.5) / SH_C0
    f_rest = s.f_rest[idx].astype(np.float64)
    if f_rest.shape[1]:
        g_rest = f_rest.mean(2, keepdims=True)
        f_rest = g_rest * gain[:, None, None] + (f_rest - g_rest) * cg[:, None, None]
    added = SplatSet(
        xyz=s.xyz[idx].copy(), f_dc=f_dc.astype(np.float32),
        f_rest=f_rest.astype(np.float32), opacity=s.opacity[idx].copy(),
        scale=s.scale[idx].copy(), rot=s.rot[idx].copy(),
        extra={k: v[idx].copy() for k, v in s.extra.items()})
    share = clamped / max(len(want), 1)
    warn = None
    if share > 0.5:
        warn = ("the spot is already bright: %d %% of the splats were raised only to white"
                % round(100 * share))
    info = {"mode": "lift", "factor": round(f, 3),
            "meanGain": round(float(np.exp(lg).mean()), 3),
            "shadow": False, "reflex": False, "lifted": int(len(idx)),
            "painted": int(n_painted), "picture": reach is not None,
            "clamped": clamped, "feather": round(float(feather), 5),
            **pinfo, "warning": warn}
    return RepairResult(removed=idx, added=added, info=info)


# Refusals of the shadow part that a wide view can answer: a shadow was
# found and is deeper than the grain, only its edge pairs unlike surfaces.
_VIEW_REASONS = ("the surface changes at the shadow edge",
                 "there is no material on both sides of the shadow edge",
                 "a material in the shadow")


def _view_shadow(s, sel, P, params, W=768, sigma=3.0):
    """The shadow from a wide 2-D view: ``(xyz -> log gain, gain)``.

    The windowed pairing needs the same material lit and shadowed side by
    side along the edge. On a strongly curved crossing the edge runs along
    the knick, the windows pair smooth white midsole with dotted sole and
    the brush refused (71 per cent of the edge). A wide view knows more:
    rendered straight down the normal with three stroke radii of
    surroundings, LaMa paints the stroke as the lit layout around it --
    the strip, the leather, the sole continue -- and the picture's log
    deficit against that is the shadow, per pixel. On the curved bench
    site: refused -> 60.6 to 10.2 levels; from three views 52-60 -> 18.

    It is the fallback, not the method: where the windows work they are
    better (bench_shadow 3.3 against 4.5-9.7, the edge sites 6.7-11.3
    against 7.0-15.4), because LaMa's texture is not the surface's and
    some of it rides on the deficit.
    """
    import torch
    from . import gpu
    from .deep import load_inpainter
    normal = P.basis[2] * P.outward
    idx = np.where(sel)[0]
    xyz = s.xyz[idx].astype(np.float64)
    w = np.clip(s.alpha[idx], 1e-3, None)
    centre = (xyz * w[:, None]).sum(0) / w.sum()
    d = xyz - centre
    rad = float(np.percentile(np.linalg.norm(d - np.outer(d @ normal, normal), axis=1), 98))
    fov = 30.0
    dist = rad * params.view_context / np.tan(np.radians(fov / 2))
    cam = gpu.look_at_camera(centre + normal * dist, centre, P.basis[1], W, W, fov)
    # Only the surface and what sits on it: anything between the eye and the
    # surface would be painted into the "lit" picture (see viewfill).
    from .viewfill import view_subset
    sub = view_subset(s, centre, normal, rad * params.view_context, idx)
    shown = s.take(np.isin(np.arange(s.n), sub)) if len(sub) < s.n else s
    tt = gpu.to_tensors(shown, torch.device("cuda"))
    with torch.no_grad():
        img, a, _ = gpu.render(tt, [cam], sh_degree=0, antialiased=True, with_depth=False)
    rgb = gpu.to_uint8(img[0])
    alpha = a[0, ..., 0].cpu().numpy()

    def project(p):
        q = np.asarray(p, np.float64) @ cam.view[:3, :3].T + cam.view[:3, 3]
        uv = q @ cam.K.T
        z = np.where(np.abs(uv[:, 2]) > 1e-9, uv[:, 2], 1e-9)
        return uv[:, 0] / z - 0.5, uv[:, 1] / z - 0.5, q[:, 2]

    u, v, z = project(s.xyz[idx[s.alpha[idx] > 0.2]])
    m = np.zeros((W, W), bool)
    ok = (u >= 0) & (u < W - 1) & (v >= 0) & (v < W - 1) & (z > 0)
    m[np.round(v[ok]).astype(int), np.round(u[ok]).astype(int)] = True
    g = max(2, W // 170)
    m = ndimage.binary_closing(ndimage.binary_dilation(m, iterations=g), iterations=2 * g)
    m = ndimage.binary_fill_holes(m)
    if m.sum() < 64 or m.mean() > 0.6:
        raise RepairError("the spot does not fit in the view")
    # the mask is grown so the penumbra's rim is not taken as lit context
    fn, _ = load_inpainter("lama")
    mk = ndimage.binary_dilation(m, iterations=3 * g)
    lit = fn(rgb, (mk * 255).astype(np.uint8))
    D = (np.log(np.clip(rgb.astype(np.float64) / 255.0, 0.01, 1.0))
         - np.log(np.clip(lit.astype(np.float64) / 255.0, 0.01, 1.0)))
    mm = (m & (alpha > 0.5)).astype(float)

    def smooth(x, sg):
        return ndimage.gaussian_filter(x * mm, sg) / np.maximum(ndimage.gaussian_filter(mm, sg), 1e-6)

    Ds = np.stack([smooth(D[..., c], sigma) for c in range(3)], -1)
    Ds = np.where(m[..., None], np.clip(Ds, -np.log(params.lift_cap), 0.0), 0.0)
    # only where the broad deficit says shadow: lit texture in the stroke
    # keeps its own dark phases (lit error 0.010 -> 0.003 on the curved site)
    Ds = Ds * np.clip(-smooth(D.mean(-1), 4 * sigma) / 0.1, 0.0, 1.0)[..., None]
    core = m & (alpha > 0.5) & (Ds.mean(-1) < -0.06)
    if core.sum() < 50:
        raise RepairError("no shadow found in the stroke")
    gain = np.exp(np.median(Ds[core], 0))
    # No pixel lifts past the shadow's own gain: a dark logo in the shadow
    # is a deficit LaMa paints away too, and was lifted 0.90 log past its
    # truth; capped at the core's median it is lifted by the shadow only.
    Ds = np.maximum(Ds, params.view_cap * np.log(gain)[None, None, :])

    def at(p):
        uu, vv, zz = project(p)
        out = np.stack([ndimage.map_coordinates(Ds[..., c], [vv, uu], order=1, mode="constant",
                                                cval=0.0) for c in range(3)], 1)
        out[zz <= 0] = 0.0
        return out * float(np.clip(params.strength, 0, 1))

    return at, gain


def _faint_under(s, P, alpha, cand):
    """The faint splats in the slab under the stroke: ``(index, uvh)``.

    The patch keeps only splats over 0.2 opacity -- the surface is found
    and measured on those. But a scan bakes a shadow into every splat that
    made the pixel, and the translucent layers are many, most of all at a
    crease: on the curved bench site 488 faint splats under the stroke kept
    the whole bluish shadow, 7 to 10 per cent of what is seen there from
    three views -- the glassy blue patch at the fold. They get the gain of
    where they lie. Their depth is taken against the face of their own
    cell, with the slab's bounds, so haze in front and a far wall stay out.
    Image error from three views, curved site 16.8 -> 11.8; on every other
    bench site lower too, the benches' numbers (opaque splats) unchanged.
    """
    stroke, nv, nu = P.stroke, P.nv, P.nu
    lo_u, lo_v = P.lo
    xyz = s.xyz
    pts = xyz[cand]
    lo, hi = pts.min(0), pts.max(0)
    inc = np.zeros(s.n, bool)
    inc[cand] = True
    pick = np.where(np.all((xyz >= lo) & (xyz <= hi), axis=1) & ~inc & (alpha <= 0.2))[0]
    if len(pick) == 0:
        return pick, np.zeros((0, 3))
    uvh = (xyz[pick].astype(np.float64) - P.origin) @ P.basis.T
    iu = np.floor((uvh[:, 0] - lo_u) / P.cell).astype(np.int64)
    iv = np.floor((uvh[:, 1] - lo_v) / P.cell).astype(np.int64)
    ok = (iu >= 0) & (iu < nu) & (iv >= 0) & (iv < nv)
    ok[ok] = stroke[iv[ok], iu[ok]]
    pick, uvh, iu, iv = pick[ok], uvh[ok], iu[ok], iv[ok]
    # the face's height per cell, as the patch measured it
    from .repair import _fill_nearest
    hf = np.full(nv * nu, np.nan)
    hf[P.kv * nu + P.ku] = P.uvh[:, 2] - P.outward * P.depth
    hf = hf.reshape(nv, nu)
    hf = _fill_nearest(np.where(np.isfinite(hf), hf, 0.0), np.isfinite(hf))
    depth = P.outward * (uvh[:, 2] - hf[iv, iu])
    sp = P.spacing
    keep = (np.abs(depth) < 3.0 * sp) | ((depth < 0) & (depth > -10.0 * sp))
    return pick[keep], uvh[keep]


def _cell_normals(uvh, face, kv, ku, nv, nu, cell, smooth=3):
    """Unit surface normal per cell, (u, v, h) components, from the face height."""
    from .repair import _fill_nearest
    h = np.zeros((nv, nu))
    c = np.zeros((nv, nu))
    np.add.at(h, (kv[face], ku[face]), uvh[face, 2])
    np.add.at(c, (kv[face], ku[face]), 1.0)
    has = c > 0
    h = _fill_nearest(np.where(has, h / np.maximum(c, 1), 0.0), has)
    if smooth > 1:
        h = ndimage.uniform_filter(h, smooth, mode="nearest")
    gy, gx = np.gradient(h / cell)
    n = np.stack([-gx, -gy, np.ones_like(gx)], -1)
    return n / np.linalg.norm(n, axis=-1, keepdims=True)


def _smoothstep(x):
    x = np.clip(x, 0.0, 1.0)
    return x * x * (3.0 - 2.0 * x)


def _labels(Lc, known, stroke, params, comp=None, rng=None):
    """Material labels per cell, or -1 where nothing is known.

    Features: low-passed log-chromaticity (blind to a neutral shadow; ``comp``
    subtracts a known coloured shadow's chroma shift) and the local roughness
    of the log brightness. One reference across brown tread and white
    midsole blended the two under the stroke, Otsu then found the material
    edge, and the brush refused or lifted brown by 0.3 log; per material the
    reference stays on its own side.
    """
    nv, nu, _ = Lc.shape
    chroma = Lc - Lc.mean(-1, keepdims=True)
    if comp is not None:
        chroma = chroma - comp
    lum = Lc.mean(-1)
    m = known.astype(np.float64)
    sm = params.chroma_smooth
    den = np.maximum(ndimage.uniform_filter(m, sm), 1e-9)
    low = np.stack([ndimage.uniform_filter(chroma[..., c] * m, sm) / den
                    for c in range(3)], -1)
    lum_low = ndimage.uniform_filter(lum * m, sm) / den
    rough = np.sqrt(np.maximum(ndimage.uniform_filter(lum ** 2 * m, sm) / den - lum_low ** 2, 0))
    X = np.concatenate([low, 0.5 * rough[..., None]], -1)
    pts = X[known]
    k = params.k_labels
    rng = rng or np.random.default_rng(params.seed)
    # k-means++ seeding, a dozen Lloyd steps
    cent = [pts[rng.integers(len(pts))]]
    for _ in range(k - 1):
        d2 = np.min([((pts - c) ** 2).sum(1) for c in cent], 0)
        cent.append(pts[rng.choice(len(pts), p=d2 / d2.sum())])
    cent = np.array(cent)
    for _ in range(15):
        lab = np.argmin(((pts[:, None] - cent[None]) ** 2).sum(-1), 1)
        for j in range(k):
            if np.any(lab == j):
                cent[j] = pts[lab == j].mean(0)
    lab = np.argmin(((pts[:, None] - cent[None]) ** 2).sum(-1), 1)
    # Two clusters that are halves of one continuum -- the dark and light
    # phases of a tread, or a coloured shadow against its lit material -- are
    # not two materials. Materials stand apart by several within-spreads
    # (brown/white on the shoe: 5 to 10); a split texture by 2 to 3. Merge
    # the closest pair while it is closer than ``label_sep``.
    alive = list(range(k))
    while len(alive) > 1:
        best = None
        for ia, a in enumerate(alive):
            for b in alive[ia + 1:]:
                pa, pb = pts[lab == a], pts[lab == b]
                if len(pa) < 2 or len(pb) < 2:
                    sep = 0.0
                else:
                    u = pb.mean(0) - pa.mean(0)
                    d = np.linalg.norm(u)
                    u = u / max(d, 1e-9)
                    sep = d / max(np.sqrt(0.5 * ((pa @ u).var() + (pb @ u).var())), 1e-9)
                if best is None or sep < best[0]:
                    best = (sep, a, b)
        if best[0] >= params.label_sep:
            break
        lab[lab == best[2]] = best[1]
        alive.remove(best[2])
    for a in alive:
        cent[a] = pts[lab == a].mean(0)
    cent = cent[alive]
    k = len(alive)
    full = np.argmin(((X.reshape(-1, X.shape[-1])[:, None] - cent[None]) ** 2).sum(-1), 1).reshape(nv, nu)
    full = np.where(known, full, -1)
    # A label is a material only if it reaches the stroke's immediate
    # surroundings: the shadow is inside the stroke, a material crosses its
    # edge. (A wide ring was tried: dark cells elsewhere in it 'supported'
    # the shadow's own label.) An unsupported label joins the neighbour it
    # shares the longest border with -- a shadow is surrounded by its own
    # lit material.
    ring = ndimage.binary_dilation(stroke, np.ones((3, 3)), iterations=3) & ~stroke & known
    support = np.array([(full[ring] == j).sum() for j in range(k)])
    valid = support >= params.label_min_ring
    if not valid.any():
        valid[np.argmax(support)] = True
    while not valid.all():
        j = int(np.where(~valid)[0][0])
        mask = full == j
        grown = ndimage.binary_dilation(mask, np.ones((3, 3))) & ~mask & known
        border = np.array([(full[grown] == q).sum() if valid[q] else -1 for q in range(k)])
        target = int(np.argmax(border))
        full[mask] = target
        valid[j] = True
        support[target] += support[j]
        support[j] = 0
    valid = support >= params.label_min_ring
    # a lone cell's label is noise: majority over 3x3 among known
    labs = np.where(valid)[0]
    votes = np.stack([ndimage.uniform_filter((full == j).astype(float), 3) for j in labs], -1)
    out = np.where(known, labs[np.argmax(votes, -1)], -1)
    return out, labs, cent, support


def _otsu(vals, bins=64):
    hist, edges = np.histogram(vals, bins=bins)
    mids = 0.5 * (edges[1:] + edges[:-1])
    c = np.cumsum(hist)
    tot = max(c[-1], 1)
    cs = np.cumsum(hist * mids)
    w0 = c / tot
    mu0 = cs / np.maximum(c, 1)
    mu1 = (cs[-1] - cs) / np.maximum(tot - c, 1)
    between = w0 * (1 - w0) * (mu0 - mu1) ** 2
    return float(mids[int(np.argmax(between[:-1]))])


def _band_excess(s, P, cand, face, kv, ku, nv, nu, stroke, alpha, gain, params,
                 view=None):
    """The view-dependent bands' local excess under the stroke, or None.

    Returns ``(A, info)``: ``A`` is (nv, nu, 1 + 3K), per cell a matte and the
    surrounding surface's SH bands carried in across the stroke. Where the
    matte is one a splat's own bands are replaced by the surroundings'; it is
    non-zero only where the colour those bands give *from the painting view*
    stands clearly out from what the surroundings give from there, with a
    soft edge. None when no part of the stroke does.

    Replaced, not corrected. Subtracting the cell-mean excess took the
    stand's red cast from 0.72 to 0.71 (red over brightness; the sole beside
    it 0.56): the cast is not a shift shared by a cell's splats but sits in
    a few of them, at a finer grain than any cell mean. Setting the bands of
    the painted splats to zero took it to 0.52 -- so the bands are where it
    lives, and the surroundings' bands are what the surface there should
    have.

    Judged in colour, from the view, and cell by cell. The forty-five raw
    coefficients averaged over the stroke were the first try: on this scan
    they vary a lot from cell to cell everywhere, a reflection covers only
    part of a generous stroke, and the red cast beside the stand came out at
    1.37 times the surroundings' own variation -- indistinguishable. What the
    eye sees is the colour the bands add from where it is looking.
    """
    from .shtools import sh_basis
    K = s.f_rest.shape[1]
    if K == 0 or not params.reflex:
        return None, {}
    deg = int(round(np.sqrt(K + 1))) - 1
    # From the painting view and from a ring of views around the surface's
    # normal. A reflection moves with the eye: judged from one direction
    # only, the stand's cast was gone from where it was painted and still
    # red a step to the side.
    look = -P.basis[2] * P.outward
    a1, a2 = P.basis[0], P.basis[1]
    dirs = [look]
    for tilt in (30.0, 55.0):
        for az in np.arange(6) * np.pi / 3:
            r = np.deg2rad(tilt)
            dirs.append(np.cos(r) * look + np.sin(r) * (np.cos(az) * a1 + np.sin(az) * a2))
    if view is not None and np.linalg.norm(view) > 0:
        v = np.asarray(view, float) / np.linalg.norm(view)
        if float(v @ look) > 0.1:
            dirs[0] = v
    D = np.array(dirs)
    D /= np.linalg.norm(D, axis=1, keepdims=True)
    Y = sh_basis(D, deg)[:, 1:]                                # (m, K)
    m_dirs = len(D)

    f = np.where(face)[0]
    R = s.f_rest[cand[f]].astype(np.float64) / gain[f][:, None, :]   # (n, K, 3)
    vd = np.einsum("mk,nkc->nmc", Y, R).reshape(len(f), -1)    # colour from each view
    F = np.concatenate([vd, R.reshape(len(f), -1)], 1)
    w = np.clip(alpha[cand[f]], 0.05, 1.0)
    sumF, _ = _accum(F * w[:, None], kv[f], ku[f], nv, nu)
    sumw, cnt = _accum(w, kv[f], ku[f], nv, nu)
    known = cnt > 0
    m = known.astype(float)
    den = ndimage.uniform_filter(m, 5)
    Fs = (ndimage.uniform_filter(sumF / np.maximum(sumw, 1e-9)[..., None] * m[..., None],
                                 (5, 5, 1)) / np.maximum(den, 1e-9)[..., None])
    away = ~ndimage.binary_dilation(stroke, np.ones((3, 3)), iterations=2)
    ref = fit_reference_surface(Fs, known & away, smooth_cells=params.tone_smooth,
                                plate=False)
    dev = Fs - ref
    devV = np.linalg.norm(dev[..., :3 * m_dirs].reshape(nv, nu, m_dirs, 3), axis=-1)
    ring = (ndimage.binary_dilation(stroke, np.ones((3, 3)), iterations=6)
            & ~stroke & known)
    inside = stroke & (den > 0.2)
    if ring.sum() < 20 or inside.sum() < 20:
        return None, {}
    # The surroundings' own variation, measured the way the stroke is: the
    # reference above passes close to the ring cells it was fitted to, and
    # their deviation from it understated the noise so much that a clean
    # surface scored 3.5-4.1 and the stand's cast 4.7. Fitted without a
    # band around the stroke, the band is carried across like the stroke.
    vv = slice(0, 3 * m_dirs)
    far = ~ndimage.binary_dilation(stroke, np.ones((3, 3)), iterations=8)
    band_cv = ~far & away & known
    ref_cv = fit_reference_surface(Fs[..., vv], known & far,
                                   smooth_cells=params.tone_smooth, plate=False)
    dev_cv = np.linalg.norm((Fs[..., vv] - ref_cv).reshape(nv, nu, m_dirs, 3), axis=-1)
    if band_cv.sum() < 20:
        return None, {}
    noise = np.sqrt((dev_cv[band_cv] ** 2).mean(0))            # per view
    z = (devV / np.maximum(noise, 1e-9)).max(-1)
    # A reflection is one coherent patch, noise is speckle. Per cell the two
    # overlap: thirteen views over thousands of cells put the largest noise
    # cell of a clean surface at 3.2-3.8 times the noise, the stand's cast
    # at 5.0. As patches they do not: the largest connected run above twice
    # the noise held at most 39 cells on clean surfaces (a summed excess
    # under 20), the cast 495 (540). A patch counts by that summed excess.
    lab, n = ndimage.label(inside & (z > 2.0))
    mass = (ndimage.sum(z - 2.0, lab, np.arange(1, n + 1)) if n else np.zeros(0))
    peak = (ndimage.maximum(z, lab, np.arange(1, n + 1)) if n else np.zeros(0))
    good = 1 + np.where((mass >= params.reflex_mass) & (peak >= params.reflex_min))[0]
    hot = np.isin(lab, good)
    hot = ndimage.binary_closing(hot, np.ones((3, 3))) & inside
    info = {"reflexPeak": round(float(z[inside].max()), 2),
            "reflexMass": round(float(mass.max()), 1) if n else 0.0,
            "reflexCells": int(hot.sum())}
    if hot.sum() < 10:
        return None, info
    # A reflection is coloured; a material's own sheen is mostly brightness.
    # Where the stroke crosses two materials -- dotted rubber, white leather
    # -- the carried-in bands blend them, and the difference read as a
    # reflex on all four two-material bench sites, shadowed or not (mass 250
    # to 940; the stand's red cast 151): the bands of 340 to 930 cells were
    # replaced by the blend, the rubber lost its sparkle and a dark dotted
    # ghost stayed, image error up by as much as 4.4 levels. As the share of
    # the excess that is colour they part: 0.16 to 0.22 there, 0.36 for the
    # stand's cast, 0.82 for a red lobe.
    dv = dev[..., :3 * m_dirs].reshape(nv, nu, m_dirs, 3)[hot]
    chroma = (np.linalg.norm(dv - dv.mean(-1, keepdims=True), axis=-1).sum()
              / max(float(np.linalg.norm(dv, axis=-1).sum()), 1e-12))
    info["reflexChroma"] = round(float(chroma), 2)
    if chroma < params.reflex_chroma:
        info["reflexCells"] = 0
        return None, info
    # soft edge: replaced fully inside, fading over three cells outside it
    # (a smoothstep on the distance -- a membrane matte leaks past its edge)
    t_out = np.clip(ndimage.distance_transform_edt(~hot) / 3.0, 0.0, 1.0)
    matte = np.where(stroke, 1.0 - t_out * t_out * (3 - 2 * t_out), 0.0)
    A = np.concatenate([matte[..., None], ref[..., 3 * m_dirs:]], -1)
    return A, info


def unshadow(s: SplatSet, selection, params: Optional[ShadowParams] = None,
             view=None, volume=None) -> RepairResult:
    """Lift the shadow inside the painted ``selection``.

    Returns a :class:`RepairResult` whose ``removed`` are the splats that
    changed colour and whose ``added`` are those same splats with their
    colour corrected -- positions, shapes and opacities untouched.

    ``params.mode`` "lift" brightens instead of detecting (`_lift`).
    Where the cells refuse, the stroke is looked at as a picture along
    ``view`` (`_unshadow_picture`); ``volume``, the painted volume, gives
    that picture its mask.
    """
    params = params or ShadowParams()
    if params.mode == "lift":
        return _lift(s, selection, params, view, volume)
    if params.mode != "auto":
        raise ValueError("shadow mode must be auto or lift, not %r" % params.mode)
    rng = np.random.default_rng(params.seed)
    sel = np.zeros(s.n, bool)
    sel[selection] = True
    if not sel.any():
        raise RepairError("nothing painted")
    alpha = s.alpha
    P = surface_patch(s, sel, rng)
    cand, uvh, ku, kv, nu, nv = P.cand, P.uvh, P.ku, P.kv, P.nu, P.nv
    lo_u, lo_v = P.lo
    cell, face, body, stroke = P.cell, P.face, P.body, P.stroke

    # --- where the stroke is, and whether it is on the surface ------------
    face_cells = np.zeros((nv, nu), bool)
    face_cells[kv[face], ku[face]] = True
    # Against the surroundings, not against one. On a sparse surface a third
    # of the cells hold nothing by chance, everywhere; a stroke that runs off
    # the surface holds markedly less than the surface around it does.
    ring = (ndimage.binary_dilation(stroke, np.ones((3, 3)), iterations=6)
            & ~stroke)
    occ_in = float(face_cells[stroke].mean()) if stroke.any() else 0.0
    occ_ring = float(face_cells[ring].mean()) if ring.any() else 1.0
    coverage = occ_in / max(occ_ring, 1e-6)
    if coverage < params.min_coverage:
        raise RepairError(
            "the stroke runs off the surface (only %d %% covered); "
            "stay on the surface" % round(100 * coverage))

    # A hole is not a shadow. Where an object stood, what is left under the
    # stroke is dark and lies far below where the surrounding surface would
    # continue; lifting it brightens the inside of a hole, which the brush did
    # fourfold over the stand's. Per stroke cell, the outermost material
    # against the carried-in face: a hole is more than 25 spacings under it
    # (the stand's: a median of 148), curvature is within about 13.
    ok_s = alpha[cand] > 0.2
    lin = (kv * nu + ku)
    outer = np.full(nv * nu, -np.inf)
    np.maximum.at(outer, lin[ok_s], P.depth_carried[ok_s] / P.spacing)
    # a cell empty by chance takes its neighbours' value; a hole is wider
    outer = ndimage.maximum_filter(outer.reshape(nv, nu), size=3, mode="nearest")
    hole = float((outer[stroke] < -25.0).mean()) if stroke.any() else 0.0
    if hole > 0.3:
        raise RepairError(
            "there is a hole under the stroke, not a shadow (%d %% of the spot); "
            "fill it with the Repair Brush first" % round(100 * hole))

    # Per cell, how far the face lies from where the surroundings say the
    # surface goes on, in spacings -- the hole guard's measure, as a mean.
    fl = lin[face]
    off_s = np.zeros(nv * nu)
    off_n = np.zeros(nv * nu)
    np.add.at(off_s, fl, np.abs(P.depth_carried[face]) / P.spacing)
    np.add.at(off_n, fl, 1.0)
    off_cell = np.where(off_n > 0, off_s / np.maximum(off_n, 1), np.nan).reshape(nv, nu)

    cells = {}                        # what the cells measured, for the picture's guard

    def shadow_field():
        """The base-colour shadow as a dict, or a RepairError saying why not."""
        # --- per-cell log colour ------------------------------------------------
        f = face
        L = np.log(np.clip(s.rgb[cand[f]], 0.02, 1.0))
        w = np.clip(alpha[cand[f]], 0.05, 1.0)
        sumL, _ = _accum(L * w[:, None], kv[f], ku[f], nv, nu)
        sumw, cnt = _accum(w, kv[f], ku[f], nv, nu)
        Lc = sumL / np.maximum(sumw, 1e-9)[..., None]
        known = cnt > 0
        pen = int(max(params.penumbra, 1))
        st = int(max(params.strip, 1))
        # Where to measure is not how soft to blend. Strips right at the edge
        # pick up the penumbra and the threshold's own wobble; set back a few
        # cells they see clean shadow and clean light.
        gp = int(max(params.gap, pen))
        lum = Lc.mean(-1)

        comp = None
        for pas in range(2):
            # --- the lit reference, one per material ---------------------------
            lab, labs, _, _ = _labels(Lc, known, stroke, params, comp, rng)
            # The lit reference per material. A membrane alone smears the
            # surface's own shading: on a curved surface one material is
            # brighter where it faces the light and darker where it turns
            # away, and the membrane carried that step under the shadow as if
            # it were the shadow (0.10 log too dark on its own ring, 0.43 in
            # the core). So per material the log colour is first regressed on
            # the cell's surface normal -- curvature shading follows the
            # normal, a cast shadow does not -- and the membrane carries only
            # what is left.
            T = np.zeros_like(Lc)
            nrm = (_cell_normals(uvh, face, kv, ku, nv, nu, cell)
                   if params.shading else None)

            def lit(kn):
                if nrm is None or kn.sum() < 30:
                    return fit_reference_surface(Lc, kn, smooth_cells=params.tone_smooth,
                                                 plate=False)
                X = np.concatenate([np.ones((nv, nu, 1)), nrm], -1)
                S = np.zeros_like(Lc)
                for c in range(3):
                    y, A, wgt = Lc[..., c][kn], X[kn], np.ones(int(kn.sum()))
                    for _ in range(3):                     # Cauchy at two MADs
                        beta = np.linalg.lstsq(A * wgt[:, None], y * wgt, rcond=None)[0]
                        r = y - A @ beta
                        sc = 1.4826 * np.median(np.abs(r - np.median(r))) + 1e-6
                        wgt = 1.0 / np.sqrt(1.0 + (r / (2.0 * sc)) ** 2)
                    S[..., c] = X @ beta
                return fit_reference_surface(Lc - S, kn, smooth_cells=params.tone_smooth,
                                             plate=False) + S

            plain = lit(known & ~stroke)
            for j in labs:
                kn = known & ~stroke & (lab == j)
                Tj = lit(kn) if kn.sum() >= 8 else plain
                T[lab == j] = Tj[lab == j]
            T[lab < 0] = plain[lab < 0]

            # --- find the shadow's edge inside the stroke ----------------------
            # The deficit against the lit tone, smoothed over a few cells: a
            # single cell's deficit is mostly texture -- this sole's own log
            # brightness varies half as much as the shadow being looked for.
            dev3 = Lc - T
            dev = np.where(known, dev3.mean(-1), 0.0)
            m_ = (known & stroke).astype(float)
            dev_s = (ndimage.uniform_filter(dev * m_, 5)
                     / np.maximum(ndimage.uniform_filter(m_, 5), 1e-9))
            vals = dev_s[known & stroke]
            how, sep, lit_frac, core = "stroke", float("nan"), float("nan"), None
            if params.edge == "detect" and len(vals) >= 40:
                thr = _otsu(vals)
                lo, hi = vals[vals < thr], vals[vals >= thr]
                if len(lo) > 2 and len(hi) > 2:
                    lit_frac = len(hi) / len(vals)
                    sep = (hi.mean() - lo.mean()) / max(np.sqrt(0.5 * (lo.var() + hi.var())), 1e-9)
                    # A real split: both sides substantial, clearly apart, and
                    # the dark side darker by a margin worth calling a shadow.
                    # Otherwise Otsu has just halved the texture, and the
                    # stroke's own edge is the better guess.
                    if (0.12 <= lit_frac <= 0.88 and sep > 2.0
                            and hi.mean() - lo.mean() > params.min_depth):
                        cc = stroke & (dev_s < thr)
                        cc = ndimage.binary_closing(ndimage.binary_opening(cc, np.ones((3, 3))),
                                                    np.ones((3, 3)))
                        # A shadow is one region. Otsu on a patterned surface
                        # happily splits the pattern into its dark and light
                        # halves -- dozens of blobs, every one "shadow" -- and
                        # the brush then brightened plain texture by 1.8.
                        lb, nlab = ndimage.label(cc)
                        if nlab:
                            sizes = np.bincount(lb.ravel())[1:]
                            if sizes.max() >= 0.6 * sizes.sum():
                                how = "detect"
                                core = lb == (int(np.argmax(sizes)) + 1)
            if how == "stroke":
                core = stroke & (ndimage.distance_transform_edt(stroke) > pen + 1)
            if core.sum() < 6:
                raise RepairError("no shadow found in the stroke")
            # A shadow lies on the surface. A stroke over the hut's sand along
            # its wall found the dark gap under the hut -- 5.2 times darker
            # than the rest, its face a median 46 spacings off the surface the
            # sand around it continues -- and lifted it x4.9. Bench shadow
            # cores lie 1.2 to 19.1 spacings off, holes from 25 on.
            oc = off_cell[core & np.isfinite(off_cell)]
            if len(oc):
                cells["off"] = float(np.median(oc))
            if len(oc) and float(np.median(oc)) > params.off_surface:
                raise RepairError(
                    "the dark part does not lie on the surface (%d spacings away): "
                    "a gap or another part, not a shadow" % round(float(np.median(oc))))
            # A shadow is a level shift larger than the surface's own grain.
            # On lit tread Otsu found a genuinely darker half -- 0.14 log, ten
            # times what a shifted core measures -- but the tread's own grain
            # is 0.26, and the brush lifted intact sole by 13 percent. The
            # smoothed deficit cannot tell, its texture is averaged away and
            # the separation test passed at 2.6. Real shadows stand 2 to 5
            # grains deep. Medians, so a logo inside the core does not move
            # the step.
            step_grain = None
            if params.min_step_grain > 0:
                ring_g = ndimage.binary_dilation(stroke, np.ones((3, 3)), iterations=6)
                g_in = core & known
                g_out = ring_g & ~core & known
                if g_in.sum() >= 6 and g_out.sum() >= 6:
                    step = float(np.median(dev[g_out]) - np.median(dev[g_in]))
                    grain = float(np.sqrt(0.5 * (dev[g_in].var() + dev[g_out].var())))
                    step_grain = step / max(grain, 1e-6)
                    if step_grain < params.min_step_grain:
                        raise RepairError(
                            "no shadow found in the stroke (step %s times the grain)"
                            % ("%.1f" % step_grain))

            # --- the gain, from local windows along the edge -------------------
            # Each edge cell gets a window: the strip cells either side, of the
            # edge cell's own material, whose nearest edge cell lies within
            # ``window`` cells of it. Its ratio is the median deficit inside
            # minus outside -- same material, same place, different light;
            # on the deficit, the reference's level error cancels in the
            # difference and its gradient in the deficit. The texture guard
            # is per window, so a material edge crossing the shadow spoils
            # only the windows that straddle it, and the gain is the weighted
            # median over the good ones. One global pair of strips was pulled
            # +0.19 log by a black logo in the inner strip.
            d_in = ndimage.distance_transform_edt(core)
            d_out = ndimage.distance_transform_edt(~core)
            s_in = core & (d_in > gp) & (d_in <= gp + st) & known
            s_out = ~core & (d_out > gp) & (d_out <= gp + st) & known
            if s_in.sum() < 6 or s_out.sum() < 6:
                s_in, s_out = core & known, ~core & known & (d_out <= pen + 3 * st)
            edge = core & (d_in <= 1.0)
            ey, ex = np.where(edge)
            if len(ey) < 6:
                raise RepairError("no shadow found in the stroke")
            tree = cKDTree(np.stack([ey, ex], 1))
            _, (iy, ix) = ndimage.distance_transform_edt(~edge, return_indices=True)
            lin_e = np.full((nv, nu), -1, int)
            lin_e[ey, ex] = np.arange(len(ey))
            own = lin_e[iy, ix]
            win = tree.query_ball_point(np.stack([ey, ex], 1), params.window)
            sy, sx = np.where(s_in | s_out)
            own_s, inside_s, lab_s = own[sy, sx], core[sy, sx], lab[sy, sx]
            order = np.argsort(own_s)
            own_sorted = own_s[order]
            starts = np.searchsorted(own_sorted, np.arange(len(ey)), "left")
            ends = np.searchsorted(own_sorted, np.arange(len(ey)), "right")
            vals_e = np.zeros((len(ey), 3))
            tex_e = np.full(len(ey), np.nan)
            n_e = np.zeros(len(ey), int)
            lab_e = lab[ey, ex]
            for e in range(len(ey)):
                if not win[e]:
                    continue
                members = np.concatenate([order[starts[q]:ends[q]] for q in win[e]])
                mem = members[lab_s[members] == lab_e[e]]
                ia, ib = mem[inside_s[mem]], mem[~inside_s[mem]]
                if len(ia) >= 10 and len(ib) >= 10:
                    vals_e[e] = (np.median(dev3[sy[ia], sx[ia]], 0)
                                 - np.median(dev3[sy[ib], sx[ib]], 0))
                    la, lb_ = lum[sy[ia], sx[ia]], lum[sy[ib], sx[ib]]
                    # A shadow multiplies: in log space it shifts a texture
                    # without changing its spread; a change of material
                    # changes the spread too.
                    tex_e[e] = (np.std(la - np.median(la))
                                / max(np.std(lb_ - np.median(lb_)), 1e-6))
                    n_e[e] = min(len(ia), len(ib))
            ok_e = n_e > 0
            if ok_e.sum() < 6:
                raise RepairError("there is no material on both sides of the shadow edge")
            good_e = ok_e & (tex_e >= params.texture_low_w) & (tex_e <= params.texture_tol_w)
            if how == "detect" and good_e.sum() < max(6, params.min_good_windows * ok_e.sum()):
                tex = float(np.median(tex_e[ok_e]))
                raise RepairError(
                    "the surface changes at the shadow edge (texture ×%s, %d %% of the edge); "
                    "paint over one surface only"
                    % (("%.1f" % tex),
                       round(100 * (1 - good_e.sum() / ok_e.sum()))))
            if not good_e.any():
                good_e = ok_e
            wts = n_e[good_e].astype(float)
            vg = vals_e[good_e]
            log_gain = np.zeros(3)
            for c in range(3):
                o = np.argsort(vg[:, c])
                cw = np.cumsum(wts[o])
                log_gain[c] = vg[o][np.searchsorted(cw, 0.5 * cw[-1])][c]
            texture_ratio = float(np.median(tex_e[good_e]))
            log_gain = np.clip(log_gain, -np.log(params.lift_cap), 0.0)
            # Under six per cent is no shadow anyone sees; the app's own test
            # stroke over lit sole was lifted by five.
            if float(log_gain.mean()) > -0.06:
                raise RepairError("no shadow found in the stroke")
            # A coloured shadow shifts the chroma of what it covers and can
            # come out as its own "material"; with the gain known, take that
            # shift out and label once more.
            shift = log_gain - log_gain.mean()
            if pas == 0 and np.abs(shift).max() > 0.03:
                comp = np.where(core, 1.0, 0.0)[..., None] * shift[None, None, :]
                continue
            break

        # The materials share one light, so their gains must agree. Only
        # materials that carry a fifth of the windows' weight count: a sliver
        # label fired the warning on a clean crossing.
        tot = float(n_e[good_e].sum())
        per_label = {}
        for j in labs:
            m = good_e & (lab_e == j)
            if m.sum() >= 3 and n_e[m].sum() >= 0.2 * tot:
                per_label[int(j)] = (vals_e[m] * n_e[m, None]).sum(0) / n_e[m].sum()
        gl = np.array([v.mean() for v in per_label.values()])
        spread = float(gl.max() - gl.min()) if len(gl) > 1 else 0.0
        # A material that holds a good part of the stroke and of the shadow
        # but got no window of its own was never measured -- on a strongly
        # curved crossing it was left dark while the brush reported success.
        unmeasured = 0.0
        for j in labs:
            share = float(((lab == j) & stroke).sum()) / max(int(stroke.sum()), 1)
            if (share > 0.1 and not (good_e & (lab_e == j)).any()
                    and ((lab == j) & core).sum() > 0.1 * core.sum()):
                unmeasured = max(unmeasured, share)
        if unmeasured > 0:
            raise RepairError(
                "a material in the shadow (%d %% of the stroke) has no bright counterpart "
                "at the edge; extend the stroke across its shadow edge" % round(100 * unmeasured))

        # --- the matte: a fitted profile of the signed distance to the edge ----
        # A membrane matte does not hold its known cells: it was still 0.08
        # three cells outside the edge, which brightened lit surface into a
        # halo. One smoothstep with a fitted offset and width does not leak;
        # the lit error went from 0.015-0.033 to 0.000.
        sd = d_out - d_in                     # negative inside the core
        gm = float(log_gain.mean())
        band = known & stroke & (np.abs(sd) <= params.pen_max + 2) & (lab >= 0)
        q = dev[band] / gm                    # 1 in the shadow, 0 outside, noisy
        xs = sd[band]
        best = (np.inf, 0.0, 2.0)
        for sd0 in np.arange(-3.0, 3.01, 0.25):
            for wid in np.arange(1.0, 2.0 * params.pen_max + 0.01, 0.5):
                e = np.abs(q - (1.0 - _smoothstep((xs - sd0) / wid + 0.5))).mean()
                if e < best[0]:
                    best = (e, sd0, wid)
        _, sd0, wid = best
        matte = (1.0 - _smoothstep((sd - sd0) / wid + 0.5)) * stroke
        G = matte[..., None] * log_gain[None, None, :] * float(np.clip(params.strength, 0, 1))
        return dict(G=G, log_gain=log_gain, core=core, how=how, sep=sep,
                    lit_frac=lit_frac, texture_ratio=texture_ratio, Lc=Lc, known=known,
                    sd=np.where(stroke, sd, 1e3), sd0=float(sd0), wid=float(wid),
                    spread=spread, windows=int(ok_e.sum()), good=int(good_e.sum()),
                    labels=len(labs), passes=pas + 1)

    G, log_gain, why_not, info_dc, F = None, None, None, {}, None
    try:
        F = shadow_field()
        G, log_gain = F["G"], F["log_gain"]
        info_dc = {"edge": F["how"],
                   "separation": None if not np.isfinite(F["sep"]) else round(float(F["sep"]), 1),
                   "litFraction": (None if not np.isfinite(F["lit_frac"])
                                   else round(float(F["lit_frac"]), 2)),
                   "coreCells": int(F["core"].sum()),
                   "textureRatio": round(F["texture_ratio"], 2),
                   "materials": F["labels"], "labelSpread": round(F["spread"], 3),
                   "windows": F["windows"], "goodWindows": F["good"],
                   "matteOffset": F["sd0"], "matteWidth": F["wid"], "passes": F["passes"]}
    except RepairError as e:
        why_not = str(e)
    V = None
    if why_not and params.view_fallback and why_not.startswith(_VIEW_REASONS):
        try:
            V, vgain = _view_shadow(s, sel, P, params)
            log_gain, G = np.log(vgain), True
            info_dc = {"edge": "view", "viewFallback": why_not}
            why_not = None
        except Exception:                        # noqa: BLE001 -- no LaMa, no GPU
            V = None

    # --- the picture: what the cells could not find ---------------------------
    # Soft, shallow shadows on grained surfaces, shadows wider than the
    # stroke, strokes of a hundred splats: the cells refuse them. Looked at
    # as a picture they are found, and every splat that makes the picture is
    # lifted, not only the painted ones (`_picture_shadow`, `_picture_fit`).
    # A dark part far off the surface is another object, not a gap in the
    # light: the black rest of the stand lies 132 spacings off the sole, and
    # the picture lifted it x2. The hut's sand along its wall, 36 off, is the
    # picture's (its gap guard leaves the gap itself).
    if why_not and params.picture and cells.get("off", 0.0) <= 2.0 * params.off_surface:
        look = (np.asarray(view, np.float64) / np.linalg.norm(view)
                if view is not None and np.linalg.norm(view) > 0
                else -P.basis[2] * P.outward)
        try:
            res = _unshadow_picture(s, np.where(sel)[0], look, params, volume)
            res.info.update(cellsWhyNot=why_not, coverage=round(coverage, 2),
                            holeShare=round(hole, 2))
            return res
        except RepairError as e:
            why_not = str(e)
        except Exception:                        # noqa: BLE001 -- no GPU: the cells' answer
            pass

    # --- the reflection: what the view-dependent bands carry extra ---------
    # A coloured reflection -- the red cast the stand threw onto the sole --
    # is not in the base colour at all. Rendered without the SH bands the
    # patch beside the stand is plain sole; with them it is red. So it is
    # measured where it lives: per cell, the bands' mean against the same
    # bands of the surrounding surface carried in across the stroke (the
    # method of the tone field, forty-five channels instead of three),
    # smoothed over a few cells so each splat keeps its own view-dependence
    # and only the shared excess goes. Taken after the shadow gain, so a
    # shadow in the bands is not removed twice.
    def gain_uv(uv, xyz):
        """Log gain per splat: the profile at each splat's own distance."""
        if V is not None:
            return V(xyz)
        if F is None:
            return np.zeros((len(uv), 3))
        x = (uv[:, 0] - lo_u) / cell - 0.5
        y = (uv[:, 1] - lo_v) / cell - 0.5
        sd_b = ndimage.map_coordinates(F["sd"], [y, x], order=1, mode="nearest")
        mat = 1.0 - _smoothstep((sd_b - F["sd0"]) / F["wid"] + 0.5)
        return mat[:, None] * log_gain[None, :] * float(np.clip(params.strength, 0, 1))

    def gain_at(rows):
        return gain_uv(uvh[rows], s.xyz[cand[rows]])

    g_all = gain_at(np.arange(len(cand)))
    A, info_sh = _band_excess(s, P, cand, face, kv, ku, nv, nu, stroke, alpha,
                              np.exp(g_all), params, view)
    if G is None and A is None:
        raise RepairError((why_not + "; no reflection either") if why_not
                          else "neither shadow nor reflection found in the stroke")

    # --- apply ---------------------------------------------------------------
    # The shadow is in the surface; a reflection is also carried by the faint
    # splats in front of it that the scan grew to explain the glare -- the
    # painted ones up to the eye, which the slab leaves out.
    #
    # Painted splats behind the slab are under the shadow too: where a fold
    # turns the surface away, the stroke reaches past the slab's ten
    # spacings, and left alone they kept the whole shadow (edge sites of
    # bench_shadow_multi 6.7 / 8.1 / 11.3 / 7.0 -> 4.5 / 6.0 / 9.2 / 6.1
    # levels, curved 11.0 -> 9.5). And the faint splats in the slab, which
    # the patch does not hold (see _faint_under).
    deep = sel[cand] & ~body & (P.depth < 0)
    painted = sel[cand] & ~body & ~deep
    b = np.where(body | deep | (painted if A is not None else False))[0]
    fi, fuvh = (_faint_under(s, P, alpha, cand) if params.faint
                else (np.zeros(0, np.int64), np.zeros((0, 3))))
    gidx = np.concatenate([cand[b], fi])
    guvh = np.concatenate([uvh[b], fuvh])
    x = (guvh[:, 0] - lo_u) / cell - 0.5
    y = (guvh[:, 1] - lo_v) / cell - 0.5
    g = gain_uv(guvh, s.xyz[gidx])
    g[:len(b)][painted[b]] = 0.0
    K = s.f_rest.shape[1]
    mt = np.zeros(len(gidx))
    rb = np.zeros((len(gidx), K, 3))
    if A is not None:
        at = np.stack([ndimage.map_coordinates(A[..., c], [y, x], order=1, mode="nearest")
                       for c in range(A.shape[-1])], 1)
        mt = np.clip(at[:, 0], 0.0, 1.0) * float(np.clip(params.strength, 0, 1))
        rb = at[:, 1:].reshape(len(gidx), K, 3)
    touched = (np.abs(g).max(1) > 1e-3) | (mt > 1e-3)
    n_faint = int(touched[len(b):].sum())
    gidx, g, mt, rb = gidx[touched], g[touched], mt[touched], rb[touched]
    if len(gidx) == 0:
        raise RepairError(why_not or "no shadow found in the stroke")
    idx = gidx
    # With a view (the app), the shadow found in the cells is also taken out
    # of the picture: every splat seen there gets its share (`_picture_fit`),
    # not only the painted ones -- on real scans those make a fraction of
    # what is seen. The benches pass no view and measure the painted splats.
    n_picture = 0
    if (G is not None and params.picture and view is not None
            and np.linalg.norm(view) > 0):
        try:
            look = np.asarray(view, np.float64) / np.linalg.norm(view)
            fit = _picture_of_field(
                s, idx, -g, look, params, volume,
                lambda xyz: gain_uv((xyz - P.origin) @ P.basis.T, xyz))
        except Exception:                        # noqa: BLE001 -- no GPU: the cells' splats
            fit = None
        if fit is not None:
            ix, lg = fit
            pos = {int(i): k for k, i in enumerate(idx)}
            at_ = np.array([pos.get(int(i), -1) for i in ix])
            mt = np.where(at_ >= 0, mt[np.maximum(at_, 0)], 0.0)
            rb = np.where((at_ >= 0)[:, None, None], rb[np.maximum(at_, 0)], 0.0)
            n_picture = int((at_ < 0).sum())
            idx, g = ix, -lg
    rgb = 0.5 + SH_C0 * s.f_dc[idx].astype(np.float64)
    # No channel past white. A colour the gain would lift over 1 was never in
    # the shadow -- lit grains and haze inside the stroke; on the hut's sand
    # 111 of 360 lifted splats went to 1.4-3.6 and rendered as white
    # sparkles. The bands follow the clamped gain.
    lrgb = np.log(np.clip(rgb, 1e-6, None))
    n_white = int(((lrgb - g) > 0.0).any(1).sum())
    g = np.maximum(g, np.minimum(lrgb, 0.0))
    gain = np.exp(g)
    f_dc = (rgb / gain - 0.5) / SH_C0
    f_rest = s.f_rest[idx].astype(np.float64)
    if f_rest.shape[1]:
        f_rest = (f_rest / gain[:, None, :] * (1.0 - mt)[:, None, None]
                  + rb * mt[:, None, None])
    added = SplatSet(
        xyz=s.xyz[idx].copy(), f_dc=f_dc.astype(np.float32),
        f_rest=f_rest.astype(np.float32), opacity=s.opacity[idx].copy(),
        scale=s.scale[idx].copy(), rot=s.rot[idx].copy(),
        extra={k: v[idx].copy() for k, v in s.extra.items()})

    warn = []
    if F is not None:
        Lc, core, known = F["Lc"], F["core"], F["known"]
        core_val = (float(np.exp(np.median(Lc[core & known])).mean())
                    if (core & known).any() else 1.0)
        lift = float(np.exp(-log_gain).max())
        if lift > 5.0:
            warn.append("lift ×%s: the noise in the shadow becomes visible"
                        % ("%.1f" % lift))
        if core_val < 16.0 / 255.0:
            warn.append("the shadow is almost black: hardly any color is left there")
        if np.exp(log_gain).max() / max(np.exp(log_gain).min(), 1e-9) > 1.6:
            warn.append("strongly colored shadow; check the material at the edge")
        if F["spread"] > params.label_tol:
            warn.append("the materials see the shadow differently (%s log)"
                        % ("%.2f" % F["spread"]))
    info = {
        "mode": "unshadow",
        "gain": None if log_gain is None else [round(float(v), 3) for v in np.exp(log_gain)],
        "shadow": G is not None, "shadowWhyNot": why_not,
        "reflex": A is not None, **info_dc, **info_sh,
        "strokeCells": int(stroke.sum()), "lifted": int(len(idx)), "faint": n_faint,
        "picture": n_picture,
        "clamped": n_white,
        "coverage": round(coverage, 2), "strength": float(params.strength),
        "holeShare": round(hole, 2),
        "warning": "; ".join(warn) or None,
    }
    return RepairResult(removed=idx, added=added, info=info)
