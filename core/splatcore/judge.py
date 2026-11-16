"""Is the refined fill visibly better than the quick fill? Look, then decide.

The background refine (`Document.refine_last`) used to swap in whenever the
deep fit's own guard let it through. That guard compares the fit with the
inpainted picture carried into side views, and an error against a picture
rewards blur: on the shoe's stand hole it went 0.069 -> 0.045 for the
refine, while the eye saw the quick fill keep the diamond lattice and the
refine wash it out. On the EHBO letters the same refine took out dark
vertical streaks and was plainly better. So the question is put to what the
eye sees instead, and it is put the same way to both candidates.

Both fills have the same splats in the same places; only their colours
differ. The patch is rendered from the front and from two oblique views,
with its splats once white and once black: the difference is exactly how
much the patch shows in each pixel, occlusion included. Each candidate is
then compared with the ring of surface around it, on what a person
notices:

* **foreign structure** -- band-pass gradient amplitude, per orientation
  and on two scales, beyond half again what the ring has (streaks across
  the planks, copied stitching, a bright blob's edges);
* **missing structure** -- the other way: less than two thirds of what the
  ring has (a lattice washed out, planks that fade);
* **lines** -- the share of the fill that stands off its own surroundings
  by more than the ring ever does in that orientation;
* **tone** -- the fill's coarse colour against what the ring leads one to
  expect at that spot, beyond how far the ring itself strays from it there
  (a dark blob in grass, a pale patch on a sole);
* **blobs** -- the share of the fill whose colour (Lab: a*b*, and L* at a
  coarse scale) is not among the colours of the ring pixels nearest to it
  on screen. Each fill pixel is held to its own neighbourhood, so on a ring
  that is half pink stone, half grass, a grey-green patch on the stone side
  counts: it is neither (the castle floor, where the quick fill kept such
  a patch and the other terms did not see it). A soft fill in the right
  colours stays quiet (the stand).

Each is scaled so that about 1 is plainly visible; the sum is the score,
lower is better, and the refine is kept only if its score is lower than the
quick fill's by a margin. Added things -- foreign structure and blobs --
count far more than missing detail. Calibrated on the 2026-11-09 sheet
(stand, EHBO, heel logo, hut corner) and checked on the cross, the sole's
hole and edge and five spots on the castle, two of them with a ring of two
materials; a seam term and structure compared locally were tried and did
not help.
"""
from __future__ import annotations

import time
from dataclasses import dataclass
from typing import Optional

import numpy as np

from .io import SH_C0, SplatSet


@dataclass
class JudgeParams:
    size: int = 384           # pixels across the long side of each view
    frame: float = 1.5        # half-width of the view in patch half-lengths
    fov: float = 30.0         # horizontal
    oblique: float = 35.0     # tilt of the two side views, degrees
    margin: float = 0.06      # how much lower the refine's score has to be (share)
    bins: int = 8             # gradient orientations
    tol: float = 1.5          # texture within this ratio of the ring's is no flaw
    # what counts as plainly visible (term = 1). Structure: gradient
    # amplitude as a share of the ring's brightness (Weber contrast); added
    # structure catches the eye far sooner than lost structure.
    c_foreign: float = 0.005
    c_missing: float = 0.03
    c_lines: float = 0.02     # share of the fill standing off
    c_tone: float = 0.03      # mean coarse colour off, 0..1
    w_foreign: float = 1.0
    w_missing: float = 1.0
    w_lines: float = 1.0
    w_tone: float = 1.0
    # blobs: colour (Lab) a fill pixel's own ring neighbourhood does not have
    blob_k: int = 48          # ring pixels per fill pixel
    blob_ring: int = 3000     # ring pixels sampled
    blob_fill: int = 4000     # fill pixels sampled
    blob_de: tuple = (7.0, 13.0)  # ΔE from "same colour" to "plainly another"
    c_blob: float = 0.05      # share of the fill (5 %) that is plainly visible
    w_blob: float = 1.0


TERMS = ("foreign", "missing", "lines", "tone", "blobs")


# ------------------------------------------------------------------ frame
def patch_frame(quick: SplatSet, ctx: Optional[dict] = None, view=None, backing=None) -> dict:
    """Where the patch is: centre, outward normal, long and short tangent
    axes and its half-lengths along them.

    The normal is the fitted surface's (each face's, averaged by size on a
    fill made face by face), turned toward the painting camera; without a
    context, the camera's own direction; without either, the patch's
    flattest direction.
    """
    xyz = np.asarray(quick.xyz, np.float64)
    if backing is not None and len(backing) == len(xyz) and (~np.asarray(backing, bool)).sum() >= 8:
        xyz = xyz[~np.asarray(backing, bool)]
    centre = np.median(xyz, 0)
    n = None
    ctx = ctx or {}
    charts = ctx.get("charts") or []
    if len(charts) > 1:
        acc = np.zeros(3)
        spans = ctx.get("chart_spans") or []
        for k, c in enumerate(charts):
            if "basis" not in c:
                continue
            wk = float(spans[k][1] - spans[k][0]) if k < len(spans) else 1.0
            acc += wk * np.asarray(c["basis"][2], float) * float(c.get("outward", 1.0))
        if np.linalg.norm(acc) > 1e-9:
            n = acc
    elif "basis" in ctx:
        n = np.asarray(ctx["basis"][2], float) * float(ctx.get("outward", 1.0))
    v = None
    if view is not None and len(view) == 3 and np.linalg.norm(view) > 1e-9:
        v = np.asarray(view, float) / np.linalg.norm(view)
    if n is None:
        if v is not None:
            n = -v
        else:
            X = xyz - centre
            n = np.linalg.eigh(X.T @ X)[1][:, 0]
    n = n / max(np.linalg.norm(n), 1e-12)
    if v is not None and float(n @ v) > 0:
        n = -n
    d = xyz - centre
    tang = d - np.outer(d @ n, n)
    # the long axis of the patch in its plane
    ev = np.linalg.eigh(tang.T @ tang)[1]
    t1 = ev[:, 2] - n * float(ev[:, 2] @ n)
    if np.linalg.norm(t1) < 1e-9:
        tmp = np.array([0.0, 1.0, 0.0]) if abs(n[1]) < 0.9 else np.array([1.0, 0.0, 0.0])
        t1 = np.cross(n, tmp)
    t1 /= np.linalg.norm(t1)
    t2 = np.cross(n, t1)
    floor = 4.0 * float(ctx["spacing"]) if "spacing" in ctx else 1e-6
    a = max(float(np.percentile(np.abs(tang @ t1), 95)) if len(xyz) else 0.0, floor)
    b = max(float(np.percentile(np.abs(tang @ t2), 95)) if len(xyz) else 0.0, floor, 0.25 * a)
    return {"centre": centre, "n": n, "t1": t1, "t2": t2, "a": a, "b": b}


def _rot(v, axis, deg):
    k = axis / np.linalg.norm(axis)
    t = np.radians(deg)
    return v * np.cos(t) + np.cross(k, v) * np.sin(t) + k * np.dot(k, v) * (1 - np.cos(t))


def cameras(fr: dict, p: JudgeParams):
    """Front, and tilted about each tangent axis; framed on the patch with
    room for the ring (the long axis across, so an elongated patch fills
    a wide, low picture)."""
    from . import gpu
    a, b = fr["a"], fr["b"]
    half_w = a * p.frame
    half_h = b + (p.frame - 1.0) * a
    W = int(p.size)
    H = int(np.clip(round(W * half_h / half_w / 16.0) * 16, 0.4 * W, W))
    tanh = np.tan(np.radians(p.fov) * 0.5)
    dist = half_w / tanh
    fov_v = 2.0 * np.degrees(np.arctan(tanh * H / W))
    n, t1, t2, c = fr["n"], fr["t1"], fr["t2"], fr["centre"]
    dirs = [n, _rot(n, t1, p.oblique), _rot(n, t2, -p.oblique)]
    return [gpu.look_at_camera(c + d * dist, c, -t2, W, H, fov_v) for d in dirs], dist


# ---------------------------------------------------------------- render
def _render_all(scene: SplatSet, quick: SplatSet, refined: SplatSet, cams, deg):
    """From every camera: quick, refined, how much the patch shows, and the
    quick render's alpha and depth.

    How much the patch shows: its splats once white and once black, the
    difference -- exact, occlusion included.

    Classic rasterisation, as the viewport and the sheets draw: the
    antialiased mode thins small splats and showed the refine's backing
    through it as a grey cloud nobody sees in the app.
    """
    import torch
    from . import gpu
    both = scene.concat(quick)
    t = gpu.to_tensors(both, torch.device("cuda"))
    n0 = scene.n
    colq = t["colors"].clone()
    refined_sh = np.concatenate([refined.f_dc[:, None, :], refined.f_rest], axis=1)
    colr = colq.clone()
    colr[n0:] = torch.tensor(np.ascontiguousarray(refined_sh), device=colq.device,
                             dtype=colq.dtype)
    flat = []
    for v in (1.0, 0.0):
        c = colq.clone()
        c[n0:] = 0.0
        c[n0:, 0, :] = (v - 0.5) / SH_C0
        flat.append(c)
    out = []
    with torch.no_grad():
        for k, col in enumerate((colq, colr, flat[0], flat[1])):
            t["colors"] = col
            img, alpha, dep = gpu.render(t, cams, sh_degree=deg, antialiased=False,
                                         with_depth=(k == 0))
            out.append(img.clamp(0, 1).cpu().numpy().astype(np.float64))
            if k == 0:
                a0 = alpha[..., 0].cpu().numpy().astype(np.float64)
                d0 = dep.cpu().numpy().astype(np.float64)
                d0[a0 < 0.5] = np.inf
    del t
    return out[0], out[1], out[2] - out[3], a0, d0


# ---------------------------------------------------------------- measure
def _lum(img):
    return img @ np.array([0.299, 0.587, 0.114])


def _blur(x, sigma):
    """Gaussian over the image plane; a wide one on a smaller copy (time)."""
    from scipy import ndimage
    extra = (0,) * (x.ndim - 2)
    f = int(sigma // 3)
    if f < 2:
        return ndimage.gaussian_filter(x, (sigma, sigma) + extra)
    H, W = x.shape[:2]
    Hc, Wc = -(-H // f) * f, -(-W // f) * f
    pad = np.pad(x, ((0, Hc - H), (0, Wc - W)) + ((0, 0),) * (x.ndim - 2), mode="edge")
    small = pad.reshape((Hc // f, f, Wc // f, f) + x.shape[2:]).mean((1, 3))
    small = ndimage.gaussian_filter(small, (sigma / f, sigma / f) + extra)
    return ndimage.zoom(small, (f, f) + (1,) * (x.ndim - 2), order=1)[:H, :W]


def _lab(img):
    """sRGB (0..1) -> CIE Lab (D65)."""
    c = np.where(img <= 0.04045, img / 12.92, ((img + 0.055) / 1.055) ** 2.4)
    xyz = c @ np.array([[0.4124, 0.3576, 0.1805],
                        [0.2126, 0.7152, 0.0722],
                        [0.0193, 0.1192, 0.9505]]).T
    xyz = xyz / np.array([0.95047, 1.0, 1.08883])
    f = np.where(xyz > 0.008856, np.cbrt(xyz), 7.787 * xyz + 16.0 / 116.0)
    return np.stack([116.0 * f[..., 1] - 16.0, 500.0 * (f[..., 0] - f[..., 1]),
                     200.0 * (f[..., 1] - f[..., 2])], -1)


def _colours(img, sc_l):
    """a*b* lightly smoothed, L* at the coarse scale: colour, not texture."""
    from scipy import ndimage
    lab = _lab(np.clip(img, 0.0, 1.0).astype(np.float32))
    return np.stack([_blur(lab[..., 0], sc_l),
                     ndimage.gaussian_filter(lab[..., 1], 1.5),
                     ndimage.gaussian_filter(lab[..., 2], 1.5)], -1)


def masks(w, alpha, depth, img, dist, a):
    """Where the patch shows (``core``) and the ring around it.

    The ring is a band just off the core where the patch shows less than a
    third. A fuzzy fill -- grass -- reaches far past its core in faint
    wisps, so "not at all" would push the ring out of the picture; what
    little of either candidate is still in it is evened out by measuring the
    ring on the mean of both (``img``). It is cleaned of what is plainly
    another material -- the red cross next to the EHBO letters, a shutter's
    edge: pixels whose coarse colour is far off the ring's own (robustly),
    as long as most of the ring stays.
    """
    from scipy import ndimage
    patch = w.mean(-1)
    core = ndimage.binary_erosion(patch > 0.35, iterations=1)
    if core.sum() < 120:
        return None
    rw = int(np.clip(0.5 * np.sqrt(core.sum() / np.pi), 4, 24))
    dt = ndimage.distance_transform_edt(~core)
    gap = 3
    band = (dt > gap) & (dt <= gap + rw) & (patch < 0.35)
    fin = band & np.isfinite(depth)
    dz = float(np.median(depth[fin])) if fin.any() else dist
    # the same surface, not what lies far behind or in front of it
    usable = (alpha > 0.95) & np.isfinite(depth) & (np.abs(depth - dz) < 1.2 * a)
    ring = band & usable
    if ring.sum() < 0.4 * band.sum():
        ring = (dt > gap) & (dt <= gap + 2 * rw) & (patch < 0.35) & usable
    if ring.sum() < 120:
        return None
    coarse = np.stack([ndimage.gaussian_filter(img[..., c], 1.5) for c in range(3)], -1)
    med = np.median(coarse[ring], 0)
    d = np.linalg.norm(coarse - med, axis=-1) / np.sqrt(3.0)
    dr = d[ring]
    thr = max(3.5 * 1.4826 * float(np.median(dr)), 0.06)
    if float((dr <= thr).mean()) < 0.7:
        thr = float(np.percentile(dr, 70))
    ring0 = ring.copy()
    ring &= d <= thr
    return core, ring, rw, ring0


def _bins(gx, gy, bins):
    th = np.mod(np.arctan2(gy, gx), np.pi)
    return np.minimum((th / np.pi * bins).astype(int), bins - 1)


SCALES = (0.8, 2.0)


def _energy(L, s, bins):
    from scipy import ndimage
    g = ndimage.gaussian_filter(L, s)
    gy, gx = np.gradient(g)
    gx, gy = gx * s, gy * s                     # scale-normalised
    return gx * gx + gy * gy, _bins(gx, gy, bins)


def expect(img, core, ring, rw, p: JudgeParams, ring0=None) -> dict:
    """What the ring says the fill should look like.

    Structure: the ring's gradient amplitude per orientation and scale.
    Colour: carried in from the ring with a normalised Gaussian, so a fill
    between a pale floor and the grass beside it is held to floor on the
    floor's side and to grass on the grass's side; and how far the ring
    itself strays from that, near each pixel.
    """
    from scipy import ndimage
    sig = max(1.5 * rw, 6.0)
    rm = ring.astype(np.float64)
    den = _blur(rm, sig) + 1e-6
    nc = lambda x: _blur(x * (rm if x.ndim == 2 else rm[..., None]), sig) / (
        den if x.ndim == 2 else den[..., None])
    M = nc(img)
    L = _lum(img)
    XR = []
    for s in SCALES:
        e, b = _energy(L, s, p.bins)
        XR.append(np.sqrt(np.bincount(b[ring], weights=e[ring], minlength=p.bins)
                          / max(int(ring.sum()), 1)))
    # lines: how far the ring stands off its own surroundings, per orientation
    H = np.abs(L - ndimage.gaussian_filter(L, 2.5))
    e1, b1 = _energy(L, 1.0, p.bins)
    tau = np.full(p.bins, float(np.percentile(H[ring], 99)))
    for k in range(p.bins):
        m = ring & (b1 == k)
        if m.sum() >= 40:
            tau[k] = float(np.percentile(H[m], 99))
    sc = max(rw / 2.0, 2.0)
    C = _blur(img, sc)
    dev = np.sqrt(((C - M) ** 2).mean(-1))
    # blobs: for each fill pixel, the ring pixels nearest to it on screen --
    # its own neighbourhood, so stone is held to stone and grass to grass
    from scipy.spatial import cKDTree
    r0 = ring0 if ring0 is not None else ring
    ry, rx = np.nonzero(r0)
    rng = np.random.default_rng(0)
    if len(ry) > p.blob_ring:
        k = rng.choice(len(ry), p.blob_ring, replace=False)
        ry, rx = ry[k], rx[k]
    fy, fx = np.nonzero(core)
    if len(fy) > p.blob_fill:
        k = rng.choice(len(fy), p.blob_fill, replace=False)
        fy, fx = fy[k], fx[k]
    kk = min(p.blob_k, len(ry))
    _, nb = cKDTree(np.column_stack([ry, rx])).query(np.column_stack([fy, fx]), k=kk)
    col = _colours(img, sc)
    return {"M": M, "XR": XR, "lumR": max(float(np.mean(L[ring])), 0.1),
            "tau": np.maximum(1.3 * tau, 0.03), "base": nc(dev), "sc": sc,
            "ringCol": col[ry, rx], "nb": nb.reshape(len(fy), kk), "fill": (fy, fx)}


def raw(img, core, ex: dict, p: JudgeParams) -> dict:
    """What one candidate looks like in one view, against `expect`, unweighted."""
    from scipy import ndimage
    L = _lum(img)
    fo = mi = 0.0
    for j, s in enumerate(SCALES):
        e, b = _energy(L, s, p.bins)
        aF = np.sqrt(np.bincount(b[core], weights=e[core], minlength=p.bins)
                     / max(int(core.sum()), 1))
        aR = ex["XR"][j]
        fo += float(np.maximum(aF - p.tol * aR, 0).sum()) / ex["lumR"] / len(SCALES)
        mi += float(np.maximum(aR / p.tol - aF, 0).sum()) / ex["lumR"] / len(SCALES)
    H = np.abs(L - ndimage.gaussian_filter(L, 2.5))
    _, b1 = _energy(L, 1.0, p.bins)
    C = _blur(img, ex["sc"])
    dev = np.sqrt(((C - ex["M"]) ** 2).mean(-1))
    fy, fx = ex["fill"]
    col = _colours(img, ex["sc"])[fy, fx]
    dE = np.linalg.norm(col[:, None, :] - ex["ringCol"][ex["nb"]], axis=-1)
    j = min(2, dE.shape[1] - 1)
    near = np.partition(dE, j, axis=1)[:, j]      # third-closest: one stray ring pixel is no excuse
    return {"foreign": fo, "missing": mi,
            "lines": float((H[core] > ex["tau"][b1[core]]).mean()),
            "tone": float(np.mean(np.maximum(dev[core] - ex["base"][core], 0.0))),
            # share of the fill off its neighbourhood's colours, ramped over
            # blob_de (ΔE) so one threshold does not make it jump
            "blobs": float(np.clip((near - p.blob_de[0]) / (p.blob_de[1] - p.blob_de[0]),
                                   0.0, 1.0).mean())}


def terms(r: dict, p: Optional[JudgeParams] = None) -> dict:
    """The raw look of a candidate as the flaw terms, ~1 = plainly visible."""
    p = p or JudgeParams()
    return {"foreign": r["foreign"] / p.c_foreign,
            "missing": r["missing"] / p.c_missing,
            "lines": max(r["lines"] - 0.002, 0.0) / p.c_lines,
            "tone": r["tone"] / p.c_tone,
            "blobs": r.get("blobs", 0.0) / p.c_blob}


def score(t: dict, p: JudgeParams) -> float:
    return (p.w_foreign * t["foreign"] + p.w_missing * t["missing"]
            + p.w_lines * t["lines"] + p.w_tone * t["tone"] + p.w_blob * t["blobs"])


def look(scene: SplatSet, quick: SplatSet, refined: SplatSet, fr: dict,
         p: Optional[JudgeParams] = None, keep_images: bool = False) -> dict:
    """Render both candidates on ``scene`` and measure them (no verdict yet)."""
    p = p or JudgeParams()
    cams, dist = cameras(fr, p)
    Iq, Ir, W, A, D = _render_all(scene, quick, refined, cams, scene.sh_degree)
    per, ms = [], []
    for k in range(len(cams)):
        both = 0.5 * (Iq[k] + Ir[k])
        m = masks(W[k], A[k], D[k], both, dist, fr["a"])
        ms.append(m)
        if m is None:
            per.append(None)
            continue
        core, ring, rw, ring0 = m
        ex = expect(both, core, ring, rw, p, ring0)
        per.append({"quick": raw(Iq[k], core, ex, p),
                    "refined": raw(Ir[k], core, ex, p),
                    "pixels": int(core.sum())})
    out = {"per": per}
    if keep_images:
        out["images"] = (Iq, Ir, W, ms)
    return out


def verdict(per, p: Optional[JudgeParams] = None) -> dict:
    """Scores, terms, the decision and why, from `look`'s measurements."""
    p = p or JudgeParams()
    ok = [v for v in per if v is not None]
    if not ok:
        return {"keep": None, "views": 0, "why": "the fill was not visible in any view"}
    tv = [{c: terms(v[c], p) for c in ("quick", "refined")} for v in ok]
    mean = {c: {t: float(np.mean([v[c][t] for v in tv])) for t in TERMS}
            for c in ("quick", "refined")}
    sq, sr = score(mean["quick"], p), score(mean["refined"], p)
    keep = sr < sq * (1.0 - p.margin)
    return {"keep": bool(keep), "views": len(ok), "quick": round(sq, 3), "refined": round(sr, 3),
            "terms": {c: {t: round(v, 3) for t, v in mean[c].items()} for c in mean},
            "perView": [{c: round(score(v[c], p), 3) for c in ("quick", "refined")} for v in tv],
            "why": _why(mean, keep)}


def judge_scene(scene: SplatSet, quick: SplatSet, refined: SplatSet, fr: dict,
                p: Optional[JudgeParams] = None) -> dict:
    """Keep the refined fill, judged on ``scene`` (the surroundings, patch left out)?"""
    return verdict(look(scene, quick, refined, fr, p)["per"], p)


def _why(mean, keep):
    """The term that tipped it, as a short reason (German, for the status line)."""
    q, r = mean["quick"], mean["refined"]
    if keep:
        gain = {t: q[t] - r[t] for t in TERMS}
        return {"foreign": "the Fast Fill had distracting content",
                "missing": "the Fast Fill was too smooth",
                "lines": "the Fast Fill had streaks",
                "tone": "the Fast Fill had blotches",
                "blobs": "the Fast Fill had foreign colors"}[max(gain, key=gain.get)]
    loss = {t: r[t] - q[t] for t in TERMS}
    t = max(loss, key=loss.get)
    if loss[t] <= 0:
        return "the refinement was no better"
    return {"foreign": "the refinement added distracting content",
            "missing": "it was sharper",
            "lines": "the refinement added streaks",
            "tone": "the refinement added blotches",
            "blobs": "the refinement added foreign colors"}[t]


def judge(world: SplatSet, removed, quick: SplatSet, refined: SplatSet,
          ctx: Optional[dict] = None, view=None, backing=None,
          params: Optional[JudgeParams] = None) -> dict:
    """Keep the refined fill? ``world``: the live set the quick fill was made
    on; ``removed``: what that fill took out of it (indices into ``world``).

    Returns ``keep`` (None when it could not look: no GPU, the patch not in
    view), both scores, the terms, ``why`` and ``seconds``.
    """
    t0 = time.time()
    p = params or JudgeParams()
    from . import gpu
    if not gpu.available()[0]:
        return {"keep": None, "why": "no GPU for the comparison", "seconds": 0.0}
    fr = patch_frame(quick, ctx, view, backing)
    reach = fr["a"] * p.frame
    xyz = np.asarray(world.xyz)
    d = xyz - fr["centre"].astype(np.float32)
    # the surroundings the views can show; what stands well in front is left out
    keep = ((np.einsum("ij,ij->i", d, d) < (2.5 * reach) ** 2)
            & (d @ fr["n"].astype(np.float32) < 0.6 * reach))
    keep[np.asarray(removed, np.int64)] = False
    scene = world.take(np.where(keep)[0])
    res = judge_scene(scene, quick, refined, fr, p)
    res["seconds"] = round(time.time() - t0, 3)
    return res
