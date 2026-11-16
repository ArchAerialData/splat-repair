"""Select the object, not the brush stroke: what to take out before a repair.

Painted over the raised EHBO letters on the hut, the repair left pale ghost
letters in all three views; over the green logo on the brush's heel, greenish
smears. Rendering what survived the repair showed why. The brush selects what
lies within a depth window of the surface the pointer sees, so it catches the
letters' faces but not all of their sides, not the faint, large splats that
blur their edges and not the part of the drop shadow outside the dilated
mask. Those leftovers sit on the rim of the hole, and the fill both copies from
the rim and blends into it: the outline of every letter came back.

Photoshop answers the same problem with "Select Subject": you point at the
thing, it finds its edges. Here the scene is in 3-D, which makes the question
easier rather than harder:

* **the base surface** is the surface around the stroke, fitted robustly
  (a quadratic height field over a plane through the ring of splats around
  the stroke, outliers down-weighted), facing the camera the stroke was
  painted from;
* **the object** is whatever stands proud of that surface, or whose colour
  is far from the surface's own material -- seeded inside the stroke, then
  grown through touching splats (its sides, its thickness, faint splats, a
  little past the stroke);
* **its footprint** on the surface is closed into one region, widened to
  take the drop shadow of a raised object, and everything in that column in
  front of the surface's skin goes, so no part of the object is left on the
  rim for the fill to copy.

Then one of two things happens:

* ``"lift"``: the object stands clear of a surface that is whole behind it
  (the antenna pole in front of the hut's wall). Only the object is deleted;
  the wall the brush also touched stays as it is.
* ``"fill"``: the surface behind it is missing or stained (letters fixed to a
  wall, a print on leather). The object and its footprint go, and the repair
  fills the footprint.

``"none"``: no object in the stroke (a hole, a stain on a fold, a stroke over
plain surface). The selection is left exactly as painted.

`Document.repair` uses all three. `Document.deep_repair` uses "lift" only:
the inpainter takes the painted mask as it is, and on the hut's cross the
grown footprint left a faint pale cross where the painted one left none.

Everything is local to the stroke and runs in well under a second.
"""
from __future__ import annotations

from dataclasses import dataclass, field
from typing import Optional

import numpy as np
from scipy import ndimage
from scipy.sparse import coo_matrix
from scipy.sparse.csgraph import connected_components
from scipy.spatial import cKDTree


# For diagnosis only: with DEBUG on, the last call's working arrays are kept
# in _LAST (the surface fit, the footprint grids, the thresholds).
DEBUG = False
_LAST: dict = {}


@dataclass
class ObjParams:
    reach: float = 10.0        # how far past the stroke the object may grow, in spacings
    proud: float = 3.0         # standing this many spacings off the surface is "object"
    link: float = 2.0          # splats this close (in spacings) touch
    colour_k: float = 3.0      # chroma this many ring spreads off the surface's is foreign
    min_core: float = 0.08     # the object must be this share of the stroke
    min_core_n: int = 40       # ... and at least this many splats
    intact: float = 0.6        # a cell holding this share of the ring's surface is whole
    lift_share: float = 0.8    # this share of whole cells under the object: just delete it
    close: int = 2             # closing of the footprint, in cells
    margin: int = 1            # the footprint widened by this many cells
    shadow: bool = True        # take a raised object's drop shadow along
    ring_inliers: float = 0.5  # the ring must be this much one surface
    cut: Optional[float] = None  # how deep under the skin the footprint is cleared, in spacings


@dataclass
class ObjectSelection:
    mode: str                              # "none" | "lift" | "fill"
    indices: np.ndarray                    # what to take out (lift) or repair (fill)
    object: np.ndarray = field(default_factory=lambda: np.zeros(0, np.int64))
    info: dict = field(default_factory=dict)


# ------------------------------------------------------------------ helpers

def _lab(rgb):
    """sRGB in [0, 1] to CIE L*a*b* (D65)."""
    c = np.clip(np.asarray(rgb, np.float64), 0.0, 1.0)
    lin = np.where(c <= 0.04045, c / 12.92, ((c + 0.055) / 1.055) ** 2.4)
    M = np.array([[0.4124, 0.3576, 0.1805],
                  [0.2126, 0.7152, 0.0722],
                  [0.0193, 0.1192, 0.9505]])
    xyz = lin @ M.T / np.array([0.95047, 1.0, 1.08883])
    f = np.where(xyz > 0.008856, np.cbrt(xyz), 7.787 * xyz + 16.0 / 116.0)
    L = 116.0 * f[:, 1] - 16.0
    a = 500.0 * (f[:, 0] - f[:, 1])
    b = 200.0 * (f[:, 1] - f[:, 2])
    return np.stack([L, a, b], 1)


def _spacing(P, rng, sample=3000):
    tree = cKDTree(P)
    q = P if len(P) <= sample else P[rng.choice(len(P), sample, replace=False)]
    d = tree.query(q, k=2)[0][:, 1]
    d = d[np.isfinite(d) & (d > 0)]
    return float(np.median(d)) if len(d) else 0.0


def _local_nanmedian(a, size, at):
    """Per cell of ``at``: the median of what is not NaN in the ``size`` window round it.

    What ``ndimage.generic_filter(a, np.nanmedian, size=size, mode="constant",
    cval=nan)`` gives, on the cells of ``at`` only; NaN everywhere else and
    where a window holds nothing. That filter calls Python once per cell of
    the whole grid, and its result is read on the ring and the band alone:
    a tenth to a fifth of the grid round a print or a sign. Cutting the grid
    to those cells' bounding box would not do it: the ring goes all the way
    round, and round the sign the box was the whole grid. ``size`` is odd.
    """
    h = size // 2
    win = np.lib.stride_tricks.sliding_window_view(
        np.pad(a, h, constant_values=np.nan), (size, size))
    out = np.full(a.shape, np.nan)
    ys, xs = np.where(at)
    step = max(1, 2_000_000 // (size * size))            # 16 MB of windows at a time
    for k in range(0, len(ys), step):
        y, x = ys[k:k + step], xs[k:k + step]
        w = np.sort(win[y, x].reshape(len(y), -1), axis=1)       # NaN sorts to the end
        n = np.count_nonzero(~np.isnan(w), axis=1)
        hi = n // 2
        lo = np.maximum(hi - (n % 2 == 0), 0)
        i = np.arange(len(y))
        # the middle value, or the mean of the middle two; all NaN stays NaN
        with np.errstate(invalid="ignore"):
            out[y, x] = (w[i, lo] + w[i, hi]) / 2.0
    return out


def _poly_cols(u, v):
    return np.stack([np.ones_like(u), u, v, u * u, u * v, v * v], 1)


def _front_cells(u, v, h, cell, q=0.8, min_n=3):
    """Per cell of ``cell`` size: its centre and the ``q`` quantile of ``h``.

    The hut's wall is two layers in the scan, its skin and, nine spacings
    behind, a denser layer of whatever the scanner saw through the boards.
    A fit through all the splats took the back layer for the wall and the
    wall's own skin for an object standing on it. The front of each cell is
    the skin wherever there is one; the cells an object stands in are few,
    and the robust fit drops them.
    """
    iu = np.floor(u / cell).astype(np.int64)
    iv = np.floor(v / cell).astype(np.int64)
    iu -= iu.min()
    iv -= iv.min()
    key = iv * (iu.max() + 1) + iu
    order = np.lexsort((h, key))
    ks, hs = key[order], h[order]
    starts = np.r_[0, np.where(np.diff(ks))[0] + 1]
    cnt = np.diff(np.r_[starts, len(ks)])
    ok = cnt >= min_n
    pick = starts + np.floor(q * (cnt - 1)).astype(np.int64)
    uu = np.add.reduceat(u[order], starts) / cnt
    vv = np.add.reduceat(v[order], starts) / cnt
    return uu[ok], vv[ok], hs[pick][ok], cnt[ok].astype(float)


def _robust_surface(u, v, h, w, floor):
    """A quadratic ``h(u, v)``, Tukey-reweighted so objects on it do not tip it.

    Returns ``(fn, scale, inlier)``. ``floor``: the smallest residual scale
    believed (a perfectly flat ring would otherwise call every speck an
    outlier).
    """
    sc = max(float(np.sqrt(np.percentile(u * u + v * v, 90))), 1e-9)
    A = _poly_cols(u / sc, v / sc)
    ridge = np.diag([0, 0, 0, 1, 1, 1]) * 1e-3
    wt = np.asarray(w, float).copy()
    coef = np.zeros(6)
    # Start from the plane most cells agree with (RANSAC, deterministic).
    # A histogram of heights was tried first: round the pole the plane
    # through the ring sat 20 degrees off the wall, the wall's heights ran
    # over 75 spacings along the stroke, and the "mode" was the roof.
    rng = np.random.default_rng(0)
    A1 = A[:, :3]
    best, best_r = -1.0, h - np.median(h)
    tol = 2.0 * floor
    n = len(h)
    for _ in range(min(300, max(30, n))):
        k = rng.choice(n, 3, replace=False)
        try:
            cf = np.linalg.solve(A1[k], h[k])
        except np.linalg.LinAlgError:
            continue
        r_ = h - A1 @ cf
        score = float(wt[np.abs(r_) < tol].sum())
        if score > best:
            best, best_r = score, r_
    r = best_r
    scale = max(1.4826 * float(np.median(np.abs(r[np.abs(r) < 3 * tol]))), floor) \
        if np.any(np.abs(r) < 3 * tol) else floor
    for _ in range(6):
        t = r / (4.685 * scale)
        tw = np.where(np.abs(t) < 1, (1 - t * t) ** 2, 0.0) * wt
        if tw.sum() < 1e-9:
            break
        Aw = A * tw[:, None]
        coef = np.linalg.solve(A.T @ Aw + ridge * tw.sum(), Aw.T @ h)
        r = h - A @ coef
        # the MAD over every cell: most of the ring is the surface, so its
        # median is the surface's spread. Over the cells still weighted it
        # crept up each round wherever an object's sides stood in the ring.
        scale = max(1.4826 * float(np.median(np.abs(r))), floor)
    inl = np.abs(r) < 3.0 * scale

    def f(uu, vv):
        return _poly_cols(uu / sc, vv / sc) @ coef

    f.slope = (float(coef[1] / sc), float(coef[2] / sc))
    return f, scale, inl


def _components(P, seeds_mask, radius):
    """Points connected (within ``radius``, chained) to any seed."""
    if len(P) == 0:
        return np.zeros(0, bool)
    tree = cKDTree(P)
    pairs = tree.query_pairs(radius, output_type="ndarray")
    n = len(P)
    if len(pairs) == 0:
        return seeds_mask.copy()
    g = coo_matrix((np.ones(len(pairs), np.int8), (pairs[:, 0], pairs[:, 1])), shape=(n, n))
    _, lab = connected_components(g, directed=False)
    keep = np.unique(lab[seeds_mask])
    return np.isin(lab, keep)


# --------------------------------------------------------------------- main

def find_object(s, selection, view=None, params: Optional[ObjParams] = None,
                seed: int = 0) -> ObjectSelection:
    """What in the stroke is an object, and what to remove with it.

    ``s``: the live splats; ``selection``: indices (or mask) into ``s``;
    ``view``: the painting camera's forward direction (the side facing it is
    the front). Returns an `ObjectSelection` with indices into ``s``.
    """
    p = params or ObjParams()
    sel = np.asarray(selection)
    if sel.dtype == bool:
        sel = np.where(sel)[0]
    sel = np.unique(sel.astype(np.int64))
    none = ObjectSelection("none", sel, info={"objsel": "none"})
    if len(sel) < max(p.min_core_n, 20):
        none.info["why"] = "too little selected"
        return none
    rng = np.random.default_rng(seed)

    xyz32 = s.xyz
    P = xyz32[sel].astype(np.float64)
    c = np.median(P, 0)
    r_sel = float(np.percentile(np.linalg.norm(P - c, axis=1), 99))
    sp = _spacing(P, rng)
    if not sp > 0:
        none.info["why"] = "no spacing measurable"
        return none
    ringw = float(np.clip(0.5 * r_sel, 15.0 * sp, 30.0 * sp))
    d_c = np.linalg.norm(xyz32 - c.astype(np.float32), axis=1)
    L = np.where(d_c < r_sel + ringw)[0]
    if len(L) > 1_500_000:
        none.info["why"] = "surroundings too large"
        return none
    XL = xyz32[L].astype(np.float64)
    in_sel = np.isin(L, sel)
    d_sel = cKDTree(P).query(XL, distance_upper_bound=ringw)[0]
    # The surface's spacing, not the stroke's or the object's: measured on
    # the outer half of the ring, away from an object's dense sides.
    outer = np.isfinite(d_sel) & ~in_sel & (d_sel > 0.5 * ringw)
    if outer.sum() >= 100:
        sp = _spacing(XL[outer], rng) or sp
    ring = (~in_sel) & np.isfinite(d_sel) & (d_sel > 2.0 * sp)
    if ring.sum() < 60:
        none.info["why"] = "no rim"
        return none
    alpha = 1.0 / (1.0 + np.exp(-s.opacity[L].astype(np.float64)))

    # --- the base surface ---------------------------------------------------
    # Heights along the painting camera's axis first: the brush paints a
    # surface it faces. The plane through the ring by PCA was tried and,
    # round the pole, stood at right angles to the wall: a long thin ring of
    # wall strips, pole stubs and roof has no reliable smallest axis. The
    # height field's own tilt then turns the frame onto the surface.
    R = XL[ring]
    w = alpha[ring]
    cr = np.average(R, 0, weights=w)
    vw = None if view is None else np.asarray(view, float)
    if vw is not None and (vw.shape != (3,) or not np.linalg.norm(vw) > 0):
        vw = None
    if vw is not None:
        vw = vw / np.linalg.norm(vw)
        n = -vw
    else:
        X = R - cr
        n = np.linalg.eigh((X * w[:, None]).T @ X / w.sum())[1][:, 0]

    def frame(n_):
        a_ = np.array([1.0, 0.0, 0.0]) if abs(n_[0]) < 0.9 else np.array([0.0, 1.0, 0.0])
        e1_ = np.cross(n_, a_)
        e1_ /= np.linalg.norm(e1_)
        e2_ = np.cross(n_, e1_)
        return e1_, e2_, np.stack([(XL - cr) @ e1_, (XL - cr) @ e2_], 1), (XL - cr) @ n_

    def fit(UV_, H_):
        cu, cv, ch, cn = _front_cells(UV_[ring, 0], UV_[ring, 1], H_[ring], 4.0 * sp, min_n=2)
        if len(cu) < 12:
            return None
        # one vote per cell: what covers the ring, not what is densest
        fn, sc_, inl_ = _robust_surface(cu, cv, ch, np.ones_like(cn), floor=1.0 * sp)
        return fn, sc_, float(np.mean(inl_))

    got = None
    for it in range(3):
        e1, e2, UV, H = frame(n)
        got = fit(UV, H)
        if got is None:
            none.info["why"] = "no rim"
            return none
        if vw is None and it == 0:
            # no camera: the side the stroke stands out to is the front
            rel0 = H[in_sel] - got[0](UV[in_sel, 0], UV[in_sel, 1])
            if np.median(rel0) < 0:
                n = -n
                continue
        bu, bv = got[0].slope
        n_new = n - bu * e1 - bv * e2
        n_new /= np.linalg.norm(n_new)
        if vw is not None and n_new @ vw > -0.2:
            break                     # never turn edge-on to the camera
        if float(n_new @ n) > np.cos(np.radians(4.0)):
            break
        n = n_new
    surf_fn, s_h, inl_share = got
    relief = H - surf_fn(UV[:, 0], UV[:, 1])
    if DEBUG:
        _LAST.clear()
        _LAST.update(locals())
    info = {"objsel": "none", "spacing": round(sp, 6), "ringInliers": round(inl_share, 2),
            "surfaceSpread": round(s_h / sp, 2)}
    if inl_share < p.ring_inliers:
        # not one surface round the stroke (a fold, a corner): not ours to judge
        info["why"] = "the rim is not one surface"
        return ObjectSelection("none", sel, info=info)

    proud_thr = max(p.proud * sp, 3.0 * s_h)
    skin = max(2.0 * sp, 2.5 * s_h)                 # the surface's own thickness, in front

    # --- the surface's material --------------------------------------------
    lab = _lab(np.clip(0.5 + 0.28209479 * s.f_dc[L].astype(np.float64), 0, 1))
    base = ring & (np.abs(relief) < skin)
    if base.sum() < 30:
        info["why"] = "no surface at the rim"
        return ObjectSelection("none", sel, info=info)
    L0 = float(np.median(lab[base, 0]))
    a0, b0 = np.median(lab[base, 1]), np.median(lab[base, 2])
    d_ab = np.hypot(lab[:, 1] - a0, lab[:, 2] - b0)
    dL = lab[:, 0] - L0
    s_ab = float(np.percentile(d_ab[base], 75))
    s_L = float(np.percentile(np.abs(dL[base]), 75))
    thr_ab = max(p.colour_k * s_ab, 10.0)
    thr_ab_weak = max(0.6 * thr_ab, 6.0)
    thr_L = max(3.0 * s_L, 25.0)
    proud = relief > proud_thr
    foreign = d_ab > thr_ab
    weak = d_ab > thr_ab_weak
    bright = np.abs(dL) > thr_L

    core = in_sel & (proud | foreign | (bright & (relief > -skin)))
    n_core = int(core.sum())
    info.update({"core": n_core, "proudSel": int((in_sel & proud).sum()),
                 "foreignSel": int((in_sel & foreign).sum())})
    if n_core < max(p.min_core_n, p.min_core * len(sel)):
        info["why"] = "no object in the stroke"
        return ObjectSelection("none", sel, info=info)
    sgn = np.sign(np.median(dL[core]))

    # --- grow through touching splats --------------------------------------
    reach = p.reach * sp
    near = np.isfinite(d_sel) & (d_sel <= reach) | in_sel
    cand = near & (relief > -skin) & (proud | weak | (bright & (np.sign(dL) == sgn)) | core)
    ci = np.where(cand)[0]
    grown = _components(XL[ci], core[ci], p.link * sp)
    obj = np.zeros(len(L), bool)
    obj[ci[grown]] = True
    obj |= core
    n_obj = int(obj.sum())
    proud_share = float(np.average(proud[obj], weights=alpha[obj]))
    height = float(np.percentile(relief[obj], 95))

    # --- its footprint on the surface ----------------------------------------
    cell = 2.0 * sp
    o_uv = UV[obj]
    lo = o_uv.min(0) - (p.margin + p.close + 14) * cell
    hi = o_uv.max(0) + (p.margin + p.close + 14) * cell
    nu, nv = (np.ceil((hi - lo) / cell).astype(int) + 1)
    if nu * nv > 8_000_000:
        info["why"] = "object too large"
        return ObjectSelection("none", sel, info=info)
    gi = np.floor((UV - lo) / cell).astype(np.int64)
    ing = (gi[:, 0] >= 0) & (gi[:, 0] < nu) & (gi[:, 1] >= 0) & (gi[:, 1] < nv)
    gu, gv = np.clip(gi[:, 0], 0, nu - 1), np.clip(gi[:, 1], 0, nv - 1)
    O = np.zeros((nv, nu), bool)
    O[gv[obj], gu[obj]] = True
    if p.close:
        O = ndimage.binary_closing(np.pad(O, p.close + 1), np.ones((3, 3)),
                                   iterations=p.close)[p.close + 1:-p.close - 1, p.close + 1:-p.close - 1]
    O = ndimage.binary_fill_holes(O)

    # The surface still there behind it, per cell, against the ring's: how
    # opaque what is left of the column is once the object is gone, and in
    # which colour. The slab from the skin down a little way, not the skin
    # alone: the hut's wall is a thin skin over a denser layer nine spacings
    # in, and behind the pole the skin is thin where the pole hid it from
    # most photos, yet from the front the wall there looks whole. Behind
    # the EHBO letters the column is as opaque -- but dark red, the letters'
    # backs: not the wall.
    slab = (relief < skin) & (relief > -max(30.0 * sp, 4.0 * skin))
    rem = np.where(slab & ing & ~obj)[0]
    # composited front to back, as the camera sees the column
    key = gv[rem] * nu + gu[rem]
    order = np.lexsort((-relief[rem], key))
    rem, key = rem[order], key[order]
    lgt = np.log1p(-0.99 * alpha[rem])
    cs = np.cumsum(lgt)
    first = np.r_[True, key[1:] != key[:-1]]
    base_ = np.maximum.accumulate(np.where(first, np.arange(len(key)), 0))
    excl = cs - lgt - np.r_[0.0, cs][base_]
    wgt = alpha[rem] * np.exp(excl)
    grids = np.zeros((4, nv, nu))
    for k_, val in enumerate((-lgt, wgt * lab[rem, 0], wgt * lab[rem, 1], wgt * lab[rem, 2])):
        np.add.at(grids[k_], (gv[rem], gu[rem]), val)
    wsum = np.zeros((nv, nu))
    np.add.at(wsum, (gv[rem], gu[rem]), wgt)
    # A splat covers more than the cell its centre falls in: spread each
    # one's optical depth a little, or a wall of large, sparse splats reads
    # as mostly holes. Not by much, or the wall round the letters closes
    # over the letters' own footprint.
    sig_s = np.exp(s.scale[L[rem]].astype(np.float64)).max(1) if len(rem) else np.ones(1)
    s_c = float(np.clip(np.median(sig_s) / cell, 0.5, 1.5))
    grids = np.stack([ndimage.gaussian_filter(g_, s_c, mode="constant") for g_ in grids])
    tot = np.maximum(ndimage.gaussian_filter(wsum, s_c, mode="constant"), 1e-12)
    have = 1.0 - np.exp(-grids[0] * (2.0 * np.pi * s_c * s_c))
    c_L = grids[1] / tot - L0
    c_ab = np.hypot(grids[2] / tot - a0, grids[3] / tot - b0)
    ring_cells = np.zeros((nv, nu), bool)
    rc = ring & ing & slab
    ring_cells[gv[rc], gu[rc]] = True
    ring_cells &= ~ndimage.binary_dilation(O, iterations=2)
    dens = float(np.median(have[ring_cells])) if ring_cells.sum() >= 8 else 0.0
    # Judged against the ring's columns, composited the same way: single
    # splats' colours scatter far more than what a camera sees of them (the
    # hut's wall: 10 units of chroma per splat, under 3 per column).
    if ring_cells.sum() >= 8:
        t_ab = max(3.0 * float(np.percentile(c_ab[ring_cells], 75)), 6.0)
        t_L = max(3.0 * float(np.percentile(np.abs(c_L[ring_cells]), 75)), 15.0)
    else:
        t_ab, t_L = thr_ab_weak, thr_L
    # Chroma decides; lightness only in the large. Behind the pole the wall
    # is in the pole's own shadow -- darker, the same cream (median 1.5
    # units of chroma off the ring); behind the letters it is their dark
    # red backs (10.6 off).
    whole = (have >= p.intact * dens) & (c_ab <= t_ab) if dens > 0 else np.zeros_like(O)
    intact_share = float(whole[O].mean()) if O.any() else 0.0
    lit_share = float((np.abs(c_L) < t_L)[O].mean()) if O.any() else 0.0
    standoff = float(np.median(relief[obj])) / sp
    if DEBUG:
        _LAST.update(locals())
    info.update({"object": n_obj, "proudShare": round(proud_share, 2),
                 "height": round(height / sp, 1), "standoff": round(standoff, 1),
                 "cells": int(O.sum()), "intact": round(intact_share, 2),
                 "lit": round(lit_share, 2)})

    if proud_share >= 0.5 and standoff >= 2.0 and intact_share >= p.lift_share \
            and lit_share >= 0.6:
        # Standing clear of a whole surface: take the object, and anything
        # else standing proud in its column (haze it dragged along); the
        # surface the brush touched behind it stays.
        Ow = ndimage.binary_dilation(O, iterations=p.margin)
        col = ing & Ow[gv, gu] & (relief > proud_thr)
        out = obj | (col & (near | in_sel))
        idx = L[out]
        info.update({"objsel": "lift", "removed": int(len(idx)),
                     "keptSurface": int((in_sel & ~out).sum())})
        return ObjectSelection("lift", idx, L[obj], info)

    # --- fill: the footprint, the drop shadow, everything in front of the skin
    F = ndimage.binary_dilation(O, iterations=p.margin) if p.margin else O.copy()
    raised = proud_share >= 0.3 and height > proud_thr
    n_shadow = 0
    if p.shadow and dens > 0:
        # What still looks wrong round the footprint goes with it: a raised
        # object's drop shadow, a halo of its colour. The rim is what the
        # fill copies from and blends into -- left there, the EHBO letters'
        # shadow came back as a dark band through the middle of the fill.
        # Judged per column as the camera sees it, against the ring; no
        # wider than a raised object stands out (two cells for a print).
        S = int(np.clip(round(0.75 * height / cell), 2, 6)) if raised else 2
        t_dark = max(2.0 * float(np.percentile(np.abs(c_L[ring_cells]), 75)), 8.0) \
            if ring_cells.sum() >= 8 else thr_L
        seen = have >= 0.5 * dens
        odd = seen & ((c_L < -t_dark) | (c_ab > t_ab))
        band = ndimage.binary_dilation(F, iterations=S) & ~F
        # Darker than the wall just past the band, too. The ring's own
        # spread is no measure where the surface round the object is lit
        # unevenly: above the EHBO letters the eave's shadow doubled it, the
        # O's drop shadow passed for wall (14 cells taken, the crescent
        # stayed on the rim), and the fill drew the O's outline from it.
        # Against a local median of what lies beyond the band the
        # crescent stands out; the eave's shadow is its own reference.
        beyond = seen & ~(band | F)
        if beyond.sum() >= 8:
            # only where it is read below: the ring's cells and the band
            loc = _local_nanmedian(np.where(beyond, c_L, np.nan), 4 * S + 3,
                                   (ring_cells & beyond) | (seen & band))
            dev = c_L - loc
            rr = dev[ring_cells & beyond & np.isfinite(dev)]
            if len(rr) >= 8:
                t_loc = max(2.0 * float(np.percentile(np.abs(rr), 75)), 8.0)
                odd |= seen & band & np.isfinite(dev) & (dev < -t_loc)
        lab_, nl = ndimage.label((odd & band) | F)
        keep = np.unique(lab_[F])
        sh = np.isin(lab_, keep[keep > 0]) & band
        n_shadow = int(sh.sum())
        F |= sh
        F = ndimage.binary_fill_holes(ndimage.binary_closing(F, np.ones((3, 3))) | F)
    inF = ing & F[gv, gu]
    # The column goes a little below the skin too: under the EHBO letters
    # dark splats five spacings in, just under the skin, were what the
    # camera saw in the hole, and they showed through the fill as smudges.
    # The repair lays its own backing under the new face. Under a flat print
    # only the skin goes: below it on the heel was the shoe's dark inside,
    # and with a deeper cut it showed through the fill as a grey-green smear.
    cut = max(2.0 * skin, 8.0 * sp) if raised else skin
    if p.cut is not None:
        cut = p.cut * sp
    out = obj | (inF & (relief > -cut))
    # and the object's colour deeper in: the letters' backs, dark red behind
    # the wall's skin, showed through the fill as red ghosts
    out |= inF & weak & slab
    # faint, wide splats of the object's colour reaching into the footprint
    dist = ndimage.distance_transform_edt(~F) * cell
    sig = np.exp(s.scale[L].astype(np.float64)).max(1)
    reachF = ing & (dist[gv, gu] <= 2.0 * sig) & weak & (relief > -skin)
    out |= reachF
    idx = np.union1d(L[out], sel)
    info.update({"objsel": "fill", "shadowCells": n_shadow, "footprint": int(F.sum()),
                 "added": int(len(idx) - len(sel)), "faint": int((reachF & ~obj & ~inF).sum())})
    if DEBUG:
        _LAST.update(locals())
    return ObjectSelection("fill", idx, L[obj], info)
