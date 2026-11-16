"""The surface under a brush stroke, found the way the repair brush finds it.

The shadow and solidify brushes edit what is already there instead of
filling a hole, but they need the same three things the repair brush works
out first: a tangent frame, a grid over it, and which splats are the face of
the surface, which are the slab behind it and which are haze in front.
"""
from __future__ import annotations

from dataclasses import dataclass

import numpy as np
from scipy import ndimage
from scipy.spatial import cKDTree

from .io import SplatSet
from .repair import (RepairError, RepairParams, _cells, _cell_quantile,
                     _fill_nearest, _outer_shell, _rim_band,
                     fit_reference_surface, fit_tangent_frame)


@dataclass
class Patch:
    cand: np.ndarray        # indices of the splats considered, stroke included
    uvh: np.ndarray         # their (u, v, h) in the tangent frame
    ku: np.ndarray          # grid column of each
    kv: np.ndarray          # grid row of each
    nu: int
    nv: int
    lo: tuple               # (lo_u, lo_v) of the grid
    cell: float
    spacing: float
    origin: np.ndarray
    basis: np.ndarray       # rows (e_u, e_v, n)
    outward: float          # +1 if the face is at higher h
    depth: np.ndarray       # outward distance from the face; 0 at it, < 0 inside
    depth_carried: np.ndarray  # the same from the face carried in from outside
    face: np.ndarray        # bool over cand: the outermost dense layer
    body: np.ndarray        # bool over cand: face plus the slab behind it
    stroke: np.ndarray      # (nv, nu) cells the stroke covers
    radius: float
    width: float


def surface_patch(s: SplatSet, sel: np.ndarray, rng, view=None,
                  thinned: bool = False) -> Patch:
    """The surface around and under the painted ``sel`` (a bool mask).

    ``view``, the camera's forward direction, settles which side is the
    front. Without it the side holding less material is taken as outward,
    which is right on a shell and a coin toss on a slab that is as thick on
    one side of its middle as on the other.

    ``thinned`` says the surface under the stroke may have lost its
    opacity -- the solidify brush's whole subject -- so it is found there
    by position among layers dense by count, not by opacity.
    """
    if not sel.any():
        raise RepairError("nothing painted")
    xyz = s.xyz.astype(np.float64)
    alpha = s.alpha
    _, radius, width, spacing, cand, rim = _rim_band(xyz, sel, alpha,
                                                      RepairParams(), rng)
    # Unlike the repair brush, the painted splats stay: they are the subject.
    # Faint ones only when the brush is about faintness; otherwise a stroke
    # over a hole counts the hole's leftover wisps as surface, and the
    # shadow brush took the dark remains of a removed stand for a shadow and
    # lifted them fourfold.
    painted = sel if thinned else (sel & (alpha > 0.2))
    cand = np.unique(np.concatenate([cand, np.where(painted)[0]]))
    origin, basis, _ = fit_tangent_frame(xyz[rim], w=alpha[rim])
    uvh = (xyz[cand] - origin) @ basis.T
    cell = max(2.0 * spacing, 1e-9)
    lo_u, lo_v = uvh[:, 0].min() - cell, uvh[:, 1].min() - cell
    nu = int((uvh[:, 0].max() - lo_u) / cell) + 2
    nv = int((uvh[:, 1].max() - lo_v) / cell) + 2
    if nu * nv > 6_000_000:
        raise RepairError("the stroke is too large for one pass")
    ku, kv, _ = _cells(uvh[:, 0], uvh[:, 1], cell, lo_u, lo_v, nu, nv)
    rim_h = float(np.median(((xyz[rim] - origin) @ basis.T)[:, 2]))
    # The material count where it is clear -- on a shell, most of the mass
    # lies on one side -- and the camera where it is not. Letting the
    # camera overrule a clear count made one site of the solidify test
    # worse; letting the count decide a symmetric slab raised the haze.
    below = float(np.mean(uvh[:, 2] < rim_h))
    outward = 1.0 if below >= 0.5 else -1.0
    # The count is a coin toss on a curved or thin surface: on the curved
    # shadow site it came out at 0.46, the wrong side, and 82 percent of the
    # stroke was then judged not to be on the face. The painted splats are
    # the subject and sit on the face, so per painted splat take the share
    # of its own cell's material that lies above it -- little on the outward
    # side. Per column, so a folded surface does not mix its walls. Right on
    # all nine shadow sites, where the count was wrong on two.
    lin_c = kv * nu + ku
    order_c = np.lexsort((uvh[:, 2], lin_c))
    lin_s = lin_c[order_c]
    first = np.searchsorted(lin_s, lin_s, "left")
    last = np.searchsorted(lin_s, lin_s, "right")
    rank = np.arange(len(order_c)) - first
    size = last - first
    above = np.empty(len(cand))
    above[order_c] = np.where(size > 1, (size - 1 - rank) / np.maximum(size - 1, 1), 0.5)
    multi = np.empty(len(cand), bool)
    multi[order_c] = size > 1
    pm = sel[cand] & multi
    if pm.sum() >= 20:
        above_p = float(np.mean(above[pm]))
        if abs(above_p - 0.5) >= 0.05:
            outward = 1.0 if above_p < 0.5 else -1.0
    if (view is not None and np.linalg.norm(view) > 0
            and abs(below - 0.5) < 0.2):
        outward = 1.0 if float(np.dot(basis[2], view)) < 0 else -1.0
    in_sel = sel[cand]
    stroke = np.zeros((nv, nu), bool)
    stroke[kv[in_sel], ku[in_sel]] = True
    stroke = ndimage.binary_fill_holes(ndimage.binary_closing(stroke, np.ones((3, 3))))

    solid = alpha[cand] > 0.2
    q_out = 0.80 if outward > 0 else 0.20
    if not thinned:
        # The outermost dense layer, exactly as the repair brush takes it.
        surf = np.zeros(len(cand), bool)
        ok = np.where(solid)[0]
        surf[ok] = _outer_shell(uvh[ok, 2], (kv * nu + ku)[ok], outward,
                                gap_tol=8.0 * spacing)
        h_out, _ = _cell_quantile(uvh[surf, 2], kv[surf], ku[surf], nv, nu, q_out)
        h_out = ndimage.median_filter(_fill_nearest(h_out, np.isfinite(h_out)),
                                      size=3, mode="nearest")
    else:
        # Under the stroke the surface is the thing being edited, and its
        # opacity cannot be trusted to say where it is: made transparent it
        # is no longer "dense", and the outermost dense layer under it was
        # the shell's inner wall, eighteen spacings down -- every painted
        # splat then sat "in front of the face" and was left alone. Carrying
        # the face across from the surroundings fixed flat stretches and
        # undershot by ten spacings on the curved edge of the sole.
        #
        # Thinning moves nothing and removes nothing, though. So under the
        # stroke the face is the outermost layer by position among layers
        # dense by *count*: a splat of haze has a few neighbours, one of a
        # surface three or four times as many, thinned or not.
        under_stroke = stroke[kv, ku]
        pts = s.xyz[cand].astype(np.float64)
        nbrs = np.asarray(cKDTree(pts).query_ball_point(pts, 2.5 * spacing,
                                                        return_length=True))
        ref_set = solid & ~under_stroke
        ref_n = float(np.median(nbrs[ref_set] if ref_set.any() else nbrs))
        # a thinned surface keeps its full count, so the bar can sit close to
        # it; at 0.4 a chance clump of four haze splats passed as surface, at
        # 0.6 the sole lost real surface where its count varies
        dense = nbrs >= 0.5 * ref_n
        usable = np.where(solid | (under_stroke & dense))[0]
        surf = np.zeros(len(cand), bool)
        surf[usable] = _outer_shell(uvh[usable, 2], (kv * nu + ku)[usable],
                                    outward, gap_tol=8.0 * spacing)
        h_out, _ = _cell_quantile(uvh[surf, 2], kv[surf], ku[surf], nv, nu, q_out)
        known = np.isfinite(h_out)
        if known.sum() >= 8 and not known.all():
            h_out = fit_reference_surface(np.where(known, h_out, 0.0), known,
                                          smooth_cells=1.0, plate=True)
        elif not known.all():
            h_out = _fill_nearest(h_out, known)
        h_out = ndimage.median_filter(h_out, size=3, mode="nearest")
    # Where the surrounding surface says the face continues under the
    # stroke: measured outside it, carried across with a thin plate. Not used
    # to place anything -- on the curved edge of the sole it misses by ten
    # spacings -- but a hole is a hundred and more below it, which is how a
    # brush tells a hole from a surface. Over the stand's hole the material
    # left lay a median 148 spacings under it; under real shadows, within 13.
    hc, _ = _cell_quantile(uvh[surf, 2], kv[surf], ku[surf], nv, nu, q_out)
    away = ~ndimage.binary_dilation(stroke, np.ones((3, 3)), iterations=2)
    kn = np.isfinite(hc) & away
    if kn.sum() >= 8:
        hc = fit_reference_surface(np.where(kn, hc, 0.0), kn, smooth_cells=3.0,
                                   plate=True)
    else:
        hc = h_out
    depth_carried = outward * (uvh[:, 2] - hc[kv, ku])
    depth = outward * (uvh[:, 2] - h_out[kv, ku])
    face = surf & (np.abs(depth) < 3.0 * spacing)
    body = (np.abs(depth) < 3.0 * spacing) | ((depth < 0) & (depth > -10.0 * spacing))
    if face.sum() < 30:
        raise RepairError("no surface can be found here")
    return Patch(cand=cand, uvh=uvh, ku=ku, kv=kv, nu=nu, nv=nv, lo=(lo_u, lo_v),
                 cell=cell, spacing=spacing, origin=origin, basis=basis,
                 outward=outward, depth=depth, depth_carried=depth_carried,
                 face=face, body=body,
                 stroke=stroke, radius=radius, width=width)
