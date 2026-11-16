"""A coarse layer behind a new skin, for viewers that stop drawing small splats.

The picture engines lift every pixel of a picture to one small flat splat,
three to five to a splat spacing. Close up that is what makes the skin crisp.
But viewers drop a splat once it is smaller on screen than a pixel or two
(PlayCanvas: ``minPixelSize``, 2 by default), and the skin's splats get there
long before the scan's own, larger ones do: zoom out and the fill is gone
while everything around it still stands, and what lies under it shows.

`backing` gives the skin what the scan has: large splats. Under the fine
skin it lays sparse layers made from the skin's own splats, one splat per
cell of the skin:

* the layers have cells one, two, four and eight spacings across
  (``pitches``): about one splat per square spacing in the first, a quarter
  of that in each further one. A splat is as large as its cell (its sigma
  ``sigma`` cells, taken from how the cell's own splats are spread, so a
  cell the skin's rim cuts gets a splat that ends with it). The first layer
  is what shows once the skin's own splats are dropped; the coarser ones
  are what is left from very far, as the scan's own largest splats are.
  (A layer of half-spacing cells keeps a stamp's pattern crisper over a
  short range of distances and costs a third more splats: left out.)
* it lies flat in the plane of the skin there, under the lowest of the fine
  skin within its reach (relief, an edge): by ``deep`` of its own sigma, at
  least ``drop`` spacings. That far, because a viewer draws splats in the
  order of their centres: looked at from a slant, a large splat lying just
  under small ones counts as nearer than those over its far half and is
  drawn over them (a fill lost a tenth of its fine contrast at 35
  degrees). And not into scan that stays below it (the far side of a thin
  part): where there is no room, the splat is smaller, or left out;
* it never reaches past the fine skin: its sigma is at most the distance to
  the skin's rim over ``inside``. Close up the fine skin is then above all
  of it and hides it; a splat that would have to be smaller than the fine
  ones it stands for is left out;
* its colour and its SH are the mean of the fine splats it stands for, and
  it is opaque.

A cell gets a splat only where the skin there is made of splats smaller than
the scan's own (``small``; copied Gaussians have the scan's sizes and need
none), and only in the layers that are clearly coarser than the skin itself
(``larger``; a large fill is lifted from a coarse picture and needs the
coarse layers only).

Far away, when the fine splats are dropped, the layers are what is left: the
surface stays closed and keeps its colour, a little soft, which at that size
on screen is hard to tell.
"""
from __future__ import annotations

import math
from dataclasses import dataclass

import numpy as np
from scipy.spatial import cKDTree

from .io import SplatSet
from .skin import Cells, cells, planes, tangent_sigma, tangents


@dataclass
class LodParams:
    pitches: tuple = (1.0, 2.0, 4.0, 8.0)   # the layers' cell widths, in spacings
    sigma: float = 0.6        # a backing splat's sigma in the skin, in cell widths
    flat: float = 0.15        # ... and across the skin, as a share of that
    drop: float = 0.4         # it lies at least this far under the fine skin (spacings),
    deep: float = 1.25        # ... and at least this many of its own sigmas
    above: float = 0.75       # ... and this far above scan that stays below it (spacings)
    drop_max: float = 2.5     # no splat where the skin's relief alone is deeper than this (cell widths)
    inside: float = 2.5       # the skin's rim is at least this many of its sigmas away
    small: float = 0.7        # a cell gets a splat if its skin's splats are smaller than this (sigma, spacings)
    larger: float = 1.5       # ... and the layer's are at least this much larger than they are
    least: float = 0.15       # ... and the cell holds at least this share of a whole cell's skin
    opacity: float = 0.97


def _low_quantile(of, h, m, q=0.05):
    """The ``q`` quantile of ``h`` within every one of ``m`` groups ``of`` (inf where empty)."""
    order = np.lexsort((h, of))
    cnt = np.bincount(of, minlength=m)
    start = np.concatenate([[0], np.cumsum(cnt)[:-1]])
    pick = np.minimum(start + np.floor(q * cnt).astype(np.int64), len(h) - 1)
    out = h[order[pick]]
    return np.where(cnt > 0, out, np.inf)


def _layer(grid: Cells, w, ts, f_dc, f_rest, P: LodParams, level: int, floor=None):
    """One layer: positions, quaternions (w, x, y, z), sigmas, f_dc, f_rest of its splats."""
    A, nrm, sp = grid.xyz, grid.nrm, grid.sp
    p = float(P.pitches[level]) * sp
    sig = P.sigma * p
    if abs(p - sp) < 1e-9 * sp:
        C, of, cell_of = grid.centres, grid.of, np.arange(len(grid.centres))
    else:
        # wider cells are made of the skin's own cells, not of its splats again
        C, cell_of, _ = cells(grid.centres, p, np.bincount(grid.of, w, minlength=len(grid.centres)))
        of = cell_of[grid.of]
    m = len(C)
    W = np.bincount(of, w, minlength=m)
    N, _ = planes(A, of, C, nrm, max(1.5 * p, 2.0 * sp), w)
    # how the cell's own splats are spread in the plane, their own size included
    d = A - C[of]
    h = (d * N[of]).sum(1)
    u, v = tangents(N)
    x, y = (d * u[of]).sum(1), (d * v[of]).sum(1)
    own = 0.5 * (ts[:, 0] ** 2 + ts[:, 1] ** 2)
    Wc = np.maximum(W, 1e-30)
    cxx = np.bincount(of, w * (x * x + own), minlength=m) / Wc
    cyy = np.bincount(of, w * (y * y + own), minlength=m) / Wc
    cxy = np.bincount(of, w * x * y, minlength=m) / Wc
    mid = 0.5 * (cxx + cyy)
    rad = np.hypot(0.5 * (cxx - cyy), cxy)
    ang = 0.5 * np.arctan2(2.0 * cxy, cxx - cyy)
    # a whole square cell of width p has a variance of p^2 / 12 along each side
    k = P.sigma * math.sqrt(12.0)
    # ... and never past the fine skin's rim
    lim = np.minimum(grid.rim[grid.tree.query(C)[1]] / P.inside, 1.3 * sig)
    s1 = np.minimum(k * np.sqrt(np.maximum(mid + rad, 0.0)), lim)
    s2 = np.minimum(k * np.sqrt(np.maximum(mid - rad, 0.0)), lim)
    e1 = np.cos(ang)[:, None] * u + np.sin(ang)[:, None] * v
    e2 = np.cross(N, e1)
    # under the lowest of the fine skin within its reach: its own cell's, and its neighbours'
    low = _low_quantile(of, h, m)
    low_pt = C + np.where(np.isfinite(low), low, 0.0)[:, None] * N
    pr = cKDTree(C).query_pairs(2.2 * p, output_type="ndarray")
    if len(pr):
        i, j = np.concatenate([pr[:, 0], pr[:, 1]]), np.concatenate([pr[:, 1], pr[:, 0]])
        rel = low_pt[j] - C[i]
        hj = (rel * N[i]).sum(1)
        lat = np.linalg.norm(rel - hj[:, None] * N[i], axis=1)
        reach = lat < P.inside * s1[i] + 0.5 * p
        np.minimum.at(low, i[reach], hj[reach])
    relief = np.maximum(-low, 0.0)
    # how far under the splat scan that stays begins: under what it covers itself, two sigmas around
    room = np.full(m, np.inf)
    there = np.where(np.isfinite(floor))[0] if floor is not None else np.zeros(0, np.int64)
    if len(there):
        lists = cKDTree(grid.centres[there]).query_ball_point(C, 2.0 * s1 + 0.5 * sp)
        cnt = np.fromiter((len(q) for q in lists), np.int64, m)
        if cnt.sum():
            flat = np.fromiter((g for q in lists for g in q), np.int64, int(cnt.sum()))
            np.minimum.at(room, np.repeat(np.arange(m), cnt), floor[there[flat]])
    # deep enough to stay behind the fine skin from a slant, and clear of what stays below
    fit = np.maximum(room - P.above * sp - relief, 0.0) / P.deep
    s1, s2 = np.minimum(s1, fit), np.minimum(s2, fit)
    down = relief + np.maximum(P.drop * sp, P.deep * s1)
    # the skin a whole cell holds: what a cell in the middle of the skin holds
    whole = float(np.percentile(W, 75)) if m else 0.0
    fine = np.sqrt(np.bincount(of, w * own, minlength=m) / Wc)     # the size of the skin's own splats there
    ok = ((fine < P.small * sp) & (sig >= P.larger * fine) & (s1 > P.larger * fine)
          & (W >= P.least * whole) & (relief <= P.drop_max * max(p, sp)))
    pos = C - down[:, None] * N
    dc = np.stack([np.bincount(of, w * f_dc[:, c], minlength=m) for c in range(3)], 1) / Wc[:, None]
    K = f_rest.shape[1]
    rest = np.zeros((m, K, 3), np.float32)
    for kk in range(K):
        for c in range(3):
            rest[:, kk, c] = np.bincount(of, w * f_rest[:, kk, c], minlength=m) / Wc
    from scipy.spatial.transform import Rotation
    q = Rotation.from_matrix(np.stack([e1, e2, N], axis=2)[ok]).as_quat() if ok.any() else np.zeros((0, 4))
    quat = np.concatenate([q[:, 3:4], q[:, :3]], 1)
    sg = np.stack([s1, np.maximum(s2, 0.25 * s1), P.flat * s1], 1)[ok]
    return pos[ok], quat, sg, dc[ok], rest[ok]


def backing(skin: SplatSet, nrm, spacing, params=None, grid=None, floor=None):
    """The coarse layers for a new ``skin``: (splats to add after it, info).

    ``nrm``: the outward normal of every splat of the skin (`skin.normals`);
    ``spacing``: the scan's splat spacing there; ``grid``: the skin's cells
    (`skin.Cells`) if the caller has them already; ``floor``: per cell of
    the grid, how far under the skin scan that stays begins (`Cells.floor`),
    None if nothing is known to lie there. ``info["added"]`` is how many
    splats the layers have, ``info["levels"]`` how many each.
    """
    P = params or LodParams()
    sp = float(spacing)
    empty = skin.take(np.zeros(0, np.int64))
    info = {"added": 0, "levels": []}
    if nrm is None:
        info["why"] = "without a view direction there is no telling which way is down"
        return empty, info
    if skin.n < 8:
        return empty, info
    ts = tangent_sigma(skin)
    if not (ts[:, 0] < P.small * sp).any():
        return empty, info                                  # nothing in it is finer than the scan
    grid = grid or Cells(skin, nrm, sp)
    w = ts[:, 0] * ts[:, 1] * skin.alpha                    # the skin each splat stands for
    f_dc = skin.f_dc.astype(np.float64)
    f_rest = skin.f_rest.astype(np.float64)
    parts = []
    for level in range(len(P.pitches)):
        if not (P.larger * ts[:, 0] < P.sigma * float(P.pitches[level]) * sp).any():
            info["levels"].append(0)                        # nothing this layer would be coarser than
            continue
        pos, quat, sg, dc, rest = _layer(grid, w, ts, f_dc, f_rest, P, level, floor)
        info["levels"].append(int(len(pos)))
        if not len(pos):
            continue
        n = len(pos)
        parts.append(SplatSet(
            xyz=pos.astype(np.float32), f_dc=dc.astype(np.float32), f_rest=rest.astype(np.float32),
            opacity=np.full(n, math.log(P.opacity / (1.0 - P.opacity)), np.float32),
            scale=np.log(np.maximum(sg, 1e-12)).astype(np.float32), rot=quat.astype(np.float32),
            extra={k: np.zeros((n,) + v.shape[1:], v.dtype) for k, v in skin.extra.items()}))
    if not parts:
        return empty, info
    out = parts[0]
    for part in parts[1:]:
        out = out.concat(part)
    info["added"] = int(out.n)
    return out, info
