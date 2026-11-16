"""A freshly made skin of splats: where it faces, the cells it falls into, and finishing it.

The picture fill, the picture stamp and the deep fill all end the same way:
old splats go, a skin of new ones is put in their place. Two things are then
done to every such skin, whichever engine made it (`finish`):

* what the engine left of the old scene *under* the skin is cleared
  (`under.py`), so it cannot show through gaps, at a slant, or once a viewer
  stops drawing the skin's small splats;
* the skin gets a sparse layer of large splats under it (`lod.py`), so that a
  viewer that drops small splats still shows a closed surface of the right
  colour.

Both need the same two things of the skin, which are here: which way each of
its splats faces (`normals`) and a division into cells of about one splat
spacing with a plane through each (`cells`, `planes`).

NumPy and SciPy only: nothing here needs a GPU.
"""
from __future__ import annotations

from typing import Optional

import numpy as np
from scipy.spatial import cKDTree

from .io import SplatSet


def rotmat(quat) -> np.ndarray:
    """Rotation matrices (N, 3, 3) of unit quaternions (w, x, y, z); the columns are the splat's axes."""
    q = np.asarray(quat, np.float64)
    w, x, y, z = q[:, 0], q[:, 1], q[:, 2], q[:, 3]
    R = np.empty((len(q), 3, 3))
    R[:, 0, 0] = 1 - 2 * (y * y + z * z)
    R[:, 0, 1] = 2 * (x * y - w * z)
    R[:, 0, 2] = 2 * (x * z + w * y)
    R[:, 1, 0] = 2 * (x * y + w * z)
    R[:, 1, 1] = 1 - 2 * (x * x + z * z)
    R[:, 1, 2] = 2 * (y * z - w * x)
    R[:, 2, 0] = 2 * (x * z - w * y)
    R[:, 2, 1] = 2 * (y * z + w * x)
    R[:, 2, 2] = 1 - 2 * (x * x + y * y)
    return R


def normals(skin: SplatSet, view=None, signed: Optional[bool] = None):
    """The outward unit normal of every splat of a new skin, or None if it cannot be told.

    A splat's normal is its thinnest axis. Which of the two ways along it is
    *out* the axis does not say. The picture engines lift every pixel to a
    flat splat whose third axis points at the camera of its picture: there
    the sign is known (``signed``; None: taken as known when the third axis
    is the thinnest for practically all of them). Copied Gaussians carry no
    such sign; they are turned to face the user (against ``view``, the
    direction the user looked along).
    """
    if skin.n == 0:
        return np.zeros((0, 3))
    R = rotmat(skin.quat)
    thin = np.argmin(skin.scale, axis=1)
    if signed is None:
        signed = bool((thin == 2).mean() > 0.99)
    if signed:
        return R[:, :, 2]
    if view is None:
        return None
    v = np.asarray(view, np.float64)
    v = v / max(float(np.linalg.norm(v)), 1e-12)
    n = R[np.arange(skin.n), :, thin]
    return np.where(((n @ v) > 0)[:, None], -n, n)


def spacing(xyz, alpha, c, r):
    """The scan's splat spacing around ``c``: the median distance between
    neighbouring opaque splats within ``r`` (as `imfill._spacing` measures it); None if too few."""
    near = np.where((np.linalg.norm(np.asarray(xyz, np.float64) - np.asarray(c, np.float64), axis=1) < max(r, 1e-6))
                    & (np.asarray(alpha) > 0.3))[0]
    if len(near) < 20:
        return None
    pts = np.asarray(xyz, np.float64)[near]
    d = cKDTree(pts).query(pts[:: max(1, len(near) // 4000)], k=2)[0][:, 1]
    d = d[d > 0]
    return float(np.median(d)) if len(d) else None


def tangent_sigma(skin: SplatSet) -> np.ndarray:
    """Each splat's two larger sigmas (N, 2), larger first: its size in the skin."""
    s = np.sort(skin.sigma, axis=1)
    return s[:, :0:-1]


def cells(xyz, pitch: float, weight=None, relax: int = 2):
    """Divide the points of a skin into cells about ``pitch`` across.

    Returns (centres (M, 3), the cell of every point (N,), each cell's weight
    (M,)). The points are binned into cubes first; a sheet cuts its cubes
    into pieces of every size, so cells that came out next to a heavier one
    are given up and the points go to the nearest centre that is left, a few
    times over (``relax``): the centres end up spread evenly over the sheet,
    however it lies or bends.
    """
    X = np.asarray(xyz, np.float64)
    w = np.ones(len(X)) if weight is None else np.asarray(weight, np.float64)
    key = np.floor(X / pitch).astype(np.int64)
    key -= key.min(0)
    dims = key.max(0) + 1
    flat = (key[:, 0] * dims[1] + key[:, 1]) * dims[2] + key[:, 2]
    _, of = np.unique(flat, return_inverse=True)

    def centre(of, m):
        W = np.bincount(of, w, minlength=m)
        C = np.stack([np.bincount(of, w * X[:, k], minlength=m) for k in range(3)], 1)
        ok = W > 0
        return C[ok] / W[ok, None], W[ok], ok

    C, W, _ = centre(of, int(of.max()) + 1)
    for it in range(max(int(relax), 0) + 1):
        if len(C) > 1 and it < 2:
            # of two centres closer than 0.6 pitch the lighter goes
            pr = cKDTree(C).query_pairs(0.6 * pitch, output_type="ndarray")
            if len(pr):
                drop = np.zeros(len(C), bool)
                light = np.where(W[pr[:, 0]] < W[pr[:, 1]], pr[:, 0], pr[:, 1])
                drop[light] = True
                C = C[~drop]
        of = cKDTree(C).query(X)[1]
        C, W, ok = centre(of, len(C))
        if not ok.all():
            of = (np.cumsum(ok) - 1)[of]
    return C, of, W


def planes(xyz, of, centres, nrm, radius: float, weight=None):
    """A plane through the skin at every cell: (unit normals (M, 3), roughness (M,)).

    The plane is fitted to the points of the cell and of the cells within
    ``radius`` of it, and faces the way their splats do (``nrm``, per point).
    Roughness is the points' RMS distance from the plane. Where too few
    points lie around a cell to fit one, the mean of their normals is taken.
    """
    X = np.asarray(xyz, np.float64)
    m = len(centres)
    w = np.ones(len(X)) if weight is None else np.asarray(weight, np.float64)
    W = np.bincount(of, w, minlength=m)
    cnt = np.bincount(of, minlength=m).astype(np.float64)
    S1 = np.stack([np.bincount(of, w * X[:, k], minlength=m) for k in range(3)], 1)
    Sn = np.stack([np.bincount(of, w * nrm[:, k], minlength=m) for k in range(3)], 1)
    S2 = np.empty((m, 3, 3))
    for a in range(3):
        for b in range(a, 3):
            S2[:, a, b] = S2[:, b, a] = np.bincount(of, w * X[:, a] * X[:, b], minlength=m)
    pr = cKDTree(centres).query_pairs(radius, output_type="ndarray")
    if len(pr):
        i, j = np.concatenate([pr[:, 0], pr[:, 1]]), np.concatenate([pr[:, 1], pr[:, 0]])
        def around(a):
            """Each cell's own value plus its neighbours' (a: (M,) or (M, ...))."""
            flat = a.reshape(m, -1)
            return np.stack([flat[:, q] + np.bincount(i, flat[j, q], minlength=m)
                             for q in range(flat.shape[1])], 1).reshape(a.shape)
        W, cnt, S1, Sn, S2 = around(W), around(cnt), around(S1), around(Sn), around(S2)
    Wc = np.maximum(W, 1e-30)
    mu = S1 / Wc[:, None]
    cov = S2 / Wc[:, None, None] - mu[:, :, None] * mu[:, None, :]
    ev, evec = np.linalg.eigh(cov)
    N = evec[:, :, 0]
    mean_n = Sn / np.maximum(np.linalg.norm(Sn, axis=1, keepdims=True), 1e-30)
    N = np.where(((N * mean_n).sum(1) < 0)[:, None], -N, N)
    # a plane needs points spread in two directions
    poor = (cnt < 5) | (ev[:, 1] < 1e-4 * np.maximum(ev[:, 2], 1e-30))
    N = np.where(poor[:, None], mean_n, N)
    rough = np.sqrt(np.maximum(ev[:, 0], 0.0))
    return N, np.where(poor, 0.0, rough)


def tangents(n):
    """Two unit vectors across each normal: (N, 3) -> (N, 3), (N, 3), right-handed with it."""
    a = np.where((np.abs(n[:, :1]) < 0.9), np.array([[1.0, 0.0, 0.0]]), np.array([[0.0, 1.0, 0.0]]))
    e1 = a - (a * n).sum(1, keepdims=True) * n
    e1 /= np.maximum(np.linalg.norm(e1, axis=1, keepdims=True), 1e-12)
    return e1, np.cross(n, e1)


def rim_distance(centres, N, pitch: float, gap_deg: float = 135.0):
    """How far inside the skin's rim every cell lies (M,): inf for a skin without one.

    A cell is at the rim when its neighbours do not stand all around it:
    seen in its own plane, some sector wider than ``gap_deg`` holds none. A
    hole in the skin has a rim like any other. Every cell is then as far
    inside as the nearest such cell is away, plus the half cell that one has
    beyond its own centre.
    """
    m = len(centres)
    if m < 4:
        return np.full(m, 0.5 * pitch)
    tree = cKDTree(centres)
    pr = tree.query_pairs(1.8 * pitch, output_type="ndarray")
    widest = np.full(m, 2.0 * np.pi)
    if len(pr):
        i, j = np.concatenate([pr[:, 0], pr[:, 1]]), np.concatenate([pr[:, 1], pr[:, 0]])
        u, v = tangents(N)
        d = centres[j] - centres[i]
        ang = np.arctan2((d * v[i]).sum(1), (d * u[i]).sum(1))
        order = np.lexsort((ang, i))
        i_s, a_s = i[order], ang[order]
        gap = np.where(i_s[1:] == i_s[:-1], a_s[1:] - a_s[:-1], 0.0)
        big = np.zeros(m)
        np.maximum.at(big, i_s[1:], gap)
        first, last = np.full(m, np.inf), np.full(m, -np.inf)
        np.minimum.at(first, i_s, a_s)
        np.maximum.at(last, i_s, a_s)
        has = np.isfinite(first)
        widest[has] = np.maximum(big[has], 2.0 * np.pi - (last[has] - first[has]))   # ... and round the back
    edge = widest > np.radians(gap_deg)
    if not edge.any():
        return np.full(m, np.inf)
    return cKDTree(centres[edge]).query(centres)[0] + 0.5 * pitch


class Cells:
    """A new skin divided into cells one spacing across, as both finishing steps read it.

    ``xyz``, ``nrm``: the skin's splats and their outward normals; ``centres``,
    ``of``, ``weight``: the cells (`cells`); ``normal``: the skin's plane at each,
    smooth over three spacings (`planes`); ``rim``: how far inside the skin's
    rim each lies (`rim_distance`); ``tree``: a k-d tree over the centres.
    """

    def __init__(self, skin: SplatSet, nrm, spacing: float):
        self.sp = float(spacing)
        self.xyz = skin.xyz.astype(np.float64)
        self.nrm = np.asarray(nrm, np.float64)
        self.centres, self.of, self.weight = cells(self.xyz, self.sp)
        self.normal, _ = planes(self.xyz, self.of, self.centres, self.nrm, 3.0 * self.sp)
        self.rim = rim_distance(self.centres, self.normal, self.sp)
        self.tree = cKDTree(self.centres)

    def below(self, p, deep: float, foot: float = 0.8):
        """Is the skin above the points ``p``? (yes/no, depth under it, the cell above each)

        ``deep``: how far under the skin to look (units); ``foot``: a cell of
        the skin must lie within this many spacings across of the point
        straight above. In two steps: the cell nearest to ``p`` says which
        way is up there; ``p`` is carried up along that onto the skin, and
        only there is it asked whether a cell lies close by. (From deep down
        the nearest cells are any of those overhead: a little relief decides
        which, not who is straight above.)
        """
        sp, C, N = self.sp, self.centres, self.normal
        p = np.asarray(p, np.float64).reshape(-1, 3)
        dist, j = self.tree.query(p, distance_upper_bound=deep + 1.5 * sp)
        got = np.isfinite(dist)
        j = np.where(got, j, 0)
        d0 = -((p - C[j]) * N[j]).sum(1)
        at = p + d0[:, None] * N[j]
        kq = min(3, len(C))
        dist, k = self.tree.query(at, k=kq, distance_upper_bound=4.0 * sp)
        dist, k = dist.reshape(len(p), kq), k.reshape(len(p), kq)
        ok = np.isfinite(dist) & got[:, None]
        k = np.where(ok, k, 0)
        r = at[:, None, :] - C[k]
        h = (r * N[k]).sum(2)
        lat = np.linalg.norm(r - h[..., None] * N[k], axis=2)
        ok &= (lat <= foot * sp) & (np.abs(h) <= 3.0 * sp)
        best = np.argmin(np.where(ok, lat, np.inf), axis=1)
        row = np.arange(len(p))
        return ok.any(1), d0 - h[row, best], k[row, best]

    def floor(self, s: SplatSet, gone, deep: float, top: float = 3.0):
        """How far under each cell the old scene that stays begins: (M,), inf where nothing does.

        ``gone``: mask over ``s`` of what leaves with the edit. Counted from
        where a splat's own body begins (its centre less its largest sigma),
        between ``top`` spacings and ``deep`` (units) under the skin: what
        lies higher is the old skin's own lower layers, left at the new
        skin's rim, and no floor.
        """
        out = np.full(len(self.centres), np.inf)
        pad = deep + 2.0 * self.sp
        box = np.where(((s.xyz >= (self.xyz.min(0) - pad).astype(np.float32))
                        & (s.xyz <= (self.xyz.max(0) + pad).astype(np.float32))).all(1) & ~gone)[0]
        if len(box):
            box = box[s.alpha[box] > 0.3]
        if not len(box):
            return out
        on, dep, k = self.below(s.xyz[box].astype(np.float64), deep)
        on &= dep > top * self.sp
        begins = np.maximum(dep[on] - s.sigma[box[on]].max(1), 0.0)
        np.minimum.at(out, k[on], begins)
        return out


def finish(s: SplatSet, removed, added: SplatSet, view, spacing, face=None, signed: Optional[bool] = None,
           clear: bool = True, back: bool = True, under_params=None, lod_params=None):
    """Finish a new skin: clear what lies under it, lay the coarse layer behind it.

    ``s``: the scene the engine worked on; ``removed``: indices into it the
    engine takes out; ``added``: its new splats; ``view``: the direction the
    user looked along; ``spacing``: the scan's splat spacing there.
    ``face``: which of ``added`` are the skin itself (the copy engines also
    add a slab behind it); None: all. ``signed``: see `normals`.

    Returns (removed, added, info): the engine's ``removed`` plus what was
    cleared, its ``added`` with the backing layer appended (the engine's own
    splats stay first, in their order), and what each step did:
    ``info["under"]`` and ``info["lod"]``, None for a step that was off, and
    ``info["seconds"]`` for both with the dividing into cells they share.
    Neither step ever puts an old splat back.
    """
    import time
    t0 = time.time()
    removed = np.asarray(removed, np.int64).ravel()
    info = {"under": None, "lod": None, "seconds": 0.0}
    sp = float(spacing or 0.0)
    if not (clear or back) or added.n == 0 or not np.isfinite(sp) or sp <= 0:
        return removed, added, info
    skin = added if face is None else added.take(np.asarray(face, bool))
    nrm = normals(skin, view, signed)
    grid = Cells(skin, nrm, sp) if nrm is not None and skin.n >= 8 else None
    if clear:
        from .under import clear_under
        t = time.time()
        extra, got = clear_under(s, removed, skin, nrm, sp, under_params, grid=grid)
        got["seconds"] = round(time.time() - t, 3)
        info["under"] = got
        if len(extra):
            removed = np.union1d(removed, extra)
    if back:
        from .lod import backing
        t = time.time()
        floor = None
        if grid is not None:
            gone = np.zeros(s.n, bool)
            gone[removed] = True
            floor = grid.floor(s, gone, 12.0 * sp)
        layer, got = backing(skin, nrm, sp, lod_params, grid=grid, floor=floor)
        got["seconds"] = round(time.time() - t, 3)
        info["lod"] = got
        if layer.n:
            added = added.concat(layer)
    info["seconds"] = round(time.time() - t0, 3)
    return removed, added, info
