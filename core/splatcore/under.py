"""Clear what an engine left of the old scene under its new skin.

A fill or a stamp takes out what the user painted and what stands in the way
above the new surface. What lies *under* it mostly stays: the lower layers of
the old skin, inner layers, the dark mush a reconstruction leaves inside an
object (the remains of the scanning stand inside the shoe). Under a closed
skin nobody would see it. But the skin has gaps, it thins out towards its
rim, it is looked at from the side, and a viewer far away stops drawing its
small splats: then what is underneath shows through.

`clear_under` removes the old splats that lie under the new skin:

* **under** means straight under it: below the skin along the skin's own
  normal (not along the user's view: a stamp's skin faces its surface, a
  fill's flank faces sideways), from just under it down to ``depth``
  spacings, with the skin itself above the splat (its foot point lies on the
  skin);
* **not at the rim**: a splat closer to the skin's rim than about its own
  size is left alone. It also shows *beside* the new skin, and outside the
  skin nothing may change;
* **not a real surface that carries on**: the tube under the skin also holds
  things that belong to the object. The shoe's side wall runs on under the
  sole's edge; behind leather lies the lining; a thin part has a far side.
  Two things tell such a surface from mush, and a candidate needs both.
  It is a *sheet*: the old splats around it (a few spacings, the near ones
  counting most) lie in a plane that is thin, and thin against how far they
  spread in it, and it lies in that plane itself. Scattered blobs do not.
  And the sheet *carries on*: it is joined to scan that stays, beside the
  tube, below it or at the skin's rim. What has been found to stay counts
  as staying for the next: the surface grows into the tube from where it
  enters it, round by round, as far as it is one surface. A flake that
  floats in the tube joined to nothing goes; so does a stub that is odd in
  colour next to what goes;
* **but not the old skin itself**: its lower layers are a sheet as well, and
  one that carries on all around. A sheet that lies parallel to the new skin
  and within ``skin`` spacings under it is what the new skin replaces.

Nothing is ever put back: the result only adds to what the engine removes.
"""
from __future__ import annotations

from dataclasses import dataclass

import numpy as np
from scipy.spatial import cKDTree

from .io import SplatSet, SH_C0
from .skin import Cells


@dataclass
class UnderParams:
    depth: float = 8.0        # the old scene is cleared down to this far under the skin (spacings)
    top: float = 0.25         # ... from this far under it (what lies higher was the engine's to clear)
    foot: float = 0.8         # "the skin is above it": a cell of the skin within this, across (spacings)
    rim: float = 1.0          # nothing is touched closer to the skin's rim than this (spacings) ...
    rim_sigma: float = 1.0    # ... plus this many of the splat's own sigmas,
    rim_reach: float = 3.0    # ... which count up to this many spacings
    faint: float = 0.05       # splats fainter than this are left where they are (opacity)
    skin: float = 3.0         # a sheet parallel to the new skin and this close under it is the old skin
    parallel: float = 30.0    # "parallel": within this many degrees
    # the sheet rule: who stays because it is surface that carries on
    reach: float = 3.5        # neighbours are looked for this far (spacings)
    near: float = 1.5         # ... and count less with distance: a Gaussian this wide (spacings)
    least: int = 5            # a sheet needs this many neighbours
    thick: float = 1.0        # their plane must be thinner than this (RMS, spacings),
    flat: float = 0.6         # ... and than this share of how far they spread in it,
    off_max: float = 1.2      # and the candidate this close to it (spacings)
    link: int = 3             # it carries on if this many neighbours in its plane stay
    rounds: int = 80          # the surface grows inwards at most this many times
    odd: float = 0.15         # a stub differs from its staying neighbours' colour by more than this


def _sheet(pts, X, idx, w, sp, P):
    """The plane through each ``X``'s neighbours ``pts[idx]`` with weights ``w`` (0: not a neighbour).

    Returns (is ``X`` in a sheet, the plane's unit normal, a point of it).
    """
    cnt = (w > 0).sum(1)
    W = np.maximum(w.sum(1), 1e-30)
    Pn = pts[idx]
    mu = (w[..., None] * Pn).sum(1) / W[:, None]
    d_ = Pn - mu[:, None]
    ev, evec = np.linalg.eigh(np.einsum("nk,nki,nkj->nij", w, d_, d_) / W[:, None, None])
    n = evec[:, :, 0]
    thick = np.sqrt(np.maximum(ev[:, 0], 0.0))
    spread = np.sqrt(np.maximum(ev[:, 1], 0.0))
    off = np.abs(((X - mu) * n).sum(1))
    ok = (cnt >= P.least) & (thick < P.thick * sp) & (thick < P.flat * spread) & (off < P.off_max * sp)
    return ok, n, mu


def clear_under(s: SplatSet, removed, skin: SplatSet, nrm, spacing, params=None, debug=None, grid=None):
    """Old splats of ``s`` under the new ``skin`` that should go with what the engine removed.

    ``removed``: indices into ``s`` the engine already takes out. ``nrm``:
    the outward normal of every splat of the skin (`skin.normals`).
    ``spacing``: the scan's splat spacing there. ``grid``: the skin's cells
    (`skin.Cells`) if the caller has them already.

    Returns (indices into ``s``, none of them in ``removed``; info):
    ``candidates`` lay in the tube under the skin, ``kept`` of them stay as
    surface that carries on, ``cleared`` go. ``debug``: a dict that is given
    the candidates and which of them stay.
    """
    P = params or UnderParams()
    sp = float(spacing)
    none = np.zeros(0, np.int64)
    info = {"cleared": 0, "candidates": 0, "kept": 0, "depth": float(P.depth)}
    if nrm is None:
        info["why"] = "without a view direction there is no telling which way is down"
        return none, info
    if skin.n < 20 or s.n == 0:
        return none, info
    grid = grid or Cells(skin, nrm, sp)
    A = grid.xyz
    rem = np.zeros(s.n, bool)
    rem[np.asarray(removed, np.int64)] = True
    D = P.depth * sp
    pad = D + 2.0 * sp
    inbox = ((s.xyz >= (A.min(0) - pad).astype(np.float32)) & (s.xyz <= (A.max(0) + pad).astype(np.float32))).all(1)
    box = np.where(inbox & ~rem)[0]
    if not len(box):
        return none, info
    alpha_b = s.alpha[box]
    X = s.xyz[box].astype(np.float64)

    on, dep, cell = grid.below(X, D, P.foot)
    nx, inside = grid.normal[cell], grid.rim[cell]
    cand = on & (dep > P.top * sp) & (dep <= D) & (alpha_b > P.faint)
    box, X, dep, nx, inside = box[cand], X[cand], dep[cand], nx[cand], inside[cand]
    if len(box):
        # ... and the skin is above it all around, as far as the splat itself reaches: not at the rim
        sig = s.sigma[box].max(1)
        keep = inside >= P.rim * sp + np.minimum(P.rim_sigma * sig, P.rim_reach * sp)
        box, X, dep, nx = box[keep], X[keep], dep[keep], nx[keep]
    info["candidates"] = int(len(box))
    if not len(box):
        return none, info

    # ---- the sheet rule: which of them are surface that carries on
    ci = box
    nc = len(ci)
    reach = P.reach * sp
    lo, hi = (X.min(0) - reach).astype(np.float32), (X.max(0) + reach).astype(np.float32)
    near = np.where(((s.xyz >= lo) & (s.xyz <= hi)).all(1))[0]
    is_c = np.zeros(s.n, bool)
    is_c[ci] = True
    solid = s.alpha[near] > 0.3
    ri = near[~rem[near] & ~is_c[near] & solid]            # the scan as it stands, around them
    zi = near[rem[near] & solid]                           # what goes whatever happens
    stays_c = np.zeros(nc, bool)
    if len(ri) >= P.link:
        ids = np.concatenate([ri, ci, zi])
        pts = s.xyz[ids].astype(np.float64)
        nr = len(ri)
        k = int(min(40, len(pts)))
        dist, idx = cKDTree(pts).query(X, k=k, distance_upper_bound=reach)
        dist, idx = dist.reshape(nc, k), idx.reshape(nc, k)
        nb = np.isfinite(dist) & (idx != (nr + np.arange(nc))[:, None])
        idx = np.where(nb, idx, 0)
        wgt = np.where(nb, np.exp(-0.5 * (np.where(nb, dist, 0.0) / (P.near * sp)) ** 2), 0.0)
        old = idx < nr + nc                                # the old scene that is still there (not what the engine took)
        # is it a sheet, among the old scene around it?
        is_sheet, m_n, mu = _sheet(pts, X, idx, wgt * old, sp, P)
        # the old skin's own lower layers: a sheet parallel to the new skin, close under it
        cos_par = np.cos(np.radians(P.parallel))
        skin_c = is_sheet & (np.abs((m_n * nx).sum(1)) > cos_par) & (dep < P.skin * sp)
        if skin_c.any():
            # ... which is not what the others are to be judged among: a wall
            # that passes through those layers is a sheet without them
            other = np.ones(len(pts), bool)
            other[nr + np.where(skin_c)[0]] = False
            again = np.where(~skin_c)[0]
            ok2, n2, mu2 = _sheet(pts, X[again], idx[again], (wgt * old)[again] * other[idx[again]], sp, P)
            is_sheet[again], m_n[again], mu[again] = ok2, n2, mu2
            is_sheet[skin_c] = False
        # does it carry on? joined, in its own plane, to scan that stays
        in_plane = nb & (np.abs(((pts[idx] - mu[:, None]) * m_n[:, None]).sum(2)) < P.off_max * sp)
        stays = np.zeros(len(pts), bool)
        stays[:nr] = True
        todo = np.where(is_sheet)[0]
        for _ in range(int(P.rounds)):
            if not len(todo):
                break
            joined = (in_plane[todo] & stays[idx[todo]]).sum(1) >= P.link
            if not joined.any():
                break
            stays[nr + todo[joined]] = True
            todo = todo[~joined]
        col = 0.5 + SH_C0 * s.f_dc[ids].astype(np.float64)
        for _ in range(3):
            gi = np.where(stays[nr:nr + nc])[0]
            if not len(gi):
                break
            edge = (nb[gi] & ~stays[idx[gi]]).any(1)       # beside something that goes
            if not edge.any():
                break
            gi = gi[edge]
            w = in_plane[gi] & stays[idx[gi]]              # the sheet it lies in, as far as it stays
            with np.errstate(all="ignore"):
                med = np.nanmedian(np.where(w[..., None], col[idx[gi]], np.nan), axis=1)
            odd = ~w.any(1) | (np.abs(col[nr + gi] - np.nan_to_num(med)).max(1) > P.odd)
            if not odd.any():
                break
            stays[nr + gi[odd]] = False
        stays_c = stays[nr:nr + nc]
    if debug is not None:
        debug.update({"candidates": ci, "stays": stays_c, "depth": dep / sp})
    out = ci[~stays_c]
    info["kept"] = int(stays_c.sum())
    info["cleared"] = int(len(out))
    return out, info
