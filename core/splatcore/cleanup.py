"""Finding the junk a scan leaves behind.

Photogrammetry and 3DGS training both produce debris that no amount of
careful brushing is worth spending on. Two kinds are found here:

* **floaters** -- splats in empty space, cut off from everything else. They
  come from under-constrained regions: the optimiser had no photo that
  disagreed, so it left something there.
* **hidden haze** -- splats behind what the cameras saw and yet visible from
  where no camera stood: the fog under a beach's sand, seen from below, or
  the coloured fringe under a tree crown. Every training photo had something
  opaque in front of them, so nothing ever told the optimiser they were wrong.

The first version scored everything on isolation -- distance to the k-th
neighbour against the scene's median spacing -- combined with opacity and
size. On a shoe that was fine. On a beach hut it took 49.8k splats out of
shutters and the roof, and holes out of a castle's tree crowns: a shutter or
a crown is *locally sparse* (median isolation 5, size 1.1x its neighbours),
which is exactly what a floater looks like to a nearest-neighbour count.
Measuring against the neighbours' own spacing kept the structure but found
almost nothing (1175 / 191 splats) and left the haze. Sparse is not wrong.

So isolation is no longer evidence on its own. It still says *how*
suspicious a splat is (``base``), but it only counts where a second, separate
reason says the splat is junk:

* *detached* -- it sits in a small connected group of its own. Two splats are
  linked when they are within a few median spacings or their extents touch,
  which keeps a wire, a railing, a shutter or a leafy crown in one piece with
  the object while a floater stays alone.
* *hidden* -- seen from a sphere of cameras around the object (world up as
  the app's "Oben" toggle declares it, -y by default, the 3DGS convention),
  either it lies behind the blended surface from
  every view at the horizon or above *and* light from below reaches it, or
  most of what it shows, it shows to the views from below. The first means
  no plausible camera saw it, and the light from below separates haze from
  an object's interior, which is behind a floor or a sole from there too and
  stays. The second catches the fog at a scan's rim, which a low view from
  above glimpses but which shows mainly from underneath.

A third kind needed its own test: the purple fringe under a castle's tree
crowns, about 300 big opaque blobs seen from the side. From there they are in
front of everything, so they are not hidden, and they hang off the crown, so
they are not detached. Big, opaque and isolated alone took leaves out of the
crown tops; colour alone took 18.6k splats of real terrain. What gives them
away is where they can be seen from: nothing covers them from below, and the
steep views from above, where the photos were taken, see hardly any of them
(see ``fringe`` in :func:`analyse`).

Hidden haze is not always better gone. Under a beach hut it is the ground
nobody photographed: removed, it opened a square hole in the sand, and from
below the hut's streaky, never-seen underside showed through it. So before
the haze counts, the scene is rendered from below without it (``_reveal``):
where that would show empty space or sand seen from above, the haze goes;
where it would show a surface no camera from above saw, the haze stays
(``guarded``). Same for the inside of a shoe seen through the hole in its
sole.

A splat scores ``max(detached, base x hidden, fringe)``, the hidden part
damped by that guard. Without a GPU only the detached part is measured, and
that is a yes or no: the score is 0 or 0.95, so the threshold slider changes
nothing between them.
"""
from __future__ import annotations

from dataclasses import dataclass, field

import numpy as np
from scipy.spatial import cKDTree

from .io import SplatSet


@dataclass
class FloaterStats:
    """Per-splat measurements, computed once and re-thresholded cheaply."""

    isolation: np.ndarray      # (N,) distance to the k-th neighbour, in units of the median
    alpha: np.ndarray          # (N,) activated opacity
    bulk: np.ndarray           # (N,) size relative to the local neighbourhood
    score: np.ndarray          # (N,) 0..1, higher means more likely junk
    median_spacing: float
    k: int
    base: np.ndarray = None    # (N,) isolation/opacity/size suspicion alone
    detached: np.ndarray = None  # (N,) bool, in a small group of its own
    behind: np.ndarray = None  # (N,) least depth behind the surface from above, in spacings
    exposure: np.ndarray = None  # (N,) best transmittance from below, 0..1
    below: np.ndarray = None   # (N,) share of its blending weight seen from below
    hidden: np.ndarray = None  # (N,) 0..1, behind from above and lit from below
    steep: np.ndarray = None   # (N,) share of its blending weight seen from steeply above
    fringe: np.ndarray = None  # (N,) 0..1, big lone blob seen from the side and below
    reveal: np.ndarray = None  # (N,) share of its view from below that hides an unseen surface
    guarded: np.ndarray = None  # (N,) bool, hidden haze kept because of what it covers
    hidden_raw: np.ndarray = None  # (N,) ``hidden`` before that guard
    gpu: bool = False
    info: dict = field(default_factory=dict)

    def select(self, threshold: float) -> np.ndarray:
        """Indices whose score is at or above ``threshold``."""
        return np.where(self.score >= threshold)[0]

    def kept(self, threshold: float) -> int:
        """Hidden haze that would be at or above ``threshold`` but stays,
        because without it a surface no camera saw would show."""
        if self.hidden_raw is None:
            return 0
        raw = self.base * self.hidden_raw >= threshold
        return int((raw & (self.score < threshold)).sum())

    def histogram(self, bins: int = 24) -> list[int]:
        h, _ = np.histogram(self.score, bins=bins, range=(0.0, 1.0))
        return [int(v) for v in h]


def _soft(x, lo, hi):
    """Ramp from 0 at ``lo`` to 1 at ``hi``, clamped."""
    return np.clip((x - lo) / max(hi - lo, 1e-9), 0.0, 1.0)


def analyse(s: SplatSet, k: int = 8, sample: int = 120000,
            seed: int = 0, use_gpu: bool | None = None,
            up=(0.0, -1.0, 0.0)) -> FloaterStats:
    """Measure every splat. The expensive parts are one k-d tree query and,
    with a GPU, a few dozen small renders.

    ``up``: the scene's up direction in file coordinates. Which cameras count
    as "from below" depends on it: on a +y-up file measured with the default
    -y the sand haze is seen from above and the views from below look down
    through the roof.
    """
    n = s.n
    if n < k + 2:
        z = np.zeros(n)
        return FloaterStats(z, s.alpha, z, z, 1.0, k, base=z,
                            detached=np.zeros(n, bool))

    xyz = s.xyz.astype(np.float64)
    tree = cKDTree(xyz)
    d, nbr = tree.query(xyz, k=k + 1, workers=-1)
    iso = d[:, -1]                                   # distance to the k-th neighbour

    rng = np.random.default_rng(seed)
    probe = iso if n <= sample else rng.choice(iso, sample, replace=False)
    med = float(np.median(probe[np.isfinite(probe) & (probe > 0)]))
    med = max(med, 1e-9)
    iso_rel = iso / med

    alpha = s.alpha
    size = s.sigma.mean(axis=1)
    # Compare each splat's size against its neighbours' rather than a global
    # figure: "too big" only means anything locally. A wall and a leaf are both
    # fine, each at its own scale.
    nb_size = size[nbr[:, 1:]].mean(axis=1)
    bulk = size / np.maximum(nb_size, 1e-9)

    # Each term is a soft vote, and they multiply rather than add: a splat has
    # to be suspicious on several counts at once. Summing would let one extreme
    # measurement convict an otherwise ordinary splat.
    v_iso = _soft(iso_rel, 2.0, 6.0)
    v_faint = _soft(0.35 - alpha, 0.0, 0.3)
    v_bulk = _soft(bulk, 2.5, 8.0)
    base = np.clip(v_iso * (0.45 + 0.55 * np.maximum(v_faint, v_bulk))
                   + 0.35 * v_iso * v_faint * v_bulk, 0.0, 1.0)

    detached = _detached(s, d, nbr, med)
    score = np.where(detached, 0.95, 0.0)

    st = FloaterStats(isolation=iso_rel, alpha=alpha, bulk=bulk, score=score,
                      median_spacing=med, k=k, base=base, detached=detached)

    if use_gpu is None:
        from . import gpu
        use_gpu = gpu.available()[0]
    if use_gpu:
        try:
            h = _hidden(s, base, med, up)
        except RuntimeError as e:          # out of GPU memory, say: still answer
            st.info["gpuError"] = str(e).strip().splitlines()[-1][:160] if str(e) else "?"
            h = None
        if h is not None:
            st.behind, st.exposure, st.below, st.steep, st.hidden = (
                h["behind"], h["exposure"], h["below"], h["steep"], h["hidden"])
            # What taking the haze away would show (see _reveal). Under a hut
            # the haze is the ground the photos never saw; gone, it opens a
            # hole onto the hut's streaky underside, so it stays. Judged per
            # patch, not per splat: one splat's share is noisy, a patch of
            # haze is not.
            cand = h["cand"]
            rv = _smooth_among(h["reveal"], cand, nbr)
            st.reveal = rv
            st.guarded = cand & (rv >= REVEAL[0])
            st.hidden_raw = st.hidden
            st.hidden = st.hidden * (1.0 - _soft(rv, *REVEAL))
            # The fringe under a tree crown: big, opaque blobs far from
            # anything, with nothing in front of them from below and seen
            # hardly at all from steeply above, where the photos were taken.
            # Each vote alone is common -- a hut's wall panels are big and
            # sparse, a crown's underside is lit from below -- together they
            # are not. Size is against the scene's median splat on purpose:
            # the fringe blobs sit among each other, so next to their own
            # neighbours they look ordinary (1.8x).
            rel = size / max(float(np.median(size)), 1e-12)
            st.fringe = (_soft(iso_rel, 4.0, 8.0) * _soft(rel, 4.0, 10.0)
                         * _soft(alpha, 0.2, 0.5) * _soft(st.exposure, 0.3, 0.6)
                         * _soft(0.45 - st.steep, 0.0, 0.2))
            st.score = np.maximum(score, np.maximum(base * st.hidden, st.fringe))
            st.gpu = True
    st.info.update(detached=int(detached.sum()),
                   hidden=int(((base * st.hidden) >= 0.3).sum()) if st.gpu else 0,
                   fringe=int((st.fringe >= 0.3).sum()) if st.gpu else 0,
                   guarded=int(st.guarded.sum()) if st.gpu else 0)
    return st


# Haze stays where this share of its weight (all 48 views) lies on pixels
# that would show, without it, a surface no camera from above saw. Ramp.
REVEAL = (0.35, 0.5)
# Depth behind the steep views from above, in spacings, from which a surface
# counts as unseen. Less is the far side of a thin layer whose top was seen.
THIN = (1.5, 3.0)


def _smooth_among(v, mask, nbr, rounds: int = 4):
    """Average ``v`` over each splat's k-NN, counting only splats in ``mask``."""
    m = mask.astype(float)
    x = v * m
    for _ in range(rounds):
        num = x[nbr[:, 1:]].sum(1) + x
        den = m[nbr[:, 1:]].sum(1) + m
        x = np.where(mask, num / np.maximum(den, 1.0), 0.0)
    return x


def _detached(s: SplatSet, d, nbr, med, reach: float = 3.0,
              frac: float = 0.001, cap: int = 5000) -> np.ndarray:
    """Splats in a small connected group of their own.

    Linked along the k-NN edges when two splats are within ``reach`` median
    spacings or their 2-sigma extents touch. The second clause is what keeps
    big sparse splats -- a shutter, a roof panel -- attached; the first keeps
    small sparse ones -- leaves in a crown, a lace -- attached. A group with
    fewer than ``frac`` of all splats (at most ``cap``) is cut off.
    """
    from scipy.sparse import coo_matrix
    from scipy.sparse.csgraph import connected_components
    n, k1 = nbr.shape
    smax = s.sigma.max(axis=1)
    i = np.repeat(np.arange(n), k1 - 1)
    j = nbr[:, 1:].ravel()
    dd = d[:, 1:].ravel()
    ok = dd <= np.maximum(reach * med, 2.0 * (smax[i] + smax[j]))
    g = coo_matrix((np.ones(int(ok.sum()), np.int8), (i[ok], j[ok])), shape=(n, n))
    nc, lab = connected_components(g, directed=False)
    size = np.bincount(lab, minlength=nc)
    limit = min(max(frac * n, 10.0), cap)
    return size[lab] < limit


def _sphere(n):
    i = np.arange(n) + 0.5
    phi = np.arccos(1.0 - 2.0 * i / n)
    th = np.pi * (1.0 + 5.0 ** 0.5) * i
    return np.stack([np.cos(th) * np.sin(phi), np.cos(phi),
                     np.sin(th) * np.sin(phi)], axis=1)


def _hidden(s: SplatSet, base, med, up, n_views: int = 48, res: int = 256,
            fov: float = 45.0, horizon: float = -0.2, peel_level: float = 0.25,
            peels: int = 2, steep_alt: float = 0.5):
    """Behind what the cameras saw, and visible from where none stood.

    48 cameras on a sphere, 2.3 robust radii out, split at ``horizon``
    (``d . up``) into the views a camera plausibly stood at and the views
    from below. Three measurements come out of them:

    * ``behind`` -- from every upper view, the splat's depth against the
      blended depth of the pixel under it; the least of those, in median
      spacings. Large means something lay in front of it from everywhere a
      camera stood. Expected depth rather than the first hit on purpose: a
      translucent shutter averages in what is behind it, so what is seen
      *through* it counts as seen.
    * ``exposure`` -- from below, the best transmittance reaching the splat:
      its blending weight (the gradient of the image with respect to its
      colour) over its unoccluded footprint. Haze under sand is lit from
      below; a hut's interior is behind its floor and is not. The haze is a
      thick layer, so it is peeled: what was found is taken out and the views
      from below rendered again, ``peels`` times. More peels eat through a
      floor into the room above it (on the hut: 2 peels take the sand haze,
      4 start on the shutters from behind).
    * ``below`` -- the share of the splat's whole blending weight that comes
      from the views below. This catches the fog at the rim of a scan, which
      a low view from above does see a little of, so it is never fully
      ``behind``, but which shows mainly from below.
    * ``steep`` -- the share of the splat's whole blending weight that comes
      from the upper views looking down by more than 30 degrees
      (``steep_alt``). Used by the fringe test in :func:`analyse`, not here.

    ``hidden = max(behind gate x exposure gate, below gate)``, 0..1.

    Also returned: ``cand``, the haze at ``peel_level``, and ``reveal``, what
    taking it away would show (:func:`_reveal`); :func:`analyse` turns that
    into the guard.
    """
    from . import gpu
    torch, gsplat = gpu.require()
    dev = torch.device("cuda")
    n = s.n
    p = s.xyz.astype(np.float64)
    lo, hi = np.percentile(p, 2, axis=0), np.percentile(p, 98, axis=0)
    c = 0.5 * (lo + hi)
    r = max(0.5 * float(np.linalg.norm(hi - lo)), med)
    up = np.asarray(up, float)
    up = up / np.linalg.norm(up)
    dirs = _sphere(n_views)
    alt = dirs @ up

    def cams_for(ds):
        out = []
        for dd in ds:
            u = up if abs(float(dd @ up)) < 0.95 else np.array([1.0, 0.0, 0.0])
            out.append(gpu.look_at_camera(c + dd * r * 2.3, c, -u, res, res, fov))
        return out

    upper, lower = cams_for(dirs[alt > horizon]), cams_for(dirs[alt <= horizon])
    # Which of the upper views look down steeply (more than 30 degrees):
    # where a drone or a person standing over the object took its photos.
    steep_cam = torch.tensor(alt[alt > horizon] > steep_alt, device=dev,
                             dtype=torch.float32)
    tt = gpu.to_tensors(s, dev)
    batch = 8
    kw = dict(means=tt["means"], quats=tt["quats"], scales=tt["scales"],
              width=res, height=res, sh_degree=None, rasterize_mode="antialiased")

    def stack(cb):
        vm = torch.tensor(np.stack([x.view for x in cb]), device=dev, dtype=torch.float32)
        Ks = torch.tensor(np.stack([x.K for x in cb]), device=dev, dtype=torch.float32)
        return vm, Ks

    behind = torch.full((n,), float("inf"), device=dev)
    behind_steep = torch.full((n,), float("inf"), device=dev)
    w_up = torch.zeros((n,), device=dev)
    w_steep = torch.zeros((n,), device=dev)
    for b in range(0, len(upper), batch):
        cb = upper[b:b + batch]
        C = len(cb)
        vm, Ks = stack(cb)
        with torch.no_grad():
            pc = torch.einsum("cij,nj->cni", vm[:, :3, :3], tt["means"]) + vm[:, None, :3, 3]
            z = pc[..., 2]
        col = torch.stack([z, torch.ones_like(z), torch.ones_like(z)], -1).requires_grad_(True)
        out, _, _ = gsplat.rasterization(opacities=tt["opacities"], colors=col,
                                         viewmats=vm, Ks=Ks, **kw)
        out[..., 2].sum().backward()
        w_up += col.grad[..., 2].sum(0)
        w_steep += (col.grad[..., 2] * steep_cam[b:b + C, None]).sum(0)
        with torch.no_grad():
            cover = out[..., 1]
            ed = out[..., 0] / cover.clamp(min=1e-6)
            zz = z.clamp(min=1e-6)
            u_ = (pc[..., 0] / zz * Ks[:, None, 0, 0] + Ks[:, None, 0, 2]).long()
            v_ = (pc[..., 1] / zz * Ks[:, None, 1, 1] + Ks[:, None, 1, 2]).long()
            inside = (u_ >= 0) & (u_ < res) & (v_ >= 0) & (v_ < res) & (z > 0)
            uu, vv = u_.clamp(0, res - 1), v_.clamp(0, res - 1)
            ci = torch.arange(C, device=dev)[:, None].expand(C, n)
            beh = (z - ed[ci, vv, uu]) / med
            # Where the pixel is mostly empty there is no surface to be behind.
            beh = torch.where(cover[ci, vv, uu] < 0.3, torch.zeros_like(beh), beh)
            beh = torch.where(inside, beh, torch.full_like(beh, float("inf")))
            behind = torch.minimum(behind, beh.min(0).values)
            beh = torch.where(steep_cam[b:b + C, None] > 0, beh, torch.full_like(beh, float("inf")))
            behind_steep = torch.minimum(behind_steep, beh.min(0).values)
    behind = behind.cpu().numpy()
    behind_steep = behind_steep.cpu().numpy()
    w_up = w_up.cpu().numpy()
    w_steep = w_steep.cpu().numpy()

    hb = _soft(behind, 4.0, 12.0)
    exposure = np.zeros(n)
    below = np.zeros(n)
    hidden = np.zeros(n)
    steep = w_steep / np.maximum(w_up, 1e-12)
    if not lower:
        return {"behind": behind, "exposure": exposure, "below": below,
                "steep": steep, "hidden": hidden, "reveal": np.zeros(n),
                "cand": np.zeros(n, bool)}

    removed = np.zeros(n, bool)
    for it in range(max(peels, 1)):
        keep = torch.tensor(~removed, device=dev, dtype=torch.float32)
        opac = tt["opacities"] * keep
        T = torch.zeros((n,), device=dev)
        w_lo = torch.zeros((n,), device=dev)
        for b in range(0, len(lower), batch):
            cb = lower[b:b + batch]
            C = len(cb)
            vm, Ks = stack(cb)
            col = torch.ones((C, n, 1), device=dev, requires_grad=True)
            out, _, meta = gsplat.rasterization(opacities=opac, colors=col,
                                                viewmats=vm, Ks=Ks, **kw)
            out.sum().backward()
            w = col.grad[..., 0]                               # (C, N) weight
            w_lo += w.sum(0)
            con = meta["conics"].detach()
            det = (con[:, 0] * con[:, 2] - con[:, 1] ** 2).clamp(min=1e-12)
            foot = meta["opacities"].detach() * (2.0 * np.pi) / det.sqrt()
            gi, cid = meta["gaussian_ids"], meta["camera_ids"]
            t = w[cid, gi] / foot.clamp(min=1e-9)
            t = torch.where(foot > 1e-6, t, torch.zeros_like(t))
            T.scatter_reduce_(0, gi, t.clamp(max=1.0), reduce="amax")
        if it == 0:
            w_lo = w_lo.cpu().numpy()
            w_lo0 = w_lo
            below = w_lo / np.maximum(w_lo + w_up, 1e-12)
            steep = w_steep / np.maximum(w_lo + w_up, 1e-12)
            hidden = _soft(below, 0.45, 0.75)
        Tn = T.cpu().numpy()
        exposure = np.maximum(exposure, Tn)
        hidden = np.maximum(hidden, hb * _soft(Tn, 0.03, 0.12))
        new = (base * hidden >= peel_level) & ~removed
        if it + 1 >= peels or new.sum() <= 10:
            break
        removed |= new
    cand = base * hidden >= peel_level
    reveal = _reveal(gsplat, torch, dev, tt, kw, stack, upper, lower, cand,
                     w_lo0 + w_up, _soft(behind_steep, *THIN), batch)
    return {"behind": behind, "exposure": exposure, "below": below,
            "steep": steep, "hidden": hidden, "reveal": reveal, "cand": cand}


def _reveal(gsplat, torch, dev, tt, kw, stack, upper, lower, cand, w_all, deep, batch,
            unseen=(0.5, 0.75), cover=(0.2, 0.5)):
    """For each splat, the share of its weight (``w_all``, every view) that
    sits on pixels of the views from below which, with the haze candidates
    ``cand`` gone, would show a surface no camera from above saw.

    The scene is measured again without the candidates: how much of what is
    left of a splat's blending weight comes from below (``below'``)? Sand
    photographed from above shows from both sides, about half and half; a
    hut's underside and the posts under it show almost only from below. And
    how deep is it behind what the steep views from above see (``deep``, a
    0..1 ramp)? The lower layer of a thin ground sits right under its top and
    counts as seen; the hut's underside sits under the whole hut. Unseen is
    both at once. Each view from below is rendered without the candidates,
    coloured by that unseen-ness (ramp ``unseen`` on ``below'``, times
    ``deep``) and weighted by coverage (ramp ``cover``), and the gradient of
    the original render against that picture says how much of each splat lies
    on such pixels. Where the picture is empty -- fog in front of nothing --
    or shows sand seen from above, the share is low and the haze may go.
    Dividing by the weight from every view, not only from below, lets haze
    that also shows from the side go: there it is in the way.
    """
    n = len(cand)
    if cand.sum() < 10:
        return np.zeros(n)
    after = tt["opacities"] * torch.tensor(~cand, device=dev, dtype=torch.float32)

    def weights(cams, opac):
        w = torch.zeros((n,), device=dev)
        for b in range(0, len(cams), batch):
            cb = cams[b:b + batch]
            vm, Ks = stack(cb)
            col = torch.ones((len(cb), n, 1), device=dev, requires_grad=True)
            out, _, _ = gsplat.rasterization(opacities=opac, colors=col,
                                             viewmats=vm, Ks=Ks, **kw)
            out.sum().backward()
            w += col.grad[..., 0].sum(0)
        return w

    wu, wl = weights(upper, after), weights(lower, after)
    bl = wl / (wu + wl).clamp(min=1e-12)
    un = ((bl - unseen[0]) / (unseen[1] - unseen[0])).clamp(0, 1)
    un = un * torch.tensor(deep, device=dev, dtype=torch.float32)
    paint = torch.stack([un, torch.ones_like(un), torch.zeros_like(un)], -1)
    bad_w = torch.zeros((n,), device=dev)
    for b in range(0, len(lower), batch):
        cb = lower[b:b + batch]
        C = len(cb)
        vm, Ks = stack(cb)
        with torch.no_grad():
            pic, _, _ = gsplat.rasterization(opacities=after,
                                             colors=paint[None].expand(C, n, 3).contiguous(),
                                             viewmats=vm, Ks=Ks, **kw)
            cov = pic[..., 1]
            bad = (((cov - cover[0]) / (cover[1] - cover[0])).clamp(0, 1)
                   * pic[..., 0] / cov.clamp(min=1e-6))
        col = torch.ones((C, n, 1), device=dev, requires_grad=True)
        out, _, _ = gsplat.rasterization(opacities=tt["opacities"], colors=col,
                                         viewmats=vm, Ks=Ks, **kw)
        (out[..., 0] * bad).sum().backward()
        bad_w += col.grad[..., 0].sum(0)
    bw = bad_w.cpu().numpy()
    return np.where(w_all > 1e-9, bw / np.maximum(w_all, 1e-12), 0.0)


def crop_outside(s: SplatSet, lo, hi) -> np.ndarray:
    """Indices outside an axis-aligned box."""
    lo = np.asarray(lo, float)
    hi = np.asarray(hi, float)
    p = s.xyz.astype(np.float64)
    inside = np.all((p >= lo) & (p <= hi), axis=1)
    return np.where(~inside)[0]
