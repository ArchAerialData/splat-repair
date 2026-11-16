"""Fill a hole in the surface's own picture, then lift the picture to splats.

The quick fill (`repair.py`) copies whole Gaussians from elsewhere on the
surface. A Gaussian is an extended, irregular thing: copied in blocks, the
blocks cut through one another, and where the blocks disagree the result is
scrambled texture at the wrong height. Every fix there fights the same fact.

This engine never copies a Gaussian. It works the way an image editor does,
on a picture of the surface, and only then goes back to 3-D:

1. **Picture.** The surface around the hole is rendered orthographically
   along the user's own line of sight, a few pixels per splat spacing: its
   colour, and its *height* -- the middle of the skin, read off level
   surfaces from where the opacity seen from the eye sets in; not the
   expected depth (which mixes the surface with whatever shows through its
   gaps) and not the height of splat centres.
2. **Base.** The height is split into a smooth base and the relief on it.
   The base is continued through the hole biharmonically, so it leaves the
   rim with the rim's own height *and slope*: a dome stays a dome. At an
   edge only the layer that was painted is continued, not what shows behind
   it: strung between the two, the fill would hang from the edge as a
   curtain. Which layer that is, the old scene says where a print or a scuff
   was painted over; where a thing stood on the surface, the picture itself
   has to tell (the piece with most of the rim, and what lies far behind
   it).
3. **Colour.** LaMa fills the colour picture. Seen square-on, at this
   resolution, with the hole a fraction of the frame, that is what LaMa is
   good at: it continues lattices, seams and grain.
4. **Relief and grain by analogy.** For every patch of the filled picture
   the most similar real patch of the same surface is looked up (PatchMatch
   on the GPU). The real patches bring what a picture cannot: the relief
   under them, and -- where LaMa had to work at a reduced size -- their
   finest grain. Both are voted into the hole. Nothing is iterated on its
   own output, so nothing drifts into mush.
5. **Seam.** The relief the patches vote just outside the hole is compared
   with the relief that is really there; the difference is spread through
   the hole as a membrane and taken off.
6. **Lift.** Every filled pixel becomes one small flat Gaussian on the
   continued surface, turned to the local normal. What was in the way in
   the hole's column goes.
7. **Side views.** One picture sees a flank that turns away only squeezed;
   lifted from there the fill is a stretched strip. Where the continued
   surface stands steeper than 55 degrees to the view, a further picture is
   taken from the side that shows most of it (`fill`), with the first
   picture's fill already in the scene. The user's own view stays the
   authority: what a side picture adds may not stand before the surface the
   first one shows, nor land outside what was painted.

Needs CUDA + gsplat; LaMa is optional (without it the colour is a membrane
of the rim's colours plus the grain of matched patches).
"""
from __future__ import annotations

import math
import time
import warnings
from dataclasses import dataclass, field, replace
from typing import Optional

import numpy as np
from scipy import ndimage, sparse
from scipy.sparse.linalg import splu
from scipy.spatial import cKDTree

from .io import SplatSet, SH_C0
from . import gpu


@dataclass
class ImFillParams:
    pxps: float = 5.0           # pixels per splat spacing
    context: float = 3.5        # picture half-size, in hole radii
    max_px: int = 1024          # longest picture side
    patch: int = 7
    base_sigma: float = 3.0     # low-pass of the base surface, in spacings
    guard: float = 1.0          # the base is not read this close to the hole (spacings)
    tension: float = 0.3        # membrane share of the base plate (0 = pure thin plate)
    lama: bool = True
    lama_hole: int = 288        # LaMa sees the hole at most this many pixels wide
    grain: float = 0.0          # share of the matched patches' finest detail
    relief: float = 1.0         # share of the matched patches' relief
    seam: bool = True
    surface_thr: float = 0.3    # opacity the skin must gather to count as surface
    skin: float = 3.0           # thickness of the skin, in spacings
    envelope: float = 1.0       # pits narrower than this many spacings are leaks, not relief
    shape_prior: bool = True    # where what was painted lay flat on the surface, keep its shape
    layers: bool = True         # at an edge, continue the painted layer only (not what shows behind it)
    view_tone: bool = True      # the fill changes with the viewing angle as its surroundings do
    grazing: bool = True        # ... and loses its fine contrast at a slant as they do
    grazing_band: float = 2.0   # "fine": finer than this many spacings
    passes: int = 3             # pictures at most: the user's view, then sides for what it left
    steep: float = 1.43         # slope (tan) above which a flank is left to a side view: 55 degrees
    steep_wide: float = 1.0     # ... if the surface at large (4 spacings) turns away too: 45 degrees
    side_tilt: float = 65.0     # how far a side view leans off the user's own, degrees
    side_context: float = 2.6   # picture half-width of a side view, in hole radii
    side_px: int = 768          # ... and its size at most (a side view only has to do a flank)
    side_left: float = 0.01     # share of the selection the first picture must leave for a side view to look
    side_area: float = 0.03     # ... and of the fill's area that must show of it from the side
    side_area_max: float = 150.0  # ... but never more than this many square spacings
    opacity: float = 0.97
    splat_sigma: float = 0.62   # tangent sigma of a lifted splat, in pixels
    seed: int = 0


# ------------------------------------------------------------------ camera

@dataclass
class Ortho:
    """An orthographic view down ``-n`` on ``c``. Image x = t1, y = t2."""
    c: np.ndarray
    n: np.ndarray
    t1: np.ndarray
    t2: np.ndarray
    px: float            # world units per pixel
    W: int
    H: int
    dist: float          # the eye sits at c + n * dist

    @property
    def viewmat(self):
        R = np.stack([self.t1, self.t2, -self.n])
        v = np.eye(4)
        v[:3, :3] = R
        v[:3, 3] = -R @ (self.c + self.n * self.dist)
        return v

    @property
    def K(self):
        f = 1.0 / self.px
        return np.array([[f, 0, self.W * 0.5], [0, f, self.H * 0.5], [0, 0, 1.0]])

    def world(self, ix, iy, height):
        X = (np.asarray(ix, float) + 0.5 - self.W * 0.5) * self.px
        Y = (np.asarray(iy, float) + 0.5 - self.H * 0.5) * self.px
        return (self.c[None] + X[:, None] * self.t1[None] + Y[:, None] * self.t2[None]
                + np.asarray(height, float)[:, None] * self.n[None])

    def pixel(self, P):
        d = np.asarray(P, float) - self.c
        x = d @ self.t1 / self.px + self.W * 0.5 - 0.5
        y = d @ self.t2 / self.px + self.H * 0.5 - 0.5
        return x, y, d @ self.n


def _frame(n, hint=None):
    n = np.asarray(n, float)
    n = n / max(np.linalg.norm(n), 1e-12)
    a = np.asarray(hint, float) if hint is not None else None
    if a is None or abs(a @ n) > 0.95 * np.linalg.norm(a):
        a = np.array([1.0, 0, 0]) if abs(n[0]) < 0.9 else np.array([0, 1.0, 0])
    t1 = a - (a @ n) * n
    t1 /= np.linalg.norm(t1)
    t2 = np.cross(-n, t1)            # rows [t1, t2, -n] are a proper rotation
    return n, t1, t2


class _Scene:
    """A set of splats on the GPU, rendered orthographically (as the app draws)."""

    def __init__(self, s: SplatSet):
        self.torch, self.gsplat = gpu.require()
        self.dev = self.torch.device("cuda")
        self.t = gpu.to_tensors(s, self.dev)
        self.deg = s.sh_degree
        self.n = s.n

    def raster(self, cam: Ortho, keep, colors, sh_degree, depth):
        torch = self.torch
        t = self.t
        sel = (lambda a: a) if keep is None else (lambda a: a[keep])
        if (self.n if keep is None else len(keep)) == 0:
            # nothing to draw. gsplat's projection divides by the number of
            # splats in native code: with none the whole process dies (an
            # integer division by zero, no Python error to catch).
            return (torch.zeros((cam.H, cam.W, 4 if depth else 3), device=self.dev),
                    torch.zeros((cam.H, cam.W), device=self.dev))
        vm = torch.tensor(cam.viewmat, device=self.dev, dtype=torch.float32)[None]
        K = torch.tensor(cam.K, device=self.dev, dtype=torch.float32)[None]
        with torch.no_grad():
            out, alpha, _ = self.gsplat.rasterization(
                means=sel(t["means"]), quats=sel(t["quats"]), scales=sel(t["scales"]),
                opacities=sel(t["opacities"]), colors=colors,
                viewmats=vm, Ks=K, width=cam.W, height=cam.H, sh_degree=sh_degree,
                render_mode="RGB+ED" if depth else "RGB", rasterize_mode="classic",
                camera_model="ortho", near_plane=1e-3, far_plane=1e8)
        return out[0], alpha[0, ..., 0]

    def picture(self, cam: Ortho, keep=None):
        """(rgb HxWx3, alpha HxW, expected height HxW) as numpy."""
        c = self.t["colors"] if keep is None else self.t["colors"][keep]
        out, a = self.raster(cam, keep, c, self.deg, True)
        rgb = out[..., :3].clamp(0, 1).cpu().numpy()
        h = cam.dist - out[..., 3].cpu().numpy()
        return rgb, a.cpu().numpy(), h

    def shown(self, cam: Ortho, flag):
        """How much of each pixel is the flagged splats (occlusion included)."""
        torch = self.torch
        f = torch.as_tensor(flag, device=self.dev, dtype=torch.float32)
        col = torch.stack([f, torch.zeros_like(f), torch.zeros_like(f)], 1)
        out, _ = self.raster(cam, None, col, None, False)
        return out[..., 0].cpu().numpy()

    def surface(self, cam: Ortho, keep_np, height_at, guess, sp, thr=0.4,
                top=4.0, step=0.25, nlev=44, shell=0.0, onset=0.1):
        """Height of the visible surface, read off level surfaces.

        ``nlev`` level surfaces ``step`` spacings apart, riding on the smooth
        guess ``guess`` (H,W; ``height_at`` is each kept splat's height minus
        the guess at its own pixel), are rendered at once: channel k shows
        only the splats above level k, so it holds the opacity the eye has
        gathered down to that level.

        ``shell`` = 0: the level where that opacity passes ``thr``.
        ``shell`` > 0: the skin's own centre. From where the opacity sets in
        (``onset``), the next ``shell`` spacings are the skin; its height is
        the mean of those levels weighted by the opacity each adds, and it
        counts as found when they add up to ``thr``. What lies deeper (the
        inside of a shoe seen through a thin spot) has no say, which the
        expected depth of everything cannot offer; and unlike a threshold the
        mean does not jump between the layers of a fuzzy skin.

        Returns (height HxW, found HxW) as numpy.
        """
        torch = self.torch
        lev = torch.tensor((top - step * np.arange(nlev)) * sp, device=self.dev, dtype=torch.float32)
        rel = torch.tensor(height_at, device=self.dev, dtype=torch.float32)
        feat = (rel[:, None] > lev[None]).float()
        keep = torch.tensor(keep_np, device=self.dev)
        V, _ = self.raster(cam, keep, feat, None, False)
        del feat
        if shell > 0:
            m = max(1, int(round(shell / step)))
            on = V >= onset
            k0 = on.float().argmax(2)
            has = on.any(2)
            dV = torch.diff(V, dim=2, prepend=torch.zeros_like(V[..., :1])).clamp(min=0)
            mid = lev + 0.5 * step * sp                       # centre of slab k
            ks = torch.arange(nlev, device=self.dev)[None, None]
            inshell = (ks >= k0[..., None]) & (ks < (k0 + m)[..., None])
            w = dV * inshell
            mass = w.sum(2)
            hgt = (w * mid[None, None]).sum(2) / mass.clamp(min=1e-6)
            found = has & (mass >= thr) & (k0 > 0)
            out = guess + hgt.cpu().numpy().astype(np.float64)
            return out, found.cpu().numpy()
        above = V >= thr
        found = above.any(2)
        kf = above.float().argmax(2)
        k1 = kf.clamp(1, nlev - 1)
        v1 = V.gather(2, k1[..., None])[..., 0]
        v0 = V.gather(2, (k1 - 1)[..., None])[..., 0]
        frac = ((thr - v0) / (v1 - v0).clamp(min=1e-6)).clamp(0.0, 1.0)
        delta = torch.where(kf == 0, lev[0].expand_as(frac), lev[k1 - 1] - step * sp * frac)
        return guess + delta.cpu().numpy().astype(np.float64), found.cpu().numpy()


# ------------------------------------------------------------ small solvers

def _lap(h, w):
    """5-point Laplacian with replicated borders on an h x w grid (sparse)."""
    idx = np.arange(h * w).reshape(h, w)
    rows, cols, vals = [], [], []
    for dy, dx in ((0, 1), (0, -1), (1, 0), (-1, 0)):
        nb = idx[np.clip(np.arange(h)[:, None] + dy, 0, h - 1),
                 np.clip(np.arange(w)[None, :] + dx, 0, w - 1)]
        rows.append(idx.ravel()); cols.append(nb.ravel()); vals.append(np.ones(h * w))
    rows.append(idx.ravel()); cols.append(idx.ravel()); vals.append(-4.0 * np.ones(h * w))
    return sparse.csr_matrix((np.concatenate(vals), (np.concatenate(rows), np.concatenate(cols))),
                             shape=(h * w, h * w))


def _solve_fill(values, unknown, order=2):
    """Continue ``values`` (.., h, w) into ``unknown``: harmonic (1) or biharmonic (2).

    Known cells are fixed. The biharmonic fill leaves the known region with
    its value and its slope, which is what carries a curved surface through a
    hole; the harmonic one is a membrane (value only).
    """
    v = np.asarray(values, np.float64)
    lead = v.shape[:-2]
    h, w = v.shape[-2:]
    u = np.asarray(unknown, bool).ravel()
    if not u.any() or u.all():
        return v.copy()
    L = _lap(h, w)
    A = ((L @ L) if order == 2 else L).tocsr()
    Auu = A[u][:, u].tocsc()
    Auk = A[u][:, ~u]
    flat = v.reshape(-1, h * w).copy()
    lu = splu(Auu)
    for k in range(flat.shape[0]):
        flat[k, u] = lu.solve(-(Auk @ flat[k, ~u]))
    return flat.reshape(*lead, h, w)


def _plate(d, w, lam, L2=None, tension=0.0):
    """Thin-plate smoothing of ``d`` (h, w) with weights ``w``: the surface z
    that minimises  sum w (z - d)^2 + lam * |laplace z|^2.

    Where the weight is zero (a hole, the band around it) the plate simply
    continues, with the height, slope and curvature the data lead up to; where
    there is data it is smoothed over about lam^(1/4) cells. One fit for both,
    so the smoothing has no edge: a local average next to a hole sees only
    the far side and flattens exactly the slope the continuation needs.
    """
    h, wd = d.shape
    if L2 is None:
        L = _lap(h, wd)
        L2 = (L, L @ L)
    L, LL = L2
    # ``tension`` adds a membrane's share: a pure plate swings freely where it
    # has data on one side only (a hole at the object's edge)
    M = (sparse.diags(np.asarray(w, np.float64).ravel() + 1e-9) + lam * LL - (tension * lam ** 0.5) * L).tocsc()
    return splu(M).solve((np.asarray(w, np.float64) * d).ravel()).reshape(h, wd)


def _loclin(z, known, sigma):
    """Low-pass of ``z`` over the known cells by a local *plane* fit (Gaussian window).

    A local average next to a hole sees only the far side: on a slope it is
    biased, and it flattens exactly the slope a continuation through the hole
    has to start with (a dome is filled flat). The intercept of a plane fitted
    in the same window has neither fault. Returns the smoothed grid (defined
    wherever the window holds enough known cells; elsewhere the local mean).
    """
    h, w = z.shape
    k = known.astype(np.float64)
    yy, xx = np.mgrid[0:h, 0:w].astype(np.float64)
    G = lambda a: ndimage.gaussian_filter(a, sigma, mode="constant")
    zk = np.where(known, z, 0.0)
    M0, Mx, My = G(k), G(k * xx), G(k * yy)
    Mxx, Mxy, Myy = G(k * xx * xx), G(k * xx * yy), G(k * yy * yy)
    Z0, Zx, Zy = G(zk), G(zk * xx), G(zk * yy)
    mx, my = Mx - xx * M0, My - yy * M0
    mxx = Mxx - 2 * xx * Mx + xx * xx * M0
    mxy = Mxy - xx * My - yy * Mx + xx * yy * M0
    myy = Myy - 2 * yy * My + yy * yy * M0
    zx, zy = Zx - xx * Z0, Zy - yy * Z0
    reg = 0.05 * sigma * sigma * M0 + 1e-9                 # leans on the mean where the fit is weak
    A = np.empty((h, w, 3, 3))
    A[..., 0, 0] = M0 + 1e-9
    A[..., 0, 1] = A[..., 1, 0] = mx
    A[..., 0, 2] = A[..., 2, 0] = my
    A[..., 1, 1] = mxx + reg
    A[..., 1, 2] = A[..., 2, 1] = mxy
    A[..., 2, 2] = myy + reg
    b = np.stack([Z0, zx, zy], -1)[..., None]
    sol = np.linalg.solve(A, b)[..., 0, 0]
    mean = Z0 / np.maximum(M0, 1e-9)
    return np.where(M0 > 0.02, sol, mean)


def _gauss_known(a, known, sigma):
    """Gaussian low-pass of ``a`` over the known cells only."""
    num = ndimage.gaussian_filter(np.where(known, a, 0.0), sigma, mode="nearest")
    den = ndimage.gaussian_filter(known.astype(np.float64), sigma, mode="nearest")
    return num / np.maximum(den, 1e-6), den


def _down_known(a, known, f):
    """Mean of the known pixels in f x f blocks -> (coarse, share known)."""
    H, W = known.shape
    h, w = H // f, W // f
    k = known[:h * f, :w * f].reshape(h, f, w, f).astype(np.float64)
    v = np.where(known, a, 0.0)[:h * f, :w * f].reshape(h, f, w, f)
    cnt = k.sum((1, 3))
    return v.sum((1, 3)) / np.maximum(cnt, 1e-9), cnt / (f * f)


def _up(a, shape):
    """Smooth (cubic) upsampling of a coarse grid to ``shape``."""
    H, W = shape
    h, w = a.shape[-2:]
    yy = (np.arange(H) + 0.5) * h / H - 0.5
    xx = (np.arange(W) + 0.5) * w / W - 0.5
    grid = np.meshgrid(yy, xx, indexing="ij")
    if a.ndim == 2:
        return ndimage.map_coordinates(a, grid, order=3, mode="nearest")
    return np.stack([ndimage.map_coordinates(c, grid, order=3, mode="nearest") for c in a])


def _block_quantile(a, ok, f, q):
    """Quantile ``q`` of the ok pixels in f x f blocks (NaN where a block has none)."""
    H, W = ok.shape
    h, w = H // f, W // f
    b = np.where(ok, a, np.nan)[:h * f, :w * f].reshape(h, f, w, f).transpose(0, 2, 1, 3).reshape(h, w, -1)
    b = np.sort(b, axis=2)                      # NaN sorts last
    cnt = np.isfinite(b).sum(2)
    k = np.clip(np.floor(q * (cnt - 1) + 0.5).astype(int), 0, f * f - 1)
    out = np.take_along_axis(b, k[..., None], 2)[..., 0]
    return np.where(cnt > 0, out, np.nan)


def _nearest_fill(a, known):
    """``a`` with every unknown cell taking its nearest known cell's value."""
    if known.all() or not known.any():
        return a
    near = ndimage.distance_transform_edt(~known, return_distances=False, return_indices=True)
    return a[..., near[0], near[1]]


# ------------------------------------------------------- patch matching

class _Match:
    """Nearest patches on the GPU (PatchMatch, coarse to fine) and their votes.

    The target picture is complete (LaMa has filled the hole); for every
    patch that touches the hole the most similar real patch of the same
    surface is found. What the real patches carry besides their colour --
    the relief under them, their finest grain -- is then voted into the hole.
    Nothing is iterated on its own output, so nothing can drift into mush.
    """

    def __init__(self, patch=7, seed=0):
        self.torch, _ = gpu.require()
        self.dev = self.torch.device("cuda")
        self.g = self.torch.Generator(device=self.dev)
        self.g.manual_seed(int(seed) + 12345)
        self.p = int(patch)
        self.h = h = self.p // 2
        oy, ox = np.mgrid[-h:h + 1, -h:h + 1]
        self.oy = self.torch.tensor(oy.ravel(), device=self.dev)
        self.ox = self.torch.tensor(ox.ravel(), device=self.dev)

    def blur(self, x, sigma):
        torch = self.torch
        if sigma <= 0.05:
            return x
        r = max(1, int(3 * sigma + 0.5))
        k = torch.exp(-torch.arange(-r, r + 1, device=self.dev, dtype=torch.float32) ** 2
                      / (2 * sigma * sigma))
        k = k / k.sum()
        F = torch.nn.functional
        C = x.shape[0]
        x = F.pad(x[None], (r, r, r, r), mode="replicate")
        x = F.conv2d(x, k.view(1, 1, 1, -1).repeat(C, 1, 1, 1), groups=C)
        x = F.conv2d(x, k.view(1, 1, -1, 1).repeat(C, 1, 1, 1), groups=C)
        return x[0]

    def _nnf(self, FT, FS, hole, valid, nnf_up, iters):
        torch = self.torch
        F = torch.nn.functional
        h, p = self.h, self.p
        C, H, W = FS.shape
        src_ok = -F.max_pool2d(-(valid & ~hole).float()[None, None], p, 1, h)[0, 0] > 0.5
        tgt = F.max_pool2d(hole.float()[None, None], p, 1, h)[0, 0] > 0.5
        border = torch.zeros_like(tgt)
        border[h:H - h, h:W - h] = True
        src_ok &= border
        tgt &= border
        ty, tx = torch.where(tgt)
        N = len(ty)
        sl = torch.stack(torch.where(src_ok), 1)
        if N == 0 or len(sl) == 0:
            return None
        tidx = torch.full((H, W), -1, device=self.dev, dtype=torch.long)
        tidx[ty, tx] = torch.arange(N, device=self.dev)
        FSf = FS.reshape(C, -1)
        tp = FT.reshape(C, -1)[:, (ty[:, None] + self.oy[None]) * W + (tx[:, None] + self.ox[None])]

        def rand_src(n):
            return sl[torch.randint(len(sl), (n,), device=self.dev, generator=self.g)]

        if nnf_up is None:
            nnf = rand_src(N)
        else:
            nnf = nnf_up(ty, tx)
            bad = (nnf[:, 0] < h) | (nnf[:, 0] >= H - h) | (nnf[:, 1] < h) | (nnf[:, 1] >= W - h)
            bad |= ~src_ok[nnf[:, 0].clamp(0, H - 1), nnf[:, 1].clamp(0, W - 1)]
            if bad.any():
                nnf[bad] = rand_src(int(bad.sum()))
        inf = torch.tensor(float("inf"), device=self.dev)
        ones = torch.ones(N, dtype=torch.bool, device=self.dev)

        def dist(cy, cx, ok):
            cy = cy.clamp(h, H - h - 1)
            cx = cx.clamp(h, W - h - 1)
            idx = (cy[:, None] + self.oy[None]) * W + (cx[:, None] + self.ox[None])
            d = ((tp - FSf[:, idx]) ** 2).sum((0, 2))
            return torch.where(ok & src_ok[cy, cx], d, inf), cy, cx

        best, _, _ = dist(nnf[:, 0], nnf[:, 1], ones)

        def take(cy, cx, ok):
            nonlocal best
            d, cy, cx = dist(cy, cx, ok)
            up = d < best
            nnf[up, 0] = cy[up]
            nnf[up, 1] = cx[up]
            best = torch.where(up, d, best)

        for it in range(iters):
            step = 8
            while step >= 1:
                for dy, dx in ((0, step), (0, -step), (step, 0), (-step, 0)):
                    ny, nx = ty + dy, tx + dx
                    inside = (ny >= 0) & (ny < H) & (nx >= 0) & (nx < W)
                    nb = tidx[ny.clamp(0, H - 1), nx.clamp(0, W - 1)]
                    ok = inside & (nb >= 0)
                    nb = nb.clamp(min=0)
                    take(nnf[nb, 0] - dy, nnf[nb, 1] - dx, ok)
                step //= 2
            rad = max(H, W) // 2
            while rad >= 1:
                ry = torch.randint(-rad, rad + 1, (N,), device=self.dev, generator=self.g)
                rx = torch.randint(-rad, rad + 1, (N,), device=self.dev, generator=self.g)
                cy, cx = nnf[:, 0] + ry, nnf[:, 1] + rx
                take(cy, cx, (cy >= h) & (cy < H - h) & (cx >= h) & (cx < W - h))
                rad //= 2
            for _ in range(2):
                r = rand_src(N)
                take(r[:, 0], r[:, 1], ones)
        return nnf, tidx, best, ty, tx

    def vote(self, Vsrc, nn, sigma_o=1.6):
        """What the matched patches say each pixel is: (K,H,W), weight (H,W)."""
        torch = self.torch
        nnf, tidx, best, ty, tx = nn
        K, H, W = Vsrc.shape
        fin = torch.isfinite(best)
        if fin.any():
            b = best[fin]
            q = torch.quantile(b[:: max(1, len(b) // 100000)], 0.75)
        else:
            q = torch.tensor(1.0, device=self.dev)
        wp = torch.exp(-best.clamp(max=1e30) / (2.0 * q.clamp(min=1e-8)))
        wp = torch.where(fin, wp, torch.zeros_like(wp))
        Sf = Vsrc.reshape(K, -1)
        acc = torch.zeros(K, H * W, device=self.dev)
        wa = torch.zeros(H * W, device=self.dev)
        for k in range(len(self.oy)):
            oy, ox = int(self.oy[k]), int(self.ox[k])
            w = wp * math.exp(-(oy * oy + ox * ox) / (2.0 * sigma_o * sigma_o))
            tpix = (ty + oy) * W + (tx + ox)
            spix = (nnf[:, 0] + oy) * W + (nnf[:, 1] + ox)
            acc.index_add_(1, tpix, Sf[:, spix] * w[None])
            wa.index_add_(0, tpix, w)
        return (acc / wa.clamp(min=1e-8)).reshape(K, H, W), wa.reshape(H, W)

    def run(self, tgt, src, hole, valid, band=0.0, iters=3):
        """NNF from the patches of ``tgt`` touching ``hole`` to valid patches of ``src``.

        ``tgt``/``src``: torch (C,H,W). ``band``: blur both by this before
        comparing (match what both can show). Returns the finest level's
        ``(nnf, tidx, best, ty, tx)`` or None.
        """
        torch = self.torch
        F = torch.nn.functional
        if band > 0:
            tgt, src = self.blur(tgt, band), self.blur(src, band)
        levels = [(tgt, src, hole, valid)]
        while len(levels) < 6:
            T, S, ho, va = levels[-1]
            H, W = ho.shape
            ys, xs = torch.where(ho)
            ext = int(min(ys.max() - ys.min(), xs.max() - xs.min())) + 1 if len(ys) else 0
            if min(H, W) < 72 or ext <= 14:
                break
            levels.append((F.avg_pool2d(T[None], 2)[0], F.avg_pool2d(S[None], 2)[0],
                           F.avg_pool2d(ho.float()[None, None], 2)[0, 0] > 0,
                           F.avg_pool2d(va.float()[None, None], 2)[0, 0] > 0.7))
        nn = None
        for li in range(len(levels) - 1, -1, -1):
            T, S, ho, va = levels[li]
            up = None
            if nn is not None:
                pn, pt = nn[0], nn[1]

                def up(ty, tx, pn=pn, pt=pt):
                    Hc, Wc = pt.shape
                    par = pt[(ty // 2).clamp(max=Hc - 1), (tx // 2).clamp(max=Wc - 1)]
                    ok = par >= 0
                    par = par.clamp(min=0)
                    out = torch.stack([2 * pn[par, 0] + ty % 2, 2 * pn[par, 1] + tx % 2], 1)
                    out[~ok] = -10 ** 6
                    return out
            got = self._nnf(T, S, ho, va, up, iters + (2 if nn is None else 0))
            nn = got if got is not None else None
        return nn


# ----------------------------------------------------------------- LaMa

_LAMA: dict = {}


def _lama(rgb, mask):
    """LaMa on a float picture (H,W,3 in 0..1) and a bool mask -> float picture."""
    torch, _ = gpu.require()
    if "m" not in _LAMA:
        from simple_lama_inpainting import SimpleLama
        _LAMA["m"] = SimpleLama()
    model = _LAMA["m"].model
    dev = _LAMA["m"].device
    H, W = mask.shape
    ph, pw = (-H) % 8, (-W) % 8
    img = torch.tensor(rgb, device=dev, dtype=torch.float32).permute(2, 0, 1)[None]
    m = torch.tensor(mask, device=dev, dtype=torch.float32)[None, None]
    if ph or pw:
        img = torch.nn.functional.pad(img, (0, pw, 0, ph), mode="reflect")
        m = torch.nn.functional.pad(m, (0, pw, 0, ph), mode="reflect")
    # The model is TorchScript: left to itself it re-specialises for every new
    # picture size, which costs over a second each time, and every hole has
    # another size. Run as written it takes 30 to 100 ms from the first call.
    with torch.inference_mode(), torch.jit.optimized_execution(False):
        out = model(img, m)
    return out[0, :, :H, :W].permute(1, 2, 0).clamp(0, 1).float().cpu().numpy()


def _lama_scaled(rgb, mask, hole_px, box=None):
    """LaMa on a window around the hole, the hole at most ``hole_px`` wide.

    Returns (picture the size of ``rgb``, scale LaMa worked at). The window
    is the hole's box plus as much again on every side: enough for LaMa to
    read the pattern, and a third of the pixels of the whole picture.
    ``box``: the hole proper, if ``mask`` hides more than it (what LaMa is
    not to read from).
    """
    from PIL import Image
    ys, xs = np.where(mask if box is None else box)
    ext = max(np.ptp(ys), np.ptp(xs)) + 1
    mar = int(max(1.0 * ext, 96))
    H0, W0 = mask.shape
    y0, y1 = max(0, ys.min() - mar), min(H0, ys.max() + mar + 1)
    x0, x1 = max(0, xs.min() - mar), min(W0, xs.max() + mar + 1)
    y1 -= (y1 - y0) % 8
    x1 -= (x1 - x0) % 8
    full = rgb
    rgb, mask = rgb[y0:y1, x0:x1], mask[y0:y1, x0:x1]

    def back(win):
        out = full.copy()
        out[y0:y1, x0:x1] = win
        return out

    sc = min(1.0, hole_px / float(ext))
    if sc > 0.9:
        return back(_lama(rgb, mask)), 1.0
    H, W = mask.shape
    h, w = max(64, int(round(H * sc / 8)) * 8), max(64, int(round(W * sc / 8)) * 8)
    small = np.asarray(Image.fromarray((rgb * 255 + 0.5).astype(np.uint8)).resize((w, h), Image.LANCZOS),
                       np.float32) / 255.0
    msmall = np.asarray(Image.fromarray(mask.astype(np.uint8) * 255).resize((w, h), Image.BILINEAR)) > 20
    out = _lama(small, msmall)
    big = np.asarray(Image.fromarray((out * 255 + 0.5).astype(np.uint8)).resize((W, H), Image.BICUBIC),
                     np.float32) / 255.0
    return back(big), sc


# ------------------------------------------------- tone that turns with the view

def _view_tone(scene, cam: Ortho, rest, rgb, Hc, surf, hole, f, sp):
    """How the surface around the hole changes colour with the viewing angle,
    continued through the hole, as SH coefficients per coarse cell.

    A splat lifted from one picture has one colour from everywhere. The scan
    around it does not: leather goes lighter at a slant, a rubber sole
    darker. Seen square-on the fill is invisible and from the side it is a
    patch. So the rest of the scene is looked at from tilted
    directions (18: three rings of six); for every cell of the surface (one spacing) the colour seen
    from each is compared with the colour in the picture; those differences
    are smoothed, carried through the hole as a membrane, and fitted with SH
    up to degree 2. Only the smooth part of the view dependence is taken: it
    is what makes a patch.

    ``Hc``: surface height per cell (h, w). Returns (coef (8, 3, h, w), basis
    at the picture's own direction (8,)) or None.
    """
    from .shtools import sh_basis
    h, w = Hc.shape
    cy, cx = np.mgrid[0:h, 0:w]
    Pc = cam.world((cx.ravel() + 0.5) * f - 0.5, (cy.ravel() + 0.5) * f - 0.5, Hc.ravel())
    ref, shr = [], None
    for k in range(3):
        m, shr = _down_known(rgb[..., k], surf, f)
        ref.append(m)
    ref = np.stack(ref)                                    # 3, h, w
    kc = shr > 0.5
    holec = _down_known(hole.astype(float), np.ones_like(hole), f)[0] > 0
    if kc.sum() < 20:
        return None
    dirs, deltas, oks = [-cam.n], [np.zeros((3, h, w))], [kc.copy()]
    pxd = cam.px * f / 2.0
    for tilt in (30.0, 52.0, 72.0):
        for a in range(6):
            az = np.radians(60.0 * a + (30.0 if 40 < tilt < 60 else 0.0))
            tl = np.radians(tilt)
            om = cam.n * np.cos(tl) + (cam.t1 * np.cos(az) + cam.t2 * np.sin(az)) * np.sin(tl)
            n_, t1_, t2_ = _frame(om, cam.t1)
            Wd = int(cam.W * cam.px / pxd * 1.25) + 8
            Wd += (-Wd) % 8
            cd = Ortho(cam.c, n_, t1_, t2_, pxd, Wd, Wd, cam.dist)
            img, al, hd = scene.picture(cd, rest)
            x, y, hh = cd.pixel(Pc)
            co = [y, x]
            samp = np.stack([ndimage.map_coordinates(img[..., k], co, order=1, mode="nearest")
                             for k in range(3)]).reshape(3, h, w)
            a_s = ndimage.map_coordinates(al, co, order=1, mode="nearest").reshape(h, w)
            h_s = ndimage.map_coordinates(hd, co, order=1, mode="nearest").reshape(h, w)
            inside = ((x >= 1) & (x < Wd - 2) & (y >= 1) & (y < Wd - 2)).reshape(h, w)
            ok = kc & inside & (a_s > 0.7) & (np.abs(h_s - hh.reshape(h, w)) < 2.5 * sp)
            dirs.append(-om)
            deltas.append(np.where(ok[None], samp - ref, 0.0))
            oks.append(ok)
    D = np.stack(deltas)                                   # d, 3, h, w
    OK = np.stack(oks)
    for i in range(1, len(D)):
        if OK[i].sum() < 10:
            D[i] = 0.0
            continue
        for k in range(3):
            D[i, k] = _gauss_known(D[i, k], OK[i], 1.5)[0]
        D[i] = _nearest_fill(D[i], OK[i])
    if holec.any() and not holec.all():
        D[1:] = _solve_fill(D[1:], holec, order=1)
    # Anchored at the picture's own direction: there the difference is zero
    # by definition, so the model is sum_k (Y_k(d) - Y_k(ref)) c_k and nothing
    # of the change can hide in a constant.
    Y = sh_basis(np.stack(dirs), 2)[:, 1:]                 # d, 8
    A = Y[1:] - Y[:1]
    pinv = np.linalg.solve(A.T @ A + 0.01 * np.eye(8), A.T)   # 8, d-1
    C = np.einsum("kd,dchw->kchw", pinv, D[1:])
    return C, Y[0]


def _grazing_fade(scene, cam: Ortho, rest, rgb, H, surf, hole, sp, pxsp, band=2.0):
    """How much of the surface's fine texture is left when it is seen at a slant.

    Lifted from one picture, a texture keeps its full contrast from every
    side. A scan does not: seen at a slant the pits between a sole's pebbles
    hide behind the pebbles and its large soft splats run together, so the
    same patch that is invisible square-on stands out from the side as crisp
    dots on a blurred ground. Measured here on the surface next to the hole:
    it is looked at from three directions 60 degrees off, each view is laid
    back onto the picture through the surface's height, and the share of the
    picture's fine detail (finer than ``band`` spacings) that is still there is read
    off (see below for how). Returns ``a`` in  kept(theta) = 1 - a * sin^2(theta).
    """
    near = surf & ndimage.binary_dilation(hole, iterations=max(4, int(8 * pxsp))) \
        & ~ndimage.binary_dilation(hole, iterations=max(1, int(pxsp)))
    ys, xs = np.where(near)
    if len(ys) < 400:
        return 0.0
    if len(ys) > 60000:
        pick = np.random.default_rng(0).choice(len(ys), 60000, replace=False)
        ys, xs = ys[pick], xs[pick]
    sig = max(1.0, band * pxsp)
    fine0 = rgb.mean(2) - ndimage.gaussian_filter(rgb.mean(2), sig)
    d0 = fine0[ys, xs]
    Pw = cam.world(xs, ys, H[ys, xs])
    kept = []
    tl = np.radians(60.0)
    for a in range(3):
        az = np.radians(120.0 * a + 15.0)
        om = cam.n * np.cos(tl) + (cam.t1 * np.cos(az) + cam.t2 * np.sin(az)) * np.sin(tl)
        n_, t1_, t2_ = _frame(om, cam.t1)
        Wd = int(cam.W * 1.25) + 8
        Wd += (-Wd) % 8
        cd = Ortho(cam.c, n_, t1_, t2_, cam.px, Wd, Wd, cam.dist)
        img, al, hd = scene.picture(cd, rest)
        lum = img.mean(2)
        # the same band as in the picture: along the slant a spacing is shorter
        fine = lum - ndimage.gaussian_filter(lum, sig)
        x, y, hh = cd.pixel(Pw)
        co = [y, x]
        s_f = ndimage.map_coordinates(fine, co, order=1, mode="nearest")
        a_s = ndimage.map_coordinates(al, co, order=1, mode="nearest")
        h_s = ndimage.map_coordinates(hd, co, order=1, mode="nearest")
        ok = (x >= 1) & (x < Wd - 2) & (y >= 1) & (y < Wd - 2) & (a_s > 0.7) & (np.abs(h_s - hh) < 2.0 * sp)
        if ok.sum() < 300:
            continue
        den = float((d0[ok] ** 2).sum())
        if den < 1e-9:
            continue
        # Two readings. By correlation with the picture's own detail: too low,
        # for laid back through a height that is off by a pixel the detail no
        # longer lines up, and that is not loss. By energy alone: too high,
        # for at a slant coarser structure is squeezed into the same band.
        # Their geometric mean sits between the two.
        by_corr = max(float((s_f[ok] * d0[ok]).sum()) / den, 0.0)
        by_energy = float(np.sqrt((s_f[ok] ** 2).sum() / den))
        kept.append(float(np.sqrt(by_corr * min(by_energy, 1.0))))
    if not kept:
        return 0.0
    g = float(np.clip(np.median(kept), 0.0, 1.0))
    return float(np.clip((1.0 - g) / np.sin(tl) ** 2, 0.0, 1.15))


# ----------------------------------------------------------------- the fill

@dataclass
class ImFillResult:
    removed: np.ndarray
    added: SplatSet
    info: dict = field(default_factory=dict)
    debug: dict = field(default_factory=dict)
    # for a further picture from the side: which of the selection this one
    # dealt with, which of its new splats sit on a steep flank, their normals
    passdata: dict = field(default_factory=dict)


def _spacing(xyz, alpha, c, r):
    near = np.where((np.linalg.norm(xyz - c, axis=1) < max(r, 1e-6)) & (alpha > 0.3))[0]
    if len(near) < 20:
        return None
    pts = xyz[near[:: max(1, len(near) // 4000)]]
    d = cKDTree(xyz[near]).query(pts, k=2)[0][:, 1]
    d = d[d > 0]
    return float(np.median(d)) if len(d) else None


def _fill_one(s: SplatSet, selection, view, params: Optional[ImFillParams] = None,
              debug: bool = False, volume=None, way_top=None, spacing=None, sheet=None,
              fade=None, check=None) -> ImFillResult:
    """One picture: remove ``selection`` from ``s`` and close the hole as seen along ``view``.

    ``way_top``: only splats up to this many spacings in front of the new
    surface count as in the way (a side view is not what the user looked
    through; what floats further out there is none of its business).
    ``spacing``: the scan's splat spacing if already known (with an earlier
    picture's dense fill in the scene it cannot be measured any more).
    ``fade``: the grazing fade if an earlier picture measured it already (it
    belongs to the surface, not to the view).
    ``check``: called with the new splats' positions as soon as they are
    known; if it says no, the picture stops there (mode "rejected") and the
    view tone, the costly half of the rest, is never worked out.
    ``sheet``: splats of the selection that are an earlier picture's fill.
    They mark where the hole is, but they are not what was there: the shape
    of "the surface as it was" is read without them.

    ``view`` is the direction the user looked along when painting (from the
    eye into the scene): the picture is taken along it.

    ``volume``: the painted volume (`volume.Volume`, something with
    ``inside(points)`` and ``capsules``). With it the hole is also where the
    continued surface runs through what was painted, splats or not: a hole in
    the scan has nothing to select, and is filled all the same.
    """
    P = params or ImFillParams()
    T0 = time.time()
    xyz = s.xyz.astype(np.float64)
    alpha = s.alpha
    sel = np.zeros(s.n, bool)
    sel[np.asarray(selection, dtype=np.int64)] = True
    v = np.asarray(view, float)
    v = v / max(np.linalg.norm(v), 1e-12)
    n0 = -v
    vol_pts = None
    if volume is not None and getattr(volume, "capsules", None):
        cp = [c for c in volume.capsules if getattr(c, "add", True)]
        if cp:
            vol_pts = np.array([[*np.asarray(c.a, float), float(c.r)] for c in cp]
                               + [[*np.asarray(c.b, float), float(c.r)] for c in cp])
    if sel.sum() < 3 and vol_pts is None:
        raise ValueError("the selection is empty")
    pts0 = xyz[sel] if sel.sum() >= 3 else vol_pts[:, :3]
    if vol_pts is not None and sel.sum() >= 3:
        pts0 = np.concatenate([pts0, vol_pts[:, :3]])
    c0 = np.median(xyz[sel], 0) if sel.sum() >= 30 else pts0.mean(0)
    d = pts0 - c0
    lat = d - np.outer(d @ n0, n0)
    r_sel = float(np.percentile(np.linalg.norm(lat, axis=1), 97)) + 1e-6
    if vol_pts is not None:
        dv = vol_pts[:, :3] - c0
        lv = np.linalg.norm(dv - np.outer(dv @ n0, n0), axis=1) + vol_pts[:, 3]
        r_sel = max(r_sel, float(lv.max()))
    sp = spacing or _spacing(xyz, alpha, c0, 2.5 * r_sel) or (r_sel / 10.0)
    sig_max = np.exp(s.scale.astype(np.float64)).max(1)

    def column(c, n, half):
        dd = xyz - c
        l = dd - np.outer(dd @ n, n)
        return np.abs(l).max(1) < half * 1.45 + 3.0 * sig_max

    extent = float(np.linalg.norm(xyz.max(0) - xyz.min(0)))

    # ---- 1. where is the visible surface that was painted, and how does it lie?
    n, t1, t2 = _frame(n0)
    half0 = 1.6 * r_sel
    W0 = int(np.clip(2 * half0 / sp, 32, 384))
    cam0 = Ortho(c0, n, t1, t2, 2 * half0 / W0, W0, W0, 2.0 * extent)
    keep0 = np.where(column(c0, n, half0))[0]
    sc0 = _Scene(s.take(keep0))
    vis0 = sc0.shown(cam0, sel[keep0])
    _, a0, h0 = sc0.picture(cam0)
    m0 = vis0 > 0.4
    c = c0
    if m0.sum() < 12 and vol_pts is not None:
        # nothing of a selection to see: stand on the surface under the stroke
        yy0, xx0 = np.mgrid[0:W0, 0:W0]
        rr = np.hypot(xx0 - W0 / 2, yy0 - W0 / 2) * cam0.px
        ring0 = (a0 > 0.5) & (rr < 1.3 * r_sel)
        if ring0.sum() >= 12:
            c = c0 + n * float(np.median(h0[ring0]))
    if m0.sum() >= 12:
        ys, xs = np.where(m0)
        pts = cam0.world(xs, ys, h0[ys, xs])
        c = pts.mean(0)
        X = pts - c
        _, vec = np.linalg.eigh(X.T @ X)
        nn = vec[:, 0]
        if nn @ n0 < 0:
            nn = -nn
        # The picture is taken along the user's own line of sight: what they
        # painted is what they saw, and from there the mask is theirs. (The
        # plane through the painted surface tilts with every wrinkle and with
        # whatever else the stroke's depth caught; it only serves as a check.)
        tilt = float(np.degrees(np.arccos(np.clip(nn @ n0, -1, 1))))
        r_hole = float(np.percentile(np.linalg.norm(X - np.outer(X @ n, n), axis=1), 98))
    else:
        r_hole = r_sel
        tilt = 0.0
    del sc0

    # ---- 2. the picture: colour, hole, and the height of the visible surface
    half = P.context * max(r_hole, 4 * sp)
    px = sp / P.pxps
    W = int(math.ceil(2 * half / px))
    if W > P.max_px:
        W = P.max_px
        px = 2 * half / W
    W += (-W) % 8
    cam = Ortho(c, n, t1, t2, px, W, W, 2.0 * extent)
    keep = np.where(column(c, n, half))[0]
    scene = _Scene(s.take(keep))
    selk = sel[keep]
    pxsp = sp / px                                     # pixels per spacing
    vis = scene.shown(cam, selk)
    rest_np = np.where(~selk)[0]
    if len(rest_np) < 20:
        # everything in the picture is selected: there is no surface left to
        # continue (Ctrl+A, then "Repair")
        raise ValueError("nothing is left around the selection for the fill to connect to")
    rest = scene.torch.tensor(rest_np, device=scene.dev)
    rgb, a_rest, h_all = scene.picture(cam, rest)

    rgb_all, a_all, h_all_full = scene.picture(cam)
    seen = vis > 0.25
    painted = np.zeros_like(seen)
    if vol_pts is not None:
        # first guess of the painted footprint: the stroke's discs, seen from here
        yy, xx = np.mgrid[0:W, 0:W]
        for q in vol_pts:
            qx, qy, _ = cam.pixel(q[None, :3])
            painted |= np.hypot(xx - qx[0], yy - qy[0]) * px < q[3]
        k2 = len(vol_pts) // 2
        for i in range(k2):                       # and the strip between a capsule's ends
            ax_, ay_, _ = cam.pixel(vol_pts[i][None, :3])
            bx_, by_, _ = cam.pixel(vol_pts[i + k2][None, :3])
            dx_, dy_ = bx_[0] - ax_[0], by_[0] - ay_[0]
            L2 = dx_ * dx_ + dy_ * dy_
            if L2 > 1e-9:
                tpar = np.clip(((xx - ax_[0]) * dx_ + (yy - ay_[0]) * dy_) / L2, 0, 1)
                painted |= np.hypot(xx - ax_[0] - tpar * dx_, yy - ay_[0] - tpar * dy_) * px < vol_pts[i][3]
    if seen.sum() < 9 and painted.sum() < 9:
        raise ValueError("none of the selection is visible from here")
    hole_wide = ndimage.binary_dilation(seen | painted, iterations=max(2, int(round(2 * pxsp))))

    # a smooth guess of the front, from what is left around the hole
    fcell = max(2, int(round(pxsp)))
    Hh = W // fcell
    front = _block_quantile(h_all, (a_rest > 0.3) & ~hole_wide, fcell, 0.8)
    okf = np.isfinite(front)
    if not okf.any():
        raise ValueError("there is no surface around the selection to connect to")
    front = _nearest_fill(np.where(okf, front, 0.0), okf)
    front = ndimage.median_filter(front, 3, mode="nearest")
    holec0 = _down_known(hole_wide.astype(float), np.ones_like(hole_wide), fcell)[0] > 0.5
    if holec0.any() and not holec0.all():
        front = _solve_fill(front, holec0, order=2)
    front = ndimage.gaussian_filter(front, 1.5, mode="nearest")
    xk, yk, hk = cam.pixel(xyz[keep])

    def at_splats(coarse):
        return ndimage.map_coordinates(coarse, [(yk + 0.5) / fcell - 0.5, (xk + 0.5) / fcell - 0.5],
                                       order=1, mode="nearest")

    # The expected depth is only a first guess: under a thin outer layer it
    # lies between that layer and the next. A coarse pass over a wide range
    # (one spacing apart, 12 up, 20 down) finds where the opacity really
    # starts; the fine pass then rides on that.
    h1, f1 = scene.surface(cam, rest_np, (hk - at_splats(front))[~selk], _up(front, (W, W)), sp,
                           thr=P.surface_thr, top=12.0, step=1.0, nlev=33)
    fr1 = _block_quantile(h1, f1 & ~hole_wide, fcell, 0.5)
    ok1 = np.isfinite(fr1)
    if ok1.any():
        fr1 = _nearest_fill(np.where(ok1, fr1, 0.0), ok1)
        fr1 = ndimage.median_filter(fr1, 3, mode="nearest")
        if holec0.any() and not holec0.all():
            fr1 = _solve_fill(fr1, holec0, order=2)
        front = ndimage.gaussian_filter(fr1, 1.0, mode="nearest")
    guess_at = at_splats(front)
    guess = _up(front, (W, W))
    def clean(hv, fd):
        # A scan's skin is thin in places: where it lets the eye through, the
        # level found is the next layer down. Such leaks are narrow pits, not
        # relief (from the side the skin looks closed), and lifted as they are
        # they would stand in the fill as pegs and holes. Close them.
        hf_, _ = _gauss_known(hv, fd, max(1.0, 0.4 * pxsp))
        hv = np.where(fd, hv, hf_)
        k = max(3, int(round(P.envelope * pxsp)) | 1)
        hv = ndimage.grey_closing(hv, size=(k, k), mode="nearest")
        return ndimage.median_filter(hv, 5, mode="nearest")

    thr = P.surface_thr
    h_vis, found = scene.surface(cam, rest_np, (hk - guess_at)[~selk], guess, sp, thr=thr,
                                 shell=P.skin)
    h_rest = clean(h_vis, found)
    # the surface as it was, selection and all, riding on its own coarse pass
    # (what was painted may stand well off what the rim suggests)
    allk = np.arange(len(keep))
    if sheet is not None and len(sheet):
        shs = np.zeros(s.n, bool)
        shs[np.asarray(sheet, dtype=np.int64)] = True
        allk = np.where(~shs[keep])[0]
    h1w, f1w = scene.surface(cam, allk, (hk - guess_at)[allk], guess, sp, thr=P.surface_thr,
                             top=16.0, step=1.0, nlev=33)
    frw = _block_quantile(h1w, f1w, fcell, 0.5)
    okw = np.isfinite(frw)
    if okw.any():
        frw = _nearest_fill(np.where(okw, frw, 0.0), okw)
        frw = ndimage.gaussian_filter(ndimage.median_filter(frw, 3, mode="nearest"), 1.0, mode="nearest")
    else:
        frw = front
    h_was, found_was = scene.surface(cam, allk, (hk - at_splats(frw))[allk], _up(frw, (W, W)), sp, thr=thr,
                                     shell=P.skin)
    h_was = clean(h_was, found_was)

    # The hole is every pixel the removal changed: where the selection showed,
    # where the colour is another now, where the visible surface dropped. A
    # brush's soft edge thins the front layer well beyond where it "showed".
    changed = (np.abs(rgb_all - rgb).max(2) > 0.06) | (found_was & (np.abs(h_was - h_rest) > 0.6 * sp)) \
        | (found_was & ~found)
    hole = seen | (changed & hole_wide)
    hole = ndimage.binary_opening(hole, iterations=1) | seen
    hole_painted = np.zeros_like(hole)
    if vol_pts is not None and painted.any():
        # where the painted volume holds the continued surface: what was
        # painted is rebuilt, whether splats were there to select or not
        yy_, xx_ = np.where(painted)
        Pp = cam.world(xx_, yy_, guess[yy_, xx_])
        ok_in = np.asarray(volume.inside(Pp), bool)
        hole_painted[yy_[ok_in], xx_[ok_in]] = True
        hole_painted = ndimage.binary_opening(hole_painted, iterations=1)
        hole |= hole_painted
    hole = ndimage.binary_closing(hole, iterations=max(1, int(round(0.5 * pxsp))))
    hole = ndimage.binary_fill_holes(hole)
    # What the user took out of the stroke again on purpose stays as it is,
    # however the hole comes to enclose it (closing, the painted footprint,
    # the layers further down): a pixel whose surface as it was lies inside
    # something un-painted and outside what is still painted is kept out of
    # the hole for good -- not filled, and nothing of it removed.
    cuts = [c_ for c_ in list(getattr(volume, "capsules", ())) + list(getattr(volume, "regions", ()))
            if not getattr(c_, "add", True)] if volume is not None else []
    kept_out = np.zeros_like(hole)
    if cuts:
        from .volume import capsule_inside
        yy_, xx_ = np.where(ndimage.binary_dilation(hole, iterations=max(1, int(pxsp))))
        if len(yy_):
            Pp = cam.world(xx_, yy_, np.where(found_was, h_was, guess)[yy_, xx_])
            un = np.zeros(len(yy_), bool)
            for c_ in cuts:
                un |= np.asarray(c_.inside(Pp) if hasattr(c_, "inside") else capsule_inside(c_, Pp), bool)
            un &= ~np.asarray(volume.inside(Pp), bool)
            kept_out[yy_[un], xx_[un]] = True
            kept_out = ndimage.binary_opening(kept_out, iterations=max(1, int(round(0.5 * pxsp))))
            hole &= ~kept_out
    hole = ndimage.binary_dilation(hole, iterations=max(1, int(round(0.4 * pxsp)))) & ~kept_out

    hole0 = hole.copy()                               # the hole as painted, before anything leaves it
    solid = (a_rest > 0.7) & (ndimage.gaussian_filter(found.astype(float), 0.5 * pxsp) > 0.3)
    f = fcell

    def same_surface(hole):
        """The surface the hole lies in: reachable from the rim without a jump in height."""
        known = solid & ~ndimage.binary_dilation(hole, iterations=1)
        hs, _ = _gauss_known(h_rest, known, 1.0 * pxsp)
        gy, gx = np.gradient(hs)
        calm = known & (np.hypot(gx, gy) / px < 3.7)       # no step: under 75 degrees
        lab, _ = ndimage.label(calm)
        ring = ndimage.binary_dilation(hole, iterations=max(3, int(3 * pxsp))) & calm
        ids = np.unique(lab[ring])
        ids = ids[ids > 0]
        if len(ids):
            cnt = ndimage.sum(ring, lab, ids)
            ids = ids[cnt >= 0.1 * cnt.max()]
        return ndimage.binary_closing(np.isin(lab, ids), iterations=max(1, int(pxsp))) & known

    _L2 = {}

    def base_of(hole, surf):
        """Smooth base per cell, continued through the hole with value and slope."""
        zc, share = _down_known(h_rest, surf, f)
        kc = share > 0.5
        zc = _nearest_fill(zc, kc)
        holec = _down_known(hole.astype(float), np.ones_like(hole), f)[0] > 0
        # The continuation takes the rim's height *and slope*, and right at a
        # hole both are the least trustworthy thing in the picture: torn skin
        # sags, a brush's soft edge thins it. A sagging rim continued with
        # its slope is a bowl. So the base is not read right at the hole.
        kfit = kc & ~ndimage.binary_dilation(holec, iterations=max(1, int(round(P.guard))))
        if kfit.sum() < 30:
            kfit = kc
        zl = _loclin(zc, kfit, P.base_sigma)
        ys, xs = np.where(holec)
        zl = _nearest_fill(zl, kfit)
        Bc = zl.copy()
        if len(ys):
            m = int(max(8, 0.5 * max(np.ptp(ys), np.ptp(xs)))) + int(round(P.guard))
            y0, y1 = max(0, ys.min() - m), min(zc.shape[0], ys.max() + m + 1)
            x0, x1 = max(0, xs.min() - m), min(zc.shape[1], xs.max() + m + 1)
            Bc[y0:y1, x0:x1] = _solve_fill(zl[y0:y1, x0:x1], ~kfit[y0:y1, x0:x1], order=2)
        return Bc, kc, holec, ys, xs

    front_info = {}
    behind = {}                                       # the surface far behind the main piece, if there is one

    def front_only(surf, hole):
        """Of the surface round the hole, drop what lies far behind the main piece.

        Painted over an edge, the hole has the surface the user looked at on
        one side and, past the edge, whatever lies behind and below it (under
        a sole's edge: the shoe's side, running away from the eye). Joined by
        one smooth base they give a ramp from the one down to the other, and
        the fill hangs over the edge. So the piece with most of the rim is
        continued on its own first; what is not part of that piece and lies
        well below its continuation everywhere (eight spacings at its
        nearest) is another layer and is left out. What it shows is then
        "past the edge" (step 4); where it is missing itself it is filled as
        a layer of its own (step 7).
        """
        if not P.layers:
            return surf
        # pieces: joined without a jump in height (`same_surface` closes small
        # gaps afterwards, which can weld two layers along an edge)
        hs, _ = _gauss_known(h_rest, surf, 1.0 * pxsp)
        gy_, gx_ = np.gradient(hs)
        labs, ns = ndimage.label(surf & (np.hypot(gx_, gy_) / px < 3.7))
        if ns < 2:
            return surf
        ring = ndimage.binary_dilation(hole, iterations=max(3, int(3 * pxsp)))
        cnt = ndimage.sum(ring, labs, np.arange(1, ns + 1))
        if cnt.max() < 20:
            return surf
        main = labs == int(np.argmax(cnt)) + 1
        # Only where the picture faces that piece (under 45 degrees). Seen at
        # a slant, what lies "behind" is mostly the same surface running away
        # from the eye, and no layer of its own: there the base bridges as
        # before and the side view sorts it out.
        if float(np.median((np.hypot(gx_, gy_) / px)[main & ring])) > 1.0:
            return surf
        Bd = _up(base_of(hole, main)[0], hole.shape)
        # ... and nowhere comes near it. A surface that meets the main piece
        # at a crease and then falls away (the leather under a sole's wall)
        # is no other layer: it stays, and the base bends round the crease.
        cand = ndimage.binary_opening(surf & ~main & (h_rest < Bd - 4.0 * sp), iterations=1)
        labf, nf = ndimage.label(cand)
        if nf == 0:
            return surf
        nearest = ndimage.labeled_comprehension((Bd - h_rest) / sp, labf, np.arange(1, nf + 1),
                                                lambda v: np.percentile(v, 5), float, 0.0)
        far = np.isin(labf, np.where(nearest >= 8.0)[0] + 1)
        if far.sum() < 20:
            return surf
        out = surf & ~ndimage.binary_dilation(far, iterations=1)
        if out.sum() < 50:
            return surf
        front_info["farSurface"] = round(float((far & ring).sum() / max((surf & ring).sum(), 1)), 2)
        behind["surf"] = far
        return out

    surf = same_surface(hole)
    # Which layer was painted? At an edge the picture holds two: the surface
    # that was painted and, past its edge, whatever lies behind (the inside of
    # a collar behind its rim). Taken as one, the base is strung between the
    # two: the fill sags towards the far layer and ends in a curtain hanging
    # from the edge. The scene as it was says which is which: the layer most
    # of the hole lay in, followed outwards without a jump in height, is the
    # one to continue. What the old scene showed behind it, inside the hole,
    # was seen past its edge and is no hole at all.
    layer_info = {}
    was_layer = None                                  # where the painted layer lay, if the old scene says so
    # Does the old scene explain the hole? Painted over a print or a scuff,
    # what was there *was* the surface: one walks from the rim onto it without
    # a climb (under 55 degrees). Then the old scene says where the layer lay
    # and where it ended, and that is used. A thing that stood on the surface
    # has walls; what it hid is not known, and there the picture itself has
    # to tell the layers apart (`front_only`).
    explained = False
    old_ok = found_was & (hole | solid)
    if P.layers and old_ok.sum() > 50:
        hso, _ = _gauss_known(h_was, old_ok, 1.0 * pxsp)
        gyo, gxo = np.gradient(hso)
        calm_o = old_ok & (np.hypot(gxo, gyo) / px < 3.7)
        labo, _ = ndimage.label(calm_o)
        cnt_h = np.bincount(labo[hole & calm_o].ravel(), minlength=int(labo.max()) + 1)
        cnt_h[0] = 0
        ring_s = ndimage.binary_dilation(hole, iterations=max(3, int(3 * pxsp))) & surf
        if ring_s.any():
            # (the climb is measured against the lie of the surface at large,
            # not against the picture: seen at a slant, a flat wall "climbs")
            reg = old_ok & ndimage.binary_dilation(hole, iterations=max(4, int(6 * pxsp)))
            yy_, xx_ = np.where(reg)
            sx_, sy_ = 0.0, 0.0
            if len(yy_) > 100:
                pick = slice(None, None, max(1, len(yy_) // 20000))
                A_ = np.stack([xx_[pick], yy_[pick], np.ones(len(yy_[pick]))], 1).astype(np.float64)
                sx_, sy_, _ = np.linalg.lstsq(A_, hso[yy_[pick], xx_[pick]], rcond=None)[0]
            labg, _ = ndimage.label(old_ok & (np.hypot(gxo - sx_, gyo - sy_) / px < 1.43))
            idg = np.unique(labg[ring_s])
            idg = idg[idg > 0]
            if len(idg):
                cg = ndimage.sum(ring_s, labg, idg)
                idg = idg[cg >= 0.1 * cg.max()]
            explained = bool((np.isin(labg, idg) & hole).sum() > 0.3 * hole.sum())
        if cnt_h.max() > 0 and ring_s.any():
            near_old = np.isin(labo, np.where(cnt_h >= 0.1 * cnt_h.max())[0])
            near_c = ndimage.binary_closing(near_old, iterations=max(1, int(pxsp)))
            surf_n = surf & near_c
            if (near_old & hole).sum() > 0.3 * hole.sum() and (near_c & ring_s).sum() > 0.25 * ring_s.sum() \
                    and surf_n.sum() >= 50:
                hn, wn = _gauss_known(h_was, near_old, 2.0 * pxsp)
                h_near = _nearest_fill(hn, wn > 0.05)
                beyond = hole & found_was & ~ndimage.binary_dilation(near_old, iterations=max(1, int(0.6 * pxsp))) \
                    & (h_was < h_near - 4.0 * sp)
                beyond = ndimage.binary_opening(beyond, iterations=1)
                left = hole & ~beyond
                layer_info = {"farLayer": round(float((surf & ~surf_n & ring_s).sum() / max(ring_s.sum(), 1)), 2),
                              "beyond": int(beyond.sum())}
                surf = surf_n
                was_layer = near_c
                if beyond.any() and left.sum() > 0.3 * hole.sum():
                    hole = ndimage.binary_opening(left, iterations=1) & ~kept_out
    if not explained:
        surf = front_only(surf, hole)
    if surf.sum() < 50:
        raise ValueError("there is no surface around the selection to connect to")

    # ---- 3. base surface and relief
    Bc, kc, holec, ys, xs = base_of(hole, surf)
    # A hole's rim is rarely clean: where the skin was torn it is thin, and
    # the surface found there is the layer below. Left as "known", those pits
    # pull the base down and the fill ends at a cliff. What lies well under
    # the continued base right at the hole belongs to the hole.
    near_h = ndimage.binary_dilation(hole, iterations=max(2, int(4 * pxsp)))
    dev0 = (h_rest - _up(Bc, hole.shape))
    sd0 = 1.4826 * float(np.median(np.abs(dev0[surf] - np.median(dev0[surf])))) if surf.any() else 0.0
    pit = solid & ~hole & near_h & (dev0 < -max(2.0 * sp, 4.0 * sd0))
    labp, _ = ndimage.label(pit | hole)
    idp = np.unique(labp[hole])
    grown = np.isin(labp, idp[idp > 0]) & (pit | hole)
    if grown.sum() > hole.sum() + 8:
        hole = ndimage.binary_fill_holes(ndimage.binary_closing(grown, iterations=max(1, int(0.5 * pxsp))))
        hole = ndimage.binary_dilation(hole, iterations=1) & ~kept_out
        surf = same_surface(hole)
        if not explained:
            surf = front_only(surf, hole)
        if surf.sum() < 50:
            raise ValueError("there is no surface around the selection to connect to")
        Bc, kc, holec, ys, xs = base_of(hole, surf)
    known = solid & ~ndimage.binary_dilation(hole, iterations=1)
    prior_share = 0.0
    prior_dbg = {}
    if P.shape_prior:
        # What was painted over had a shape of its own. Where it lay *in* the
        # surface -- a print, a stain, a scuff: one can walk from the rim onto
        # it without climbing a step -- that shape is the surface, known
        # rather than guessed, and its smooth part is kept. A thing standing
        # on the surface has walls; behind them the rim's continuation stays.
        was_ok = (ndimage.binary_dilation(hole, iterations=3) & found_was) | surf
        hsw, _ = _gauss_known(h_was, was_ok, 1.0 * pxsp)
        gyw, gxw = np.gradient(hsw)
        calm_w = was_ok & (np.hypot(gxw, gyw) / px < 3.7)
        labw, _ = ndimage.label(calm_w)
        rim_px = surf & ndimage.binary_dilation(hole, iterations=max(2, int(2 * pxsp)))
        idw = np.unique(labw[rim_px])
        idw = idw[idw > 0]
        if len(idw):
            cw = ndimage.sum(rim_px, labw, idw)
            idw = idw[cw >= 0.1 * cw.max()]
        level = np.isin(labw, idw) & hole
        zw, sw = _down_known(h_was, was_ok, f)
        kw = sw > 0.5
        zwl, _ = _gauss_known(zw - Bc, kw, P.base_sigma)       # the old shape against the base
        zwl = zwl + Bc
        ext_c = max(np.ptp(ys), np.ptp(xs)) + 1
        dlt = np.where(kw, zwl - Bc, 0.0)
        # Above the continued base the old shape is believed wherever one can
        # walk onto it. *Below* it only where the walk is gentle all the way
        # (under 35 degrees): a dent in a whole skin is smooth, while what
        # shows inside a hole in the scan -- the layer below, torn skin -- lies
        # behind a drop at the tear. Believed downwards, that is a bowl.
        gentle = was_ok & (np.hypot(gxw, gyw) / px < 0.7)
        labg, _ = ndimage.label(gentle)
        idg = np.unique(labg[rim_px])
        idg = idg[idg > 0]
        if len(idg):
            cg = ndimage.sum(rim_px, labg, idg)
            idg = idg[cg >= 0.1 * cg.max()]
        smooth_c = _down_known((np.isin(labg, idg) & hole).astype(float), np.ones_like(hole), f)[0] > 0.7
        tau = np.where(smooth_c, max(4.0 * sp, 0.25 * ext_c * f * px), max(3.0 * sp, 0.12 * ext_c * f * px))
        dlt = np.where(smooth_c, dlt, np.maximum(dlt, 0.0))
        joined = float(smooth_c[holec].mean()) if holec.any() else 0.0
        wgt = (_down_known(level.astype(float), np.ones_like(level), f)[0] > 0.7) & kw & holec
        wgt = wgt * np.clip(2.0 - np.abs(dlt) / tau, 0.0, 1.0)
        wgt = ndimage.gaussian_filter(wgt.astype(float), 0.7) * holec
        prior_share = float(wgt[holec].mean()) if holec.any() else 0.0
        prior_dbg = {"wasFound": round(float(found_was[hole].mean()), 2),
                     "level": round(float(level[hole].mean()), 2),
                     "priorShift": round(float(np.abs(dlt[holec]).mean() / sp), 2),
                     "gentle": round(joined, 2)}
        Bc = Bc + wgt * dlt
    B = _up(Bc, hole.shape)
    # Is there anything to fill? Where taking the selection away uncovers
    # skin that is whole and lies in the continued surface (a pole before a
    # wall, a sticker on leather), nothing is missing: those pixels leave the
    # hole and count as surface. If hardly anything is left, it is a delete.
    def nothing(why, share):
        K0 = s.f_rest.shape[1]
        empty = SplatSet(xyz=np.zeros((0, 3), np.float32), f_dc=np.zeros((0, 3), np.float32),
                         f_rest=np.zeros((0, K0, 3), np.float32), opacity=np.zeros(0, np.float32),
                         scale=np.zeros((0, 3), np.float32), rot=np.zeros((0, 4), np.float32),
                         extra={k: vv[:0].copy() for k, vv in s.extra.items()})
        return ImFillResult(removed=np.where(sel)[0], added=empty,
                            info={"engine": "image", "mode": "delete", "removed": int(sel.sum()),
                                  "added": 0, "why": why, "whole": round(float(share), 2),
                                  "seconds": round(time.time() - T0, 2)})

    # ... and only where what was taken stood clear of it. Under a skin that
    # was painted away lies the next layer, often within a spacing or two:
    # that is the inside showing, not a surface that was whole all along.
    # (far off counts as clear; and where the surface did not move at all,
    # what went was only a veil before it)
    clear = np.where(found_was, (h_was - h_rest > 3.0 * sp) | (np.abs(h_was - h_rest) < 0.3 * sp), True)
    # ... and looks like the surface around it: its colours lie within what
    # the rim shows (the inside of a shoe behind white leather does not)
    rim_c = surf & ndimage.binary_dilation(hole, iterations=max(3, int(4 * pxsp)))
    soft = np.stack([ndimage.gaussian_filter(rgb[..., k], 0.5 * pxsp) for k in range(3)], 2)
    if rim_c.sum() > 30:
        lo_c = np.percentile(soft[rim_c], 2, axis=0) - 0.08
        hi_c = np.percentile(soft[rim_c], 98, axis=0) + 0.08
        alike = ((soft >= lo_c) & (soft <= hi_c)).all(2)
    else:
        alike = np.ones(hole.shape, bool)
    whole = hole & found & (a_rest > 0.7) & (np.abs(h_rest - B) < 2.0 * sp) & clear & alike
    whole = ndimage.binary_opening(whole, iterations=max(1, int(0.5 * pxsp)))
    # slivers along the outline of what stood there are its soft edge, not holes
    todo = hole & ~whole
    if whole.sum() > 0.3 * hole.sum():
        todo = ndimage.binary_opening(todo, iterations=max(1, int(round(0.75 * pxsp))))
    n_hole = int(hole.sum())
    if todo.sum() < max(9, 0.08 * n_hole):
        return nothing("the surface behind it is intact", whole.sum() / max(n_hole, 1))
    if whole.sum() > 0.1 * n_hole:
        hole = ndimage.binary_dilation(todo, iterations=max(1, int(round(0.4 * pxsp)))) & hole
        surf = surf | (whole & ~ndimage.binary_dilation(hole, iterations=1))
    relief = np.where(surf, h_rest - B, 0.0)
    rel_sd = 1.4826 * float(np.median(np.abs(relief[surf] - np.median(relief[surf])))) + 1e-9
    wild = np.abs(relief) > max(8.0 * rel_sd, 3.0 * sp)
    valid = surf & ~ndimage.binary_dilation(wild, iterations=1)
    valid = ndimage.binary_opening(valid, iterations=1)

    # ---- 4. colour: LaMa on the picture (or a membrane of the rim's colours)
    lama_mask = ndimage.binary_dilation(hole, iterations=max(1, int(round(0.4 * pxsp))))
    # Does the hole touch an edge of the painted surface? Past an edge the
    # picture shows nothing, or another layer far behind (the inside of a
    # shoe past its collar). Neither is the surface to continue: LaMa is not
    # to read colours from there (it would paint the dark inside onto the
    # rim), so that part of the picture is hidden from it as well.
    other = ~lama_mask & ((a_rest < 0.5) | (found & (h_rest < B - 4.0 * sp) & P.layers))
    # Nor is LaMa to read from surface the picture sees at a slant (over 60
    # degrees): squeezed to under half, its texture is a blur, and LaMa mixes
    # that blur into the fill. A side view has the first picture's fill in
    # just that state along one rim; the scan's own wall, seen square-on, is
    # along the other, and that is the texture to continue. Only where enough
    # of the rim is seen square-on to read from.
    slanted = np.zeros_like(hole)
    hsm_, _ = _gauss_known(h_rest, surf, 1.0 * pxsp)
    gys_, gxs_ = np.gradient(hsm_)
    steep_ctx = surf & (np.hypot(gxs_, gys_) / px > 1.73)
    ring_c = ndimage.binary_dilation(hole, iterations=max(4, int(4 * pxsp))) & surf
    if ring_c.sum() > 50:
        share_s = float(steep_ctx[ring_c].mean())
        if 0.1 < share_s < 0.7:
            slanted = ndimage.binary_dilation(steep_ctx, iterations=1) & ~lama_mask
            if (valid & ~slanted).sum() > 2000:
                valid = valid & ~slanted               # ... and the relief is not copied from there either
    other = ndimage.binary_opening(other, iterations=1)
    ring_a = ndimage.binary_dilation(hole, iterations=max(2, int(2 * pxsp))) & ~hole
    at_edge = bool(ring_a.any() and float(other[ring_a].mean()) > 0.03)
    guide, lama_scale = None, 1.0
    t_l = time.time()
    if P.lama:
        try:
            guide, lama_scale = _lama_scaled(rgb, ((lama_mask | other) if at_edge else lama_mask) | slanted, P.lama_hole,
                                             box=lama_mask)
        except Exception:                         # noqa: BLE001
            guide = None
    used_lama = guide is not None
    cover = np.ones(hole.shape, bool)
    if used_lama and at_edge:
        # Where does the surface end inside the hole? LaMa continues the edge
        # in a picture of "the painted layer is here" as it would in colours;
        # where it says "not here", there is nothing to put a surface on.
        try:
            ag, _ = _lama_scaled(np.repeat((~other).astype(np.float32)[..., None], 3, 2), lama_mask, P.lama_hole)
            cover = ag.mean(2) > 0.5
        except Exception:                         # noqa: BLE001
            pass
        if was_layer is not None:
            # Where the layer was, it is again: that much is known. "Was"
            # means it lay where the new surface lies -- a thing that stood
            # on the surface was there too, a few spacings up, and says
            # nothing about where the surface ends. The layer's soft outer
            # edge (fuzz, where no skin is found) counts with it: what the
            # old picture showed solid right beside it was it.
            same = was_layer & found_was & (np.abs(h_was - B) < 3.0 * sp)
            cover |= same
            if sheet is None or not len(sheet):
                cover |= ndimage.binary_dilation(same, iterations=max(1, int(pxsp))) & (a_all > 0.5)
    if guide is None:
        gc = np.stack([_down_known(rgb[..., k], ~lama_mask & solid, f)[0] for k in range(3)])
        hc = _down_known((lama_mask | ~solid).astype(float), np.ones_like(hole), f)[0] > 0.5
        gc = _solve_fill(_nearest_fill(gc, ~hc), hc, order=1)
        guide = np.clip(_up(gc, hole.shape).transpose(1, 2, 0), 0, 1)
        lama_scale = 0.0
    t_l = time.time() - t_l

    # ---- 5. relief and grain by analogy: what real patches like these carry
    t_s = time.time()
    M = _Match(P.patch, P.seed)
    torch = M.torch
    tt = lambda a: torch.tensor(np.ascontiguousarray(a), device=M.dev, dtype=torch.float32)
    src = tt(rgb).permute(2, 0, 1)
    hole_t = torch.tensor(hole, device=M.dev)
    tgt = torch.where(hole_t[None], tt(guide).permute(2, 0, 1), src)
    # what LaMa could resolve: its own pixel when it ran reduced, a soft one otherwise
    if lama_scale >= 0.9:
        band = 0.7
    elif lama_scale > 0:
        band = min(0.6 / max(lama_scale, 0.15), 6.0)
    else:
        band = min(2.0 * pxsp, 6.0)
    nn = M.run(tgt, src, hole_t, torch.tensor(valid, device=M.dev), band=band)
    col = np.where(hole[..., None], guide, rgb)
    rel = np.zeros_like(relief)
    seam_info = {}
    if nn is not None:
        fine = src - M.blur(src, band)
        Vv, wv = M.vote(torch.cat([tt(relief)[None], fine]), nn)
        Vv, wv = Vv.cpu().numpy(), wv.cpu().numpy()
        rel = ndimage.gaussian_filter(np.where(wv > 0, Vv[0], 0.0), 0.3 * pxsp)
        rel = np.where(hole & (wv > 0), rel, 0.0) * P.relief
        g_share = P.grain if lama_scale >= 0.9 else 1.0
        if g_share > 0:
            g_fine = (tgt - M.blur(tgt, band)).cpu().numpy().transpose(1, 2, 0)
            add = Vv[1:4].transpose(1, 2, 0) - g_fine
            col = np.where((hole & (wv > 0))[..., None], col + g_share * add, col)
        # ---- 6. seam of the relief: gradients from the patches, values from the rim
        if P.seam:
            rim = ndimage.binary_dilation(hole, iterations=max(2, int(pxsp))) & ~hole & valid & (wv > 0)
            rc = _down_known(rim.astype(float), np.ones_like(rim), f)[1] > 0.3
            hc = (_down_known(hole.astype(float), np.ones_like(hole), f)[0] > 0) & ~rc
            if rc.sum() > 4 and hc.any():
                Ec = _down_known(Vv[0] - relief, rim, f)[0]
                Ec = np.clip(Ec, -1.5 * sp, 1.5 * sp)
                Ec = _solve_fill(_nearest_fill(Ec, rc), hc, order=1)
                rel = np.where(hole, rel - P.relief * _up(Ec, hole.shape), rel)
                seam_info = {"seamHeight": round(float(np.abs(Ec[rc]).mean() / sp), 3)}
    col = np.clip(col, 0.0, 1.0)
    t_s = time.time() - t_s

    # ---- 7. lift every filled pixel to a splat on the continued surface
    Hf = np.where(hole, B + rel, np.where(surf, h_rest, B))
    lift = hole & cover & ~kept_out
    # The layer behind. Where the front surface ends inside the hole, the
    # hole goes on in the layer behind it (a stroke over a wall and the
    # leather it stands on). That layer is continued the same way, from its
    # own surface and with its own colours: LaMa reads only that layer. It is
    # filled only where it is missing; where it shows whole past the front
    # layer's edge, nothing is wanted.
    far_info = {}
    far_s = behind.get("surf")
    if far_s is not None and used_lama:
        opened = ndimage.binary_opening(hole & ~cover, iterations=max(1, int(0.3 * pxsp)))
        ring_f = ndimage.binary_dilation(opened, iterations=max(3, int(3 * pxsp))) & far_s
        if opened.sum() >= 2.0 * pxsp * pxsp and ring_f.sum() >= 50:
            try:
                Bf = _up(base_of(opened, far_s)[0], hole.shape)
                whole_f = found & (a_rest > 0.7) & (np.abs(h_rest - Bf) < 2.0 * sp)
                lift_f = opened & ~ndimage.binary_dilation(whole_f & opened, iterations=1)
                lift_f = ndimage.binary_opening(lift_f, iterations=1)
                if lift_f.sum() >= 2.0 * pxsp * pxsp:
                    g_far, _ = _lama_scaled(rgb, lama_mask | ~far_s, P.lama_hole,
                                            box=ndimage.binary_dilation(lift_f, iterations=2))
                    Hf = np.where(lift_f, Bf, Hf)
                    col = np.where(lift_f[..., None], g_far, col)
                    lift = (lift | lift_f) & ~kept_out
                    far_info = {"behind": int(lift_f.sum())}
            except Exception:                         # noqa: BLE001  (the front layer's fill stands without it)
                pass
    Hs = ndimage.gaussian_filter(Hf, max(0.8, 0.6 * pxsp))
    gy, gx = np.gradient(Hs)
    if lift.sum() < 9:
        return nothing("there is nothing behind it to connect to", 0.0)
    ys, xs = np.where(lift)
    pos = cam.world(xs, ys, Hf[ys, xs])
    if check is not None and not check(pos):
        out = nothing("discarded", 0.0)
        out.info["mode"] = "rejected"
        return out
    sx, sy = gx[ys, xs] / px, gy[ys, xs] / px
    nl = n[None] - sx[:, None] * t1[None] - sy[:, None] * t2[None]
    s2 = sx * sx + sy * sy
    # how much more surface one pixel covers: from the smooth lie of the
    # surface only (a stretched splat on a rough spot is a needle)
    Hb = ndimage.gaussian_filter(Hf, 1.5 * pxsp)
    by_, bx_ = np.gradient(Hb)
    bsx, bsy = bx_[ys, xs] / px, by_[ys, xs] / px
    slope = np.sqrt(1.0 + np.minimum(bsx * bsx + bsy * bsy, 6.25))
    nl /= np.linalg.norm(nl, axis=1, keepdims=True)
    # the first axis runs up the slope: there one pixel covers `slope` times more
    up_dir = bsx[:, None] * t1[None] + bsy[:, None] * t2[None]
    up_dir = np.where((bsx * bsx + bsy * bsy > 1e-4)[:, None], up_dir, t1[None])
    e1 = up_dir - (np.einsum("ij,ij->i", up_dir, nl))[:, None] * nl
    e1 /= np.maximum(np.linalg.norm(e1, axis=1, keepdims=True), 1e-9)
    e2 = np.cross(nl, e1)
    from scipy.spatial.transform import Rotation
    q = Rotation.from_matrix(np.stack([e1, e2, nl], axis=2)).as_quat()      # x,y,z,w
    quat = np.concatenate([q[:, 3:4], q[:, :3]], 1)
    st = P.splat_sigma * px
    scale = np.log(np.stack([st * slope, np.full_like(slope, st), np.full_like(slope, 0.3 * px)], 1))
    K = s.f_rest.shape[1]
    N = len(ys)
    f_rest = np.zeros((N, K, 3), np.float32)
    base = col[ys, xs]
    tone_info = {}
    fade_cf = np.zeros((N, min(K, 8), 3), np.float32)      # the grazing fade's share of the SH (kept apart)
    if P.view_tone and K >= 8:
        ft = f * max(1, int(round(W / 512.0)))             # cells of the tone: two spacings on a large picture
        Hc_ = _down_known(Hf, hole | surf, ft)[0]
        Hc_ = _nearest_fill(Hc_, _down_known((hole | surf).astype(float), np.ones_like(hole), ft)[0] > 0.3)
        got = _view_tone(scene, cam, rest, rgb, Hc_, surf, hole, ft, sp)
        if got is not None:
            coef, b_ref = got                               # (8,3,h,w), (8,)
            cf = np.stack([[ndimage.map_coordinates(coef[k, c_], [(ys + 0.5) / ft - 0.5, (xs + 0.5) / ft - 0.5],
                                                    order=1, mode="nearest")
                            for c_ in range(3)] for k in range(8)])     # 8, 3, N
            if fade is None:
                fade = _grazing_fade(scene, cam, rest, rgb, np.where(surf, h_rest, B), surf, hole, sp, pxsp,
                                     P.grazing_band) if P.grazing else 0.0
            if fade > 0.05:
                # fine detail times (1 - fade * sin^2): (d.n)^2 is an SH of degree 2
                from .shtools import sh_basis
                rng_ = np.random.default_rng(0)
                dd = rng_.normal(size=(400, 3))
                dd /= np.linalg.norm(dd, axis=1, keepdims=True)
                qk = np.linalg.lstsq(sh_basis(dd, 2), (dd @ n) ** 2, rcond=None)[0]     # 9
                soft_c = np.stack([ndimage.gaussian_filter(col[..., c_], max(1.0, P.grazing_band * pxsp))
                                   for c_ in range(3)], 2)
                detail = (col - soft_c)[ys, xs]                                        # N, 3
                fd = fade * qk[1:, None, None] * detail.T[None]
                cf = cf + fd
                fade_cf = fd.transpose(2, 0, 1).astype(np.float32)
            f_rest[:, :8, :] = cf.transpose(2, 0, 1)
            base = base - np.einsum("k,kcn->nc", b_ref, cf)   # from the picture's side: the picture
            tone_info = {"viewTone": round(float(np.abs(cf).mean()), 4), "grazingFade": round(fade, 2)}
    added = SplatSet(
        xyz=pos.astype(np.float32),
        f_dc=((base - 0.5) / SH_C0).astype(np.float32),
        f_rest=f_rest,
        opacity=np.full(N, math.log(P.opacity / (1 - P.opacity)), np.float32),
        scale=scale.astype(np.float32),
        rot=quat.astype(np.float32),
        extra={k: np.zeros((N,) + vv.shape[1:], vv.dtype) for k, vv in s.extra.items()},
    )

    # ---- 8. what is still in the way in the hole's column goes too
    core = ndimage.binary_erosion(hole, iterations=1) & ~kept_out
    xi, yi = np.round(xk).astype(int), np.round(yk).astype(int)
    inimg = (xi >= 0) & (xi < W) & (yi >= 0) & (yi < W)
    way = np.zeros(len(keep), bool)
    way[inimg] = core[yi[inimg], xi[inimg]] & (hk[inimg] > Hf[yi[inimg], xi[inimg]] - 1.0 * sp)
    if way_top is not None:
        way[inimg] &= hk[inimg] < Hf[yi[inimg], xi[inimg]] + way_top * sp
    removed = np.unique(np.concatenate([np.where(sel)[0], keep[way & ~selk]]))

    # What this picture could not do well: flanks it sees steeply (their fill
    # is a stretched strip of pixels), and whatever of the selection lies
    # outside its fill altogether (the wall of an edge seen from above).
    # A flank is where the surface as a whole turns away, not a step inside
    # the hole (the edge of a seam): from the side a step's wall is seen
    # square-on, but everything around it squeezed -- no better a picture
    # than this one. So the slope must hold at four spacings as well.
    Hw = ndimage.gaussian_filter(Hf, 4.0 * pxsp)
    wy_, wx_ = np.gradient(Hw)
    steep_img = (np.hypot(bx_, by_) / px > P.steep) & (np.hypot(wx_, wy_) / px > P.steep_wide) & lift
    steep_img = ndimage.binary_opening(steep_img, iterations=max(1, int(0.5 * pxsp)))
    wsx, wsy = wx_[ys, xs] / px, wy_[ys, xs] / px
    n_wide = n[None] - wsx[:, None] * t1[None] - wsy[:, None] * t2[None]
    area_w = np.linalg.norm(n_wide, axis=1)                 # surface per pixel, in pixels
    n_wide /= area_w[:, None]
    done_px = (lift & ~steep_img) | whole
    res_k = np.zeros(len(keep), bool)
    res_k[inimg] = done_px[yi[inimg], xi[inimg]]
    resolved = np.zeros(s.n, bool)
    resolved[keep[selk & res_k]] = True
    passdata = {"resolved": resolved, "steep": steep_img[ys, xs], "normals": n_wide,
                "area": np.minimum(area_w, 4.0) * px * px, "frame": (c, n, t1, t2), "sp": sp, "r": float(r_hole),
                "cam": cam, "lift": lift, "Hf": Hf, "col": col, "steep_img": steep_img,
                "Hsurf": np.where(lift, Hf, np.where(surf & ~hole0, h_rest, np.nan)), "foot": hole0 | lift,
                "base": B,
                "front": np.where(a_all > 0.5, h_all_full, -np.inf), "fade_cf": fade_cf}

    info = {"engine": "image", "mode": "image", "added": int(N), "removed": int(len(removed)),
            "holePainted": int(hole_painted.sum()), "uncovered": int((hole & ~lift).sum()),
            "inTheWay": int((way & ~selk).sum()), "picture": [int(W), int(W)],
            "pxPerSpacing": round(float(pxsp), 2), "spacing": float(sp),
            "holePixels": int(hole.sum()), "lama": bool(used_lama),
            "lamaScale": round(float(lama_scale), 3), "tLama": round(t_l, 2),
            "tMatch": round(t_s, 2), "seconds": round(time.time() - T0, 2),
            "shapePrior": round(prior_share, 2), "tilt": round(tilt, 1), "steepPixels": int(steep_img.sum()),
            **layer_info, **front_info, **far_info, "oldSceneExplains": explained,
            **prior_dbg, **tone_info, **seam_info}
    dbg = {}
    if debug:
        dbg = {"rgb": rgb, "hole": hole, "valid": valid, "guide": guide, "col": col,
               "h_was": h_was, "found_was": found_was,
               "B": B, "relief": relief, "rel": rel, "h_rest": h_rest, "Hf": Hf, "cam": cam,
               "vis": vis, "surf": surf}
    return ImFillResult(removed=removed, added=added, info=info, debug=dbg, passdata=passdata)


def _best_side(s2: SplatSet, flag_o, steep_a, normals, area, n0, pd, tilt_deg, used=(), least=12.0):
    """From which side is most of what the pictures so far left to be seen?

    ``flag_o``: unresolved splats of the selection (over ``s2``), ``steep_a``:
    fill on a flank (the last ``len(steep_a)`` splats of ``s2``) with the
    ``normals`` of the surface at large there and the ``area`` each stands for.
    Sixteen directions around the user's own, at ``tilt_deg`` and at two
    thirds of it (a pit's wall hides behind its rim from further out); the
    one that shows the largest area, occlusion included, is then turned to
    face the flank it sees square-on. Returns (direction towards the eye,
    which of the steep fill faces it, shown area in square spacings), or None
    if no side shows enough to be worth a picture.
    """
    _, n, t1, t2 = pd["frame"]
    sp = pd["sp"]
    xyz = s2.xyz.astype(np.float64)
    na = len(steep_a)
    flag_a = np.zeros(s2.n, bool)
    flag_a[s2.n - na:] = steep_a
    pts = xyz[flag_o | flag_a]
    if len(pts) < 20:
        return None
    c = np.median(pts, 0)
    r = float(np.percentile(np.linalg.norm(pts - c, axis=1), 95)) + 2.0 * sp
    near = np.where(np.linalg.norm(xyz - c, axis=1) < max(8.0 * r, 40.0 * sp))[0]
    sc = _Scene(s2.take(near))
    W = int(np.clip(4.0 * r / sp, 48, 192))
    W += (-W) % 8
    px = 4.0 * r / W

    def score_of(om):
        face = steep_a & (normals @ om > 0.5)
        flag = flag_o.copy()
        flag[s2.n - na:] |= face
        n_, a_, b_ = _frame(om, t1)
        cam = Ortho(c, n_, a_, b_, px, W, W, 20.0 * r + 100.0 * sp)
        return float((sc.shown(cam, flag[near]) > 0.3).sum()) * px * px / (sp * sp), face

    best, best_om, best_face = 0.0, None, None
    for tl in (np.radians(tilt_deg), np.radians(tilt_deg * 2.0 / 3.0)):
        for k in range(8):
            az = np.radians(45.0 * k)
            om = n0 * np.cos(tl) + (t1 * np.cos(az) + t2 * np.sin(az)) * np.sin(tl)
            if any(om @ u > 0.8 for u in used):
                continue
            sc_, face = score_of(om)
            if sc_ > best:
                best, best_om, best_face = sc_, om, face
    if best_om is not None and best_face.sum() >= 20:
        # square onto the flank it found (but not past 80 degrees off the user's view)
        m = (normals[best_face] * area[best_face, None]).sum(0)
        if np.linalg.norm(m) > 1e-9:
            m /= np.linalg.norm(m)
            cs = float(np.clip(m @ n0, -1, 1))
            lim = np.cos(np.radians(80.0))
            if cs < lim:
                side = m - cs * n0
                side /= max(np.linalg.norm(side), 1e-9)
                m = n0 * lim + side * np.sqrt(1 - lim * lim)
            if not any(m @ u > 0.8 for u in used):
                sc_, face = score_of(m)
                if sc_ >= 0.9 * best:
                    best, best_om, best_face = sc_, m, face
    del sc
    if best_om is None or best < least:        # square spacings in the picture
        return None
    return best_om, best_face, best


def _disagrees(pd, xyz):
    """A side picture's points, held against the first picture: (against it, already there).

    Seen along the user's view, a point goes *against* the first picture if
    it stands before the surface that picture shows (by more than two
    spacings) or far behind it (six), where that picture knows the surface --
    where it filled, off its own flanks, and on the surface around -- or if
    it lands outside what was painted (three spacings beyond the hole).
    Where the first picture left the hole open (past an edge it put nothing),
    a side picture may fill what lies below: the wall under the edge. It may
    not carry the first picture's own surface on past the edge as a shelf:
    there the first picture has said that this surface ends.

    A point is *already there* if it lies in the surface the first picture
    knows: on its own fill off the flanks, or on the scan beside the hole.
    That is no disagreement, but a second skin over the first is no use
    either.
    """
    cam, lift, steep_img, H, sp = pd["cam"], pd["foot"], pd["steep_img"], pd["Hsurf"], pd["sp"]
    left_open = pd["foot"] & ~pd["lift"]
    pxsp = sp / cam.px
    x, y, h = cam.pixel(xyz)
    xi, yi = np.round(x).astype(int), np.round(y).astype(int)
    ins = (xi >= 0) & (xi < cam.W) & (yi >= 0) & (yi < cam.H)
    xi, yi = np.clip(xi, 0, cam.W - 1), np.clip(yi, 0, cam.H - 1)
    flank = ndimage.binary_dilation(steep_img, iterations=max(1, int(round(1.5 * pxsp))))
    foot = ndimage.binary_dilation(lift, iterations=max(1, int(round(3.0 * pxsp))))
    Hs = H[yi, xi]
    known = ins & ~flank[yi, xi] & np.isfinite(Hs)
    with np.errstate(invalid="ignore"):
        bad = known & ((h > Hs + 2.0 * sp) | (h < Hs - 6.0 * sp))
    bad |= ins & left_open[yi, xi] & (h > pd["base"][yi, xi] - 2.0 * sp)
    bad |= ~ins | ~foot[yi, xi]
    return bad, known & ~bad


def _seen_by_user(s: SplatSet, removed, added: SplatSet, side_of, sides, pd, fade_cf):
    """The finished fill, looked at once more along the user's own view.

    Two things a side picture cannot know are checked against the first
    picture, which does:

    * Its splats carry a guess of how their colour turns with the view (the
      smooth view dependence of their surroundings, carried through the
      hole). From the user's side that guess may be far off: dark flecks on a
      pale rim. Rendered once with the guess and once without, the smooth
      difference of each to the first picture says how much of the guess to
      keep: all of it where it helps, none where it leads away. It is only
      ever turned down, so no other view can get worse by it. (The loss of
      fine contrast at a slant, ``fade_cf``, is no guess about tone and stays.)
    * Between the two fills a slit may gape. Returned as a mask of the first
      picture's pixels where nothing is left to see. (Not: where the eye now
      reaches further than the first fill lay. On a flank the first picture
      strings curtains across to what lies behind, and a side picture is
      right to take them down.)

    Changes ``added`` in place; returns (open pixels, mean share of the tone kept).
    """
    from .shtools import sh_basis
    cam, lift, col, sp = pd["cam"], pd["lift"], pd["col"], pd["sp"]
    rest = s.take(~removed)
    x, y, _ = cam.pixel(np.concatenate([rest.xyz, added.xyz]).astype(np.float64))
    near = np.where((x > -30) & (x < cam.W + 30) & (y > -30) & (y < cam.H + 30))[0]
    flag = np.concatenate([np.zeros(rest.n, bool), side_of > 0])
    sc = _Scene(rest.concat(added).take(near))
    img, alpha, _ = sc.picture(cam)
    shown = sc.shown(cam, flag[near])
    del sc
    open_px = lift & (alpha < 0.85)
    K = added.f_rest.shape[1]
    idx = np.where(side_of > 0)[0]
    M = lift & (shown > 0.5)
    if K < 3 or not len(idx) or M.sum() < 20:
        return open_px, 1.0
    deg = int(round(math.sqrt(K + 1))) - 1
    kf = fade_cf.shape[1]
    tone = added.f_rest.astype(np.float64)                 # the guess of the tone: all of the SH but the fade
    tone[:, :kf] -= fade_cf
    own = np.zeros((added.n, 3))                           # ... and what it comes to seen from its own picture
    for k, om in enumerate(sides, start=1):
        m = side_of == k
        if m.any():
            own[m] = np.einsum("k,nkc->nc", sh_basis((-om)[None], deg)[0, 1:K + 1], tone[m])
    side = (side_of > 0)[:, None, None]
    flat = SplatSet(xyz=added.xyz, f_dc=(added.f_dc + own / SH_C0).astype(np.float32),
                    f_rest=(added.f_rest - np.where(side, tone, 0.0)).astype(np.float32),
                    opacity=added.opacity, scale=added.scale, rot=added.rot, extra=added.extra)
    sc = _Scene(rest.concat(flat).take(near))
    img0 = sc.picture(cam)[0]
    del sc
    sig = 1.5 * sp / cam.px
    wgt = np.maximum(ndimage.gaussian_filter(M.astype(np.float64), sig), 1e-6)

    def lp(a):
        return np.stack([ndimage.gaussian_filter(np.where(M, a[..., c], 0.0), sig) for c in range(3)], 2) / wgt[..., None]

    base = lp(img0)
    want, got = lp(col) - base, lp(img) - base             # the change the first picture shows, the change guessed
    lam = np.clip((want * got).sum(2) / np.maximum((got * got).sum(2), 1e-6), 0.0, 1.0)
    lam = np.where((np.abs(got).max(2) < 0.01) | (wgt < 0.2), 1.0, lam)
    xa, ya, _ = cam.pixel(added.xyz[idx].astype(np.float64))
    ls = ndimage.map_coordinates(lam, [ya, xa], order=1, mode="nearest")
    added.f_dc[idx] = (added.f_dc[idx] + (1.0 - ls)[:, None] * own[idx] / SH_C0).astype(added.f_dc.dtype)
    added.f_rest[idx] = (added.f_rest[idx] - (1.0 - ls)[:, None, None] * tone[idx]).astype(added.f_rest.dtype)
    return open_px, round(float(ls.mean()), 3)


def fill(s: SplatSet, selection, view, params: Optional[ImFillParams] = None,
         debug: bool = False, volume=None) -> ImFillResult:
    """Remove ``selection`` from ``s`` and close the hole.

    ``view`` is the direction the user looked along when painting (from the
    eye into the scene): the first picture is taken along it.

    One picture sees a flank that turns away only squeezed, and the wall of an
    edge not at all: lifted from there, the fill is a stretched strip of
    pixels. So what the first picture leaves -- fill on slopes over 55
    degrees, selection outside its fill -- gets a further picture from the
    side that shows most of it (`_best_side`), with the first picture's fill
    already in the scene: the side view continues it as it continues the
    scan. At most ``passes - 1`` side pictures are tried.

    The first picture is what the user sees and stays the authority:

    * A side picture's fill is checked against it (`_disagrees`) and thrown
      away whole if a third of it goes against what the first picture shows.
    * Of the first fill only the flank goes, and only where the side picture
      put something in its place or looked at it and left it empty (a ramp
      strung across to the next layer). Off the flank the first fill stays.
    * At the end the fill is looked at once more along the user's view
      (`_seen_by_user`): a side splat's guessed change of tone is turned down
      where the first picture says otherwise, and where a slit has opened the
      first picture's splats come back.
    """
    P = params or ImFillParams()
    first = _fill_one(s, selection, view, P, debug, volume)
    if P.passes <= 1 or first.info.get("mode") != "image" or not first.passdata:
        return first
    T0 = time.time()
    v = np.asarray(view, float)
    n0 = -v / max(np.linalg.norm(v), 1e-12)
    sel = np.zeros(s.n, bool)
    sel[np.asarray(selection, dtype=np.int64)] = True
    opaque = s.alpha > 0.3
    n_op = max(int((sel & opaque).sum()), 1)
    removed = np.zeros(s.n, bool)
    removed[first.removed] = True
    added, pd = first.added, first.passdata
    resolved, steep, normals, area = pd["resolved"].copy(), pd["steep"].copy(), pd["normals"], pd["area"]
    info = dict(first.info)
    info["flank"] = {"unresolved": int((sel & ~resolved & opaque).sum()), "selected": n_op,
                     "steepArea": round(float(area[steep].sum() / pd["sp"] ** 2), 1),
                     "fillArea": round(float(area.sum() / pd["sp"] ** 2), 1)}
    sides, used = [], []
    was_tree = cKDTree(s.xyz[sel & opaque].astype(np.float64)) if (sel & opaque).any() \
        else cKDTree(np.full((1, 3), 1e30))
    side_of = np.zeros(added.n, np.int32)                  # which picture a new splat is from (0: the first)
    first_i = np.arange(added.n)                           # ... and which splat of the first picture it is (-1: none)
    fade_cf = pd["fade_cf"]
    P2 = replace(P, context=min(P.context, P.side_context), max_px=min(P.max_px, P.side_px))
    for _ in range(P.passes - 1):
        unres = sel & ~resolved & opaque
        few_o = unres.sum() < max(40, P.side_left * n_op)
        few_a = area[steep].sum() < max(12.0 * pd["sp"] ** 2, 0.05 * area.sum())
        if few_o and few_a:
            break
        if few_o:
            unres[:] = False
        idx_map = np.where(~removed | unres)[0]            # the scan as it stands, the unresolved back in
        n1 = len(idx_map)
        s2 = s.take(idx_map).concat(added)
        flag_o = np.concatenate([unres[idx_map], np.zeros(added.n, bool)])
        got = _best_side(s2, flag_o, steep, normals, area, n0, pd, P.side_tilt, used,
                         least=max(12.0, min(P.side_area * float(area.sum()) / pd["sp"] ** 2,
                                             P.side_area_max)))
        if got is None:
            break
        om, face, shown_area = got
        used.append(om)
        # Where is the hole, seen from the side? Where what was removed shows,
        # and where the first fill stands in for it. Not all of the first
        # fill does: at a slant it strings ramps across to whatever lies
        # behind the far rim, far from anything that was there. Only fill
        # that lies where something was removed, or inside what was painted,
        # marks the hole; the rest is left out of the side picture.
        marks = face & (was_tree.query(added.xyz.astype(np.float64))[0] < 3.0 * pd["sp"])
        if volume is not None:
            marks |= face & np.asarray(volume.inside(added.xyz.astype(np.float64)), bool)
        ramp = face & ~marks
        if ramp.any():
            s2 = s.take(idx_map).concat(added.take(~ramp))
            pos = np.cumsum(~ramp) - 1                     # index among the first fill left in the scene
            sheet = n1 + pos[marks]
            flag_o = flag_o[:n1 + int((~ramp).sum())]
        else:
            sheet = n1 + np.where(marks)[0]
        sel2 = flag_o.copy()
        sel2[sheet] = True
        if sel2.sum() < 20:
            steep &= ~face
            sides.append({"skipped": "nothing to show"})
            continue
        try:
            nxt = _fill_one(s2, np.where(sel2)[0], -om, P2, debug, None, way_top=4.0, spacing=pd["sp"],
                            sheet=sheet, fade=first.info.get("grazingFade"),
                            check=lambda q: float(_disagrees(pd, q)[0].mean()) <= 0.3)
        except (ValueError, RuntimeError) as ex:
            steep &= ~face
            sides.append({"error": str(ex)})
            continue
        if nxt.info.get("mode") == "rejected":
            steep &= ~face
            sides.append({"skipped": "does not match the first image", "early": True})
            continue
        if nxt.info.get("mode") != "image" or nxt.added.n == 0:
            # nothing to fill from there (what was removed stood before whole surface):
            # the unresolved it saw are dealt with, the steep fill stays as it is
            resolved |= unres
            steep &= ~face
            sides.append({"mode": nxt.info.get("mode"), "why": nxt.info.get("why")})
            continue
        if debug:
            first.debug.setdefault("sides", []).append(nxt.debug)
        # The user's own view is the authority. Where the first picture knows
        # the surface (off its flanks), nothing a side picture adds may stand
        # before it or far behind it, and nothing may land outside what was
        # painted. A side picture that mostly does is thrown away whole.
        bad, twice = _disagrees(pd, nxt.added.xyz.astype(np.float64))
        if bad.mean() > 0.3:
            steep &= ~face
            sides.append({"skipped": "does not match the first image", "share": round(float(bad.mean()), 2)})
            continue
        good = ~(bad | twice)
        if good.sum() < 20:
            # all it would add is there already
            resolved |= unres
            steep &= ~face
            sides.append({"skipped": "nothing new", "share": round(float(twice.mean()), 2)})
            continue
        cam2, lift2, Hf2 = nxt.passdata["cam"], nxt.passdata["lift"], nxt.passdata["Hf"]
        put = np.zeros(lift2.shape, bool)                  # where the side picture really put something
        put[lift2] = good
        rem2 = np.zeros(s2.n, bool)
        rem2[nxt.removed] = True
        # of the first fill's flank only what the side picture put something
        # in place of goes: what it could not see (behind a rim) stays
        xa, ya, ha = cam2.pixel(added.xyz.astype(np.float64))
        xi_, yi_ = np.round(xa).astype(int), np.round(ya).astype(int)
        ins = (xi_ >= 0) & (xi_ < cam2.W) & (yi_ >= 0) & (yi_ < cam2.H)
        covered = np.zeros(added.n, bool)
        covered[ins] = put[yi_[ins], xi_[ins]] & (np.abs(ha[ins] - Hf2[yi_[ins], xi_[ins]]) < 12.0 * pd["sp"])
        if marks.sum() >= 20 and covered[marks].mean() < 0.3:
            steep &= ~face
            sides.append({"skipped": "not on the flank", "share": round(float(covered[marks].mean()), 2)})
            continue
        # ... and what it saw in front of it and put nothing in place of was
        # no surface: a curtain the first picture strung across to the next
        # layer. (Where its own fill was thrown out above, the first stays.)
        shows = np.zeros(added.n, bool)                    # not hidden behind anything, seen from the side
        shows[ins] = ha[ins] > nxt.passdata["front"][yi_[ins], xi_[ins]] - 3.0 * pd["sp"]
        spare = shows & np.where(ins, ~lift2[np.clip(yi_, 0, cam2.H - 1), np.clip(xi_, 0, cam2.W - 1)], False)
        removed[idx_map[rem2[:n1]]] = True
        # (off the flank the first picture's fill is what the user sees: it
        # stays. A ramp goes in any case: it lies where nothing was. Should
        # the user's view open up without it, it comes back below.)
        ka = ~((face & (covered | spare)) | ramp)
        res2 = nxt.passdata["resolved"]
        resolved[idx_map[res2[:n1]]] = True
        new = nxt.added.take(good)
        added = added.take(ka).concat(new)
        steep = np.concatenate([steep[ka] & ~face[ka], np.zeros(new.n, bool)])
        side_of = np.concatenate([side_of[ka], np.full(new.n, len(used), np.int32)])
        first_i = np.concatenate([first_i[ka], np.full(new.n, -1, np.int64)])
        fade_cf = np.concatenate([fade_cf[ka], nxt.passdata["fade_cf"][good]])
        normals = np.concatenate([normals[ka], nxt.passdata["normals"][good]])
        area = np.concatenate([area[ka], nxt.passdata["area"][good]])
        sides.append({"area": round(shown_area, 1), "unresolved": int(unres.sum()), "steepSplats": int(face.sum()),
                      "added": int(new.n), "dropped": int((~ka).sum()), "unfit": round(float(bad.mean()), 3),
                      "dir": [round(float(x_), 4) for x_ in om],
                      "tiltDeg": round(float(np.degrees(np.arccos(np.clip(om @ n0, -1, 1)))), 1),
                      "seconds": nxt.info.get("seconds")})
    if (side_of > 0).any():
        # Seen along the user's own view nothing may be open that the first
        # picture had closed: a side picture's fill lies where *it* saw the
        # surface, and between the two a slit can gape. There the first
        # picture's splats come back.
        open_px, info["sideTone"] = _seen_by_user(s, removed, added, side_of, used, pd, fade_cf)
        gone = np.ones(first.added.n, bool)
        gone[first_i[first_i >= 0]] = False
        if gone.any():
            open_px = ndimage.binary_dilation(open_px, iterations=1)
            back = gone & open_px[pd["lift"]]
            if back.any():
                added = added.concat(first.added.take(back))
                steep = np.concatenate([steep, np.zeros(int(back.sum()), bool)])
                side_of = np.concatenate([side_of, np.zeros(int(back.sum()), np.int32)])
                normals = np.concatenate([normals, pd["normals"][back]])
                area = np.concatenate([area, pd["area"][back]])
                info["reopened"] = int(back.sum())
    if sides:
        info["sideViews"] = sides
    info["passes"] = 1 + sum(1 for v_ in sides if "added" in v_)
    info["added"] = int(added.n)
    info["removed"] = int(removed.sum())
    info["seconds"] = round(float(first.info.get("seconds", 0)) + time.time() - T0, 2)
    return ImFillResult(removed=np.where(removed)[0], added=added, info=info, debug=first.debug,
                        passdata={"resolved": resolved, "steep": steep, "normals": normals, "area": area,
                                  "frame": pd["frame"], "sp": pd["sp"], "r": pd["r"]})
