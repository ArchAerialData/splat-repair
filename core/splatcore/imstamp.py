"""The picture stamp: copy a patch of surface in its own picture, then lift it.

The stamp in `clone.py` copies Gaussians, dab by dab. A Gaussian is an
extended, irregular thing with a height of its own: the copies land above or
below the target's skin, overlapping dabs thin one another out, and what was
under the stroke shows through as grey smudges. Every fix there fights the
same fact, as it did for the fill (`imfill.py`).

This engine copies no Gaussian. It works on pictures, per stroke:

1. **Target picture.** The surface under the stroke is rendered
   orthographically, square onto it: its colour, and where its skin lies.
   A scan's skin is fuzzy, so that is a band per pixel, not a height: from
   where a fifth of the skin's opacity is gathered to where half is
   (`_skin`, `_band`). The brush in the picture is the user's: what their
   disc on the screen covers of the surface (`_brush`).
2. **Source picture.** The same picture of the source: the target's camera
   moved by the stamp's offset (the stroke's sources onto its targets) and
   turned by the rotation between the two surfaces' normals, so the source
   faces its picture as the target faces its own. Pixel (x, y) of one
   picture is then pixel (x, y) of the other.
3. **Base and relief.** Each surface is split into a smooth base (a thin
   plate, a third of a brush long) and the relief on it: the least relief
   its band allows, so a crisp groove or pebble stays and the blotches of a
   fuzzy skin do not (`_settle`). The stamped surface is the *target's* base
   plus the *source's* relief: the target keeps its own shape at large -- a
   toe cap stays a toe cap -- and gets the source's lattice, pebbles and
   seams. What stands on the target under the stroke (a sign, a bump) or is
   missing there (a hole) is left out of its base: the base runs on
   underneath from the surface around.
4. **Seam.** Colour and relief are blended with the target's own by the
   brush's falloff, in the picture. The tone of the source is moved towards
   the target's (``color_match``) by the difference of the two pictures
   around the stroke, carried through it as a membrane.
5. **Lift.** Every pixel under the brush becomes one small flat Gaussian on
   that surface (as `imfill` lifts a fill); the old skin under the stroke
   and whatever stands above the new surface goes. One skin, at a known
   height, at the same resolution everywhere.
6. **From the side.** One picture knows one side. Both surfaces are looked at
   from nineteen directions around the surface's normal (straight down and
   three rings out to 72 degrees); every pixel keeps what it showed from
   each (laid back through its own height), blended and tone-matched like
   the picture itself, as SH. Seen from the side the stamp then shows what
   the source showed from that side: the shaded wall of a groove, the gaps
   between pebbles closing up, leather going soft.

Needs CUDA + gsplat. `Document.clone_stroke` falls back to the copying stamp
when this one cannot do a stroke.
"""
from __future__ import annotations

import math
import time
from dataclasses import dataclass, field
from typing import Optional

import numpy as np
from scipy import ndimage, sparse
from scipy.sparse.linalg import splu
from scipy.spatial import cKDTree

from .io import SplatSet, SH_C0
from .imfill import (Ortho, _Scene, _frame, _lap, _plate, _solve_fill, _gauss_known,
                     _down_known, _block_quantile, _nearest_fill, _view_tone,
                     _grazing_fade, _spacing)
from . import shtools


@dataclass
class ImStampParams:
    feather: float = 0.3        # soft edge, fraction of the radius
    color_match: float = 0.6    # 0 keeps the source's tone, 1 takes the target's
    pxps: float = 3.0           # pixels per splat spacing
    max_px: int = 1024          # longest picture side
    max_splats: int = 60000     # new splats per stroke at most (the picture gets coarser)
    margin: float = 1.3         # picture beyond the stroke, in brush radii
    base_len: float = 0.3       # length of the base plate, in brush radii
    base_min: float = 3.0       # ... and at least this many spacings
    tension: float = 0.3        # membrane share of the base plate
    level: float = 0.25         # what lies further off the continued base is no surface (radii)
    crease: float = 0.2         # the base may fold where the surface turns by more than this per cell (slope)
    unroll: bool = True         # lay the source out on the target by length along each surface, not by pixel
    surface_thr: float = 0.3    # opacity the skin must gather to count as surface
    skin_max: float = 8.0       # the skin is no thicker than this many spacings
    outer: float = 0.2          # the skin's outside: where this share of its opacity is gathered
    lateral: float = 0.5        # the opacity is averaged over this many spacings before it is read
    ripple: float = 0.35        # relief finer than a spacing and lower than this many spacings is no relief
    steep: float = 1.5          # no slope of the relief steeper than this (tan): a wall becomes a ramp
    turn: float = 0.4           # a lifted splat lies in the surface smoothed over this many spacings
    old_skin: float = 1.5       # the old skin goes down to this far under its middle (spacings)
    # Where the skin under the stroke reads as the layer below it (a thin
    # spot): "edge" stamps over it and lets the brush's soft edge run out
    # onto it as it reads, "flat" as if it lay in the base, "off" leaves it.
    thin: str = "edge"
    seam: bool = True           # take the relief's offset at the rim out
    tone: str = "views"         # colour from the side: "views" | "target" | "source" | "none"
    sh_bands: int = 15          # SH coefficients fitted to the side views (3, 8 or 15)
    grazing: bool = True        # tone "target": lose fine contrast at a slant as the surroundings do
    grazing_band: float = 2.0   # "fine": finer than this many spacings
    sharpen: float = 0.7        # the picture's finest detail is raised by this share before lifting
    sharpen_px: float = 1.0     # ... "finest": finer than this many pixels
    opacity: float = 0.97
    splat_sigma: float = 0.62   # tangent sigma of a lifted splat, in pixels


@dataclass
class ImStampResult:
    removed: np.ndarray
    added: SplatSet
    info: dict = field(default_factory=dict)
    debug: dict = field(default_factory=dict)


def _unit(v):
    v = np.asarray(v, float)
    return v / max(float(np.linalg.norm(v)), 1e-12)


def _upx(a, shape, f, order=3):
    """A grid of ``f`` x ``f`` pixel cells back at full size (cell centres kept)."""
    H, W = shape
    yy = (np.arange(H) + 0.5) / f - 0.5
    xx = (np.arange(W) + 0.5) / f - 0.5
    grid = np.meshgrid(yy, xx, indexing="ij")
    if a.ndim == 2:
        return ndimage.map_coordinates(a, grid, order=order, mode="nearest")
    return np.stack([ndimage.map_coordinates(c, grid, order=order, mode="nearest") for c in a])


def _column(xyz, sig_max, c, n, half):
    """Splats that can show in a picture ``half`` wide around ``c`` along ``n``."""
    d = xyz - c
    lat = d - np.outer(d @ n, n)
    return np.abs(lat).max(1) < half * 1.45 + 3.0 * sig_max


# --------------------------------------------------------------- the skin

def _levels(scene: _Scene, cam: Ortho, rel, lev):
    """The opacity the eye has gathered down to each level, (H, W, len(lev)) on the GPU.

    ``rel``: each splat's height over a smooth guess of the surface, ``lev``
    the levels over that guess, falling. Channel k shows only the splats
    above level k (as `imfill._Scene.surface` renders them).
    """
    torch = scene.torch
    lv = torch.tensor(np.asarray(lev), device=scene.dev, dtype=torch.float32)
    feat = (torch.tensor(rel, device=scene.dev, dtype=torch.float32)[:, None] > lv[None]).float()
    V, _ = scene.raster(cam, None, feat, None, False)
    return V


def _blur_levels(scene: _Scene, V, sigma):
    """``V`` (H, W, K) smoothed across the picture by ``sigma`` pixels."""
    if sigma < 0.3:
        return V
    torch = scene.torch
    F = torch.nn.functional
    rk = max(1, int(3 * sigma + 0.5))
    k = torch.exp(-torch.arange(-rk, rk + 1, device=scene.dev, dtype=torch.float32) ** 2 / (2 * sigma * sigma))
    k = k / k.sum()
    C = V.shape[2]
    x = F.pad(V.permute(2, 0, 1)[None], (rk, rk, rk, rk), mode="replicate")
    x = F.conv2d(x, k.view(1, 1, 1, -1).repeat(C, 1, 1, 1), groups=C)
    x = F.conv2d(x, k.view(1, 1, -1, 1).repeat(C, 1, 1, 1), groups=C)
    return x[0].permute(1, 2, 0)


def _band(scene: _Scene, V, lev, step, P, sp, pxsp):
    """Between which heights the skin's surface lies: (upper, lower, found), over the guess.

    From where the opacity sets in, the next ``skin_max`` spacings are the
    skin. A scan's skin is fuzzy: a few layers of splats, each some spacings
    wide. No single reading of it is the surface. Where a fifth of its
    opacity is gathered (``outer``) is its outside, as a grazing eye sees
    it, but a faint splat floating over the leather lifts that by four
    spacings; where half is gathered is its middle, but that jumps between
    the layers from splat to splat. On a crisp skin -- the wall of a groove,
    a pebble -- the two agree to a quarter spacing; on a fuzzy one they lie
    spacings apart, and all the opacity says is that the surface is between
    them. That band is what is returned. What lies deeper than the skin (the
    inside of a shoe through a thin spot) has no say: the shares are of what
    the skin holds, not of everything.

    The opacity is averaged over half a spacing first. Between the splats of
    a skin's outer layer the eye reaches the next one down: read pixel by
    pixel that is a pit a spacing wide and two deep, crisp, and no relief
    (lifted as it is, every such pit shifts its pixel sideways when the
    surface is seen at a slant, and smooth leather turns to grit). Averaged,
    the layer is there with the share of the area it covers.
    """
    torch = scene.torch
    nlev = V.shape[2]
    V = (V - V[..., :1]).clamp(min=0)                      # what floats above the range is not counted
    V = _blur_levels(scene, V, P.lateral * pxsp)
    on = V >= 0.1
    k0 = on.float().argmax(2)
    m = max(1, int(round(P.skin_max / step)))
    total = V.gather(2, (k0 + m).clamp(max=nlev - 1)[..., None])[..., 0]
    lv = torch.tensor(np.asarray(lev), device=scene.dev, dtype=torch.float32)

    def gathered(share):
        """The level at which this share of the skin's opacity is gathered."""
        want = share * total.clamp(min=1e-3)
        kf = (V >= want[..., None]).float().argmax(2).clamp(min=1)
        v1 = V.gather(2, kf[..., None])[..., 0]
        v0 = V.gather(2, (kf - 1)[..., None])[..., 0]
        frac = ((want - v0) / (v1 - v0).clamp(min=1e-6)).clamp(0.0, 1.0)
        return lv[kf - 1] - step * sp * frac

    found = on.any(2) & (total >= P.surface_thr)
    num = lambda t: t.cpu().numpy().astype(np.float64)     # noqa: E731
    up, dn = num(gathered(P.outer)), num(gathered(0.5))
    return np.maximum(up, dn), dn, found.cpu().numpy()


def _skin(scene: _Scene, cam: Ortho, xyz_k, sp, P):
    """Colour and skin of everything in ``scene``.

    A smooth guess of the front from the expected depth, a coarse pass over
    level surfaces a spacing apart that finds where the skin really is, and
    the fine pass riding on that (a quarter spacing apart). Returns a dict:
    ``rgb``, ``alpha``, the band the surface lies in (``up``, ``dn``, see
    `_band`), ``found``, ``mid`` (the band's middle, filled where nothing was
    found) and each splat's pixel and height (``xk``, ``yk``, ``hk``).
    """
    H, W = cam.H, cam.W
    pxsp = sp / cam.px
    f = max(2, int(round(pxsp)))
    rgb, a, h_exp = scene.picture(cam)
    front = _block_quantile(h_exp, a > 0.3, f, 0.8)
    okf = np.isfinite(front)
    if okf.sum() < 4:
        raise ValueError("there is no surface in the image")
    front = _nearest_fill(np.where(okf, front, 0.0), okf)
    front = ndimage.gaussian_filter(ndimage.median_filter(front, 3, mode="nearest"), 1.5, mode="nearest")
    xk, yk, hk = cam.pixel(xyz_k)

    def at_splats(coarse):
        return ndimage.map_coordinates(coarse, [(yk + 0.5) / f - 0.5, (xk + 0.5) / f - 0.5],
                                       order=1, mode="nearest")

    # The coarse pass looks far up: through thin leather the expected depth
    # lies somewhere inside the shoe, tens of spacings under the skin, and
    # levels that start below the skin never see it (the reading was then of
    # the lining, and a whole flank of a toe cap "another surface").
    lev = (40.0 - 2.0 * np.arange(33)) * sp
    u1, d1, f1 = _band(scene, _levels(scene, cam, hk - at_splats(front), lev), lev, 2.0, P, sp, pxsp)
    fr1 = _block_quantile(0.5 * (u1 + d1) + _upx(front, (H, W), f, 1), f1, f, 0.5)
    ok1 = np.isfinite(fr1)
    if ok1.any():
        fr1 = _nearest_fill(np.where(ok1, fr1, 0.0), ok1)
        front = ndimage.gaussian_filter(ndimage.median_filter(fr1, 3, mode="nearest"), 1.0, mode="nearest")
    lev = (6.0 - 0.25 * np.arange(64)) * sp
    up, dn, found = _band(scene, _levels(scene, cam, hk - at_splats(front), lev), lev, 0.25, P, sp, pxsp)
    guess = _upx(front, (H, W), f, 1)
    up, dn = up + guess, dn + guess
    mid = 0.5 * (up + dn)
    mf, den = _gauss_known(mid, found, max(1.0, 0.5 * pxsp))
    mid = np.where(found, mid, np.where(den > 0.05, mf, guess))
    return {"rgb": rgb, "alpha": a, "up": up, "dn": dn, "found": found, "mid": mid,
            "xk": xk, "yk": yk, "hk": hk}


def _slope_limit(h, step, rounds):
    """``h`` with no slope steeper than ``step`` per pixel.

    The mean of the two surfaces that hug ``h`` from above and from below
    without ever climbing faster than twice that (the largest such below,
    the smallest above): a wall becomes a ramp centred on it, a spike a low
    cone, and what was gentle stays as it is. ``rounds``: how many pixels a
    ramp may take.
    """
    up, lo = h.copy(), h.copy()
    d1, d2 = 2.0 * step, 2.0 * step * math.sqrt(2.0)
    for _ in range(int(rounds)):
        pu = np.pad(up, 1, mode="edge")
        pl = np.pad(lo, 1, mode="edge")
        H, W = h.shape
        for dy, dx, d in ((0, 1, d1), (0, -1, d1), (1, 0, d1), (-1, 0, d1),
                          (1, 1, d2), (1, -1, d2), (-1, 1, d2), (-1, -1, d2)):
            np.minimum(up, pu[1 + dy:1 + dy + H, 1 + dx:1 + dx + W] + d, out=up)
            np.maximum(lo, pl[1 + dy:1 + dy + H, 1 + dx:1 + dx + W] - d, out=lo)
    return 0.5 * (up + lo)


def _settle(B, sk, pxsp, sp, P, ok):
    """The relief on the base ``B``: the least the skin's band allows.

    Where the band holds the base, the surface is the base: a fuzzy skin has
    nothing to say against it, and its blotches are no relief. Where the
    band lies off the base (the bottom of a groove, the top of a pebble: a
    crisp skin, a narrow band) the surface is the band's nearer edge.

    Two things a height read off splat centres has that the surface has not.
    Ripples: a splat counts with its centre's height all over, so a skin
    seen at a slant comes out in terraces a splat wide, and the levels add
    their own a quarter spacing high. Lifted, every terrace tilts its row of
    new splats like the slats of a blind, and from the side the patch is
    striped. What differs from the relief smoothed over a spacing by less
    than ``ripple`` spacings goes. Walls: where the height jumps by spacings
    from one pixel to the next, the splats of the two pixels stand apart and
    the eye goes between them, into the object. No slope may be steeper than
    ``steep``: a wall becomes a ramp.

    ``ok``: where the reading is of this surface at all; elsewhere the
    surface is the base.
    """
    known = sk["found"] & ok
    lo = np.where(known, sk["dn"] - B, -np.inf)
    hi = np.where(known, sk["up"] - B, np.inf)
    rel = np.clip(0.0, lo, hi)
    soft = ndimage.gaussian_filter(rel, 1.0 * pxsp, mode="nearest")
    d = rel - soft
    rel = soft + np.sign(d) * np.maximum(np.abs(d) - P.ripple * sp, 0.0)
    px = sp / pxsp
    return _slope_limit(rel, P.steep * px, max(2, int(math.ceil(3.0 * pxsp / P.steep))))


# ------------------------------------------------------- stroke and surface

def _brush(cam: Ortho, H, points, radii, rays, feather):
    """The stroke in the picture: the brush's falloff (0..1), the largest of all dabs.

    The brush is the user's: a disc on their screen, so in the scene a
    cylinder along their line of sight through each dab (``rays``: from the
    eye into the scene, one per dab). A pixel of the picture gets what the
    cylinder gives the point of the surface it shows (``H``: the surface's
    height), wherever the picture is taken from: seen square-on, a stroke
    painted at a slant is as much longer as the user saw it shorter.
    """
    Hh, W = cam.H, cam.W
    w = np.zeros((Hh, W))
    x, y, _ = cam.pixel(points)
    fw = max(float(feather), 1e-3)
    for k in range(len(points)):
        u = np.asarray(rays[k], float)
        rk = float(radii[k])
        reach = int(math.ceil(rk / cam.px / max(abs(float(cam.n @ u)), 0.3))) + 2
        x0, x1 = max(0, int(x[k]) - reach), min(W, int(x[k]) + reach + 2)
        y0, y1 = max(0, int(y[k]) - reach), min(Hh, int(y[k]) + reach + 2)
        if x0 >= x1 or y0 >= y1:
            continue
        yy, xx = np.mgrid[y0:y1, x0:x1]
        d = (((xx + 0.5 - W * 0.5) * cam.px)[..., None] * cam.t1 + ((yy + 0.5 - Hh * 0.5) * cam.px)[..., None] * cam.t2
             + H[y0:y1, x0:x1, None] * cam.n + (cam.c - points[k]))
        d = d - (d @ u)[..., None] * u
        t = np.clip((1.0 - np.linalg.norm(d, axis=2) / max(rk, 1e-9)) / fw, 0.0, 1.0)
        np.maximum(w[y0:y1, x0:x1], 0.5 * (1.0 - np.cos(np.pi * t)), out=w[y0:y1, x0:x1])
    return w


def _same_surface(h, found, seed, px, pxsp):
    """The surface the stroke lies on: what hangs together with it without a jump in height."""
    hs, _ = _gauss_known(h, found, 1.0 * pxsp)
    gy, gx = np.gradient(hs)
    calm = found & (np.hypot(gx, gy) / px < 3.7)            # no step: under 75 degrees
    lab, _ = ndimage.label(calm)
    ids, cnt = np.unique(lab[seed & calm], return_counts=True)
    cnt, ids = cnt[ids > 0], ids[ids > 0]
    if not len(ids):
        return np.zeros_like(found)
    ids = ids[cnt >= 0.1 * cnt.max()]
    return ndimage.binary_closing(np.isin(lab, ids), iterations=max(1, int(pxsp))) & found


def _walk(h, found, rim, px, pxsp):
    """What one can walk onto from the stroke's rim without climbing a step.

    As `_same_surface`, but stricter, because more hangs on it: this is the
    part of what lies under the stroke that the base is fitted to (`_base`).
    The ground counts where, smoothed over a spacing and a half, it rises by
    less than 63 degrees (tan 2), and only where that is broad. The wall of
    a sign is a step all round, but read coarsely (a large stroke's picture
    has little more than a pixel per spacing) the step is a ramp of some 70
    degrees, and a ramp has a gap somewhere: at 75 degrees the whole sign
    was "the same surface" and the stamp sat on it. The two sides of a
    square edge, each seen at 45 degrees, stay well under the bar.
    """
    hs, _ = _gauss_known(h, found, 1.5 * pxsp)
    gy, gx = np.gradient(hs)
    k = max(1, int(round(pxsp)))
    gentle = ndimage.binary_erosion(found & (np.hypot(gx, gy) / px < 2.0), iterations=k)
    lab, _ = ndimage.label(gentle)
    ids, cnt = np.unique(lab[rim & gentle], return_counts=True)
    cnt, ids = cnt[ids > 0], ids[ids > 0]
    if not len(ids):
        return np.zeros_like(found)
    return ndimage.binary_dilation(np.isin(lab, ids[cnt >= 0.1 * cnt.max()]), iterations=k) & found


def _close_gaps(mask, reach):
    """``mask`` with its holes up to about ``reach`` pixels across filled."""
    holes = ndimage.binary_fill_holes(mask) & ~mask
    lab, nl = ndimage.label(holes)
    if nl == 0:
        return mask
    area = np.bincount(lab.ravel(), minlength=nl + 1)
    small = area <= reach * reach
    small[0] = False
    return mask | small[lab]


def _mean_normal(cam: Ortho, h, mask, sigma):
    """The middle of the directions a surface faces over ``mask``: a unit vector.

    Not the normal of a plane through it. A plane through the two sides of
    an edge is fitted robustly, and robustly means: through the larger side,
    the other thrown out. The picture then stood square on a sole's wall and
    saw its underside edge-on: no surface there to stamp on, and the stamp
    ran on past the edge as a shelf. Nor the mean by area, which leans to
    the larger side as well (the wall at 25 degrees, the underside at 63).
    The normals are split in two groups, and the picture faces the middle
    between the two: across a square edge each side at 45 degrees; on a
    surface that turns evenly, its middle. ``sigma``: pixels the height is
    smoothed over before its slope is read, so relief does not count.
    """
    hs = ndimage.gaussian_filter(h, sigma, mode="nearest")
    gy, gx = np.gradient(hs)
    ys, xs = np.where(mask)
    if len(ys) < 12:
        return cam.n
    pick = slice(None, None, max(1, len(ys) // 20000))
    sx, sy = gx[ys[pick], xs[pick]] / cam.px, gy[ys[pick], xs[pick]] / cam.px
    N = cam.n[None] - sx[:, None] * cam.t1[None] - sy[:, None] * cam.t2[None]
    N /= np.linalg.norm(N, axis=1, keepdims=True)
    m = _unit(N.sum(0))
    # two groups: either side of the direction the normals spread most along
    X = N - m
    side = X @ np.linalg.eigh(X.T @ X)[1][:, -1] > 0
    for _ in range(6):
        if side.all() or not side.any():
            return m
        c1, c2 = _unit(N[side].sum(0)), _unit(N[~side].sum(0))
        nxt = N @ c1 > N @ c2
        if np.array_equal(nxt, side):
            break
        side = nxt
    if side.all() or not side.any():
        return m
    return _unit(_unit(N[side].sum(0)) + _unit(N[~side].sum(0)))


def _plane(h, mask):
    """Robust plane through ``h`` over ``mask``: slopes per pixel (sx, sy) and the height at (0, 0)."""
    ys, xs = np.where(mask)
    if len(ys) < 12:
        return 0.0, 0.0, float(np.median(h[mask])) if len(ys) else 0.0
    pick = slice(None, None, max(1, len(ys) // 40000))
    A = np.stack([xs[pick], ys[pick], np.ones(len(ys[pick]))], 1).astype(np.float64)
    z = h[ys[pick], xs[pick]]
    keep = np.ones(len(z), bool)
    co = np.zeros(3)
    for _ in range(3):
        co = np.linalg.lstsq(A[keep], z[keep], rcond=None)[0]
        res = z - A @ co
        mad = 1.4826 * float(np.median(np.abs(res[keep] - np.median(res[keep])))) + 1e-12
        nxt = np.abs(res) < 3.0 * mad
        if nxt.sum() < 12 or np.array_equal(nxt, keep):
            break
        keep = nxt
    return float(co[0]), float(co[1]), float(co[2])


def _steps(h, w):
    """Differences between neighbouring cells of an h x w grid, as a sparse matrix (rows: pairs)."""
    idx = np.arange(h * w).reshape(h, w)
    pairs = np.concatenate([np.stack([idx[:, :-1].ravel(), idx[:, 1:].ravel()], 1),
                            np.stack([idx[:-1, :].ravel(), idx[1:, :].ravel()], 1)])
    n = len(pairs)
    rows = np.repeat(np.arange(n), 2)
    vals = np.tile([-1.0, 1.0], n)
    return sparse.csr_matrix((vals, (rows, pairs.ravel())), shape=(n, h * w)), pairs


def _plate_soft(d, w, lam, soft, L, tension):
    """Thin plate through ``d`` with weights ``w`` (as `imfill._plate`), less stiff where ``soft`` is low.

    Minimises  sum w (z - d)^2 + lam * sum soft * (laplace z)^2 + the
    membrane's share, soft likewise. Where ``soft`` is near zero the plate
    may fold: it follows the data there.
    """
    G, pairs = _steps(*d.shape)
    sf = np.asarray(soft, np.float64).ravel()
    M = (sparse.diags(np.asarray(w, np.float64).ravel() + 1e-9)
         + lam * (L.T @ sparse.diags(sf) @ L)
         + (tension * lam ** 0.5) * (G.T @ sparse.diags(np.minimum(sf[pairs[:, 0]], sf[pairs[:, 1]])) @ G)).tocsc()
    return splu(M).solve((np.asarray(w, np.float64) * d).ravel()).reshape(d.shape)


def _base(h, ok, inside, f, length, tol, tension, crease, cell, walk):
    """The smooth base under a height picture, as a thin plate ``length`` cells long.

    First from the surface around the stroke alone (``ok`` outside
    ``inside``), continued under it with the rim's height, slope and
    curvature. Under the stroke the surface itself is then fitted too, so the
    base follows its own shape there instead of guessing it: what lies within
    ``tol`` of the continuation, and what one can walk onto from the rim
    without climbing a step (``walk``). The second is an edge: the corner of
    a sole under the stroke stands far off anything continued from around
    it, and taken for a thing standing on the surface it was cut off, five
    spacings deep. What stands off and has walls -- a sign on a wall -- or is
    missing -- a hole in the scan -- has no say, and the base runs on
    underneath.

    A plate, not a local average: it lets a surface of constant curvature
    through unchanged, where an average sinks into every dome by the square
    of its width. And a plate that may fold. A stiff one rounds every edge
    off, and the corner comes back as "relief" -- the target's lost under the
    stroke, the source's carried over as a ridge on a flat wall. Where the
    surface, smoothed over a cell and a half so that pebbles and grooves do
    not count, still turns by more than ``crease`` from one cell to the next,
    the plate is soft and takes the turn. ``f``: pixels per cell, ``cell``: a
    cell's size in the heights' own unit. Returns the base at full size.
    """
    zc, share = _down_known(h, ok, f)
    kc = share > 0.5
    zc = np.where(kc, zc, 0.0)
    ins = _down_known(inside.astype(float), np.ones_like(inside), f)[0] > 0
    wk = (_down_known(walk.astype(float), np.ones_like(walk), f)[0] > 0.5) & kc
    L = _lap(*zc.shape)
    lam = float(length) ** 4
    soft = np.ones(zc.shape)
    if crease > 0 and kc.sum() >= 12:
        zs = _nearest_fill(_gauss_known(zc, kc, 1.5)[0], kc)
        turn = np.abs(ndimage.laplace(zs, mode="nearest")) / cell      # change of slope per cell
        soft = 1.0 / (1.0 + (turn / crease) ** 4)
        soft = ndimage.gaussian_filter(ndimage.minimum_filter(soft, 3, mode="nearest"), 0.7, mode="nearest")
        # ... but only where it is fitted to the surface. Under a sign it is
        # fitted to nothing, and a plate that is soft all round the sign's
        # walls has nothing to hold it there either: it drifted up to the
        # sign's height, and the stamp with it.
        soft = np.where(~ins | wk, soft, 1.0)
    around = kc & ~ins
    if around.sum() < 12:
        around = kc
    B = _plate_soft(zc, around.astype(float), lam, soft, L, tension)
    for _ in range(2):
        level = kc & ((np.abs(zc - B) < tol) | wk)
        B = _plate_soft(zc, (kc & (~ins | level)).astype(float), lam, soft, L, tension)
    return _upx(B, h.shape, f)


def _unroll(B, px, f, anchor):
    """Where every pixel of a picture lies on its base surface when that is rolled out flat.

    Returns (du, dv): what to add to a pixel's own (x, y), in pixels; zero at
    ``anchor`` (x, y). A picture squeezes whatever turns away from it: a
    pattern taken from the flank of a heel is shorter in the picture than on
    the heel, and laid on a flat wall pixel by pixel it stays that short;
    taken from a flat wall and laid on the heel it is stretched. Lengths
    along the surface are what a stamp has to keep. The map is the one whose
    steps are, as nearly as a flat map can have them (least squares), as long
    as the surface's own: its Jacobian the square root of the surface's
    metric, I + s s^T for slopes s.
    """
    Bc = _down_known(B, np.ones(B.shape, bool), f)[0]
    gy, gx = np.gradient(Bc, f * px)
    s2 = gx * gx + gy * gy
    a = (np.sqrt(1.0 + s2) - 1.0) / np.maximum(s2, 1e-9)
    h, w = Bc.shape
    lu = splu((_lap(h, w) - 1e-6 * sparse.identity(h * w)).tocsc())
    yy, xx = np.mgrid[0:h, 0:w].astype(np.float64)
    out = []
    for Fx, Fy in ((a * gx * gx, a * gx * gy), (a * gx * gy, a * gy * gy)):
        mx, my = float(Fx.mean()), float(Fy.mean())            # a slope all over: a plain stretch
        div = np.gradient(Fx - mx, axis=1) + np.gradient(Fy - my, axis=0)
        d = lu.solve(div.ravel()).reshape(h, w) + mx * xx + my * yy
        d = _upx(d, B.shape, f) * f
        out.append(d - ndimage.map_coordinates(d, [[anchor[1]], [anchor[0]]], order=1, mode="nearest")[0])
    return out[0], out[1]


def _carry(vals, ok, ring_yx, inside, f, sigma, at, clip=None):
    """Differences measured around the stroke, carried through it as a membrane.

    ``vals`` (C, n): C quantities at the n ring pixels ``ring_yx``; ``ok``
    (C, n) says where each was really measured (None: everywhere).
    ``inside``: the stroke. Per cell of ``f`` pixels, smoothed along the ring
    over ``sigma`` cells first: the two sides have different texture there,
    and cell by cell their difference is that texture, not their tone.
    Returns each quantity at the pixels ``at`` = (ys, xs): (C, N).
    """
    H, W = inside.shape
    h, w = H // f, W // f
    ry, rx = ring_yx
    m = (ry < h * f) & (rx < w * f)
    cell = (ry[m] // f) * w + rx[m] // f
    C = vals.shape[0]
    ys, xs = at
    out = np.zeros((C, len(ys)))
    if not m.any():
        return out
    all_cnt = np.bincount(cell, minlength=h * w).reshape(h, w)
    rc0 = all_cnt > 0.3 * f * f
    ic = (_down_known(inside.astype(float), np.ones_like(inside), f)[0] > 0) & ~rc0
    grid = np.zeros((C, h, w))
    for k in range(C):
        wk = np.ones(int(m.sum())) if ok is None else ok[k][m].astype(np.float64)
        cnt = np.bincount(cell, weights=wk, minlength=h * w).reshape(h, w)
        rc = cnt > 0.3 * f * f
        if rc.sum() < 4:
            continue
        dc = np.bincount(cell, weights=vals[k][m] * wk, minlength=h * w).reshape(h, w) / np.maximum(cnt, 1e-9)
        med = float(np.median(dc[rc]))
        mad = 1.4826 * float(np.median(np.abs(dc[rc] - med)))
        dc = np.clip(dc, med - 3.0 * max(mad, 1e-3), med + 3.0 * max(mad, 1e-3))
        if clip is not None:
            dc = np.clip(dc, -clip, clip)
        grid[k] = _nearest_fill(_gauss_known(dc, rc, sigma)[0], rc)
    if ic.any():
        grid = _solve_fill(grid, ic, order=1)
    co = [(ys + 0.5) / f - 0.5, (xs + 0.5) / f - 0.5]
    for k in range(C):
        out[k] = ndimage.map_coordinates(grid[k], co, order=1, mode="nearest")
    return out


def _lift(cam: Ortho, Hf, lift, sp, P):
    """One flat Gaussian per pixel of ``lift`` on the height ``Hf``.

    Returns positions, quaternions (w, x, y, z), log scales and the pixels.
    As `imfill._fill_one` lifts a fill (step 7), with one difference. There
    a splat is turned to the surface smoothed over 0.6 spacings and
    stretched by its slope over 1.5: right for a fill, whose relief is
    voted and soft. A stamp carries the source's relief as it is, walls of
    grooves included, and on a wall splats that lie in the smooth surface
    stand like the steps of a stair: from the side the eye goes between
    them, into the object. Here every splat lies in the surface at its own
    place (smoothed over ``turn`` spacings) and is as long up the slope as
    its pixel is there. The relief has no ripples left to tilt it and no
    slope over ``steep`` to make a needle of it (`_settle`).
    """
    from scipy.spatial.transform import Rotation
    px = cam.px
    pxsp = sp / px
    n, t1, t2 = cam.n, cam.t1, cam.t2
    Hs = ndimage.gaussian_filter(Hf, max(1.0, P.turn * pxsp))
    gy, gx = np.gradient(Hs)
    ys, xs = np.where(lift)
    pos = cam.world(xs, ys, Hf[ys, xs])
    sx, sy = gx[ys, xs] / px, gy[ys, xs] / px
    nl = n[None] - sx[:, None] * t1[None] - sy[:, None] * t2[None]
    nl /= np.linalg.norm(nl, axis=1, keepdims=True)
    s2 = sx * sx + sy * sy
    slope = np.sqrt(1.0 + np.minimum(s2, 6.25))
    up_dir = sx[:, None] * t1[None] + sy[:, None] * t2[None]
    up_dir = np.where((s2 > 1e-4)[:, None], up_dir, t1[None])
    e1 = up_dir - (np.einsum("ij,ij->i", up_dir, nl))[:, None] * nl
    e1 /= np.maximum(np.linalg.norm(e1, axis=1, keepdims=True), 1e-9)
    e2 = np.cross(nl, e1)
    q = Rotation.from_matrix(np.stack([e1, e2, nl], axis=2)).as_quat()      # x, y, z, w
    quat = np.concatenate([q[:, 3:4], q[:, :3]], 1)
    st = P.splat_sigma * px
    scale = np.log(np.stack([st * slope, np.full_like(slope, st), np.full_like(slope, 0.3 * px)], 1))
    return pos, quat, scale, ys, xs


# ------------------------------------------------------- colour from the side

# The side views: (tilt off the surface's normal, azimuth), degrees. Straight
# down on the surface and three rings of six, as `imfill._view_tone` looks.
SIDES = ((0.0, 0.0),) + tuple((tilt, 60.0 * a + (30.0 if 40 < tilt < 60 else 0.0))
                              for tilt in (30.0, 52.0, 72.0) for a in range(6))


def _side_dirs(normal, hint):
    """Towards the eye of every side view, around the surface's ``normal``: (D, 3).

    Around the normal, not around the picture's own direction: the picture
    is taken as the user looked, which may be well off the normal, and views
    tilted from there leave one side of the surface unseen. From that side
    the fitted colours are a guess, and looked like it: dark streaks.
    """
    nn, a1, a2 = _frame(normal, hint)
    out = []
    for tilt, az in SIDES:
        tl, a = np.radians(tilt), np.radians(az)
        out.append(nn * np.cos(tl) + (a1 * np.cos(a) + a2 * np.sin(a)) * np.sin(tl))
    return np.stack(out)


def _seen(scene: _Scene, cam: Ortho, pts, oms, tol):
    """What each point of a surface shows from the side views: colours (D, N, 3), seen (D, N).

    ``pts``: world points on the surface as the picture ``cam`` reads it;
    ``oms``: towards the eye of each view. Each side picture is taken at the
    picture's own resolution and read at the points' places in it. A point
    counts as seen where the picture is solid and what it shows does not
    stand more than ``tol`` before the point: behind a pebble it shows the
    pebble, which is what the eye sees there too, but not another part of
    the object standing in the way. What it shows may lie behind the point:
    the depth is the expected one, and through thin leather it is somewhere
    inside the shoe (held to both sides, most of a toe cap counted as unseen).
    """
    torch = scene.torch
    F = torch.nn.functional
    P_ = torch.tensor(np.asarray(pts), device=scene.dev, dtype=torch.float32)
    c = torch.tensor(cam.c, device=scene.dev, dtype=torch.float32)
    tt = lambda a_: torch.tensor(a_, device=scene.dev, dtype=torch.float32)   # noqa: E731
    Wd = int(cam.W * 1.25) + 8
    Wd += (-Wd) % 8
    cols, oks = [], []
    for om in oms:
        n_, t1_, t2_ = _frame(om, cam.t1)
        cd = Ortho(cam.c, n_, t1_, t2_, cam.px, Wd, Wd, cam.dist)
        out, al = scene.raster(cd, None, scene.t["colors"], scene.deg, True)
        d = P_ - c
        x = d @ tt(t1_) / cam.px + Wd * 0.5 - 0.5
        y = d @ tt(t2_) / cam.px + Wd * 0.5 - 0.5
        hh = d @ tt(n_)
        feat = torch.cat([out[..., :3].clamp(0, 1), cam.dist - out[..., 3:4], al[..., None]], 2)
        grid = torch.stack([(x + 0.5) / Wd * 2 - 1, (y + 0.5) / Wd * 2 - 1], 1)[None, :, None, :]
        smp = F.grid_sample(feat.permute(2, 0, 1)[None], grid, mode="bilinear", padding_mode="border",
                            align_corners=False)[0, :, :, 0].T                   # N, 5
        ok = (x >= 1) & (x < Wd - 2) & (y >= 1) & (y < Wd - 2) & (smp[:, 4] > 0.7) & (smp[:, 3] - hh < tol)
        cols.append(smp[:, :3].cpu().numpy().astype(np.float64))
        oks.append(ok.cpu().numpy())
    return np.stack(cols), np.stack(oks)


def _sh_fit(front, oms, K):
    """Fit of SH bands 1.. (``K`` coefficients) to colour changes over the side views.

    Anchored at the picture's own direction (``front``: towards its eye),
    where the change is zero by definition: the model is
    sum_k (Y_k(d) - Y_k(front)) c_k. Returns (pinv (K, D), Y(front) (K,)).
    """
    deg = {3: 1, 8: 2, 15: 3}[K]
    Y = shtools.sh_basis(-np.concatenate([np.asarray(front, float)[None], oms]), deg)[:, 1:]
    A = Y[1:] - Y[:1]
    return np.linalg.solve(A.T @ A + 0.01 * np.eye(K), A.T), Y[0]


def _sh_near(sset: SplatSet, pts, sp, k=6):
    """The SH bands of the splats nearest each point, averaged by opacity and distance."""
    op = np.where(sset.alpha > 0.2)[0]
    if len(op) < k:
        return np.zeros((len(pts), sset.f_rest.shape[1], 3))
    d, j = cKDTree(sset.xyz[op].astype(np.float64)).query(pts, k=k)
    wgt = sset.alpha[op][j] * np.exp(-0.5 * (d / max(sp, 1e-9)) ** 2) + 1e-9
    fr = sset.f_rest[op].astype(np.float64)
    out = np.zeros((len(pts), fr.shape[1], 3))
    for i in range(k):
        out += fr[j[:, i]] * wgt[:, i, None, None]
    return out / wgt.sum(1)[:, None, None]


# ----------------------------------------------------------------- the stamp

def stamp(s: SplatSet, sources, targets, radii, view, params: Optional[ImStampParams] = None,
          eye=None, picked=None, spacing=None, debug: bool = False) -> ImStampResult:
    """Stamp one stroke: the surface around ``sources`` onto the surface around ``targets``.

    ``sources`` / ``targets``: one point per dab, ``radii`` the brush's world
    radius at each. ``view``: the direction the user looked along (from the
    eye into the scene); ``eye``: the camera's position, with which the
    pictures are taken along the ray to the stroke instead (off the middle of
    the screen the two differ). ``picked``: which dabs' sources the app
    found on the surface under its crosshair; the others only follow.
    ``spacing``: the scan's splat spacing if the caller knows it (with an
    earlier stamp's dense splats under the stroke it cannot be measured).

    Returns what to remove from ``s`` and what to add. Raises ValueError when
    there is no surface to work on (the caller then uses the copying stamp).
    """
    P = params or ImStampParams()
    T0 = time.time()
    if view is None:
        raise ValueError("no image without a view direction")
    xyz = s.xyz.astype(np.float64)
    alpha = s.alpha
    Tg = np.atleast_2d(np.asarray(targets, float))
    Sr = np.atleast_2d(np.asarray(sources, float))
    K = len(Tg)
    if K == 0 or len(Sr) != K:
        raise ValueError("the stroke is empty")
    rad = np.asarray(radii, float).reshape(-1)
    rad = np.concatenate([rad, np.full(max(0, K - len(rad)), rad[-1])])[:K]
    pk = np.ones(K, bool)
    if picked is not None and len(picked):
        pk = np.array([bool(picked[k]) if k < len(picked) else False for k in range(K)])
        if not pk.any():
            pk[:] = True
    r = float(np.median(rad))
    r_max = float(rad.max())
    v = _unit(view)

    def toward(p):
        """Towards the eye from ``p``: along the ray if the camera is known."""
        if eye is not None:
            d = _unit(np.asarray(eye, float) - p)
            if d @ -v > 0.5:
                return d
        return -v

    extent = float(np.linalg.norm(xyz.max(0) - xyz.min(0)))
    sig_max = np.exp(s.scale.astype(np.float64)).max(1)
    reach0 = float(np.linalg.norm(Tg - Tg.mean(0), axis=1).max())
    sp = spacing or _spacing(xyz, alpha, Tg.mean(0), reach0 + 2.5 * r_max) or (r / 15.0)

    def lie(points, what):
        """How a stroke's place lies, from a small picture as the user saw it:
        (the surface's normal, the stroke's points on its skin)."""
        n_0, u1, u2 = _frame(toward(points.mean(0)))
        la = np.stack([(points - points.mean(0)) @ u1, (points - points.mean(0)) @ u2], 1)
        mi = 0.5 * (la.max(0) + la.min(0))
        hf = 0.5 * float(np.ptp(la, axis=0).max()) + r_max + max(P.margin * r_max, 10.0 * sp)
        px0 = max(sp, 2.0 * hf / 512.0)
        W0 = int(math.ceil(2 * hf / px0))
        W0 += (-W0) % 8
        cam0 = Ortho(points.mean(0) + mi[0] * u1 + mi[1] * u2, n_0, u1, u2, px0, W0, W0, 2.0 * extent)
        keep_0 = np.where(_column(xyz, sig_max, cam0.c, n_0, hf))[0]
        if len(keep_0) < 30:
            raise ValueError(what)
        s0 = _skin(_Scene(s.take(keep_0)), cam0, xyz[keep_0], sp, P)
        soft = ndimage.gaussian_filter(s0["mid"], 1.0)
        foot0 = _brush(cam0, soft, points, rad, np.repeat(-n_0[None], len(points), 0), P.feather) > 0
        on_0 = _same_surface(s0["mid"], s0["found"] & (s0["alpha"] > 0.5), foot0, px0, sp / px0)
        wide0 = ndimage.binary_dilation(foot0, iterations=max(2, int(0.5 * r / px0)))
        if (on_0 & wide0).sum() < 12:
            raise ValueError(what)
        x_, y_, _ = cam0.pixel(points)
        n1 = _mean_normal(cam0, s0["mid"], on_0 & foot0 if (on_0 & foot0).sum() >= 12 else on_0 & wide0,
                          max(1.0, 0.25 * r / px0))
        pts = cam0.world(x_, y_, ndimage.map_coordinates(soft, [y_, x_], order=1, mode="nearest"))
        if float(n1 @ n_0) < 0.9:
            # Seen at more than 25 degrees the place is squeezed in this
            # picture, its skin read through more of itself, and the plane
            # through it leans. Once more, square onto it: the lie from
            # there, and the stroke's points along the user's rays onto the
            # skin as that picture reads it.
            m1, w1, w2 = _frame(n1, u1)
            cam1 = Ortho(pts.mean(0), m1, w1, w2, px0, W0, W0, 2.0 * extent)
            keep_1 = np.where(_column(xyz, sig_max, cam1.c, m1, hf))[0]
            if len(keep_1) >= 30:
                s1 = _skin(_Scene(s.take(keep_1)), cam1, xyz[keep_1], sp, P)
                soft1 = ndimage.gaussian_filter(s1["mid"], 1.0)
                foot1 = _brush(cam1, soft1, pts, rad, np.repeat(-n_0[None], len(points), 0), P.feather) > 0
                on_1 = _same_surface(s1["mid"], s1["found"] & (s1["alpha"] > 0.5), foot1, px0, sp / px0)
                wide1 = ndimage.binary_dilation(foot1, iterations=max(2, int(0.5 * r / px0)))
                if (on_1 & wide1).sum() >= 12:
                    n1 = _mean_normal(cam1, s1["mid"], on_1 & foot1 if (on_1 & foot1).sum() >= 12 else on_1 & wide1,
                                      max(1.0, 0.25 * r / px0))
                    for _ in range(2):
                        x1, y1, h1 = cam1.pixel(pts)
                        hs = ndimage.map_coordinates(soft1, [y1, x1], order=1, mode="nearest")
                        pts = pts + ((hs - h1) / max(abs(float(n_0 @ m1)), 0.2))[:, None] * n_0[None]
        return n1, pts

    # ---- 1. how both places lie, as the user saw them
    n_t, Tq = lie(Tg, "there is no surface at the target")
    n_s, Sq = lie(Sr, "there is no surface at the source")
    a_t, a_s = Tq[pk].mean(0), Sq[pk].mean(0)
    rays = np.array([-toward(q) for q in Tq])               # the user's line of sight through every dab
    ray = -toward(a_t)

    # ---- 2. the target's picture, square onto it
    # Not along the line of sight: the user rarely looks square onto what
    # they paint, and a picture from the side sees the far half of a toe cap
    # squeezed; lifted from there it had holes from the other side. The brush
    # stays the user's (`_brush`). Only a surface seen nearly edge-on keeps
    # the line of sight: square onto it, what was painted might not show.
    facing = float(n_t @ -ray)
    n, t1, t2 = _frame(n_t if facing > 0.26 else -ray)
    facing = max(float(n @ -ray), 0.3)
    lat = np.stack([(Tq - a_t) @ t1, (Tq - a_t) @ t2], 1)
    mid = 0.5 * (lat.max(0) + lat.min(0))
    c = a_t + mid[0] * t1 + mid[1] * t2
    span = 0.5 * float(np.ptp(lat, axis=0).max())
    half = span + r_max / facing + max(P.margin * r_max, 10.0 * sp)
    px = sp / P.pxps
    # the stroke's area, counted on a coarse grid: a scribble covers far less
    # than its box and far more than a line of its length
    g = 2.0 * (span + r_max) / 96.0
    gy_, gx_ = np.mgrid[0:96, 0:96]
    hit = np.zeros((96, 96), bool)
    for (qx, qy), rq in zip((lat - mid) / g + 48.0, rad / g):
        hit |= (gx_ + 0.5 - qx) ** 2 + (gy_ + 0.5 - qy) ** 2 <= rq * rq
    area = float(hit.sum()) * g * g / facing
    px = max(px, 2.0 * half / P.max_px, math.sqrt(1.1 * area / P.max_splats))
    W = int(math.ceil(2 * half / px))
    W += (-W) % 8
    pxsp = sp / px
    f = max(2, int(round(pxsp)))
    cam = Ortho(c, n, t1, t2, px, W, W, 2.0 * extent)
    keep_t = np.where(_column(xyz, sig_max, c, n, half))[0]
    if len(keep_t) < 30:
        raise ValueError("there is no surface at the target")
    scene_t = _Scene(s.take(keep_t))
    tg = _skin(scene_t, cam, xyz[keep_t], sp, P)
    wb = _brush(cam, ndimage.gaussian_filter(tg["mid"], max(1.0, 1.0 * pxsp)), Tq, rad, rays, P.feather)
    foot = wb > 0
    if foot.sum() < 9:
        raise ValueError("the stroke is not in the image")
    solid_t = (tg["alpha"] > 0.7) & (ndimage.gaussian_filter(tg["found"].astype(float), 0.5 * pxsp) > 0.3)
    # (the layer the brush's middle lies on: seeded by the whole brush, a
    # stroke over a rim also took the lining it grazes for its surface)
    heart = wb > 0.5 if (wb > 0.5).sum() >= 9 else foot
    on_t = _same_surface(tg["mid"], tg["found"] & (tg["alpha"] > 0.5), heart, px, pxsp)
    if (on_t & foot).sum() < 0.1 * foot.sum():
        raise ValueError("there is no surface at the target")
    xT, yT, _ = cam.pixel(Tq)

    # ---- 3. the source's picture: the target's own camera, moved and turned onto it
    R = shtools.rotation_between(n_s, n_t)                  # source -> target
    cam_s = Ortho(a_s + R.T @ (c - a_t), R.T @ n, R.T @ t1, R.T @ t2, px, W, W, 2.0 * extent)
    # not what stands far in front of the source's surface: from the turned
    # camera something else of the object may be in the way
    keep_s = np.where(_column(xyz, sig_max, cam_s.c, cam_s.n, half)
                      & ((xyz - a_s) @ n_s < 2.0 * r_max + 0.5 * half))[0]
    if len(keep_s) < 30:
        raise ValueError("there is no surface at the source")
    src_set = s.take(keep_s)
    scene_s = _Scene(src_set)
    sr = _skin(scene_s, cam_s, xyz[keep_s], sp, P)
    on_s = _same_surface(sr["mid"], sr["found"] & (sr["alpha"] > 0.5), heart, px, pxsp)
    if (on_s & foot).sum() < 0.1 * foot.sum():
        raise ValueError("the brush covers too little surface at the source")

    # ---- 4. base and relief of both
    length = max(P.base_len * r, P.base_min * sp)           # world
    fb = max(f, int(round(length / 2.5 / px)))              # pixels per cell of the plate
    tol = max(3.0 * sp, P.level * r)
    rim = ndimage.binary_dilation(foot, iterations=max(2, int(0.5 * r / px))) & ~foot
    walk_t = _walk(tg["mid"], tg["found"] & (tg["alpha"] > 0.5), rim, px, pxsp)
    walk_s = _walk(sr["mid"], sr["found"] & (sr["alpha"] > 0.5), rim, px, pxsp)
    B_t = _base(tg["mid"], on_t, foot, fb, length / (fb * px), tol, P.tension, P.crease, fb * px, walk_t)
    B_s = _base(sr["mid"], on_s, foot, fb, length / (fb * px), tol, P.tension, P.crease, fb * px, walk_s)
    # Where the skin is thin the eye reaches the layer below it, and the
    # reading there is that layer: a patch that does not hang together with
    # the surface, a few spacings under it. To the eye it is the same surface
    # (its colour runs on), so it is stamped from and stamped over like the
    # rest. Taken from, it brings no relief (the pit is no pit); stamped
    # over, the brush's soft edge still runs out onto it as it reads, or the
    # stamp would end in a lip above it. Only what lies far under the base
    # is something else: the inside of a shoe past its collar's edge.
    deep = max(0.6 * r, 10.0 * sp)
    near_s = ndimage.binary_closing(on_s, iterations=max(1, int(pxsp)))
    near_t = ndimage.binary_closing(on_t, iterations=max(1, int(pxsp)))
    def inside(sk, B, near):
        """Thin spots of a skin: read under the base, but not far, and with
        the surface around them. Past an edge the reading falls away too,
        and next to it lies what is far under the base or nothing at all:
        that is no thin spot, and stamped over it would be a shelf strung
        out from the edge."""
        if P.thin == "off":
            return np.zeros_like(near)
        thin = (sk["alpha"] > 0.7) & (sk["mid"] > B - deep) & ~near
        gone = ((sk["alpha"] < 0.3) | (sk["mid"] <= B - deep)) & ~near
        lab, nl = ndimage.label(thin)
        if nl == 0:
            return thin
        bad = np.unique(lab[ndimage.binary_dilation(gone, iterations=2) & thin])
        return thin & ~np.isin(lab, bad[bad > 0])

    thin_s = inside(sr, B_s, near_s)
    thin_t = inside(tg, B_t, near_t)
    solid_s = (sr["alpha"] > 0.7) & (near_s | thin_s)
    rel_s = _settle(B_s, sr, pxsp, sp, P, near_s)
    rel_t = _settle(B_t, tg, pxsp, sp, P, near_t if P.thin == "flat" else np.ones_like(near_t))
    h_t = B_t + rel_t                                       # the target's surface as it is
    h_s = B_s + rel_s
    col_s = sr["rgb"].astype(np.float64)
    col_t = tg["rgb"].astype(np.float64)
    # Which pixel of the source belongs to which of the target: the one as
    # far from the stroke's anchor along its surface as the target's pixel
    # is along its own (`_unroll`). On two flat places that is the same
    # pixel. Everything of the source is from here on read through this.
    qy, qx = np.mgrid[0:W, 0:W].astype(np.float64)
    if P.unroll:
        xa, ya, _ = cam.pixel(a_t[None])
        du_t, dv_t = _unroll(B_t, px, fb, (float(xa[0]), float(ya[0])))
        du_s, dv_s = _unroll(B_s, px, fb, (float(xa[0]), float(ya[0])))
        ux, uy = qx + du_t, qy + dv_t
        for _ in range(4):
            qx = ux - ndimage.map_coordinates(du_s, [qy, qx], order=1, mode="nearest")
            qy = uy - ndimage.map_coordinates(dv_s, [qy, qx], order=1, mode="nearest")
        from_s = lambda a_, order=1: ndimage.map_coordinates(a_, [qy, qx], order=order, mode="nearest")   # noqa: E731
        # a height over the base, measured along the picture, is less across
        # the surface where that leans: the relief keeps its thickness
        lean = lambda B_: np.sqrt(1.0 + (np.gradient(B_, axis=0) / px) ** 2 + (np.gradient(B_, axis=1) / px) ** 2)   # noqa: E731
        rel_s = from_s(rel_s / lean(B_s)) * lean(B_t)
        col_s = np.stack([from_s(col_s[..., c_]) for c_ in range(3)], 2)
        solid_s = from_s(solid_s.astype(np.float64)) > 0.5
        h_sq = from_s(h_s)                                  # the source's height where each pixel is taken
    else:
        h_sq = h_s

    # ---- 5. where the stamp lands, and how much of it
    # On the surface the stroke lies on (holes in it included), and only where
    # the source has surface to give; towards either border the brush fades
    # (over two spacings: over its whole feather, every gap in a thin scan
    # would eat a hole a feather wide into the stamp).
    paint = ndimage.binary_fill_holes(near_t) | ((tg["alpha"] > 0.7) & thin_t)
    # (a gap of a few spacings in what the source or the target reads as
    # surface is the reading's, not the surface's: the brush goes over it)
    can = _close_gaps(solid_s & paint, 4.0 * pxsp)
    edge = np.clip(ndimage.distance_transform_edt(can) / max(min(P.feather * r / px, 2.0 * pxsp), 1.0), 0.0, 1.0)
    w = np.minimum(wb, 0.5 * (1.0 - np.cos(np.pi * edge)))
    foot = w > 0
    if foot.sum() < 9:
        raise ValueError("there is nothing to stamp under the stroke")
    fy, fx = np.where(foot)

    # ---- 6. both surfaces from the side, where the stroke and its rim are
    near = max(2, int(0.5 * r / px))
    ring = ndimage.binary_dilation(foot, iterations=near) & ~foot & solid_s & solid_t & on_t
    zone = ndimage.binary_dilation(foot, iterations=max(near, int(round(2 * pxsp)) + 2))
    zy, zx = np.where(zone)
    zi = np.full(foot.shape, -1, np.int64)
    zi[zy, zx] = np.arange(len(zy))
    Kc = s.f_rest.shape[1]
    Kf = max([k_ for k_ in (3, 8, 15) if k_ <= min(Kc, int(P.sh_bands))], default=0)
    sides = P.tone == "views" and Kf > 0
    dS = dT = okS = okT = None
    if sides:
        tol_v = max(4.0 * sp, 0.3 * r)
        oms = _side_dirs(n_t, t1)
        cS, okS = _seen(scene_s, cam_s, cam_s.world(qx[zy, zx], qy[zy, zx], h_sq[zy, zx]), oms @ R, tol_v)
        cT, okT = _seen(scene_t, cam, cam.world(zx, zy, h_t[zy, zx]), oms, tol_v)
        dS = np.where(okS[..., None], cS - col_s[zy, zx][None], 0.0)       # D, Z, 3: change against the picture
        dT = np.where(okT[..., None], cT - col_t[zy, zx][None], 0.0)
        del cS, cT

    # ---- 7. seam: tone and relief of the source against the target's, read
    # around the stroke and carried through it
    ry, rx = np.where(ring)
    sig_c = max(1.0, 0.35 * r / (f * px))
    tone = np.zeros((len(fy), 3))                          # at the stroke's pixels
    tone_d = None
    seam = np.zeros(len(fy))
    if ring.sum() > 50:
        if P.color_match > 0:
            vals = [(col_t - col_s)[ry, rx].T]                              # 3, n
            oks = [np.ones((3, len(ry)), bool)]
            if sides:
                q = zi[ry, rx]
                both = okS[:, q] & okT[:, q]                                # D, n
                vals.append((dT[:, q] - dS[:, q]).transpose(0, 2, 1).reshape(-1, len(ry)))
                oks.append(np.repeat(both, 3, axis=0))
            got = P.color_match * _carry(np.concatenate(vals), np.concatenate(oks), (ry, rx), foot, f,
                                         sig_c, (fy, fx))
            tone = got[:3].T
            if sides:
                tone_d = got[3:].reshape(len(SIDES), 3, len(fy)).transpose(0, 2, 1)   # D, F, 3
        if P.seam:
            seam = _carry((rel_s - rel_t)[ry, rx][None], None, (ry, rx), foot, f, sig_c, (fy, fx),
                          clip=1.5 * sp)[0]
    # under the stroke the target may have nothing to blend with (a hole in
    # the scan): there the source stands alone
    wc = np.where(solid_t, w, np.where(foot, 1.0, 0.0))
    col = col_t.copy()
    col[fy, fx] = col_t[fy, fx] + wc[fy, fx, None] * (col_s[fy, fx] + tone - col_t[fy, fx])
    col = np.clip(col, 0.0, 1.0)
    Hf = h_t.copy()
    Hf[fy, fx] = B_t[fy, fx] + w[fy, fx] * (rel_s[fy, fx] - seam) + (1.0 - w[fy, fx]) * rel_t[fy, fx]

    # ---- 8. the old skin under the stroke goes, and what stands above the new surface
    xk, yk, hk = tg["xk"], tg["yk"], tg["hk"]
    xi, yi = np.round(xk).astype(int), np.round(yk).astype(int)
    inimg = (xi >= 0) & (xi < W) & (yi >= 0) & (yi < W)
    core = ndimage.binary_erosion(foot, iterations=1)
    low = np.minimum(Hf - 1.0 * sp, np.where(tg["found"], tg["dn"], Hf) - P.old_skin * sp)
    way = np.zeros(len(keep_t), bool)
    way[inimg] = core[yi[inimg], xi[inimg]] & (hk[inimg] > low[yi[inimg], xi[inimg]])
    rest_np = np.where(~way)[0]
    rest = scene_t.torch.tensor(rest_np, device=scene_t.dev)
    rgb_r, a_r, _ = scene_t.picture(cam, rest)
    # A removed splat reaches past the stroke: where taking it away shows
    # beside the stroke, the surface is laid again as it was.
    changed = (np.abs(rgb_r - tg["rgb"]).max(2) > 0.06) | (tg["alpha"] - a_r > 0.1)
    beside = changed & ~foot & paint & ndimage.binary_dilation(foot, iterations=max(2, int(round(2 * pxsp))))
    lift = foot | ndimage.binary_opening(beside, iterations=1)
    lift = ndimage.binary_closing(lift, iterations=1) & (foot | paint) & zone
    pos, quat, scale, ys, xs = _lift(cam, Hf, lift, sp, P)
    N = len(ys)

    # ---- 9. colour that turns with the view
    f_rest = np.zeros((N, Kc, 3), np.float32)
    base = col[ys, xs]
    if P.sharpen > 0:
        # A lifted pixel is a small Gaussian, and a skin of them shows the
        # picture a little soft. The picture's finest detail is raised by
        # what the lifting takes.
        soft = np.stack([ndimage.gaussian_filter(col[..., c_], P.sharpen_px, mode="nearest") for c_ in range(3)], 2)
        base = np.clip(base + P.sharpen * (col - soft)[ys, xs], 0.0, 1.0)
    tone_info = {}
    if sides:
        # every side view is stamped like the picture itself: the source's
        # change against its picture, tone-matched, blended with the target's
        q = zi[ys, xs]
        wl = wc[ys, xs][None, :, None]
        d_src = dS[:, q]
        if tone_d is not None:
            fi = np.full(foot.shape, -1, np.int64)
            fi[fy, fx] = np.arange(len(fy))
            j = fi[ys, xs]
            d_src = d_src + np.where((j >= 0)[None, :, None], tone_d[:, np.maximum(j, 0)], 0.0)
        delta = wl * d_src + (1.0 - wl) * dT[:, q]                          # D, N, 3
        if P.sharpen > 0:
            # Raised like the picture's own detail above, each view's change
            # by its own: with the picture's detail alone, a surface that
            # goes soft at a slant (leather) kept its full grain from the
            # side and looked like orange peel there.
            y0_, x0_ = int(ys.min()), int(xs.min())
            m = np.zeros((int(ys.max()) - y0_ + 1, int(xs.max()) - x0_ + 1))
            m[ys - y0_, xs - x0_] = 1.0
            den = np.maximum(ndimage.gaussian_filter(m, P.sharpen_px, mode="constant"), 1e-6)
            img = np.zeros_like(m)
            for k_ in range(delta.shape[0]):
                for c_ in range(3):
                    img[ys - y0_, xs - x0_] = delta[k_, :, c_]
                    soft = ndimage.gaussian_filter(img, P.sharpen_px, mode="constant") / den
                    delta[k_, :, c_] += P.sharpen * (delta[k_, :, c_] - soft[ys - y0_, xs - x0_])
        pinv, y0 = _sh_fit(n, oms, Kf)
        cf = np.einsum("kd,dnc->nkc", pinv, delta)
        f_rest[:, :Kf] = cf
        base = base - np.einsum("k,nkc->nc", y0, cf)
        tone_info = {"viewTone": round(float(np.abs(cf).mean()), 4), "shBands": int(Kf),
                     "sideSeen": round(float((okS[:, q] & okT[:, q]).mean()), 3),
                     # by ring of views (0, 30, 52, 72 degrees): target, source
                     "sideSeenBy": [[round(float(o[a_:b_, q].mean()), 2) for a_, b_ in ((0, 1), (1, 7), (7, 13), (13, 19))]
                                    for o in (okT, okS)]}
    elif P.tone == "target" and Kc >= 8:
        surf_r = on_t & ~ndimage.binary_dilation(lift, iterations=1) & (a_r > 0.7)
        ft = f * max(1, int(round(W / 512.0)))
        Hc_ = _down_known(Hf, lift | surf_r, ft)[0]
        Hc_ = _nearest_fill(Hc_, _down_known((lift | surf_r).astype(float), np.ones_like(lift), ft)[0] > 0.3)
        got = _view_tone(scene_t, cam, rest, rgb_r, Hc_, surf_r, lift, ft, sp)
        if got is not None:
            coef, b_ref = got
            cf = np.stack([[ndimage.map_coordinates(coef[k, c_], [(ys + 0.5) / ft - 0.5, (xs + 0.5) / ft - 0.5],
                                                    order=1, mode="nearest")
                            for c_ in range(3)] for k in range(8)])     # 8, 3, N
            fade = _grazing_fade(scene_t, cam, rest, rgb_r, h_t, surf_r, lift, sp, pxsp,
                                 P.grazing_band) if P.grazing else 0.0
            if fade > 0.05:
                # fine detail times (1 - fade * sin^2): (d.n)^2 is an SH of degree 2
                rng_ = np.random.default_rng(0)
                dd = rng_.normal(size=(400, 3))
                dd /= np.linalg.norm(dd, axis=1, keepdims=True)
                qk = np.linalg.lstsq(shtools.sh_basis(dd, 2), (dd @ n) ** 2, rcond=None)[0]
                soft = np.stack([ndimage.gaussian_filter(col[..., c_], max(1.0, P.grazing_band * pxsp))
                                 for c_ in range(3)], 2)
                cf = cf + fade * qk[1:, None, None] * (col - soft)[ys, xs].T[None]
            f_rest[:, :8, :] = cf.transpose(2, 0, 1)
            base = base - np.einsum("k,kcn->nc", b_ref, cf)
            tone_info = {"viewTone": round(float(np.abs(cf).mean()), 4), "grazingFade": round(float(fade), 2)}
    elif P.tone == "source" and Kc >= 3:
        # the SH of the source's own splats where each pixel was taken,
        # turned with the surface; beside the brush's middle, the target's
        deg = int(round(math.sqrt(Kc + 1))) - 1
        fs = shtools.rotate_sh(_sh_near(src_set, cam_s.world(qx[ys, xs], qy[ys, xs], h_sq[ys, xs]), sp), R)
        ws = wc[ys, xs][:, None, None]
        if (ws < 1).any():
            fs = ws * fs + (1.0 - ws) * _sh_near(s.take(keep_t), cam.world(xs, ys, h_t[ys, xs]), sp)
        f_rest[:] = fs
        b_ref = shtools.sh_basis((-n)[None], deg)[0, 1:Kc + 1]
        base = base - np.einsum("k,nkc->nc", b_ref, fs)
        tone_info = {"viewTone": round(float(np.abs(fs).mean()), 4)}

    added = SplatSet(
        xyz=pos.astype(np.float32),
        f_dc=((base - 0.5) / SH_C0).astype(np.float32),
        f_rest=f_rest,
        opacity=np.full(N, math.log(P.opacity / (1 - P.opacity)), np.float32),
        scale=scale.astype(np.float32),
        rot=quat.astype(np.float32),
        extra={k: np.zeros((N,) + vv.shape[1:], vv.dtype) for k, vv in s.extra.items()},
    )
    removed = keep_t[way]
    angle = float(np.degrees(np.arccos(np.clip((np.trace(R) - 1.0) / 2.0, -1, 1))))
    # For checks, as the copying stamp reports them: how much of what was
    # laid lies under the stroke (across the view), and the colour of the
    # copy next to what was there and what the source showed -- all three as
    # the pictures show them, weighted by the brush.
    dd, jj = cKDTree(Tq).query(pos)
    relp = pos - Tq[jj]
    dd = np.linalg.norm(relp - np.outer(relp @ ray, ray), axis=1)
    wf = w[fy, fx]
    mean_rgb = lambda a_: [round(float(x), 3) for x in (a_ * wf[:, None]).sum(0) / max(wf.sum(), 1e-9)]   # noqa: E731
    # a dab whose middle got nothing: no surface under it, or none at its source
    wT = w[np.clip(np.round(yT).astype(int), 0, W - 1), np.clip(np.round(xT).astype(int), 0, W - 1)]
    inT = (xT >= 0) & (xT < W) & (yT >= 0) & (yT < W)
    skipped = int((~inT | (wT <= 0)).sum())
    info = {"mode": "clone", "engine": "image", "dabs": int(K - skipped), "added": int(N),
            "removed": int(len(removed)), "rotation": round(angle, 2), "discarded": 0,
            "landed": round(float(np.mean(dd <= 1.15 * rad[jj])), 3),
            "newRgb": mean_rgb(np.clip(col_s[fy, fx] + tone, 0.0, 1.0)), "oldRgb": mean_rgb(col_t[fy, fx]),
            "srcRgb": mean_rgb(col_s[fy, fx]),
            "picture": [int(W), int(W)], "pxPerSpacing": round(float(pxsp), 2),
            "spacing": float(sp), "radius": round(r, 4),
            "strokePixels": int(foot.sum()), "beside": int((lift & ~foot).sum()),
            "sourceShare": round(float((can & (wb > 0)).sum() / max((wb > 0).sum(), 1)), 3),
            "toneShift": round(float(np.abs(tone).mean()), 4),
            "seamHeight": round(float(np.abs(seam).mean() / sp), 3),
            "seamShift": round(float(np.abs(seam).mean() / sp), 3),
            # where the last dab took from: its target, carried back to the source
            "source": [float(x) for x in a_s + R.T @ (Tq[-1] - a_t)],
            "seconds": round(time.time() - T0, 2), **tone_info}
    if skipped:
        info["skipped"] = skipped
        info["skippedWhy"] = "there is no surface there, at the target or at the source"
    dbg = {}
    if debug:
        dbg = {"cam": cam, "cam_s": cam_s, "rgb_t": tg["rgb"], "rgb_s": sr["rgb"], "h_t": h_t,
               "h_s": h_s, "B_t": B_t, "B_s": B_s, "w": w, "col": col, "Hf": Hf, "lift": lift,
               "on_t": on_t, "on_s": on_s, "found_t": tg["found"], "found_s": sr["found"],
               "walk_t": walk_t, "walk_s": walk_s, "mid_t": tg["mid"], "rim": rim,
               "thick_t": np.where(tg["found"], tg["up"] - tg["dn"], 0.0),
               "thick_s": np.where(sr["found"], sr["up"] - sr["dn"], 0.0),
               "n_t": n_t, "n_s": n_s, "R": R, "a_t": a_t, "a_s": a_s}
    return ImStampResult(removed=removed, added=added, info=info, debug=dbg)
