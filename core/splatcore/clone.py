"""The clone stamp: copy a patch of surface from one place to another.

The repair brush picks its own donor. This is the same transplant with the
choice handed back to you -- for when the right material is somewhere the
automatic search would never look, or when you want a specific feature copied.

What makes this more than a translation is that the two places are rarely
parallel. Source and target each get their own tangent frame, and everything
moves through the rotation between them:

* positions, so the patch lies on the target surface rather than floating at
  the source's angle;
* orientations, so the anisotropic discs stay flat against the surface instead
  of poking through it;
* **the SH bands**, which is the part that is easy to forget and impossible to
  miss once seen. A Gaussian's view-dependent lobes are expressed in world
  axes. Move a splat without turning them and the patch is right from the angle
  you cloned it at and wrong from every other -- it flashes and shifts hue as
  the camera goes round. Rotating them costs one matrix per band.
"""
from __future__ import annotations

from dataclasses import dataclass
from typing import Optional

import numpy as np
from scipy.spatial import cKDTree

from .io import SplatSet, SH_C0
from .repair import RepairError, RepairResult, fit_tangent_frame, _outer_shell
from . import shtools


# How far something has to stand out of the surface around the stamp, at
# least, to count as painted over rather than as relief: a fraction of the
# brush radius (the hut's cross stands 0.6 radii proud of its wall).
PROUD_MIN = 0.25

# How far past the brush's edge a copied splat may still reach, in radii.
OVERHANG = 0.35

# The ring the tone is compared over, in brush radii: just inside the edge,
# where copy and surface meet (0.9 beat 1.0, 1.3 and 1.7 on bench_clone);
# wider and outside the brush when what lies under it is being painted over.
TONE_RING = (0.6, 0.9)
TONE_RING_PROUD = (1.0, 1.8)

# Under a sign: how deep behind the surface around it the stamp clears what
# no camera saw, in brush radii (behind the hut's cross a dark red blob lay
# 3 cm, 0.4 radii, behind the siding).
SIGN_DEPTH = 1.0

# Around a sign, how far the wall still counts as touched by it -- its
# shadow, its colour bled into the scan -- in brush radii, across the wall.
SIGN_MARGIN = 0.3


@dataclass
class CloneParams:
    radius_scale: float = 1.0      # multiplies the requested radius
    feather: float = 0.3           # soft edge, fraction of the radius
    color_match: float = 0.6       # 0 keeps source tone, 1 matches the target
    rotate_sh: bool = True
    replace: bool = True           # remove what the stamp lands on
    seam: str = "plane"            # "plane" | "offset" | "none": seam matching
    jigsaw: bool = True            # old and new together come to one
    align: bool = True             # nudge the source to meet the pattern
    seed: int = 0
    # "image": the picture stamp (`imstamp.py`), "copy": this module's stamp,
    # "auto": what the document is set to (`Document.clone_engine`)
    engine: str = "auto"


def clone_params(d) -> CloneParams:
    """``CloneParams`` from the app's dict, ignoring keys it does not know.

    The app grows options faster than the engine: a ``volume`` sent along
    with a stroke used to end the call with a TypeError before any dab.
    """
    known = CloneParams.__dataclass_fields__
    kw = {k: v for k, v in (d or {}).items() if k in known}
    p = CloneParams(**kw)
    p.feather = float(np.clip(p.feather, 0.02, 1.0))
    p.color_match = float(np.clip(p.color_match, 0.0, 1.0))
    if p.engine not in ("auto", "image", "copy"):
        p.engine = "auto"
    return p


def _ring_fit(uv, mism, weight, mode):
    """A plane through the mismatch measured around the stamp's edge.

    The stamp carries its source's height straight into the target's frame.
    Both frames are least-squares planes through their own neighbourhood, so
    on two patches of the same wall they agree to within the noise -- but
    they are fitted over different radii and different curvature, and what is
    left is a small constant offset and a small tilt. On a tread that is
    invisible, because the tread is deeper than the error. On a smooth patch
    there is nothing to hide behind: a tenth of a splat spacing of tilt is a
    ridge running across an otherwise flat surface, and the stamp reads as a
    plate laid on top instead of as the surface continuing.

    So measure the disagreement where both sides exist -- the ring of
    surviving surface around the stamp -- and take it out. A plane removes
    offset and tilt, which is exactly the pair of things two tangent frames
    can differ by; anything beyond that is the surfaces genuinely differing,
    and taking that out would flatten the thing being copied. Colour gets the
    same treatment for the same reason, which is what turns a constant tone
    shift into one that follows the light across the patch.
    """
    if mode == "none" or weight.sum() < 8:
        return np.zeros(mism.shape[1:])[None] * 0.0, lambda q: 0.0
    w = weight / weight.sum()
    mu = (w[:, None] * mism).sum(0)
    if mode == "offset":
        return mu, lambda q: mu
    c = uv - (w[:, None] * uv).sum(0)
    A = np.concatenate([np.ones((len(uv), 1)), c], 1) * np.sqrt(weight)[:, None]
    b = mism * np.sqrt(weight)[:, None]
    coef, *_ = np.linalg.lstsq(A, b, rcond=None)
    base = (w[:, None] * uv).sum(0)
    return mu, lambda q: (np.concatenate(
        [np.ones((len(q), 1)), q - base], 1) @ coef)


def _tone_delta(s, xyz, alive, r, sp, src, tgt, proud, hide, mode, bad_src=None):
    """The colour shift that makes the copy sit in the target's light.

    Returns ``(delta(q) -> rgb shift, cells used)``; ``delta`` is None when
    there is nothing to compare. ``src`` and ``tgt`` are ``(centre, origin,
    basis, uv0)`` of each side's frame; ``q`` is a position relative to the
    brush centre in either frame.

    Compared is what the camera sees: per cell of a ring around the brush
    centre, the **opacity-weighted** colour of every splat on
    both sides, and the plane through the differences. The ring is just
    inside the edge (``TONE_RING``); where a sign is being painted over, it
    is outside the brush (``TONE_RING_PROUD``). It used to be an
    unweighted mean of each source splat against its nearest *opaque*
    target splats. On a translucent scan that is two different populations
    -- on the hut the faint splats are 0.1 darker than the opaque ones --
    and a patch stamped onto itself came out 10 grey levels brighter; along
    a stroke it reached 18 to 24, a white smear across the siding.

    ``proud(idx)``: target splats standing proud of the surface, which are
    being painted over and are no tone to match. ``hide(idx)``: target
    splats under the stroke where a sign stood -- the wall behind it, which
    no camera saw; then only the surface around the stroke counts.
    ``bad_src(idx)``: source splats that are no surface either -- more of
    the sign being painted out, which the source's ring may reach.
    """
    if mode == "none":
        return None, 0
    lo, hi = TONE_RING if hide is None else TONE_RING_PROUD
    inner, reach = lo * r, hi * r
    cell = max(2.0 * sp, r / 4.0)
    sides = []
    for k, (c, o, B, uv0) in enumerate((src, tgt)):
        near = np.linalg.norm(xyz - c, axis=1) < reach
        if alive is not None:
            near &= alive
        idx = np.where(near)[0]
        if k == 0 and len(idx) and bad_src is not None:
            idx = idx[~bad_src(idx)]
        if k == 1 and len(idx):
            bad = proud(idx)
            if hide is not None:
                bad |= hide(idx)
            idx = idx[~bad]
        if len(idx) < 8:
            return None, 0
        uvh = (xyz[idx] - o) @ B.T
        # one surface: not the far side of a thin shell, not haze in front
        hm = float(np.median(uvh[:, 2]))
        band = np.abs(uvh[:, 2] - hm) <= max(6.0 * sp, 0.5 * r)
        q = uvh[band, :2] - uv0
        dq = np.linalg.norm(q, axis=1)
        ring = (dq >= inner) & (dq <= reach)
        idx, q = idx[band][ring], q[ring]
        a = np.clip(s.alpha[idx].astype(np.float64), 0.0, 1.0)
        key = np.floor(q / cell).astype(np.int64)
        lin = (key[:, 0] + 4096) * 8192 + (key[:, 1] + 4096)
        u, inv = np.unique(lin, return_inverse=True)
        w = np.bincount(inv, a, len(u))
        rgb = np.stack([np.bincount(inv, a * s.rgb[idx][:, ch], len(u))
                        for ch in range(3)], 1)
        sides.append((u, w, rgb, (u // 8192 - 4096 + 0.5) * cell,
                      (u % 8192 - 4096 + 0.5) * cell))
    (us, ws, cs, *_), (ut, wt, ct, xt, yt) = sides
    common, i_s, i_t = np.intersect1d(us, ut, return_indices=True)
    ok = (ws[i_s] > 0.05) & (wt[i_t] > 0.05)
    if ok.sum() < 4:
        # no cell seen from both sides: the plain difference of the rings
        if ws.sum() <= 0 or wt.sum() <= 0:
            return None, 0
        d = ct.sum(0) / wt.sum() - cs.sum(0) / ws.sum()
        return (lambda q: np.broadcast_to(d, (len(q), 3))), 0
    i_s, i_t = i_s[ok], i_t[ok]
    diff = ct[i_t] / wt[i_t, None] - cs[i_s] / ws[i_s, None]
    weight = np.sqrt(ws[i_s] * wt[i_t])
    qc = np.stack([xt[i_t], yt[i_t]], 1)
    # a cell that differs wildly is a different material on one side (the
    # source's ring brushing a sign), not light: keep it out of the fit
    # -- in colour, not only in brightness: the red side of a sign beside a
    # white wall is as bright as the wall, and tinted the copy pink
    med = np.median(diff, 0)
    dev = np.linalg.norm(diff - med, axis=1)
    mad = 1.4826 * float(np.median(dev))
    good = dev <= 3.0 * max(mad, 0.02)
    if good.sum() >= 4:
        diff, weight, qc = diff[good], weight[good], qc[good]
    w = weight / weight.sum()
    mu = (w[:, None] * diff).sum(0)
    if mode == "offset" or len(diff) < 12:
        return (lambda q: np.broadcast_to(mu, (len(q), 3))), len(diff)
    # offset and tilt -- light falling off across the patch -- and no more
    base = (w[:, None] * qc).sum(0)
    A = np.concatenate([np.ones((len(qc), 1)), qc - base], 1) * np.sqrt(w)[:, None]
    coef, *_ = np.linalg.lstsq(A, diff * np.sqrt(w)[:, None], rcond=None)
    return (lambda q: np.concatenate([np.ones((len(q), 1)), np.asarray(q) - base], 1)
            @ coef), len(diff)


def _chroma(rgb):
    """Red and green share of a colour: its hue, without its brightness."""
    rgb = np.clip(np.asarray(rgb, np.float64), 0.0, None)
    return rgb[:, :2] / (rgb.sum(1, keepdims=True) + 0.05)


def _clip_to_brush(log_scale, quats, n, d, r, margin=OVERHANG):
    """Shorten copied splats that would reach far past the brush's edge.

    A splat is copied when its centre lies in the brush, but on a scan with
    long, flat splats -- the hut's planks are painted with ellipses up to a
    brush radius long -- its body reaches two or three times that far. Out
    there it shows the source's planks, shifted by the stamp's offset, over
    the target's own: doubled plank lines and streaks either side of the
    stroke, which read as a soft, smeared patch. Photoshop's stamp copies
    nothing outside the brush; this keeps each axis's in-plane reach (two
    sigma) within the edge plus ``margin`` brush radii, where the feather
    has faded the copy anyway. Splats inside that are left as they are.
    """
    sig = np.exp(log_scale.astype(np.float64))
    M = shtools.quat_to_matrix(quats)                     # columns: the axes
    inplane = np.sqrt(np.clip(1.0 - np.einsum("nij,i->nj", M, n) ** 2, 0.0, 1.0))
    reach = np.maximum(r - d, 0.0) + margin * r
    lim = reach[:, None] / np.maximum(2.0 * inplane, 1e-6)
    return np.log(np.minimum(sig, np.maximum(lim, 1e-6))).astype(np.float32)


def _along_ray(eye, p, o, n, r):
    """``p`` moved along the ray from ``eye`` onto the plane ``(o, n)``.

    Left where it is when the ray grazes the plane or the move is longer
    than the brush: then the plane is not the surface the user pointed at.
    """
    d = p - eye
    dn = float(np.dot(d, n))
    if abs(dn) < 0.3 * np.linalg.norm(d):
        return p
    q = eye + d * (float(np.dot(o - eye, n)) / dn)
    return q if np.linalg.norm(q - p) <= r else p


def _fade_alpha(alpha, f):
    """Scale a splat's coverage by ``f`` in optical depth, not in alpha.

    Two coincident layers faded linearly, to ``f a`` and ``(1 - f) a``,
    composite to ``a - f (1 - f) a^2``: a quarter short at f = 0.5 for an
    opaque splat, which is a faint translucent ring exactly where a stamp or
    a fill is meant to hand over invisibly. Scaling the optical depth
    instead, ``1 - (1 - a)^f``, makes the two halves composite back to ``a``
    exactly, whatever ``a`` is.
    """
    a = np.clip(np.asarray(alpha, np.float64), 0.0, 1.0 - 1e-6)
    return 1.0 - (1.0 - a) ** np.clip(f, 0.0, 1.0)


def _tangent_basis(n):
    """Some orthonormal tangent pair for a normal, as rows (e0, e1, n).

    Only ever chosen for one side of a stamp. The other side's basis is this
    one carried through the rotation between the two normals, so the choice
    made here cancels out of everything the user can see.
    """
    a = np.array([1.0, 0.0, 0.0]) if abs(n[0]) < 0.9 else np.array([0.0, 1.0, 0.0])
    e0 = a - n * float(np.dot(a, n))
    e0 /= max(np.linalg.norm(e0), 1e-12)
    return np.stack([e0, np.cross(n, e0), n])


def _local_surface(s: SplatSet, centre, radius, alive=None, view=None, xyz=None,
                   fit_radius=None):
    """The surface near a point: its splats, its plane and its outward normal.

    Returns ``(idx_all, idx_solid, origin, normal, spacing)``. ``idx_all`` is
    the outer layer within the radius whatever its opacity -- what a stamp
    has to remove, faint haze included, or the haze survives every dab and
    piles up. ``idx_solid`` is the part of it opaque enough to copy and to
    sample the seam from.

    Two things here used to be wrong and both were invisible on the shoe.

    The plane is fitted robustly, and the robust fit used to *decide what was
    copied*: anything further off the plane than a few median deviations was
    dropped from the stamp and from the removal. On a clean surface the
    deviation is tiny, so the rivet, the logo, the bump the user meant to
    stamp was exactly what got dropped -- measured on a plane, 0 % of a bump
    copied and 0 % of the old one removed. Now the cut only steers the plane.

    The normal's sign came from the centroid of a wider ball, which is right
    on a closed shell and a coin flip on a floor or a wall: on an open plane
    14 of 40 sites pointed the wrong way, and a stamp between two of them
    lands turned over -- relief inverted, texture mirrored. The person
    painting is looking at the surface, so when the view direction is known
    the normal faces the camera; the centroid rule is only the fallback.
    """
    xyz = s.xyz.astype(np.float64) if xyz is None else xyz
    dist = np.linalg.norm(xyz - centre, axis=1)
    near = dist < radius
    if alive is not None:
        near &= alive
    idx_all = np.where(near)[0]
    # The plane comes from ``fit_radius``, which both sides of a stamp share.
    # A plane through the centroid of a wider ball sits at a different
    # height on anything curved -- the sagitta of the extra ring -- and
    # with the target fitted over 1.4 times the source's radius a patch
    # stamped onto its own position came out four to five spacings off.
    fr = radius if fit_radius is None else fit_radius
    # The opacity bar for what counts as the surface. 0.2 on an ordinary
    # scan; a translucent one (the hut's siding) has too few splats above
    # it, and the stamp then refused or copied a third of the surface.
    bar = 0.2
    if ((s.alpha[idx_all] > bar) & (dist[idx_all] < fr)).sum() < 20:
        bar = 0.05
    fit = idx_all[(s.alpha[idx_all] > bar) & (dist[idx_all] < fr)]
    if len(fit) < 20:
        raise RepairError(
            "not enough material here (%d splats); "
            "use a larger brush or another spot" % len(fit))

    origin, basis, inl = fit_tangent_frame(xyz[fit], w=s.alpha[fit])
    if inl.sum() >= 20:
        origin, basis, _ = fit_tangent_frame(xyz[fit[inl]], w=s.alpha[fit[inl]])
    n = basis[2].copy()
    if view is not None and np.linalg.norm(view) > 0:
        if float(np.dot(n, view)) > 0:
            n = -n
    else:
        wide = np.linalg.norm(xyz - centre, axis=1) < radius * 2.5
        if alive is not None:
            wide &= alive
        bulk = xyz[wide].mean(0) if wide.sum() > 50 else xyz[fit].mean(0)
        if float(np.dot(n, origin - bulk)) < 0:
            n = -n

    # Keep only the layer facing outward, so a stamp taken from a thin shell
    # does not bring the inside of the object along with it.
    B = _tangent_basis(n)
    uvh = (xyz[idx_all] - origin) @ B.T
    sub = cKDTree(xyz[fit])
    spacing = float(np.median(sub.query(xyz[fit], k=2)[0][:, 1]))
    cell = max(2.0 * spacing, 1e-9)
    gu = np.floor(uvh[:, 0] / cell).astype(np.int64)
    gv = np.floor(uvh[:, 1] / cell).astype(np.int64)
    lin = (gv - gv.min()) * (gu.max() - gu.min() + 2) + (gu - gu.min())
    keep = _outer_shell(uvh[:, 2], lin, 1.0, gap_tol=8.0 * spacing)
    if (keep & (s.alpha[idx_all] > bar)).sum() >= 20:
        idx_all = idx_all[keep]
    idx_solid = idx_all[s.alpha[idx_all] > bar]
    return idx_all, idx_solid, origin, n, spacing


def _facing(n, view, origin=None, bulk=None):
    """``n`` turned to face the camera, else away from the bulk behind it."""
    if view is not None and np.linalg.norm(view) > 0:
        return -n if float(np.dot(n, view)) > 0 else n
    if bulk is not None and float(np.dot(n, origin - bulk)) < 0:
        return -n
    return n


def _reference(s, xyz, centre, r_fit, view, alive=None, footprint=None):
    """The surface around a point as a robust plane: ``(origin, normal, mad)``.

    Fitted over the front layer of what lies within ``r_fit`` -- and, at a
    target, outside what the stroke paints over (``footprint``: points and
    radii), because that is the surface the stamp has to continue. What the
    user paints over is what is about to go: a sign standing proud of a wall,
    a stain, a hole. Fitted over the brush itself, the stamp sat on the sign.
    """
    d = np.linalg.norm(xyz - centre, axis=1)
    near = d < r_fit
    if alive is not None:
        near &= alive
    idx = np.where(near)[0]
    bar = 0.2 if (s.alpha[idx] > 0.2).sum() >= 30 else 0.05
    idx = idx[s.alpha[idx] > bar]
    if len(idx) < 12:
        return None
    ring = idx
    if footprint is not None:
        pts, rad = footprint
        dd, jj = cKDTree(np.atleast_2d(np.asarray(pts, float))).query(xyz[idx])
        out = dd > np.asarray(rad, float).reshape(-1)[jj]
        if out.sum() >= 30:
            ring = idx[out]
    o, B, inl = fit_tangent_frame(xyz[ring], w=s.alpha[ring])
    n = _facing(B[2], view, o, xyz[idx].mean(0))
    # the front layer only: a thick shell has its inside within reach too
    Bt = _tangent_basis(n)
    uvh = (xyz[ring] - o) @ Bt.T
    sp = float(np.median(cKDTree(xyz[ring]).query(xyz[ring], k=2)[0][:, 1]))
    cell = max(2.0 * sp, 1e-9)
    gu = np.floor(uvh[:, 0] / cell).astype(np.int64)
    gv = np.floor(uvh[:, 1] / cell).astype(np.int64)
    lin = (gv - gv.min()) * (gu.max() - gu.min() + 2) + (gu - gu.min())
    keep = _outer_shell(uvh[:, 2], lin, 1.0, gap_tol=8.0 * sp)
    if keep.sum() >= 12:
        ring = ring[keep]
        o, B, inl = fit_tangent_frame(xyz[ring], w=s.alpha[ring])
        n = _facing(B[2], view, o, xyz[idx].mean(0))
    # Two layers in the ring -- a wall, and the arm of the sign on it that
    # carries on past the brush -- and the plane went between them, with a
    # spread so wide that nothing counted as standing proud. Where the ring
    # splits cleanly into a back layer and a front one, the back layer is the
    # surface and the front one is something on it.
    def split(h):
        """2-means on height: ``(back, clean)``."""
        lo, hi = np.percentile(h, [20, 80])
        back = h < 0.5 * (lo + hi)
        for _ in range(12):
            back = h < 0.5 * (lo + hi)
            if back.all() or not back.any():
                return back, False
            lo, hi = float(h[back].mean()), float(h[~back].mean())
        spread = max(sp, 1.4826 * float(np.median(np.abs(h[back] - lo))),
                     1.4826 * float(np.median(np.abs(h[~back] - hi))))
        return back, bool(back.mean() >= 0.3 and (hi - lo) > 4.0 * spread
                          and back.sum() >= 12)

    back, clean = split((xyz[ring] - o) @ n)
    if not clean:
        # With the arm on one side of the ring only, the plane through both
        # layers tilts, each layer smears across the tilt, and the split
        # looked unclean: along the hut's upright bar, 11 dabs in 22 saw
        # no sign (6 after this, 4 of them where the bar ends). Fitted
        # again through the back layer, a few times, the tilt goes and the
        # two layers part.
        o2, n2 = o, n
        for _ in range(3):
            b2, _ok = split((xyz[ring] - o2) @ n2)
            if b2.sum() < 12 or b2.all():
                break
            o2, B2, _ = fit_tangent_frame(xyz[ring[b2]], w=s.alpha[ring[b2]])
            n2 = _facing(B2[2], view, o2, xyz[idx].mean(0))
        b2, ok2 = split((xyz[ring] - o2) @ n2)
        if ok2:
            back, clean = b2, True
    if clean:
        ring = ring[back]
        o, B, inl = fit_tangent_frame(xyz[ring], w=s.alpha[ring])
        n = _facing(B[2], view, o, xyz[idx].mean(0))
    h = (xyz[ring[inl]] - o) @ n
    mad = 1.4826 * float(np.median(np.abs(h - np.median(h))))
    return o, n, mad


# How far a stroke's source may be nudged to meet the pattern around it, in
# brush radii -- half a plank of the hut's siding (planks 0.6 radii apart).
ALIGN_REACH = 0.3


def stroke_alignment(s: SplatSet, xyz, alive, sources, targets, radii, view=None):
    """A small shift of a stroke's source so its pattern meets the wall's.

    Returns ``(shift, info)``: a vector in the surface's plane to add to every
    source of the stroke, zero when nothing convincing was found.

    The offset between crosshair and brush is held in screen pixels, and the
    screen is a perspective view of a wall at an angle: an offset straight
    along the planks on screen is a few millimetres across them in the scan.
    Copied, every plank edge stepped by that much at the stroke's border --
    at the hut's upright bar and beside its shutter, plainly. Photoshop users
    line the crosshair up by eye; on a splat scan the eye cannot see a few
    millimetres. So the stamp looks for itself: the brightness of the wall
    around the stroke (outside it, and away from any sign standing on it) is
    compared with the brightness around the source, shifted by up to
    ``ALIGN_REACH`` radii, and the best match wins -- in practice half a
    plank period across the planks, so the edges continue exactly. Along
    the planks the match is flat, and a small penalty on the shift's length
    keeps it where it was.
    """
    from scipy import ndimage
    zero = np.zeros(3)
    targets = np.atleast_2d(np.asarray(targets, float))
    sources = np.atleast_2d(np.asarray(sources, float))
    radii = np.asarray(radii, float).reshape(-1)
    r = float(np.median(radii))
    mid = targets.mean(0)
    span = float(np.max(np.linalg.norm(targets - mid, axis=1)))
    ref = _reference(s, xyz, mid, span + 2.5 * r, view, alive,
                     footprint=(targets, radii))
    if ref is None:
        return zero, {"why": "no surface"}
    o, n, mad = ref
    P = _tangent_basis(n)[:2]
    d0 = np.median((sources - targets) @ P.T, axis=0)
    reach = span + float(np.linalg.norm(d0)) + 3.0 * r
    near = np.linalg.norm(xyz - mid, axis=1) < reach
    if alive is not None:
        near &= alive
    idx = np.where(near)[0]
    h = (xyz[idx] - o) @ n
    band = np.abs(h) < 0.5 * r
    proud = idx[h > max(PROUD_MIN * r, 3.0 * mad)]
    idx = idx[band]
    if len(idx) < 200:
        return zero, {"why": "not enough surface"}
    uv = (xyz[idx] - o) @ P.T
    a = np.clip(s.alpha[idx], 0.0, 1.0)
    lum = s.rgb[idx].mean(1)
    c = r / 24.0
    lo = np.minimum((targets @ P.T).min(0), (sources @ P.T).min(0)) - (o @ P.T) - 2.6 * r
    hi = np.maximum((targets @ P.T).max(0), (sources @ P.T).max(0)) - (o @ P.T) + 2.6 * r
    shape = tuple(np.ceil((hi - lo) / c).astype(int) + 1)
    if shape[0] * shape[1] > 4_000_000:
        return zero, {"why": "stroke too long"}
    ij = np.floor((uv - lo) / c).astype(int)
    ok = (ij >= 0).all(1) & (ij[:, 0] < shape[0]) & (ij[:, 1] < shape[1])
    ij, a, lum = ij[ok], a[ok], lum[ok]
    W = np.zeros(shape)
    L = np.zeros(shape)
    np.add.at(W, (ij[:, 0], ij[:, 1]), a)
    np.add.at(L, (ij[:, 0], ij[:, 1]), a * lum)
    W = ndimage.gaussian_filter(W, 1.0)
    L = ndimage.gaussian_filter(L, 1.0)
    img = L / np.maximum(W, 1e-9)
    valid = W > 0.05 * max(float(np.median(W[W > 0])), 1e-9) if (W > 0).any() else W > 0
    # away from a sign: its shadow and its colour lie on the wall beside it
    if len(proud):
        pij = np.floor(((xyz[proud] - o) @ P.T - lo) / c).astype(int)
        pok = (pij >= 0).all(1) & (pij[:, 0] < shape[0]) & (pij[:, 1] < shape[1])
        pm = np.zeros(shape, bool)
        pm[pij[pok, 0], pij[pok, 1]] = True
        pm = ndimage.binary_dilation(pm, iterations=max(1, int(round(SIGN_MARGIN * r / c))))
        valid &= ~pm
    # the wall around the stroke, outside it
    gi, gj = np.meshgrid(np.arange(shape[0]), np.arange(shape[1]), indexing="ij")
    cells = np.stack([gi.ravel(), gj.ravel()], 1) * c + lo + c / 2
    dist, _ = cKDTree(targets @ P.T - o @ P.T).query(cells)
    ring = ((dist >= 1.05 * r) & (dist <= 2.0 * r)).reshape(shape) & valid
    if ring.sum() < 100:
        return zero, {"why": "not enough wall around the stroke"}
    R = int(np.ceil(ALIGN_REACH * r / c))
    ti = np.where(ring)
    tv = img[ti]

    def sample(shift):
        q = np.stack([ti[0] + (d0[0] + shift[0]) / c, ti[1] + (d0[1] + shift[1]) / c])
        v = ndimage.map_coordinates(img, q, order=1, mode="nearest")
        m = ndimage.map_coordinates(valid.astype(float), q, order=0, mode="constant") > 0.5
        return v, m

    def ncc(shift):
        v, m = sample(shift)
        if m.sum() < max(0.25 * len(tv), 50):
            return -1.0
        x, y = tv[m] - tv[m].mean(), v[m] - v[m].mean()
        den = np.sqrt((x * x).sum() * (y * y).sum())
        return float((x * y).sum() / den) if den > 0 else -1.0

    def score(shift):
        return ncc(shift) - 0.15 * (np.linalg.norm(shift) / (ALIGN_REACH * r)) ** 2

    best, bs = None, -np.inf
    for di in range(-R, R + 1):
        for dj in range(-R, R + 1):
            sh = np.array([di, dj], float) * c
            if np.linalg.norm(sh) > ALIGN_REACH * r + 1e-12:
                continue
            v = score(sh)
            if v > bs:
                bs, best = v, sh
    # a parabola through the neighbours, per axis, for the last fraction
    for ax in range(2):
        e = np.zeros(2)
        e[ax] = c
        f0, fm, fp = bs, score(best - e), score(best + e)
        den = fm - 2 * f0 + fp
        if den < 0:
            t = float(np.clip(0.5 * (fm - fp) / den, -0.5, 0.5))
            cand = best + t * e
            vc = score(cand)
            if vc >= bs:
                best, bs = cand, vc
    n0, nb = ncc(np.zeros(2)), ncc(best)
    info = {"ncc0": round(n0, 3), "ncc": round(nb, 3),
            "shift": round(float(np.linalg.norm(best)) / r, 3)}
    if nb < 0.25 or nb - n0 < 0.02:
        return zero, info
    return best[0] * P[0] + best[1] * P[1], info


def clone(s: SplatSet, source, target, radius: float,
          params: Optional[CloneParams] = None,
          alive: Optional[np.ndarray] = None, view=None,
          xyz: Optional[np.ndarray] = None, footprint=None, eye=None,
          over_sign: bool = False, sign_chroma=None) -> RepairResult:
    """Copy the surface around ``source`` onto the surface around ``target``.

    ``view`` is the camera's forward direction; with it the stamp knows which
    side of a surface is the front. ``xyz`` is ``s.xyz`` as float64, for a
    caller that stamps many times over the same set. ``footprint`` is the
    whole stroke this dab belongs to, ``(points, radii)``: the surface the
    stamp continues is the one around the stroke, not the one under it.
    ``eye`` is the camera's position: with it the source and the target are
    moved along their rays onto their surfaces rather than along the normal.
    ``over_sign``: an earlier dab of the stroke, close by, found a sign under
    the brush (see ``Document.clone_stroke``); this one is painting it out
    too, even where little of it is left under the brush. ``sign_chroma``:
    the sign's colour as that dab measured it, for when too little of it is
    left here to measure.

    Both sides now take their orientation from a plane twice the brush wide,
    and the target its height from what lies around the stroke. Fitted over
    the brush alone, relief smaller than the brush -- the planks of the hut's
    siding, the red cross standing 4 cm proud of them -- tipped the frames by
    20 to 40 degrees from one dab to the next, and on the cross the stamp sat
    on the cross's face and left its sides standing: a red outline round a
    smear of tilted siding. What stands proud of the surrounding surface
    inside the brush is what the user is painting over, and it goes.
    """
    params = params or CloneParams()
    source = np.asarray(source, float)
    target = np.asarray(target, float)
    r = max(float(radius) * params.radius_scale, 1e-6)
    xyz = s.xyz.astype(np.float64) if xyz is None else xyz
    view = None if view is None else np.asarray(view, float)

    src_idx, _, o_s0, n_s0, sp_s = _local_surface(s, source, r, alive, view, xyz)
    all_t, tgt_idx, o_t0, n_t0, sp_t = _local_surface(s, target, r * 1.4, alive,
                                                      view, xyz, fit_radius=r)
    # Two ways to frame the two sides. The brush's own plane is exact where
    # the surface under the brush is the surface -- stamped onto its own
    # place a patch comes back unchanged, which is what the shoe's tread
    # needs. It fails in two ways, both found on the hut:
    #
    # * relief smaller than the brush -- the planks of the siding -- tips a
    #   plane fitted over the brush alone by 20 to 40 degrees from one dab
    #   to the next, and the stamp came out tilted by the difference;
    # * whatever the user paints over -- the red cross, 4 cm proud of the
    #   wall -- *is* the surface under the brush, so the stamp sat on the
    #   cross's face and left its sides standing as a red outline.
    #
    # The ring around each side out to 2.5 radii (around the target, outside
    # the whole stroke) sees neither. It is used when the brush's plane and
    # the ring's disagree by more than 12 degrees on either side, or when
    # part of the target stands proud of its ring; both sides are always
    # framed the same way, so that agreement is kept.
    ref_s = _reference(s, xyz, source, 2.5 * r, view, alive,
                       footprint=([source], [r]))
    fp = footprint if footprint is not None else ([target], [r])
    ref_t = _reference(s, xyz, target, 2.5 * r, view, alive, footprint=fp)
    o_s, n_s, o_t, n_t = o_s0, n_s0, o_t0, n_t0
    wide = False
    # Height over the target's ring, and what counts as standing proud of it.
    tol, floor, proud_share, measured = np.inf, np.inf, 0.0, 0.0
    h_ring = lambda idx: np.full(len(idx), -np.inf)               # noqa: E731
    lat_ring = lambda idx: np.full(len(idx), np.inf)              # noqa: E731
    if ref_t is not None:
        o_w, n_w, mad_w = ref_t
        h_ring = lambda idx: (xyz[idx] - o_w) @ n_w               # noqa: E731

        def lat_ring(idx):
            rel = xyz[idx] - target
            return np.linalg.norm(rel - np.outer(rel @ n_w, n_w), axis=1)
        tol = max(3.0 * sp_t, 3.0 * mad_w, PROUD_MIN * r)
        floor = np.inf
        in_brush = lat_ring(tgt_idx) <= r
        # Only for a thing on the surface: the painted point no more than
        # 0.9 brush radii in front of the ring (the cross is 0.6). Further out
        # it is haze or another object, and the stamp lands where painted --
        # from below the hut, a haze cloud lost 3,000 splats a dab to this.
        h_tgt = float(np.dot(target - o_w, n_w))
        if in_brush.any() and h_tgt <= 0.9 * r:
            proud_share = measured = float(np.mean(h_ring(tgt_idx[in_brush]) > tol))
        # After a stroke's first dab the brush is mostly its own copy, and
        # what is left of the sign is a sliver at the leading edge: 2 to 5 %
        # of the brush, under the bar for "painting over a sign". The dabs
        # along the hut's crossbar fell back to the brush's own plane -- copy
        # and sign mixed -- kept the sign's sides and the pink wall behind it
        # and matched their tone to it: the smeared band. A stroke that found
        # a sign close by is still painting it out.
        if over_sign and h_tgt <= 0.9 * r:
            proud_share = max(proud_share, 0.06)
        if proud_share > 0.05:
            # Once something is known to stand proud, all of it goes, down to
            # just above the ring's own relief: its sides reach the wall, and
            # cut at the higher bar they stayed as a red line in the siding.
            tol = max(1.5 * sp_t, 2.0 * mad_w)
            # And what lies on the surface behind it: no camera saw the wall
            # behind the cross, and those splats came out as dark smudges in
            # the new siding. The stamp brings the surface; all of it goes --
            # and what lies a little deeper too: on the hut a dark red blob
            # 3 cm behind the siding showed through the copy, which is as
            # translucent as the siding it was taken from.
            floor = -SIGN_DEPTH * r
    if ref_s is not None and ref_t is not None:
        cos_max = np.cos(np.radians(12.0))
        tilted = (abs(float(np.dot(ref_s[1], n_s0))) < cos_max
                  or abs(float(np.dot(ref_t[1], n_t0))) < cos_max)
        if tilted or proud_share > 0.05:
            wide = True
            n_s, n_t = ref_s[1], ref_t[1]
            # Height from the ring only where something stands on the
            # surface; for a mere tilt the brush's own plane keeps it, or a
            # stamp into haze landed a brush radius behind what was painted.
            if proud_share > 0.05:
                o_s, o_t = ref_s[0], ref_t[0]
    if view is None and float(np.dot(n_s, n_t)) < 0:
        n_s = -n_s
    if eye is not None:
        # Both points onto their surface along the camera's ray. They come
        # from a depth map, and where the brush is over a sign they lie on
        # the sign's face, 4 cm in front of the wall; dropped onto the wall
        # along the normal they moved sideways by up to a centimetre, a
        # different amount from one dab to the next. Overlapping dabs then
        # laid the same planks a few millimetres apart: doubled plank lines,
        # a soft patch. Along the ray, every dab is where the screen said.
        eye = np.asarray(eye, float)
        target = _along_ray(eye, target, o_t, n_t, r)
        source = _along_ray(eye, source, o_s, n_s, r)

    R = shtools.rotation_between(n_s, n_t)
    B_t = _tangent_basis(n_t)
    B_s = B_t @ R
    uvh_s = (xyz[src_idx] - o_s) @ B_s.T
    uvh_t = (xyz[tgt_idx] - o_t) @ B_t.T
    uvh_ta = (xyz[all_t] - o_t) @ B_t.T

    uv_s0 = ((source - o_s) @ B_s.T)[:2]
    uv_t0 = ((target - o_t) @ B_t.T)[:2]
    local_s = uvh_s.copy()
    local_s[:, :2] -= uv_s0

    keep = np.linalg.norm(local_s[:, :2], axis=1) <= r
    if keep.sum() < 10:
        raise RepairError("the brush covers too little surface at the source")
    src_idx, uvh_s, local_s = src_idx[keep], uvh_s[keep], local_s[keep]

    # Painting a sign out, the source must not bring it back. Beside a
    # cross the source of one stroke runs over another arm of it; copied,
    # the arm came back a stroke further on. Two marks: standing proud of
    # the source's own ring, and the sign's colour -- nearer the colour of
    # what stands proud under the brush than that of the source's surface,
    # in chromaticity, so a dark plank groove is not taken for a red sign.
    # Not where the source's middle is on the sign: a user who sets the
    # source there means to copy it.
    foreign = None
    sign_seen = None
    if proud_share > 0.05:
        bad = np.zeros(len(src_idx), bool)
        if ref_s is not None:
            o_rs, n_rs, mad_rs = ref_s
            tol_s = max(3.0 * sp_s, 3.0 * mad_rs, PROUD_MIN * r)
            bad |= (xyz[src_idx] - o_rs) @ n_rs > tol_s
        pt = tgt_idx[(h_ring(tgt_idx) > tol) & (lat_ring(tgt_idx) <= r)]
        ch_p = None
        if len(pt) >= 8:
            wp = s.alpha[pt]
            ch_p = (_chroma(s.rgb[pt]) * wp[:, None]).sum(0) / wp.sum()
            sign_seen = ch_p
        elif sign_chroma is not None:
            ch_p = np.asarray(sign_chroma, float)
        if ch_p is not None:
            # the surface's colour from what does not stand proud: a source
            # half over the sign has the sign's colour as its median
            flat = src_idx[~bad] if (~bad).sum() >= 10 else src_idx
            ch_w = np.median(_chroma(s.rgb[flat]), 0)
            if np.linalg.norm(ch_p - ch_w) > 0.05:
                def foreign(idx):
                    c = _chroma(s.rgb[idx])
                    return (np.linalg.norm(c - ch_p, axis=1)
                            < np.linalg.norm(c - ch_w, axis=1))
                bad |= foreign(src_idx)
        # Whether the user set the source on the sign is told by its middle,
        # not its share: nudged into line with the planks, a source beside
        # a bar may have more bar than wall under the brush.
        mid = np.linalg.norm(local_s[:, :2], axis=1) < 0.3 * r
        on_sign = bad[mid].mean() > 0.5 if mid.sum() >= 5 else bad.mean() >= 0.5
        if bad.any() and not on_sign and (~bad).sum() >= 10:
            src_idx, uvh_s, local_s = src_idx[~bad], uvh_s[~bad], local_s[~bad]

    # What stands proud of the surface around the target is what is being
    # painted over -- a sign on a wall -- and neither a height nor a tone to
    # match. Within the stamp it goes. Measured over the ring whichever frame
    # the stamp uses: dabs at the end of a sign's arm, where it covers too
    # little of the brush to change the frame, used to put it back.
    proud_t = h_ring(tgt_idx) > tol

    local_t = uvh_t[:, :2] - uv_t0
    tt = cKDTree(local_t)
    kk = min(4, len(local_t))
    dd, jj = tt.query(local_s[:, :2], k=kk)
    dd, jj = np.atleast_2d(dd.T).T, np.atleast_2d(jj.T).T
    near = (dd < 2.0 * max(sp_t, 1e-9)) & ~proud_t[jj]
    cnt = near.sum(1)
    denom = np.maximum(cnt, 1)
    tgt_h = np.where(near, uvh_t[jj, 2], 0.0).sum(1) / denom

    d_s = np.linalg.norm(local_s[:, :2], axis=1)
    ring = (cnt >= 2) & (d_s > 0.65 * r) & (d_s <= r)
    w_ring = ring.astype(np.float64)

    placed = local_s.copy()
    _, f_h = _ring_fit(local_s[:, :2], (local_s[:, 2] - tgt_h)[:, None],
                       w_ring, params.seam)
    h_fix = np.asarray(f_h(local_s[:, :2])).reshape(-1) if ring.sum() >= 8 else 0.0
    placed[:, 2] -= h_fix
    placed[:, 0] += uv_t0[0]
    placed[:, 1] += uv_t0[1]
    new_xyz = o_t + placed @ B_t
    picked = s.take(src_idx)

    quats = shtools.rotate_quats(picked.quat, R)
    f_rest = picked.f_rest.astype(np.float64)
    if params.rotate_sh and f_rest.shape[1]:
        f_rest = shtools.rotate_sh(f_rest, R)
    scale = _clip_to_brush(picked.scale, quats, n_t,
                           np.linalg.norm(local_s[:, :2], axis=1), r)

    f_dc = picked.f_dc.astype(np.float64)
    tone_cells = 0
    tone_applied = np.zeros((len(src_idx), 3))
    tone_centre = None
    if params.color_match > 0:
        # under the stroke, where something stood proud, nothing is surface
        # to match: the sign itself, and the wall behind it no camera saw
        hide = None
        if proud_share > 0.05:
            fp_pts, fp_rad = fp
            fp_pts = np.atleast_2d(np.asarray(fp_pts, float))
            fp_rad = np.asarray(fp_rad, float).reshape(-1)
            ftree = cKDTree(fp_pts)

            # and the sign's sides, which reach down to the wall below the
            # bar for standing proud: close beside what does stand proud
            ring_c = np.where(np.linalg.norm(xyz - target, axis=1) < 2.5 * r)[0]
            if alive is not None:
                ring_c = ring_c[alive[ring_c]]
            # -- measured across the wall, not in space: what lies behind
            # another arm of the sign is as unseen as what lies behind this
            # one, and is no nearer the arm's face than 4 cm. Counted, it
            # was the wall's darkest quarter and pulled the copy 5 levels
            # down.
            pp = ring_c[h_ring(ring_c) > tol]
            Bw = _tangent_basis(n_w)[:2]
            ptree = cKDTree((xyz[pp] - o_w) @ Bw.T) if len(pp) else None

            def hide(idx):
                d, j = ftree.query(xyz[idx])
                m = d <= fp_rad[j]
                if ptree is not None:
                    m |= np.isfinite(ptree.query((xyz[idx] - o_w) @ Bw.T,
                                                 distance_upper_bound=SIGN_MARGIN * r)[0])
                return m
        bad_src = None
        if proud_share > 0.05 and ref_s is not None:
            o_rs, n_rs, mad_rs = ref_s
            tol_rs = max(1.5 * sp_s, 2.0 * mad_rs)

            def bad_src(idx):
                m = (xyz[idx] - o_rs) @ n_rs > tol_rs
                if foreign is not None:
                    m |= foreign(idx)
                return m
        delta, tone_cells = _tone_delta(
            s, xyz, alive, r, max(sp_t, sp_s),
            (source, o_s, B_s, uv_s0), (target, o_t, B_t, uv_t0),
            lambda idx: h_ring(idx) > tol, hide, params.seam, bad_src=bad_src)
        if delta is not None:
            dq = np.asarray(delta(local_s[:, :2]), np.float64)
            d0 = np.asarray(delta(np.zeros((1, 2))), np.float64).reshape(3)
            if hide is not None:
                # Light is grey. Beside a sign the wall carries its colour
                # -- the hut's siding is pink for a few centimetres round
                # the red cross -- and a colour shift measured there tints
                # the copy. Only the brightness is matched.
                dq = np.repeat(dq.mean(1, keepdims=True), 3, 1)
                d0 = np.full(3, d0.mean())
            tone_applied = params.color_match * dq
            tone_centre = d0
            f_dc = f_dc + tone_applied / SH_C0

    dist = np.linalg.norm(local_s[:, :2], axis=1) / r
    fw = max(params.feather, 1e-3)
    t = np.clip((1.0 - dist) / fw, 0.0, 1.0)
    fade = 0.5 * (1.0 - np.cos(np.pi * t))
    a = np.clip(_fade_alpha(picked.alpha, fade), 1e-4, 1.0 - 1e-6)
    opacity = np.log(a / (1.0 - a))

    added = SplatSet(
        xyz=new_xyz.astype(np.float32), f_dc=f_dc.astype(np.float32),
        f_rest=f_rest.astype(np.float32), opacity=opacity.astype(np.float32),
        scale=scale, rot=quats.astype(np.float32),
        extra={k: v.copy() for k, v in picked.extra.items()})

    # what the copy looks like, for checks (before the old ring is added)
    wa = added.alpha
    copy_rgb = (added.rgb * wa[:, None]).sum(0) / max(float(wa.sum()), 1e-9)

    removed = np.zeros(0, np.int64)
    n_proud = 0
    n_grazed = 0
    if params.replace:
        d_t = np.linalg.norm(uvh_ta[:, :2] - uv_t0, axis=1)
        cut = d_t <= (r if params.jigsaw else r * (1.0 - fw * 0.5))
        supplied = cKDTree(placed[:, :2]).query(
            uvh_ta[:, :2], k=1, distance_upper_bound=2.0 * max(sp_t, 1e-9))[0]
        cut &= np.isfinite(supplied)
        proud_a = h_ring(all_t) > tol
        ring_t = cut & (d_t > r * (1.0 - fw)) & ~proud_a
        # everything standing proud inside the brush, the outer layer or not:
        # a sign's sides are not on the outer layer seen along the normal
        col = np.linalg.norm(xyz - target, axis=1) < 3.0 * r
        if alive is not None:
            col &= alive
        ci = np.where(col)[0]
        hc = h_ring(ci)
        lat = lat_ring(ci)
        inb = (lat <= r) & (hc < 1.5 * r)
        obj = inb & (hc > tol)
        # A stamp replaces about what it brings. Many times more standing
        # proud is not a sign but a cloud the ring fit misread: leave it.
        if obj.sum() > 4 * len(src_idx):
            obj[:] = False
        # The sign's colour, down to twice the depth cleared behind it.
        if foreign is not None:
            deep = inb & (hc > -2.0 * SIGN_DEPTH * r)
            obj[deep] |= foreign(ci[deep])
        # What stands in the brush's solid middle is what is painted out; at
        # the soft edge only what hangs together with it. Once a stroke knows
        # it is over a sign every dab looks for one, and the letters above
        # the hut's cross, grazed by the brush's edge, are another sign.
        mem = obj & (lat <= (1.0 - fw) * r)
        if (obj & ~mem).any():
            step = max(3.0 * sp_t, 0.1 * r)
            for _ in range(6):
                rest = np.where(obj & ~mem)[0]
                if not len(rest) or not mem.any():
                    break
                hit = np.isfinite(cKDTree(xyz[ci[mem]]).query(
                    xyz[ci[rest]], distance_upper_bound=step)[0])
                if not hit.any():
                    break
                mem[rest[hit]] = True
        behind = inb & (hc > floor) & (hc <= tol) if np.isfinite(floor) \
            else np.zeros(len(ci), bool)
        pr = ci[mem | behind]
        n_proud = len(pr)
        # And what stands at the soft edge without belonging to it is a
        # neighbour the brush only grazes -- the letters above the hut's
        # cross, the frame of a shutter. It keeps its splats, and the copy
        # gives way where it stands. Not what has the sign's colour: that
        # is the sign, cut off from the brush's middle by the dabs before.
        kept = np.zeros(0, np.int64)
        nb_edge = ci[obj & ~mem & (lat > (1.0 - fw) * r)]
        if len(nb_edge) and foreign is not None:
            nb_edge = nb_edge[~foreign(nb_edge)]
        if len(nb_edge):
            Bn = _tangent_basis(n_w)[:2]
            gtree = cKDTree((xyz[nb_edge] - o_w) @ Bn.T)
            under = np.isfinite(gtree.query(
                (new_xyz - o_w) @ Bn.T, distance_upper_bound=max(3.0 * sp_t, 0.05 * r))[0])
            if under.any() and (~under).sum() >= 10:
                added = added.take(np.where(~under)[0])
                tone_applied = tone_applied[~under]
            kept = nb_edge
            n_grazed = int(len(kept))
        removed = np.setdiff1d(np.union1d(all_t[cut], pr), kept)
        ring_t &= ~np.isin(all_t, kept)
        if params.jigsaw and ring_t.sum() >= 4:
            t_o = np.clip((1.0 - d_t[ring_t] / r) / fw, 0.0, 1.0)
            keepf = 1.0 - 0.5 * (1.0 - np.cos(np.pi * t_o))
            old = s.take(all_t[ring_t])
            a_o = np.clip(_fade_alpha(old.alpha, keepf), 1e-4, 1.0 - 1e-6)
            added = added.concat(SplatSet(
                xyz=old.xyz, f_dc=old.f_dc, f_rest=old.f_rest,
                opacity=np.log(a_o / (1.0 - a_o)).astype(np.float32),
                scale=old.scale, rot=old.rot,
                extra={k: v.copy() for k, v in old.extra.items()}))

    angle = float(np.degrees(np.arccos(np.clip((np.trace(R) - 1.0) / 2.0, -1, 1))))
    info = {
        "mode": "clone", "added": int(added.n), "removed": int(len(removed)),
        "rotation": round(angle, 2), "radius": round(r, 4), "seam": params.seam,
        "seamRing": int(ring.sum()), "jigsaw": bool(params.jigsaw),
        "seamShift": round(float(np.median(np.abs(np.atleast_1d(h_fix))))
                           / max(sp_t, 1e-9), 3),
        "proud": int(n_proud), "wide": bool(wide), "toneCells": int(tone_cells),
        "proudShare": round(float(measured), 3),
        "signChroma": None if sign_seen is None else [float(v) for v in sign_seen],
        "grazed": int(n_grazed),
        "shRotated": bool(params.rotate_sh and picked.f_rest.shape[1] > 0),
        "copyRgb": [float(v) for v in copy_rgb],
    }
    if tone_centre is not None:
        info["toneCentre"] = [float(v) for v in tone_centre]
    # the tone each copied splat was given, for the stroke to even out
    tone_all = np.zeros((added.n, 3))
    tone_all[:len(tone_applied)] = tone_applied
    return RepairResult(removed=removed, added=added, info=info,
                        context={"tone": tone_all})
