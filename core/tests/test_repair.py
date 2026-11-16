"""Repair-engine tests on synthetic surfaces where the right answer is known.

The shoe is the demo; these are the regression net. Each case builds a surface
whose true geometry we can assert against, removes a region, and checks that
what comes back actually sits on the surface.
"""
import os
import sys

import numpy as np

sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))
from splatcore.io import SplatSet, SH_C0                      # noqa: E402
from splatcore import repair as rp                            # noqa: E402
from splatcore import clone as cl                            # noqa: E402


def make_surface(height_fn, n=24000, extent=6.0, seed=0, period=1.1, sh=15,
                 thickness=None):
    """Splats scattered on z = height_fn(x, y) with a periodic colour pattern.

    ``thickness`` adds a second shell underneath, which is what a scanned
    object really looks like and what the surface detection has to cope with.
    """
    rng = np.random.default_rng(seed)
    xy = rng.uniform(-extent, extent, size=(n, 2))
    z = height_fn(xy[:, 0], xy[:, 1])
    xyz = np.column_stack([xy, z]).astype(np.float32)
    tone = 0.5 + 0.22 * np.sin(xy[:, 0] * 2 * np.pi / period) \
        * np.cos(xy[:, 1] * 2 * np.pi / period)
    rgb = np.column_stack([tone, tone * 0.85, tone * 0.7])
    s = SplatSet(
        xyz=xyz,
        f_dc=((rgb - 0.5) / SH_C0).astype(np.float32),
        f_rest=(rng.normal(0, 0.01, (n, sh, 3))).astype(np.float32),
        opacity=rng.uniform(1.5, 3.5, n).astype(np.float32),
        # varied, anisotropic, randomly oriented -- so "copy, don't blend" is
        # actually testable: a blended result would produce novel values
        scale=rng.uniform(np.log(0.03), np.log(0.06), (n, 3)).astype(np.float32),
        rot=(lambda q: q / np.linalg.norm(q, axis=1, keepdims=True))(
            rng.normal(size=(n, 4))).astype(np.float32),
    )
    if thickness:
        back = s.take(np.arange(n))
        back.xyz = back.xyz.copy()
        back.xyz[:, 2] -= thickness
        s = s.concat(back)
    return s


def disc(s, cx=0.0, cy=0.0, r=1.4):
    d = np.hypot(s.xyz[:, 0] - cx, s.xyz[:, 1] - cy)
    return (d < r) & (s.xyz[:, 2] > -0.5)      # front shell only


def _check_on_surface(res, height_fn, tol, label):
    a = res.added
    assert a.n > 0, "%s: nothing added" % label
    assert np.isfinite(a.xyz).all(), "%s: non-finite positions" % label
    err = np.abs(a.xyz[:, 2] - height_fn(a.xyz[:, 0], a.xyz[:, 1]))
    p95 = float(np.percentile(err, 95))
    assert p95 < tol, "%s: 95%% of new splats %.3f off the surface (tol %.3f)" % (
        label, p95, tol)
    return p95


def test_flat_plane():
    fn = lambda x, y: np.zeros_like(x)
    s = make_surface(fn)
    sel = disc(s)
    for mode in ("rim", "patch"):
        res = rp.repair(s, sel, rp.RepairParams(mode=mode, seed=1))
        p95 = _check_on_surface(res, fn, 0.09, "flat/" + mode)
        print("    flat/%-5s +%5d splats, 95%% within %.4f" % (mode, res.added.n, p95))


def test_curved_surface():
    """A dome: the fitted height field has to follow real curvature."""
    fn = lambda x, y: 0.05 * (x ** 2 + y ** 2)
    s = make_surface(fn)
    res = rp.repair(s, disc(s), rp.RepairParams(mode="rim", seed=1))
    p95 = _check_on_surface(res, fn, 0.12, "curved")
    print("    curved      +%5d splats, 95%% within %.4f" % (res.added.n, p95))


def test_shell_not_confused_by_back_face():
    """The hole must be filled on the front shell, not bridged to the back."""
    fn = lambda x, y: np.zeros_like(x)
    s = make_surface(fn, thickness=0.6)
    res = rp.repair(s, disc(s), rp.RepairParams(mode="rim", seed=1))
    z = res.added.xyz[:, 2]
    assert np.percentile(np.abs(z), 95) < 0.12, \
        "fill drifted off the front shell: z p95 = %.3f" % np.percentile(np.abs(z), 95)
    assert (z > -0.3).mean() > 0.98, "some fill landed on the back shell"
    print("    shell       +%5d splats, z p95 %.4f" % (
        res.added.n, np.percentile(np.abs(z), 95)))


def test_fill_covers_the_hole():
    """Coverage, not just plausibility: the gap should actually be closed."""
    fn = lambda x, y: np.zeros_like(x)
    s = make_surface(fn)
    sel = disc(s, r=1.4)
    res = rp.repair(s, sel, rp.RepairParams(mode="rim", seed=1))
    a = res.added
    inside = np.hypot(a.xyz[:, 0], a.xyz[:, 1]) < 1.2
    assert inside.sum() > 200, "hole interior barely populated (%d)" % inside.sum()
    # density should land near the surrounding density, not 10x off
    orig = (np.hypot(s.xyz[:, 0], s.xyz[:, 1]) < 1.2).sum()
    ratio = inside.sum() / max(orig, 1)
    assert 0.4 < ratio < 4.0, "fill density %.2fx the original" % ratio
    print("    coverage    %d in hole vs %d originally (%.2fx)" % (inside.sum(), orig, ratio))


def test_attributes_are_copied_not_blended():
    """Every new splat's covariance must equal some real donor's."""
    s = make_surface(lambda x, y: np.zeros_like(x))
    res = rp.repair(s, disc(s), rp.RepairParams(mode="rim", seed=1))
    src = set(map(tuple, np.unique(s.scale, axis=0).round(6)))
    new = set(map(tuple, np.unique(res.added.scale, axis=0).round(6)))
    assert len(src) > 1000, "test data is too uniform to detect blending"
    assert new <= src, "%d scale triples are not any donor's" % len(new - src)
    srcq = set(map(tuple, np.unique(s.rot, axis=0).round(6)))
    newq = set(map(tuple, np.unique(res.added.rot, axis=0).round(6)))
    assert newq <= srcq, "%d quaternions are not any donor's" % len(newq - srcq)
    print("    attributes  %d scales / %d quats, every one a real donor's"
          % (len(new), len(newq)))


def test_coverage_is_continuous_across_the_seam():
    """The quantity that must not jump is the total, not the fill's own alpha.

    This used to assert that the new splats fade out, which was a test of the
    mechanism rather than of the result. The fill now fades by how much
    surviving surface a cell still has, so on an evenly covered surface the
    handover is deliberately abrupt in the fill's opacity -- it stops dead
    where the old surface is whole. What has to stay smooth is old plus new
    together: a dip there is a visible gap, a bump is the pale band that used
    to lie along the rim.
    """
    s = make_surface(lambda x, y: np.zeros_like(x))
    sel = disc(s)
    res = rp.repair(s, sel, rp.RepairParams(mode="rim", seed=1, feather=0.4))

    alive = np.ones(s.n, bool)
    alive[res.removed] = False
    centre = s.xyz[sel].mean(0)[:2]
    r_hole = float(np.percentile(
        np.linalg.norm(s.xyz[sel][:, :2] - centre, axis=1), 95))

    def weight(pts, alpha, lo, hi):
        d = np.linalg.norm(pts[:, :2] - centre, axis=1)
        m = (d >= lo) & (d < hi)
        area = np.pi * (hi ** 2 - lo ** 2)
        # optical depth adds across layers; alpha does not
        return float(-np.log(np.clip(1.0 - alpha[m], 1e-6, 1.0)).sum() / area)

    step = 0.25 * r_hole
    rings = []
    for k in range(-2, 4):
        lo, hi = r_hole + k * step, r_hole + (k + 1) * step
        if lo < 0:
            continue
        w = weight(s.xyz[alive], s.alpha[alive], lo, hi) \
            + weight(res.added.xyz, res.added.alpha, lo, hi)
        rings.append(w)
    rings = np.asarray(rings)
    far = float(np.median(rings[-2:]))
    worst = float(np.max(np.abs(rings - far)) / max(far, 1e-9))
    assert worst < 0.45, (
        "coverage jumps by %.0f%% across the seam: %s"
        % (100 * worst, np.round(rings / far, 2).tolist()))
    assert (res.added.alpha > 0).all() and np.isfinite(res.added.alpha).all()
    print("    seam        coverage across the seam constant to within %.0f%%"
          % (100 * worst))


def test_sh_flat_mode_zeroes_bands():
    s = make_surface(lambda x, y: np.zeros_like(x))
    res = rp.repair(s, disc(s), rp.RepairParams(mode="rim", seed=1, sh_mode="flat"))
    assert np.abs(res.added.f_rest).max() == 0.0
    res2 = rp.repair(s, disc(s), rp.RepairParams(mode="rim", seed=1, sh_mode="donor"))
    assert np.abs(res2.added.f_rest).max() > 0.0
    print("    sh modes    flat zeroes bands, donor preserves them")


def test_apply_to_is_consistent():
    s = make_surface(lambda x, y: np.zeros_like(x))
    sel = disc(s)
    res = rp.repair(s, sel, rp.RepairParams(mode="rim", seed=1))
    out = res.apply_to(s)
    assert out.n == s.n - len(res.removed) + res.added.n
    assert out.f_rest.shape[1] == s.f_rest.shape[1]
    print("    apply_to    %d -> %d splats" % (s.n, out.n))


def test_determinism():
    s = make_surface(lambda x, y: np.zeros_like(x))
    sel = disc(s)
    a = rp.repair(s, sel, rp.RepairParams(mode="rim", seed=7)).added
    b = rp.repair(s, sel, rp.RepairParams(mode="rim", seed=7)).added
    assert np.array_equal(a.xyz, b.xyz), "same seed gave different results"
    c = rp.repair(s, sel, rp.RepairParams(mode="rim", seed=8)).added
    assert not np.array_equal(a.xyz[:50], c.xyz[:50]), "seed had no effect"
    print("    determinism same seed reproduces, different seed varies")


def test_errors_are_explained():
    s = make_surface(lambda x, y: np.zeros_like(x))
    for bad, msg in ((np.zeros(s.n, bool), "nothing selected"),):
        try:
            rp.repair(s, bad)
            raise AssertionError("expected RepairError for %s" % msg)
        except rp.RepairError as e:
            assert msg in str(e).lower(), str(e)
    for kw in ({"mode": "nope"}, {"sh_mode": "nope"}, {"surface_degree": 9},
               {"density": 0.0}):
        try:
            rp.RepairParams(**kw).validate()
            raise AssertionError("expected ValueError for %s" % kw)
        except ValueError:
            pass
    print("    errors      bad input raises with an explanation")


def test_membrane_stays_inside_its_boundary():
    """A level must not bend its way out of the range it was given.

    The reference surface is a thin plate, which continues the boundary's
    curvature across a gap -- right for a shape, because a sole that is
    curving away keeps curving away. Tone is not a shape. Fitted the same
    way it dived: 0.285 in the middle of the shoe's hole against a rim
    measured at 0.376, and it pulled the rim down to 0.339 on the way out.
    A membrane obeys the maximum principle, which is the property wanted.
    """
    nv = nu = 41
    yy, xx = np.mgrid[0:nv, 0:nu]
    data = 1.0 + 0.04 * ((xx - 20.0) ** 2 + (yy - 20.0) ** 2) / 100.0
    known = np.ones((nv, nu), bool)
    known[8:33, 8:33] = False                      # a wide hole
    lo, hi = data[known].min(), data[known].max()

    film = rp.fit_reference_surface(data, known, smooth_cells=6.0, plate=False)
    plate = rp.fit_reference_surface(data, known, smooth_cells=6.0, plate=True)
    hole = ~known
    assert film[hole].min() >= lo - 1e-6 and film[hole].max() <= hi + 1e-6, \
        "membrane left its boundary's range: %.4f..%.4f vs %.4f..%.4f" % (
            film[hole].min(), film[hole].max(), lo, hi)
    # and the fit must still honour the data it was given
    assert abs(float(np.median(film[known] - data[known]))) < 0.02
    print("    membrane    inside %.3f..%.3f in [%.3f, %.3f], plate %.3f..%.3f"
          % (film[hole].min(), film[hole].max(), lo, hi,
             plate[hole].min(), plate[hole].max()))


def test_fill_takes_the_tone_of_where_it_lands():
    """A patch that is darker than its surroundings reads as a dent.

    Nothing about the geometry has to be wrong for that. On the shoe the
    fill's outer face sat a fifth of a splat spacing *outside* the surface
    around it while looking sunken, because it was a third too dark, and on
    a real surface darker means lower. Light the test surface unevenly so a
    donor from elsewhere carries the wrong exposure, and require the fill to
    give that up for the exposure of where it lands.
    """
    s = make_surface(lambda x, y: np.zeros_like(x), thickness=0.5)
    ramp = 0.3 * (s.xyz[:, 0] / 6.0)              # bright on the right
    s.f_dc = (s.f_dc + (ramp / SH_C0)[:, None]).astype(np.float32)
    sel = disc(s, cx=3.0, r=1.4)

    def miss(**kw):
        res = rp.repair(s, sel, rp.RepairParams(mode="patch", seed=0, **kw))
        face = res.added.rgb if res.backing is None else res.added.rgb[~res.backing]
        d = np.hypot(s.xyz[:, 0] - 3.0, s.xyz[:, 1])
        ring = (d > 1.5) & (d < 2.4) & (s.xyz[:, 2] > -0.25)
        return abs(float(face.mean() - s.rgb[ring].mean()))

    with_field, without = miss(), miss(tone_smooth=0.0)
    assert with_field < 0.02, "fill is off tone by %.3f" % with_field
    assert with_field < 0.6 * without, \
        "the tone field earned nothing: %.3f vs %.3f" % (with_field, without)
    print("    tone        off by %.3f, without the tone field %.3f"
          % (with_field, without))


def test_stamp_leaves_no_halo():
    """Old and new together must come to one across the stamp's edge.

    The stamp fades its new material out over the last third of the brush,
    but used to delete the old material under all of it with a hard edge, so
    a ring was left holding a half-transparent copy of a surface that had
    been whole. On a tread nobody sees it; on a smooth patch it is a halo,
    and it was the thing that made the stamp useless on the plaque.
    """
    s = make_surface(lambda x, y: np.zeros_like(x), thickness=0.5)
    src = np.array([2.5, 0.0, 0.0])
    tgt = np.array([-2.5, 0.0, 0.0])

    def cover(**kw):
        res = cl.clone(s, src, tgt, 1.2, cl.CloneParams(seed=0, **kw))
        out = res.apply_to(s)
        d = np.hypot(out.xyz[:, 0] + 2.5, out.xyz[:, 1])
        front = out.xyz[:, 2] > -0.25
        # Optical depth, not alpha, is what adds up: two layers of a/2 do
        # not make a, two layers of half the optical depth do.
        depth = -np.log(np.clip(1.0 - out.alpha, 1e-6, 1.0))
        band = lambda a, b: float(                                # noqa: E731
            depth[front & (d >= a) & (d < b)].sum() / (np.pi * (b * b - a * a)))
        return band(0.95, 1.15), band(1.45, 1.65)   # the feather, and beyond

    seam, outside = cover()
    naive = cover(jigsaw=False)
    assert abs(seam / outside - 1.0) < 0.25, \
        "coverage jumps by %.0f%% across the seam" % (100 * abs(seam / outside - 1))
    assert abs(seam / outside - 1) < abs(naive[0] / naive[1] - 1), \
        "the complementary fade earned nothing"
    print("    stamp       coverage at the seam %.0f%% of the surroundings (before %.0f%%)"
          % (100 * seam / outside, 100 * naive[0] / naive[1]))


def test_stamp_copies_the_feature():
    """What stands proud of a smooth surface is what the user wants stamped.

    The plane fit's outlier cut used to decide what was copied, and on a
    clean surface its threshold is a hair's breadth -- a bump three spacings
    high was dropped entirely, at the source and under the target alike.
    """
    bump = lambda x, y: 0.25 * np.exp(-((x - 2.5) ** 2 + y ** 2) / 0.08)  # noqa: E731
    s = make_surface(bump)
    src_bump = int(((np.hypot(s.xyz[:, 0] - 2.5, s.xyz[:, 1]) < 1.0)
                    & (s.xyz[:, 2] > 0.1)).sum())
    res = cl.clone(s, np.array([2.5, 0.0, 0.0]), np.array([-2.5, 0.0, 0.0]),
                   1.0, cl.CloneParams(seed=0), view=np.array([0.0, 0.0, -1.0]))
    got = int((res.added.xyz[:, 2] > 0.1).sum())
    assert got >= 0.8 * src_bump, "bump copied %d of %d" % (got, src_bump)
    print("    feature     bump copied: %d of %d splats" % (got, src_bump))


def test_stamp_frame_is_continuous():
    """Two nearly parallel patches must give a nearly zero rotation.

    Each side used to pick its own tangent axes from the first world axis it
    was not too close to, and two normals either side of that threshold --
    25.5 and 26.3 degrees from z here -- came out 180 degrees apart.
    """
    def tilted(cx, deg, seed):
        p = make_surface(lambda x, y: np.zeros_like(x), n=6000, extent=2.0,
                         seed=seed)
        a = np.radians(deg)
        Rx = np.array([[1, 0, 0], [0, np.cos(a), -np.sin(a)],
                       [0, np.sin(a), np.cos(a)]])
        p.xyz = (p.xyz.astype(np.float64) @ Rx.T + [cx, 0, 0]).astype(np.float32)
        return p

    s = tilted(-3.0, 25.5, 1).concat(tilted(3.0, 26.3, 2))
    res = cl.clone(s, np.array([-3.0, 0.0, 0.0]), np.array([3.0, 0.0, 0.0]),
                   0.8, cl.CloneParams(seed=0), view=np.array([0.0, 0.0, -1.0]))
    assert res.info["rotation"] < 5.0, \
        "stamp turned %.1f degrees between patches 0.8 apart" % res.info["rotation"]
    print("    frame       rotation %.2f degrees at 0.8 degrees of true tilt"
          % res.info["rotation"])


def test_stamp_stroke_is_self_contained():
    """A painted stroke may only remove what existed before it, and redo.

    Overlapping dabs delete much of what the previous dab laid. That used to
    happen in the document itself, so the edit listed its own appended splats
    as removed; the viewport kept showing them and redo crashed.
    """
    from splatcore.document import Document
    doc = Document(make_surface(lambda x, y: np.zeros_like(x), thickness=0.5))
    n0, alive0 = doc.n_total, doc.n_alive
    view = np.array([0.0, 0.0, -1.0])
    pairs = [(np.array([2.5, -1.5 + 0.25 * k, 0.0]),
              np.array([-2.5, -1.5 + 0.25 * k, 0.0])) for k in range(8)]
    e = doc.clone_stroke(pairs, 1.0, cl.CloneParams(seed=0), view=view)
    assert len(e.removed) and int(e.removed.max()) < n0, \
        "the stroke removed splats it had appended itself"
    assert doc.n_alive == alive0 - len(e.removed) + e.added
    after = doc.n_alive
    doc.undo()
    assert doc.n_alive == alive0
    doc.redo()
    assert doc.n_alive == after
    print("    stroke      %d dabs, -%d / +%d, discarded %d, redo exact"
          % (e.info["dabs"], len(e.removed), e.added, e.info["discarded"]))


def test_long_stroke_is_sized_by_its_width():
    """A stroke along a cable costs what its area costs, not its length squared.

    Everything used to scale with the stroke's half-length: a 16-unit strip
    searched a ball 48 units across and laid seven new splats per removed
    one, most of them feather over intact surface.
    """
    s = make_surface(lambda x, y: np.zeros_like(x), n=60000, extent=7.0)
    sel = (np.abs(s.xyz[:, 0]) < 5.0) & (np.abs(s.xyz[:, 1]) < 0.2)
    res = rp.repair(s, sel, rp.RepairParams(mode="patch", seed=0))
    ratio = res.added.n / max(int(sel.sum()), 1)
    assert ratio < 3.0, "%.1f new splats per removed one" % ratio
    assert res.info["hole_width"] < 0.5 * res.info["hole_radius"]
    print("    strip       %.1f new per removed one, width %.2f at length %.2f"
          % (ratio, res.info["hole_width"], res.info["hole_radius"]))


def test_fold_is_reported():
    """Painted over the corner of two surfaces, the brush says so."""
    fl = make_surface(lambda x, y: np.zeros_like(x), n=20000, extent=4.0, seed=1)
    wl = make_surface(lambda x, y: np.zeros_like(x), n=10000, extent=4.0, seed=2)
    w = wl.xyz.astype(np.float64)
    wl.xyz = np.stack([np.full(len(w), 1.0), w[:, 0], np.abs(w[:, 1])],
                      1).astype(np.float32)
    s = fl.take(fl.xyz[:, 0] < 1.0).concat(wl)
    open_ = rp.repair(s, np.linalg.norm(s.xyz - [-2.0, 0, 0], axis=1) < 1.2,
                      rp.RepairParams(seed=0))
    corner = rp.repair(s, np.linalg.norm(s.xyz - [0.6, 0, 0], axis=1) < 1.2,
                       rp.RepairParams(seed=0))
    assert open_.info["warning"] is None, open_.info["warning"]
    assert corner.info["warning"], "no warning at a floor/wall corner"
    print("    edge        open %.0f degrees silent, corner %.0f degrees warned"
          % (open_.info["rim_fold_deg"], corner.info["rim_fold_deg"]))


def test_fill_stops_at_the_surface_edge():
    """A hole painted across the edge of a surface is filled up to the edge.

    Whatever the brush covered past the edge never held surface. Filling it,
    or letting the seam ring top up "missing" surface there, grew a fringe of
    Gaussians into empty space -- seen above the stand's hole on the shoe.
    """
    s = make_surface(lambda x, y: np.zeros_like(x), n=30000, extent=4.0)
    s = s.take(s.xyz[:, 0] < 1.0)                 # the surface ends at x = 1
    sel = np.hypot(s.xyz[:, 0] - 0.6, s.xyz[:, 1]) < 1.0
    res = rp.repair(s, sel, rp.RepairParams(mode="patch", seed=0))
    past = float((res.added.xyz[:, 0] > 1.1).mean())
    naive = rp.repair(s, sel, rp.RepairParams(mode="patch", seed=0, respect_edge=False))
    past0 = float((naive.added.xyz[:, 0] > 1.1).mean())
    assert past < 0.01, "%.1f %% of the fill lies past the edge" % (100 * past)
    print("    rim         beyond the edge %.1f %% (without the rule %.1f %%)"
          % (100 * past, 100 * past0))


def test_band_is_continued_along_its_curve():
    """A smooth band crossing the hole on a gentle arc comes back as that band.

    Two textured fields either side of a plain pale band; the hole takes a
    stretch of the band out. The curve step has to find both of the band's
    boundaries and continue them, and the fill inside the true band has to be
    band material -- without the curves barely an eighth of it was.
    """
    n, extent, half = 24000, 4.0, 0.45
    rng = np.random.default_rng(0)
    xy = rng.uniform(-extent, extent, size=(n, 2))
    band_x = lambda y: 0.15 * y ** 2 - 0.2
    inb = np.abs(xy[:, 0] - band_x(xy[:, 1])) < half
    chk = np.sign(np.sin(xy[:, 0] * 2 * np.pi / 0.4) * np.sin(xy[:, 1] * 2 * np.pi / 0.4))
    tone = np.where(inb, 0.72, 0.38 + 0.14 * chk)
    rgb = np.column_stack([tone, tone * 0.85, tone * 0.7])
    s = SplatSet(
        xyz=np.column_stack([xy, np.zeros(n)]).astype(np.float32),
        f_dc=((rgb - 0.5) / SH_C0).astype(np.float32),
        f_rest=np.zeros((n, 3, 3), np.float32),
        opacity=np.full(n, 2.5, np.float32),
        scale=np.full((n, 3), np.log(0.04), np.float32),
        rot=np.tile(np.array([1, 0, 0, 0], np.float32), (n, 1)))
    sel = np.hypot(xy[:, 0], xy[:, 1]) < 1.3
    res = rp.repair(s, sel, rp.RepairParams(mode="patch", seed=1))
    assert res.info.get("zones_curved"), "curve step did not fire"
    a = res.added
    x, y = a.xyz[:, 0], a.xyz[:, 1]
    hole = np.hypot(x, y) < 1.2
    inside = hole & (np.abs(x - band_x(y)) < half - 0.15)
    beside = hole & (np.abs(x - band_x(y)) > half + 0.15)
    f_in = float((a.rgb[inside, 0] > 0.6).mean())
    f_out = float((a.rgb[beside, 0] > 0.6).mean())
    assert f_in > 0.6, "only %.0f %% of the fill inside the band is band material" % (100 * f_in)
    assert f_out < 0.35, "band material spilled: %.0f %% beside the band" % (100 * f_out)
    print("    band arc    curved %s, band material %.0f %% inside, %.0f %% beside"
          % (res.info["zones_curved"], 100 * f_in, 100 * f_out))


def test_sh_degree_zero_input():
    """A PLY with no f_rest at all must still repair."""
    s = make_surface(lambda x, y: np.zeros_like(x), sh=0)
    res = rp.repair(s, disc(s), rp.RepairParams(mode="rim", seed=1))
    assert res.added.f_rest.shape[1] == 0 and res.added.n > 0
    print("    sh deg 0    handled")


def _pole_in_front_of_wall():
    """A thin pole standing clear of a whole wall: once it is gone, no hole."""
    wall = make_surface(lambda x, y: np.zeros_like(x), n=80000, extent=3.0, seed=3)
    rng = np.random.default_rng(0)
    m = 300
    pole = wall.take(np.arange(m))
    pole.xyz = np.column_stack([rng.normal(0, 0.02, m), rng.uniform(-1.0, 1.0, m),
                                0.2 + rng.normal(0, 0.005, m)]).astype(np.float32)
    s = wall.concat(pole)
    return s, np.arange(wall.n, s.n)


def test_pole_is_deleted_not_filled():
    """Nothing to fill behind a pole: the brush says so, the document deletes it."""
    from splatcore.document import Document
    s, sel = _pole_in_front_of_wall()
    try:
        rp.repair(s, sel, rp.RepairParams(seed=0, view=[0, 0, -1]))
        raise AssertionError("a pole in front of a whole wall was filled")
    except rp.NothingToFill as e:
        assert e.standoff > 2.0, e.standoff
        # the text points to the Delete button
        assert "Delete" in str(e), str(e)
    d = Document(s)
    e = d.repair(sel, rp.RepairParams(seed=0, view=[0, 0, -1]))
    assert e.kind == "delete" and e.info.get("mode") == "delete", (e.kind, e.info)
    assert len(e.removed) == len(sel) and d.n_alive == s.n - len(sel)
    d.undo()
    assert d.n_alive == s.n
    print("    pole        deleted instead of filled, standoff %.1f spacings" % e.info["standoff"])


def test_edge_loss_tries_the_faces():
    """A chart whose edge rule cut the whole hole away is not "nothing to fill".

    That used to raise NothingToFill, and the document deleted the stroke
    before the face-by-face fill was tried. Now it is a plain RepairError
    carrying the rim's fold, and past the warning angle the faces are tried.
    """
    orig_chart, orig_faces, orig_split = rp._repair_chart, rp._repair_across_faces, rp._chart_split
    s = make_surface(lambda x, y: np.zeros_like(x), n=6000)
    sentinel = rp.RepairResult(removed=np.zeros(0, np.int64), added=s.take(np.arange(3)),
                               info={"charts": 2})
    tried = []

    def chart(*a, fold=80.0, **k):
        err = rp.RepairError("beyond the edge")
        err.rim_fold = fold
        raise err
    try:
        rp._chart_split = lambda s_, m_, p_: (None, None)
        rp._repair_across_faces = lambda *a, **k: (tried.append(1), sentinel)[1]
        rp._repair_chart = chart
        out = rp.repair(s, disc(s), rp.RepairParams(seed=0))
        assert out is sentinel and tried, "the faces were not tried"
        # a gentle rim: no faces, the error stands
        tried.clear()
        rp._repair_chart = lambda *a, **k: chart(fold=20.0)
        try:
            rp.repair(s, disc(s), rp.RepairParams(seed=0))
            raise AssertionError("no error on a flat rim")
        except rp.NothingToFill:
            raise AssertionError("edge loss reported as nothing to fill")
        except rp.RepairError:
            pass
        assert not tried
        # a whole surface behind the stroke is still nothing to fill
        def ntf(*a, **k):
            raise rp.NothingToFill("intact", standoff=5.0)
        rp._repair_chart = ntf
        try:
            rp.repair(s, disc(s), rp.RepairParams(seed=0))
            raise AssertionError("NothingToFill swallowed")
        except rp.NothingToFill:
            pass
        assert not tried, "faces tried behind a whole surface"
    finally:
        rp._repair_chart, rp._repair_across_faces, rp._chart_split = orig_chart, orig_faces, orig_split
    print("    edge        an edge loss tries the faces, a whole surface stays NothingToFill")


def test_no_factorisation_outlives_a_repair():
    """Prefetched systems a fill never used are dropped (up to 1.5 GB each)."""
    orig = rp._find_donor_patch
    s = make_surface(lambda x, y: np.zeros_like(x), n=12000, extent=2.0)
    try:
        rp._find_donor_patch = lambda *a, **k: (None, np.inf, [])   # nothing to borrow
        res = rp.repair(s, disc(s, r=0.75), rp.RepairParams(seed=0))
        assert res.info["mode"] == "rim", res.info["mode"]
        assert not rp._LU_CACHE, "left in the cache: %d" % len(rp._LU_CACHE)
        res = rp.repair(s, disc(s, r=0.75), rp.RepairParams(seed=0))
        assert not rp._LU_CACHE
    finally:
        rp._find_donor_patch = orig
    res = rp.repair(s, disc(s, r=0.75), rp.RepairParams(seed=0))
    assert not rp._LU_CACHE, "left after a patch fill: %d" % len(rp._LU_CACHE)
    print("    breakdown   nothing in memory after rim and patch fill")


def test_document_drops_indices_out_of_range():
    """An index past the end (a mask older than an undo) is ignored, not fatal."""
    from splatcore.document import Document
    s = make_surface(lambda x, y: np.zeros_like(x), n=6000)
    d = Document(s)
    e = d.delete([0, 1, s.n, s.n + 500, -1])
    assert sorted(e.removed.tolist()) == [0, 1], e.removed
    sel = np.where(disc(s, r=1.0))[0]
    e = d.repair(np.concatenate([sel, [s.n + 3]]), rp.RepairParams(mode="rim", seed=0))
    assert e.kind == "repair" and e.added > 0
    e = d.transform([d.n_total + 9, 5], [1, 0, 0, 0], [0.1, 0, 0], 1.0)
    assert e.removed.tolist() == [5], e.removed
    print("    indices     outside the document rejected")


def test_view_fill_faces_reports_each_face():
    """Per-face fills: no left-over state, every face's source and reason kept,
    and a Photoshop that stops answering keeps the faces already done."""
    from splatcore import viewfill as vf, deep as dp, photoshop as ps
    s = make_surface(lambda x, y: np.zeros_like(x), n=2000)
    a0, a1 = s.take(np.arange(10)), s.take(np.arange(10, 25))
    base = rp.RepairResult(
        removed=np.array([0, 1]), added=a0.concat(a1), backing=np.zeros(25, bool),
        info={"charts": 2},
        context={"charts": [{"radius": 1.0}, {"radius": 1.0}],
                 "chart_spans": [(0, 10), (10, 25)],
                 "chart_sel": [np.array([0]), np.array([1])]})
    # nothing reaches the picture: no error, nothing reported
    empty = rp.RepairResult(removed=base.removed, added=s.take(np.zeros(0, np.int64)),
                            backing=np.zeros(0, bool), info={"charts": 2},
                            context={"charts": [{}, {}], "chart_spans": [(0, 0), (0, 0)],
                                     "chart_sel": [np.array([0]), np.array([1])]})
    r = vf._view_fill_faces(s, None, "lama", None, empty, None, None, 3.0, 0.5, 0.0)
    assert r.info.get("fill_fallback") is None and r.info["fill_source"] is None
    saved = (vf.view_frame, vf.view_subset, vf.render_view, vf.view_mask, vf.fill_picture,
             dp.deep_repair)
    calls = []
    try:
        vf.view_frame = lambda ck, c: (np.array([0, 0, 5.0]), np.zeros(3), np.array([0, 1.0, 0]), 30.0)
        vf.view_subset = lambda sc, *a, **k: np.arange(sc.n)
        vf.render_view = lambda *a, **k: (np.zeros((64, 64, 3), np.uint8), None, None)
        vf.view_mask = lambda *a, **k: np.pad(np.ones((16, 16), bool), 24)

        def fill(rgb, mask, source, say=None, sharpen=0.5):
            calls.append(1)
            if len(calls) == 1:
                return rgb, "Photoshop (content-aware)", "no credits"
            raise ps.PhotoshopUnavailable("Photoshop does not answer")
        vf.fill_picture = fill

        def deep(frozen, sel, params, d, base=None, on_progress=None, reference=None):
            out = base.added.take(np.arange(base.added.n))
            out.xyz = out.xyz + 1.0
            return rp.RepairResult(removed=base.removed, added=out, backing=base.backing,
                                   info={"deep": True, "fit": 0.02, "offAxisGeo": 0.1,
                                         "offAxisDeep": 0.09, "rejected": None,
                                         "inpainter": "image"},
                                   context=dict(base.context))
        dp.deep_repair = deep
        r = vf._view_fill_faces(s, None, "photoshop", None, base, None, 64, 3.0, 0.5, 0.0,
                                keep_picture=True)
    finally:
        (vf.view_frame, vf.view_subset, vf.render_view, vf.view_mask, vf.fill_picture,
         dp.deep_repair) = saved
    faces = r.info["deep_charts"]
    assert r.info["deep"] and faces[0]["deep"], faces
    assert faces[0]["fill_source"] == "Photoshop (content-aware)", faces[0]
    assert faces[0]["fill_fallback"] == "no credits", faces[0]
    assert r.info["fill_fallback"] == "no credits", r.info
    assert not faces[1]["deep"] and "Photoshop" in faces[1]["rejected"], faces[1]
    assert np.allclose(r.added.xyz[:10], a0.xyz + 1.0), "face 0's fill was lost"
    assert np.allclose(r.added.xyz[10:], a1.xyz), "face 1 should keep its quick fill"
    assert len(r.info["picture"]) == 3
    print("    faces       source and reason per face, face 1 kept")


def test_deep_step_that_cannot_run_keeps_the_quick_fill():
    from splatcore import viewfill as vf, deep as dp
    s = make_surface(lambda x, y: np.zeros_like(x), n=8000)
    sel = np.where(disc(s, r=1.0))[0]
    base = rp.repair(s, sel, rp.RepairParams(mode="rim", seed=0))
    saved = dp.deep_repair

    def boom(*a, **k):
        raise rp.RepairError("The spot is too small in the reference view")
    try:
        dp.deep_repair = boom
        r = vf.deep_patch(s, sel, None, None, base=base)
    finally:
        dp.deep_repair = saved
    assert r.added.n == base.added.n and r.info["deep"] is False
    assert "too small" in r.info["rejected"], r.info
    print("    deep        rejected, the Fast Fill stays")


def test_motif_map_prices_a_one_off():
    """A ring pattern in a lattice is priced; the lattice and a hole in it are not."""
    nv, nu = 120, 160
    yy, xx = np.mgrid[0:nv, 0:nu].astype(float)
    # a lattice that drifts a little, as a curved sole's does
    lat = np.sin(2 * np.pi * (xx + 0.002 * xx ** 2) / 8.0) * np.sin(2 * np.pi * yy / 6.0)
    ring = np.hypot(yy - 40, xx - 40)
    motif = ring < 18
    h = np.where(motif, np.cos(ring / 2.2) * 2.0, lat)
    feat = np.stack([h, 0.2 * h, 0.2 * h, 0.2 * h], -1)
    known = np.ones((nv, nu), bool)
    target = np.hypot(yy - 80, xx - 115) < 12
    known &= ~target
    got = rp._motif_map(feat, known, target)
    assert got is not None
    rare, allow = got
    in_motif = float(np.median(rare[motif & known]))
    in_lattice = float(np.median(rare[known & ~motif & (np.hypot(yy - 80, xx - 60) < 15)]))
    assert in_motif > 3 * in_lattice, (in_motif, in_lattice)
    cost = rp._motif_cost(rare, allow, np.array([40]), np.array([40]), np.array([80]), np.array([115]))
    ok = rp._motif_cost(rare, allow, np.array([80]), np.array([60]), np.array([80]), np.array([115]))
    assert cost[0] > 0 and ok[0] == 0, (cost, ok)
    print("    motif       %.2f against grid %.2f, cost %.2f" % (in_motif, in_lattice, cost[0]))


def test_bare_backing_goes_with_its_face():
    keep = np.ones(6, bool)
    back = np.array([0, 0, 1, 1, 1, 0], bool)
    key = np.array([1, 2, 1, 3, 3, 4])
    keep[1] = False                       # face copy of cell 2 dropped (no backing there)
    keep[5] = False                       # cell 4: face dropped, no backing either
    got = rp._drop_bare_backing(keep, back, key)
    assert list(got) == [True, False, True, False, False, False], got
    assert list(rp._drop_bare_backing(keep, back, None)) == list(keep)


def test_close_gaps_reseats_what_no_fill_covers():
    """A flat sheet, a painted disc, a fill that covers only its left half:
    the painted splats of the right half come back, in the sheet's colour,
    and those under the fill stay gone."""
    from splatcore.repair import RepairResult
    rng = np.random.default_rng(0)
    n = 20000
    xyz = np.column_stack([rng.uniform(-2, 2, n), rng.uniform(-2, 2, n), np.zeros(n)])
    rgb = np.tile([0.6, 0.5, 0.4], (n, 1))
    painted = np.linalg.norm(xyz[:, :2], axis=1) < 0.8
    rgb[painted] = [0.1, 0.1, 0.1]
    s = SplatSet(xyz=xyz.astype(np.float32), f_dc=((rgb - 0.5) / SH_C0).astype(np.float32),
                 f_rest=np.zeros((n, 3, 3), np.float32), opacity=np.full(n, 2.0, np.float32),
                 scale=np.full((n, 3), np.log(0.02), np.float32),
                 rot=np.tile([1.0, 0, 0, 0], (n, 1)).astype(np.float32))
    sel = np.where(painted)[0]
    left = sel[xyz[sel, 0] < 0]
    fill = s.take(left)
    fill.f_dc = np.tile(((np.array([0.6, 0.5, 0.4]) - 0.5) / SH_C0), (fill.n, 1)).astype(np.float32)
    ctx = {"basis": np.eye(3), "origin": np.zeros(3), "outward": 1.0, "uv_centre": (0.0, 0.0),
           "surface": lambda u, v: np.zeros_like(u), "spacing": 0.02}
    res = RepairResult(removed=sel, added=fill, backing=np.zeros(fill.n, bool), info={}, context=ctx)
    out = rp._close_gaps(s, sel, rp.RepairParams(), res)
    got = out.added.take(np.arange(fill.n, out.added.n))
    right = (xyz[sel, 0] > 0.1).sum()
    print("    reseated %d of %d uncovered" % (got.n, right))
    assert got.n >= 0.9 * right and (got.xyz[:, 0] > -0.06).all(), "only the uncovered half"
    assert np.abs(got.rgb - [0.6, 0.5, 0.4]).max() < 0.05, "in the colour round them"


if __name__ == "__main__":
    fails = 0
    for name, fn in sorted(globals().items()):
        if name.startswith("test_"):
            try:
                print("  " + name)
                fn()
            except AssertionError as e:
                fails += 1
                print("    FAIL:", e)
    print("all repair tests passed" if not fails else "%d FAILURES" % fails)
    sys.exit(1 if fails else 0)
