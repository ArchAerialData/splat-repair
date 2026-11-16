"""Stamp tests on synthetic surfaces: where it takes from, where it lands.

The stamp is judged by the user against two marks on screen: the crosshair
(where it takes from) and the brush (where it puts it). These cases check
that the engine keeps to both, on a flat floor, a curved one and a wall with
a sign on it -- the case from the hut that used to fail.

Two engines stamp (`Document.clone_engine`). The cases up to
``test_planks_line_up_across_the_seam`` are about the copying stamp
(`clone.py`) and ask for it by name; the ``test_picture_stamp_*`` cases are
about the picture stamp (`imstamp.py`), which needs CUDA + gsplat and is
skipped without.

Run as a script: python tests/test_clone.py
"""
import os
import sys

import numpy as np

sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))
sys.path.insert(0, os.path.dirname(__file__))
from splatcore.io import SplatSet, SH_C0                      # noqa: E402
from splatcore import clone as cl                             # noqa: E402
from splatcore.document import Document                       # noqa: E402
from test_repair import make_surface                          # noqa: E402

VIEW = np.array([0.0, 0.0, -1.0])            # camera above, looking down -z


def copy_params(**kw):
    """Parameters that ask for the copying stamp, whatever the document prefers."""
    return cl.CloneParams(engine="copy", **kw)


def picture_params(**kw):
    """Parameters that ask for the picture stamp."""
    return cl.CloneParams(engine="image", **kw)


def stripes(s, x0, x1, rgb):
    """Paint the splats with x0 <= x < x1 one colour, as a marker to trace."""
    m = (s.xyz[:, 0] >= x0) & (s.xyz[:, 0] < x1)
    s.f_dc[m] = ((np.asarray(rgb) - 0.5) / SH_C0).astype(np.float32)
    return m


def added_of(doc, e):
    return doc.generated.take(np.arange(doc.generated.n - e.added, doc.generated.n))


def test_source_is_taken_where_the_crosshair_is():
    """Picked sources are used as given, dab by dab, and land under the brush.

    A blue band sits where the crosshair runs, a red one where the carried
    world offset would have put it. Only blue may arrive.
    """
    s = make_surface(lambda x, y: np.zeros_like(x), n=30000, extent=6.0)
    stripes(s, 1.5, 2.5, (0.1, 0.2, 0.9))           # under the crosshair
    stripes(s, 3.0, 4.0, (0.9, 0.1, 0.1))           # where a drift would go
    doc = Document(s)
    pairs = [(np.array([2.0, -1.0 + 0.3 * k, 0.0]), np.array([-2.0, -1.0 + 0.3 * k, 0.0]))
             for k in range(7)]
    e = doc.clone_stroke(pairs, 0.45, copy_params(color_match=0.0), view=VIEW,
                         picked=[True] * len(pairs))
    new = added_of(doc, e)
    rgb = new.rgb[new.alpha > 0.3]
    blue = float(np.mean((rgb[:, 2] > 0.6) & (rgb[:, 0] < 0.4)))
    red = float(np.mean((rgb[:, 0] > 0.6) & (rgb[:, 2] < 0.4)))
    assert blue > 0.6 and red < 0.02, "blue %.2f red %.2f" % (blue, red)
    tg = np.array([t for _, t in pairs])
    d = np.min(np.linalg.norm(new.xyz[:, None, :2] - tg[None, :, :2], axis=2), axis=1)
    assert np.mean(d <= 0.45 * 1.15) > 0.97, "landed beside the stroke"
    assert e.info["landed"] >= 0.97 and e.info.get("skipped", 0) == 0, e.info
    print("    source      %.0f %% from the cross, %.1f %% from the stray path, %.0f %% under the stroke"
          % (100 * blue, 100 * red, 100 * e.info["landed"]))


def test_unpicked_source_keeps_its_offset():
    """Without a pick (crosshair off the object) the first offset carries on."""
    s = make_surface(lambda x, y: np.zeros_like(x), n=30000, extent=6.0)
    doc = Document(s)
    pairs = [(np.array([2.0, -1.0 + 0.3 * k, 0.0]), np.array([-2.0, -1.0 + 0.3 * k, 0.0]))
             for k in range(7)]
    e = doc.clone_stroke(pairs, 0.45, copy_params(), view=VIEW)
    src = np.array(e.info["source"])
    assert np.linalg.norm(src[:2] - pairs[-1][0][:2]) < 0.1, src
    assert e.info.get("skipped", 0) == 0, e.info


def test_curved_target_follows_the_surface():
    """On a cylinder the stamp turns with the surface and lies on it.

    The source is on the crown, the target 40 degrees round the side: the
    patch has to arrive turned by those 40 degrees and sit on the surface,
    not float at the crown's height.
    """
    R = 3.0
    rng = np.random.default_rng(3)
    n = 40000
    th = rng.uniform(-1.2, 1.2, n)
    yy = rng.uniform(-3, 3, n)
    xyz = np.column_stack([R * np.sin(th), yy, R * np.cos(th)])
    s = make_surface(lambda x, y: np.zeros_like(x), n=n)
    s.xyz = xyz.astype(np.float32)
    doc = Document(s)
    a = np.radians(40)
    src = np.array([0.0, 0.0, R])
    tgt = np.array([R * np.sin(a), 0.0, R * np.cos(a)])
    view = -tgt / np.linalg.norm(tgt)                # looking at the target
    e = doc.clone_stroke([(src, tgt)], 0.5, copy_params(), view=view,
                         picked=[True])
    new = added_of(doc, e)
    core = new.alpha > 0.3
    off = np.abs(np.linalg.norm(new.xyz[core][:, [0, 2]], axis=1) - R)
    assert np.median(off) < 0.03, "%.3f off the surface" % np.median(off)
    assert abs(e.info["rotation"] - 40.0) < 6.0, e.info["rotation"]
    print("    curved      turned %.1f degrees (surface 40), %.3f off the surface"
          % (e.info["rotation"], np.median(off)))


def test_stamp_over_a_sign_removes_it():
    """A sign standing proud of a wall goes, and the wall continues.

    The hut's cross: 0.6 brush radii in front of its siding. The stamp used
    to sit on the sign's face and leave its sides as an outline.
    """
    s = make_surface(lambda x, y: np.zeros_like(x), n=40000, extent=4.0, seed=1)
    rng = np.random.default_rng(5)
    k = 4000
    sign = make_surface(lambda x, y: np.zeros_like(x), n=k, extent=0.5, seed=2)
    p = sign.xyz.astype(np.float64)
    # a slab 1.0 x 1.0, 0.3 proud: front face and four sides
    face = np.column_stack([p[:, 0], p[:, 1], np.full(k, 0.3)])
    side_t = rng.uniform(0, 0.3, k)
    edge = rng.integers(0, 4, k)
    u = rng.uniform(-0.5, 0.5, k)
    sides = np.where(edge[:, None] == 0, np.column_stack([np.full(k, -0.5), u, side_t]),
            np.where(edge[:, None] == 1, np.column_stack([np.full(k, 0.5), u, side_t]),
            np.where(edge[:, None] == 2, np.column_stack([u, np.full(k, -0.5), side_t]),
                     np.column_stack([u, np.full(k, 0.5), side_t]))))
    slab = sign.concat(sign.take(np.arange(k)))
    slab.xyz = np.concatenate([face, sides]).astype(np.float32)
    slab.f_dc[:] = ((np.array([0.9, 0.1, 0.1]) - 0.5) / SH_C0).astype(np.float32)
    s = s.concat(slab)
    is_sign = np.zeros(s.n, bool)
    is_sign[-2 * k:] = True
    doc = Document(s)
    r = 0.5
    pairs = []
    for yv in np.arange(-0.6, 0.61, 0.2):
        for xv in np.arange(-0.6, 0.61, 0.25):
            pairs.append((np.array([xv + 2.2, yv, 0.0]), np.array([xv, yv, 0.3 if
                          abs(xv) < 0.5 and abs(yv) < 0.5 else 0.0])))
    e = doc.clone_stroke(pairs, r, copy_params(), view=VIEW,
                         picked=[True] * len(pairs))
    left = int((doc.alive[: s.n] & is_sign).sum())
    new = added_of(doc, e)
    high = int(((new.xyz[:, 2] > 0.1) & (new.alpha > 0.1)).sum())
    assert left < 0.03 * is_sign.sum(), "%d of %d sign splats left" % (left, is_sign.sum())
    assert high < 0.02 * new.n, "%d new splats float at the sign's height" % high
    print("    sign        %d of %d sign splats left, %d new at sign height"
          % (left, is_sign.sum(), high))


def test_params_ignore_unknown_keys():
    """The app may send options the engine does not know -- a volume, say."""
    p = cl.clone_params({"color_match": 0.3, "volume": {"capsules": []}, "feather": 9})
    assert p.color_match == 0.3 and p.feather == 1.0


def test_exclude_is_neither_copied_nor_counted():
    """What the crop hides is not copied, and not reported as removed."""
    s = make_surface(lambda x, y: np.zeros_like(x), n=30000, extent=6.0)
    m = stripes(s, 2.0, 2.6, (0.1, 0.9, 0.1))      # half the source disc
    doc = Document(s)
    hidden = np.where(m)[0]
    e = doc.clone_stroke([(np.array([2.0, 0.0, 0.0]), np.array([-2.0, 0.0, 0.0]))],
                         0.45, copy_params(color_match=0.0), view=VIEW,
                         picked=[True], exclude=hidden)
    assert not np.isin(e.removed, hidden).any()
    new = added_of(doc, e) if e.added else None
    if new is not None:
        g = new.rgb[new.alpha > 0.3]
        assert np.mean((g[:, 1] > 0.6) & (g[:, 0] < 0.4)) < 0.02


def translucent(s, seed=1):
    """Make half the splats faint and darker, as on the hut's siding."""
    rng = np.random.default_rng(seed)
    faint = rng.random(s.n) < 0.6
    s.opacity[faint] = np.float32(-1.8)                      # alpha 0.14
    s.opacity[~faint] = np.float32(1.5)                      # alpha 0.82
    rgb = np.where(faint[:, None], 0.45, 0.75) * np.array([1.0, 0.95, 0.85])
    s.f_dc[:] = ((rgb - 0.5) / SH_C0).astype(np.float32)
    return s


def seen_rgb(s, centre, r):
    """Opacity-weighted colour within ``r`` of ``centre`` -- what is seen."""
    m = np.linalg.norm(s.xyz[:, :2] - np.asarray(centre)[:2], axis=1) < r
    a = s.alpha[m]
    return (s.rgb[m] * a[:, None]).sum(0) / a.sum()


def test_tone_match_keeps_a_translucent_patch_as_it_was():
    """Stamped onto itself with full tone match, a translucent patch stays put.

    The faint splats are darker than the opaque ones, as on the hut. The
    tone match used to compare each source splat, faint ones included, with
    only the opaque target splats around it, unweighted: the copy came out
    brighter than what it replaced -- 10 grey levels on the hut for a patch
    stamped onto itself, 18 to 24 along a stroke.
    """
    s = translucent(make_surface(lambda x, y: np.zeros_like(x), n=30000, extent=6.0))
    p = np.array([1.0, 0.5, 0.0])
    res = cl.clone(s, p, p, 0.6, cl.CloneParams(color_match=1.0), view=VIEW)
    out = res.apply_to(s)
    before, after = seen_rgb(s, p, 0.5), seen_rgb(out, p, 0.5)
    lift = float(np.mean(after - before))
    assert abs(lift) < 0.01, "tone shifted by %.3f" % lift
    # and between two places of the same surface, likewise
    q = np.array([-2.0, -1.0, 0.0])
    res = cl.clone(s, p, q, 0.6, cl.CloneParams(color_match=1.0), view=VIEW)
    lift2 = float(np.mean(seen_rgb(res.apply_to(s), q, 0.5) - seen_rgb(s, q, 0.5)))
    assert abs(lift2) < 0.02, "tone shifted by %.3f" % lift2
    print("    tone        own %+.3f, neighbour %+.3f (before up to +0.04)" % (lift, lift2))


def test_tone_match_follows_the_target_not_the_source():
    """A darker target pulls the copy down to its own tone, seen-weighted."""
    s = translucent(make_surface(lambda x, y: np.zeros_like(x), n=30000, extent=6.0))
    dark = s.xyz[:, 0] < -1.0
    s.f_dc[dark] = s.f_dc[dark] - np.float32(0.15 / SH_C0)
    src, tgt = np.array([2.0, 0.0, 0.0]), np.array([-3.0, 0.0, 0.0])
    res = cl.clone(s, src, tgt, 0.6, cl.CloneParams(color_match=1.0), view=VIEW)
    got = float(np.mean(seen_rgb(res.apply_to(s), tgt, 0.5)))
    want = float(np.mean(seen_rgb(s, tgt, 0.5)))
    assert abs(got - want) < 0.02, "copy %.3f, target %.3f" % (got, want)
    print("    match       copy %.3f, target %.3f" % (got, want))


def test_copied_splats_stay_near_the_brush():
    """Long splats copied from the source do not reach far past the edge.

    On the hut a plank splat a brush radius long, centred inside the brush,
    painted the source's planks over the target's well outside it: doubled
    plank lines and streaks along the stroke.
    """
    s = make_surface(lambda x, y: np.zeros_like(x), n=30000, extent=6.0)
    rng = np.random.default_rng(3)
    long_ = rng.random(s.n) < 0.2
    s.scale[long_, 0] = np.float32(np.log(0.5))              # 0.5 long, along x
    s.rot[long_] = np.array([1, 0, 0, 0], np.float32)
    r = 0.6
    tgt = np.array([-2.0, 0.0, 0.0])
    # without the old ring kept at the seam: only copies in ``added``
    res = cl.clone(s, np.array([2.0, 0.0, 0.0]), tgt, r, cl.CloneParams(jigsaw=False),
                   view=VIEW)
    new = res.added
    from splatcore import shtools
    M = shtools.quat_to_matrix(new.quat)
    inplane = np.sqrt(np.clip(1.0 - M[:, 2, :] ** 2, 0.0, 1.0))   # per axis
    reach = (2.0 * new.sigma * inplane).max(1)
    d = np.linalg.norm(new.xyz[:, :2] - tgt[:2], axis=1)
    copied = d <= r
    worst = float(np.max((d + reach)[copied]))
    assert worst <= r * (1.0 + cl.OVERHANG) + 1e-3, "reaches %.2f radii" % (worst / r)
    print("    reach       at most %.2f radii" % (worst / r))


def sign_on_a_wall(seed=1, second=False):
    """A wall with a long red bar standing proud of it, as the hut's cross.

    Behind the bar lies what no camera saw: wall-level splats tinted pink by
    the sign, and a dark red blob a little behind the wall. On the hut both
    showed through the copy as a smeared, pinkish band.
    """
    s = make_surface(lambda x, y: np.zeros_like(x), n=60000, extent=4.0, seed=seed)
    rng = np.random.default_rng(seed + 10)
    k = 6000
    bar = make_surface(lambda x, y: np.zeros_like(x), n=k, extent=1.0, seed=seed + 2)
    u = rng.uniform(-1.5, 1.5, k)
    v = rng.uniform(-0.25, 0.25, k)
    side = rng.random(k) < 0.3
    z = np.where(side, rng.uniform(0.0, 0.3, k), 0.3)
    v = np.where(side, np.where(rng.random(k) < 0.5, -0.25, 0.25), v)
    bar.xyz = np.column_stack([u, v, z]).astype(np.float32)
    bar.f_dc[:] = ((np.array([0.85, 0.12, 0.12]) - 0.5) / SH_C0).astype(np.float32)
    m = 1500
    pink = make_surface(lambda x, y: np.zeros_like(x), n=m, extent=1.0, seed=seed + 3)
    pink.xyz = np.column_stack([rng.uniform(-1.5, 1.5, m), rng.uniform(-0.25, 0.25, m),
                                rng.uniform(-0.02, 0.0, m)]).astype(np.float32)
    pink.f_dc[:] = ((np.array([0.8, 0.5, 0.5]) - 0.5) / SH_C0).astype(np.float32)
    deep = make_surface(lambda x, y: np.zeros_like(x), n=300, extent=1.0, seed=seed + 4)
    deep.xyz = np.column_stack([rng.uniform(-1.2, 1.2, 300), rng.uniform(-0.2, 0.2, 300),
                                np.full(300, -0.15)]).astype(np.float32)
    deep.f_dc[:] = ((np.array([0.4, 0.12, 0.12]) - 0.5) / SH_C0).astype(np.float32)
    n0 = s.n
    s = s.concat(bar).concat(pink).concat(deep)
    kind = np.zeros(s.n, np.int8)
    kind[n0:n0 + k] = 1
    kind[n0 + k:n0 + k + m] = 2
    kind[n0 + k + m:] = 3
    if second:
        # a second, narrow bar of the same sign above the first
        b2 = bar.take(np.arange(k))
        b2.xyz = b2.xyz.copy()
        b2.xyz[:, 1] = 0.875 + 0.5 * b2.xyz[:, 1]
        s = s.concat(b2)
        kind = np.concatenate([kind, np.full(k, 1, np.int8)])
    return s, kind


def chroma(rgb):
    return rgb[:, 0] / np.maximum(rgb.sum(1), 1e-6)


def test_stroke_along_a_sign_leaves_clean_wall():
    """A stroke along a sign takes it, what lay behind it, and stays wall-coloured.

    The hut's crossbar: after the first dab the rest of the brush is the
    stroke's own copy, the sign no longer stood proud of enough of it, and
    later dabs took the brush's own plane, kept the sign's sides and the pink
    wall behind it, and matched their tone to it -- a smeared pink band.
    """
    s, kind = sign_on_a_wall()
    doc = Document(s)
    r = 0.5
    pairs = [(np.array([x, 2.0, 0.0]), np.array([x, 0.0, 0.3]))
             for x in np.arange(-1.2, 1.21, r / 4)]
    e = doc.clone_stroke(pairs, r, copy_params(), view=VIEW,
                         picked=[True] * len(pairs))
    alive = doc.alive[: s.n]
    xy = s.xyz[:, :2]
    core = (np.abs(xy[:, 0]) < 1.1) & (np.abs(xy[:, 1]) < 0.3)
    left = {kk: int((alive & core & (kind == kk)).sum()) for kk in (1, 2, 3)}
    tot = {kk: int((core & (kind == kk)).sum()) for kk in (1, 2, 3)}
    new = added_of(doc, e)
    wall = s.take(np.where(kind == 0)[0])
    dc = float(np.median(chroma(new.rgb[new.alpha > 0.3])) - np.median(chroma(wall.rgb)))
    assert left[1] < 0.03 * tot[1], "sign: %d of %d left" % (left[1], tot[1])
    assert left[2] < 0.05 * tot[2], "pink behind: %d of %d left" % (left[2], tot[2])
    assert left[3] < 0.05 * tot[3], "red blob behind: %d of %d left" % (left[3], tot[3])
    assert abs(dc) < 0.01, "copy tinted: chroma %+.3f" % dc
    print("    along       sign %d/%d, pink %d/%d, blob %d/%d left, hue %+.3f"
          % (left[1], tot[1], left[2], tot[2], left[3], tot[3], dc))


def test_source_brings_no_sign_back():
    """A source that reaches over the sign being painted out leaves it behind.

    The hut's cross: strokes along one arm take their source beside it,
    across another part of the same sign. That is what the user is
    removing; copied, it came back beside the stroke.
    """
    s, kind = sign_on_a_wall(seed=4, second=True)
    doc = Document(s)
    r = 0.5
    pairs = [(np.array([x, 1.25, 0.0]), np.array([x, 0.0, 0.3]))
             for x in np.arange(-1.2, 1.21, r / 4)]
    e = doc.clone_stroke(pairs, r, copy_params(color_match=0.0), view=VIEW,
                         picked=[True] * len(pairs))
    new = added_of(doc, e)
    red = (new.rgb[:, 0] > 0.6) & (new.rgb[:, 1] < 0.3) & (new.alpha > 0.1)
    upper = (doc.alive[: s.n] & (kind == 1) & (s.xyz[:, 1] > 0.7)).sum()
    assert red.mean() < 0.01, "%.1f %% of the copy is the sign" % (100 * red.mean())
    assert upper > 0.95 * (kind == 1).sum() / 2, "the source's own sign was taken"
    print("    source      %.1f %% of the copy from the sign" % (100 * red.mean()))

def test_planks_line_up_across_the_seam():
    """A source a few centimetres off across the planks is nudged back.

    The crosshair keeps its offset in screen pixels, and on a wall seen at
    an angle a straight offset along the planks on screen is a few
    millimetres across them in the scan: every plank edge stepped at the
    stroke's border. The stroke lines its source up with the wall around it.
    """
    s = make_surface(lambda x, y: np.zeros_like(x), n=80000, extent=4.0, seed=6)
    s.scale[:] = np.float32(np.log(0.03))
    dark = np.mod(s.xyz[:, 1], 0.5) < 0.08
    rgb = np.where(dark[:, None], 0.25, 0.8) * np.array([1.0, 0.95, 0.85])
    s.f_dc[:] = ((rgb - 0.5) / SH_C0).astype(np.float32)
    r = 0.5
    off = np.array([2.0, 0.09, 0.0])                  # 9 cm across the planks
    pairs = [(np.array([x, 0.25, 0.0]) + off, np.array([x, 0.25, 0.0]))
             for x in np.arange(-1.0, 1.01, r / 4)]
    res = {}
    for align in (False, True):
        doc = Document(s)
        e = doc.clone_stroke(pairs, r, copy_params(color_match=0.0, align=align),
                             view=VIEW, picked=[True] * len(pairs))
        new = added_of(doc, e)
        m = new.alpha > 0.3
        want = np.mod(new.xyz[m, 1], 0.5) < 0.08
        got = new.rgb[m, 0] < 0.5
        res[align] = float(np.mean(want == got))
    assert res[True] > 0.97 and res[True] > res[False] + 0.05, res
    print("    joints      %.0f %% of the copy matches the wall (without aligning %.0f %%)"
          % (100 * res[True], 100 * res[False]))


# ---------------------------------------------------------------------------
# The picture stamp (`imstamp.py`): the stroke as one picture, lifted to splats.

def _gpu():
    from splatcore import gpu
    ok, why = gpu.available()
    if not ok:
        print("    (skipped: %s)" % why)
    return ok


def _beside(xyz, pairs, axes=(0, 1)):
    """Distance of points to the nearest target of a stroke, across ``axes``."""
    tg = np.array([t for _, t in pairs])[:, list(axes)]
    return np.min(np.linalg.norm(xyz[:, None, list(axes)] - tg[None], axis=2), axis=1)


def floor_stroke(dx=-4.0):
    """A flat floor, a blue band where the crosshair runs, a red one beside it.

    Dense enough to be closed: scattered thinner, the floor has gaps a few
    splats wide, and where the source shows nothing nothing is stamped.
    """
    s = make_surface(lambda x, y: np.zeros_like(x), n=90000, extent=6.0)
    stripes(s, 1.5, 2.5, (0.1, 0.2, 0.9))
    stripes(s, 3.0, 4.0, (0.9, 0.1, 0.1))
    pairs = [(np.array([2.0, -1.0 + 0.3 * k, 0.0]), np.array([2.0 + dx, -1.0 + 0.3 * k, 0.0]))
             for k in range(7)]
    return s, pairs


def test_picture_stamp_takes_from_the_crosshair_and_lands_under_the_brush():
    """The brush's middle shows the source; what was there is gone; one skin at the floor's height."""
    if not _gpu():
        return
    s, pairs = floor_stroke()
    doc = Document(s)
    e = doc.clone_stroke(pairs, 0.45, picture_params(color_match=0.0), view=VIEW,
                         picked=[True] * len(pairs))
    assert e.info["engine"] == "image", e.info
    new = added_of(doc, e)
    d = _beside(new.xyz, pairs)
    mid = d < 0.45 * 0.6                             # the brush at full strength
    rgb = new.rgb[mid]
    blue = float(np.mean((rgb[:, 2] > 0.6) & (rgb[:, 0] < 0.4)))
    red = float(np.mean((rgb[:, 0] > 0.6) & (rgb[:, 2] < 0.4)))
    assert blue > 0.9 and red < 0.02, "blue %.2f red %.2f" % (blue, red)
    assert np.mean(d <= 0.45 * 1.15) > 0.97, "landed beside the stroke"
    # on the surface: the floor lies at z = 0, a splat spacing is about 0.03
    z = new.xyz[:, 2]
    lies = (float(np.median(z)), float(np.percentile(np.abs(z), 95)))
    assert abs(lies[0]) < 0.01 and lies[1] < 0.03, lies
    # no second skin: under the brush's middle nothing of the old floor is left
    old = _beside(s.xyz, pairs) < 0.45 * 0.6
    left = float(doc.alive[:s.n][old].mean())
    assert left < 0.03, "%.0f %% of the old surface left under the stroke" % (100 * left)
    # what the app and its self-test read
    i = e.info
    assert i["landed"] >= 0.97 and i["dabs"] == len(pairs) and i.get("skipped", 0) == 0, i
    near = lambda a, b: float(np.linalg.norm(np.array(a) - np.array(b)))      # noqa: E731
    assert near(i["newRgb"], i["srcRgb"]) < near(i["newRgb"], i["oldRgb"]), i
    print("    image       %.0f %% from the cross, %.1f %% from the stray path, height %+.4f, old surface left %.1f %%"
          % (100 * blue, 100 * red, np.median(z), 100 * left))


def test_picture_stamp_sits_on_a_curved_surface():
    """From the crown of a cylinder to 40 degrees round its side: turned with it, and on it."""
    if not _gpu():
        return
    R = 3.0
    rng = np.random.default_rng(3)
    n = 40000
    th = rng.uniform(-1.2, 1.2, n)
    yy = rng.uniform(-3, 3, n)
    s = make_surface(lambda x, y: np.zeros_like(x), n=n)
    s.xyz = np.column_stack([R * np.sin(th), yy, R * np.cos(th)]).astype(np.float32)
    doc = Document(s)
    a = np.radians(40)
    pairs = [(np.array([0.0, y, R]), np.array([R * np.sin(a), y, R * np.cos(a)]))
             for y in np.arange(-0.4, 0.41, 0.2)]
    view = -pairs[2][1] / np.linalg.norm(pairs[2][1])            # looking at the target
    e = doc.clone_stroke(pairs, 0.5, picture_params(), view=view, picked=[True] * len(pairs))
    assert e.info["engine"] == "image", e.info
    new = added_of(doc, e)
    off = np.abs(np.linalg.norm(new.xyz[:, [0, 2]], axis=1) - R)
    lies = (float(np.median(off)), float(np.percentile(off, 95)))
    assert lies[0] < 0.02 and lies[1] < 0.06, "%.3f off the surface (95 %%: %.3f)" % lies
    assert abs(e.info["rotation"] - 40.0) < 6.0, e.info["rotation"]
    print("    image       turned %.1f degrees (surface 40), %.3f off the surface"
          % (e.info["rotation"], np.median(off)))


def test_picture_stamp_onto_itself_changes_nothing_to_see():
    """Stamped onto its own place a surface keeps its height and its pattern."""
    if not _gpu():
        return
    s = make_surface(lambda x, y: 0.15 * np.sin(x), n=90000, extent=6.0)
    doc = Document(s)
    pairs = [(np.array([x, 0.5, 0.15 * np.sin(x)]),) * 2 for x in np.arange(-0.6, 0.61, 0.2)]
    e = doc.clone_stroke(pairs, 0.6, picture_params(color_match=1.0), view=VIEW,
                         picked=[True] * len(pairs))
    assert e.info["engine"] == "image", e.info
    new = added_of(doc, e)
    dz = new.xyz[:, 2] - 0.15 * np.sin(new.xyz[:, 0])
    lies = (float(np.median(dz)), float(np.percentile(np.abs(dz), 95)))
    assert abs(lies[0]) < 0.01 and lies[1] < 0.03, lies
    # the floor's own pattern (make_surface: period 1.1), where it was, in
    # the colour the new splats show along the view (not their DC alone)
    from splatcore import shtools
    xy = new.xyz[:, :2].astype(np.float64)
    tone = 0.5 + 0.22 * np.sin(xy[:, 0] * 2 * np.pi / 1.1) * np.cos(xy[:, 1] * 2 * np.pi / 1.1)
    K = new.f_rest.shape[1]
    Y = shtools.sh_basis(np.asarray(VIEW, float)[None], 3)[0, 1:K + 1]
    seen = 0.5 + SH_C0 * new.f_dc[:, 0] + np.einsum("k,nk->n", Y, new.f_rest[:, :, 0].astype(np.float64))
    r = float(np.corrcoef(tone, seen)[0, 1])
    assert r > 0.85, "pattern kept to %.2f" % r
    print("    image       onto itself: height %+.4f, pattern kept to %.2f" % (np.median(dz), r))


def test_picture_stamp_is_one_undo_step():
    """A stroke is one edit: one undo takes all of it back, redo brings exactly it."""
    if not _gpu():
        return
    s, pairs = floor_stroke()
    doc = Document(s)
    n0, alive0 = doc.n_total, doc.n_alive
    e = doc.clone_stroke(pairs, 0.45, picture_params(), view=VIEW, picked=[True] * len(pairs))
    assert e.info["engine"] == "image" and len(doc.undo_stack) == 1
    assert len(e.removed) and int(e.removed.max()) < n0, "the stroke removed splats it had appended itself"
    assert doc.n_alive == alive0 - len(e.removed) + e.added
    after, tail = doc.n_alive, added_of(doc, e).xyz.copy()
    doc.undo()
    assert doc.n_alive == alive0 and doc.n_total == n0 and doc.alive.all()
    doc.redo()
    assert doc.n_alive == after and np.array_equal(added_of(doc, e).xyz, tail)
    print("    image       one step: -%d / +%d, redo exact" % (len(e.removed), e.added))


def test_picture_stamp_falls_back_to_the_copying_stamp():
    """When the picture stamp cannot do a stroke, the copying stamp does it, and says why."""
    if not _gpu():
        return
    from splatcore import imstamp
    s, pairs = floor_stroke()
    run = imstamp.stamp

    def broken(*a, **k):
        raise RuntimeError("out of memory on the graphics card")
    imstamp.stamp = broken
    try:
        doc = Document(s)
        e = doc.clone_stroke(pairs, 0.45, picture_params(), view=VIEW, picked=[True] * len(pairs))
    finally:
        imstamp.stamp = run
    assert e.info["engine"] == "copy" and "memory" in e.info["imageFallback"], e.info
    assert len(doc.undo_stack) == 1 and e.added > 0
    # without a line of sight there is no picture: the copying stamp, and no complaint
    doc = Document(s)
    e = doc.clone_stroke(pairs, 0.45, cl.CloneParams())
    assert e.info["engine"] == "copy" and "imageFallback" not in e.info, e.info
    # and where neither can work nothing is changed: a source in empty space
    doc = Document(s)
    off = [(a + np.array([50.0, 0.0, 0.0]), b) for a, b in pairs]
    try:
        doc.clone_stroke(off, 0.45, picture_params(), view=VIEW, picked=[True] * len(off))
        raise AssertionError("a stroke without a source went through")
    except ValueError:
        pass
    assert doc.n_alive == s.n and doc.generated.n == 0 and not doc.undo_stack
    print("    image       fallback: copying stamp, reason reported; without a source nothing changes")


def test_picture_stamp_runs_the_surface_on_under_a_sign():
    """A slab standing proud of a wall goes, and the copy lies on the wall, not on the slab."""
    if not _gpu():
        return
    s = make_surface(lambda x, y: np.zeros_like(x), n=40000, extent=4.0, seed=1)
    rng = np.random.default_rng(5)
    k = 4000
    sign = make_surface(lambda x, y: np.zeros_like(x), n=k, extent=0.5, seed=2)
    p = sign.xyz.astype(np.float64)
    face = np.column_stack([p[:, 0], p[:, 1], np.full(k, 0.3)])
    side_t = rng.uniform(0, 0.3, k)
    edge = rng.integers(0, 4, k)
    u = rng.uniform(-0.5, 0.5, k)
    sides = np.where(edge[:, None] == 0, np.column_stack([np.full(k, -0.5), u, side_t]),
            np.where(edge[:, None] == 1, np.column_stack([np.full(k, 0.5), u, side_t]),
            np.where(edge[:, None] == 2, np.column_stack([u, np.full(k, -0.5), side_t]),
                     np.column_stack([u, np.full(k, 0.5), side_t]))))
    slab = sign.concat(sign.take(np.arange(k)))
    slab.xyz = np.concatenate([face, sides]).astype(np.float32)
    slab.f_dc[:] = ((np.array([0.9, 0.1, 0.1]) - 0.5) / SH_C0).astype(np.float32)
    s = s.concat(slab)
    is_sign = np.zeros(s.n, bool)
    is_sign[-2 * k:] = True
    doc = Document(s)
    r = 0.5
    pairs = []
    for yv in np.arange(-0.6, 0.61, 0.2):
        for xv in np.arange(-0.6, 0.61, 0.25):
            pairs.append((np.array([xv + 2.2, yv, 0.0]), np.array([xv, yv, 0.3 if
                          abs(xv) < 0.5 and abs(yv) < 0.5 else 0.0])))
    e = doc.clone_stroke(pairs, r, picture_params(), view=VIEW, picked=[True] * len(pairs))
    assert e.info["engine"] == "image", e.info
    left = int((doc.alive[: s.n] & is_sign).sum())
    new = added_of(doc, e)
    high = int((new.xyz[:, 2] > 0.1).sum())
    assert left < 0.03 * is_sign.sum(), "%d of %d sign splats left" % (left, is_sign.sum())
    assert high < 0.02 * new.n, "%d of %d new splats at the sign's height" % (high, new.n)
    print("    image       sign: %d of %d splats left, %d of %d new ones at sign height"
          % (left, is_sign.sum(), high, new.n))


def test_picture_stamp_keeps_an_edge_under_the_stroke():
    """Stamped across the ridge of a roof, the new skin lies on both sides and the ridge stays sharp.

    The base under a stroke used to be a stiff plate fitted to what lay
    within a quarter radius of the surface continued from around it: a ridge
    under the stroke stood further off, counted as a thing on the surface,
    and was cut off.
    """
    if not _gpu():
        return
    roof = lambda x, y: -np.abs(x)                   # noqa: E731  (two sides at 45 degrees, ridge along y)
    s = make_surface(roof, n=120000, extent=6.0)
    doc = Document(s)
    pairs = [(np.array([x + 3.0, 0.0, -abs(x + 3.0)]), np.array([x, 0.0, -abs(x)]))
             for x in np.arange(-0.45, 0.46, 0.15)]
    e = doc.clone_stroke(pairs, 0.6, picture_params(), view=VIEW, picked=[True] * len(pairs))
    assert e.info["engine"] == "image", e.info
    new = added_of(doc, e)
    new = new.take(np.exp(new.scale).max(1) < 0.05)   # the stamp's own skin, not the coarse layer behind it
    off = new.xyz[:, 2] + np.abs(new.xyz[:, 0])       # above (+) or below (-) the roof
    lies = (float(np.median(np.abs(off))), float(np.percentile(np.abs(off), 95)))
    assert lies[0] < 0.03 and lies[1] < 0.09, "%.3f off the roof (95 %%: %.3f)" % lies
    ridge = np.abs(new.xyz[:, 0]) < 0.08
    cut = float(-np.median(off[ridge]))
    assert ridge.sum() > 50 and cut < 0.06, "the ridge is cut by %.3f" % cut
    both = (float(np.mean(new.xyz[:, 0] < -0.2)), float(np.mean(new.xyz[:, 0] > 0.2)))
    assert min(both) > 0.15, "one side got %.0f %% of the stamp" % (100 * min(both))
    print("    image       across the ridge: %.3f off the roof, ridge cut by %.3f, sides %.0f / %.0f %%"
          % (lies[0], cut, 100 * both[0], 100 * both[1]))


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
    print("all clone tests passed" if not fails else "%d FAILURES" % fails)
    sys.exit(1 if fails else 0)
