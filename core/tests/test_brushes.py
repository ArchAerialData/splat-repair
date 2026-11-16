"""Shadow and solidify brushes on synthetic surfaces where the answer is known.

Run directly: ``core/.venv/Scripts/python.exe core/tests/test_brushes.py``.
The tests on the real scan, with their numbers, are ``bench_shadow.py`` and
``bench_solid.py``; these are the fast ones that pin the behaviour.
"""
import os
import sys

import numpy as np

sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))
sys.path.insert(0, os.path.dirname(__file__))
from splatcore.io import SplatSet, SH_C0                         # noqa: E402
from splatcore import repair as rp, unshadow as us              # noqa: E402
from test_repair import make_surface                            # noqa: E402


def _shadow(s, gain, r_core=1.2, pen=0.4):
    r = np.hypot(s.xyz[:, 0], s.xyz[:, 1])
    t = np.clip((r - r_core) / pen, 0, 1)
    m = 1.0 - t * t * (3 - 2 * t)
    g = 1.0 - m[:, None] * (1.0 - np.asarray(gain)[None])
    rgb = 0.5 + SH_C0 * s.f_dc.astype(np.float64)
    out = SplatSet(xyz=s.xyz, f_dc=((rgb * g - 0.5) / SH_C0).astype(np.float32),
                   f_rest=(s.f_rest * g[:, None, :]).astype(np.float32),
                   opacity=s.opacity, scale=s.scale, rot=s.rot)
    return out, m


def test_shadow_gain_is_recovered():
    """The three numbers come back, and what was lit is left alone.

    The texture repeats every 0.3 units, about six grid cells, like the
    tread. At the default 1.1 its wavelength is 22 cells, a five-cell strip
    samples one phase band of it instead of the texture, and the estimate
    is off by the pattern rather than by the method.
    """
    s = make_surface(lambda x, y: np.zeros_like(x), n=40000, extent=5.0,
                     period=0.3)
    gain = np.array([0.40, 0.45, 0.55])
    t, m = _shadow(s, gain)
    stroke = np.hypot(s.xyz[:, 0], s.xyz[:, 1]) < 2.2
    res = us.unshadow(t, stroke)
    got = np.array(res.info["gain"])
    assert np.all(np.abs(np.log(got) - np.log(gain)) < 0.08), (got, gain)
    fixed = 0.5 + SH_C0 * t.f_dc.astype(np.float64)
    fixed[res.removed] = 0.5 + SH_C0 * res.added.f_dc.astype(np.float64)
    truth = 0.5 + SH_C0 * s.f_dc.astype(np.float64)
    core = stroke & (m > 0.98)
    lit = stroke & (m < 0.02)
    e_core = float(np.median(np.abs(np.log(fixed[core] / truth[core]))))
    e_lit = float(np.median(np.abs(np.log(fixed[lit] / truth[lit]))))
    assert e_core < 0.08 and e_lit < 0.03, (e_core, e_lit)
    # the view-dependent bands came back with the colour
    rest = t.f_rest.astype(np.float64).copy()
    rest[res.removed] = res.added.f_rest
    ratio = np.median(np.abs(rest[core]).sum((1, 2)) / np.abs(s.f_rest[core]).sum((1, 2)))
    assert abs(ratio - 1.0) < 0.1, ratio
    print("    shadow      gain %s (true %s), core %.3f, lit %.3f"
          % (got.tolist(), gain.tolist(), e_core, e_lit))


def test_shadow_brush_refuses_without_a_shadow():
    """Painted over evenly lit surface, it says so instead of brightening."""
    s = make_surface(lambda x, y: np.zeros_like(x), n=40000, extent=5.0)
    stroke = np.hypot(s.xyz[:, 0], s.xyz[:, 1]) < 2.0
    try:
        res = us.unshadow(s, stroke)
    except rp.RepairError as e:
        print("    none        refused: %s" % e)
        return
    lift = float(np.max(1.0 / np.array(res.info["gain"])))
    assert lift < 1.05, "brightened an unshadowed surface by x%.2f" % lift
    print("    none        lift x%.3f, practically nothing" % lift)


def test_lift_always_brightens_and_keeps_texture():
    """"Lighten": the set factor in the middle, soft past the rim, hue kept.

    Where detection refuses -- evenly lit, nothing to find -- the manual mode
    still acts, by exactly what was asked, and nothing is pushed past white.
    A dodge lifts brightness more than colour: the channels' mean rises by
    the gain, the colour's distance from grey by the gain to the power
    ``lift_chroma`` (0.25; times the whole gain the sole's brown went
    peach). The soft edge runs from a quarter of the half width inside the
    stroke to three quarters outside it, so no edge shows where it ended.
    """
    s = make_surface(lambda x, y: np.zeros_like(x), n=40000, extent=5.0, period=0.3)
    r = np.hypot(s.xyz[:, 0], s.xyz[:, 1])
    stroke = np.where(r < 2.0)[0]
    res = us.unshadow(s, stroke, us.ShadowParams(mode="lift", lift=1.5),
                      view=[0.0, 0.0, -1.0])
    assert res.info["mode"] == "lift" and res.info["lifted"] > 0.7 * len(stroke), res.info
    rgb0 = 0.5 + SH_C0 * s.f_dc[res.removed].astype(np.float64)
    rgb1 = 0.5 + SH_C0 * res.added.f_dc.astype(np.float64)
    Y0, Y1 = rgb0.mean(1), rgb1.mean(1)
    g = Y1 / np.maximum(Y0, 1e-6)
    rr = r[res.removed]
    mid = rr < 0.8
    room = rgb0.max(1) < 1.0 / 1.5
    assert abs(float(np.median(g[mid & room])) - 1.5) < 0.01, float(np.median(g[mid & room]))
    # hue kept: the colour's distance from grey keeps its direction, and
    # grows by the gain ** 0.25 only
    d0, d1 = rgb0 - Y0[:, None], rgb1 - Y1[:, None]
    big = mid & room & (np.linalg.norm(d0, axis=1) > 1e-3)
    cg = np.linalg.norm(d1[big], axis=1) / np.linalg.norm(d0[big], axis=1)
    assert np.allclose(cg, g[big] ** 0.25, atol=1e-3), (cg[:5], g[big][:5] ** 0.25)
    cos = (d0[big] * d1[big]).sum(1) / (np.linalg.norm(d0[big], axis=1) * np.linalg.norm(d1[big], axis=1))
    assert float(cos.min()) > 0.999, float(cos.min())
    # soft edge: lifted just outside the painted rim, less than inside,
    # nothing well outside
    G = np.ones(s.n)
    G[res.removed] = g
    rim = float(np.median(G[(r > 2.1) & (r < 2.4)]))
    far = float(np.median(G[(r > 3.8) & (r < 4.4)]))
    from splatcore import gpu
    if gpu.available()[0]:
        assert 1.03 < rim < 1.35 and far < 1.01, (rim, far)
    else:
        # the soft edge comes from the rendered picture; without a GPU the
        # lift stops at the stroke
        assert rim < 1.35 and far < 1.01, (rim, far)
    # nothing past white
    assert float(rgb1.max()) <= 1.0 + 1e-4, float(rgb1.max())
    print("    brighten    middle x%.3f, just outside x%.3f, far outside x%.3f, %d splats"
          % (float(np.median(g[mid & room])), rim, far, res.info["lifted"]))


def test_lift_goes_to_the_dark_part():
    """"Lighten" is a dodge: the stroke's middle by the factor, darker parts more.

    Lifting the dark part alone left the brush looking dead (x1.5 moved the
    sole's median pixel by 0.4 per cent); the dark part fully and the rest
    by half still read as x1.22. Now a power curve through the stroke's
    median and white (`_lift_map`). A disc of shadow (x0.5) inside a wider
    stroke whose middle is lit: the lit ring comes up by the factor, the
    shadow by more, at most the factor to the power 1.5 (x1.84). Needs the
    GPU (the picture); without it the lift is uniform and the test says so.
    """
    from splatcore import gpu
    if not gpu.available()[0]:
        print("    dark        skipped (no GPU)")
        return
    s = make_surface(lambda x, y: np.zeros_like(x), n=40000, extent=5.0, period=0.3)
    t, m = _shadow(s, [0.5, 0.5, 0.5], r_core=0.8, pen=0.2)
    r = np.hypot(s.xyz[:, 0], s.xyz[:, 1])
    stroke = np.where(r < 2.0)[0]
    res = us.unshadow(t, stroke, us.ShadowParams(mode="lift", lift=1.5),
                      view=[0.0, 0.0, -1.0])
    g = np.ones(t.n)
    g[res.removed] = ((0.5 + SH_C0 * res.added.f_dc.astype(np.float64)).mean(1)
                      / (0.5 + SH_C0 * t.f_dc[res.removed].astype(np.float64)).mean(1))
    dark = float(np.median(g[(r < 0.6)]))
    lit = float(np.median(g[(r > 1.2) & (r < 1.5)]))
    assert 1.65 < dark < 1.85 and 1.4 < lit < 1.56, (dark, lit)
    print("    dark        shadow x%.2f, light in the stroke x%.2f" % (dark, lit))


def test_reflection_is_taken_out_of_the_bands():
    """A coloured glare in the SH bands goes, from the view and beside it.

    The red cast the stand threw onto the sole lives in the view-dependent
    bands: the base colour there is plain sole. Here a red lobe towards the
    viewer is added to the bands inside a disc; the brush is painted wider.
    From straight on and from 30 degrees to the side the disc must look like
    the surface around it again, and nothing outside the disc changes.
    """
    from splatcore.shtools import sh_basis
    s = make_surface(lambda x, y: np.zeros_like(x), n=40000, extent=5.0,
                     period=0.3)
    r = np.hypot(s.xyz[:, 0], s.xyz[:, 1])
    glare = np.clip((1.1 - r) / 0.3, 0, 1)
    up = np.array([[0.0, 0.0, 1.0]])
    lobe = sh_basis(up, 3)[0, 1:]
    lobe = lobe / np.linalg.norm(lobe)
    t = s.take(np.arange(s.n))
    t.f_rest = t.f_rest.copy()
    t.f_rest[:, :, 0] += (0.6 * glare[:, None] * lobe[None]).astype(np.float32)
    stroke = r < 2.0
    view = np.array([0.0, 0.0, -1.0])
    res = us.unshadow(t, stroke, view=view)
    assert res.info["reflex"], res.info
    rest = t.f_rest.astype(np.float64).copy()
    rest[res.removed] = res.added.f_rest
    inner, outside = r < 0.8, r > 2.3
    errs = []
    for tilt in (0.0, 30.0):
        a = np.deg2rad(tilt)
        d = np.array([[np.sin(a), 0.0, -np.cos(a)]])
        Y = sh_basis(-d, 3)[0, 1:]
        seen = np.einsum("k,nkc->nc", Y, rest)
        truth = np.einsum("k,nkc->nc", Y, s.f_rest.astype(np.float64))
        errs.append(float(np.abs(seen[inner] - truth[inner]).mean()))
    before = float(np.abs(0.6 * lobe @ sh_basis(up, 3)[0, 1:]))
    assert max(errs) < 0.1 * before, (errs, before)
    assert not np.isin(np.where(outside)[0], res.removed).any()
    dc = t.f_dc.copy()
    dc[res.removed] = res.added.f_dc
    assert np.abs(dc - t.f_dc).max() < 0.02
    print("    reflection  remainder %.3f / %.3f (before %.2f), %d cells"
          % (errs[0], errs[1], before, res.info["reflexCells"]))


def test_solidify_restores_coverage_and_leaves_haze():
    """A thinned patch comes back as solid as its surroundings; haze stays faint.

    Four layers of surface, a disc of them twenty times too transparent, and
    a sheet of faint haze floating in front of the disc that the stroke also
    catches. The surface must close up; the haze must not become a crust.
    """
    from splatcore import solidify as so
    layers = [make_surface(lambda x, y, k=k: np.full_like(x, -0.03 * k),
                           n=12000, extent=4.0, seed=k) for k in range(4)]
    s = layers[0]
    for l_ in layers[1:]:
        s = s.concat(l_)
    r = np.hypot(s.xyz[:, 0], s.xyz[:, 1])
    disc = r < 1.0
    a = np.clip(s.alpha * np.where(disc, 0.05, 1.0), 1e-4, 1 - 1e-6)
    s.opacity = np.log(a / (1 - a)).astype(np.float32)
    haze = make_surface(lambda x, y: np.full_like(x, 0.4), n=800, extent=1.0, seed=9)
    ha = np.full(haze.n, 0.05)
    haze.opacity = np.log(ha / (1 - ha)).astype(np.float32)
    n0 = s.n
    s = s.concat(haze)
    stroke = np.zeros(s.n, bool)
    stroke[:n0] = disc
    stroke[n0:] = np.hypot(haze.xyz[:, 0], haze.xyz[:, 1]) < 1.0
    res = so.solidify(s, stroke, view=np.array([0.0, 0.0, -1.0]),
                      params=so.SolidParams(colour="material", resolution=160))
    before, after = res.info["coverageBefore"][0], res.info["coverageAfter"][0]
    assert after > 0.97 and after > before, (before, after)
    raised = set(res.removed.tolist())
    # No crust: haze may be touched at the stroke's very edge, where a cell
    # straddles inside and out, but none of it may become visible.
    haze_hit = res.removed >= n0
    haze_after = res.added.alpha[haze_hit]
    assert haze_hit.sum() <= 0.01 * stroke[n0:].sum(), "haze raised: %d" % haze_hit.sum()
    assert (haze_after < 0.1).all(), "haze made visible: %s" % haze_after.max()
    assert all(stroke[i] for i in raised), "something outside the stroke changed"
    print("    solidify    opacity %.3f -> %.3f, haze untouched, %d splats"
          % (before, after, len(raised)))


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
    print("all brush tests passed" if not fails else "%d FAILURES" % fails)
    sys.exit(1 if fails else 0)
