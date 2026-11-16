"""Tests for the coarse layer behind a new skin (`splatcore/lod.py`). Run as a script.

The skins are made by hand, as the picture engines lay them: one small flat
splat per pixel. Only the last test needs CUDA + gsplat (it goes through the
picture fill); without them it is skipped.
"""
import os
import sys

import numpy as np

HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, os.path.join(HERE, ".."))

from splatcore.io import SplatSet, SH_C0                      # noqa: E402
from splatcore import gpu                                     # noqa: E402
from splatcore.skin import normals, rotmat                    # noqa: E402
from splatcore.lod import backing, LodParams                  # noqa: E402
from splatcore.document import Document                       # noqa: E402
from splatcore.repair import RepairParams                     # noqa: E402

SP = 0.05                               # the scan's splat spacing
VIEW = [0.0, 0.0, -1.0]                 # looking down on the skin


def skin_disc(radius=14.0, per_sp=5.0, height=None, sigma=None, seed=0):
    """A round skin ``radius`` spacings wide, ``per_sp`` flat splats to a spacing, facing +z.

    Red where x < 0, blue where x > 0. ``height(x, y)``: its relief, in units.
    ``sigma``: the splats' sigma in the skin (spacings); as lifted from a picture if None.
    """
    rng = np.random.default_rng(seed)
    px = SP / per_sp
    g = np.arange(-radius * SP, radius * SP + 1e-9, px)
    x, y = [a.ravel() for a in np.meshgrid(g, g)]
    keep = np.hypot(x, y) < radius * SP
    x, y = x[keep] + rng.normal(0, 0.05 * px, keep.sum()), y[keep] + rng.normal(0, 0.05 * px, keep.sum())
    z = np.zeros_like(x) if height is None else height(x, y)
    n = len(x)
    col = np.where((x < 0)[:, None], np.array([[0.8, 0.2, 0.2]]), np.array([[0.2, 0.2, 0.8]]))
    st = 0.62 * px if sigma is None else sigma * SP
    q = np.zeros((n, 4), np.float32)
    q[:, 0] = 1.0
    return SplatSet(
        xyz=np.stack([x, y, z], 1).astype(np.float32), f_dc=((col - 0.5) / SH_C0).astype(np.float32),
        f_rest=np.zeros((n, 15, 3), np.float32), opacity=np.full(n, 3.5, np.float32),
        scale=np.log(np.tile([st, st, 0.3 * st], (n, 1))).astype(np.float32), rot=q)


def layers(skin, **kw):
    out, info = backing(skin, normals(skin, VIEW), SP, LodParams(**kw) if kw else None)
    assert out.n == info["added"] == sum(info["levels"]), info
    return out, info


def test_a_fine_skin_gets_coarse_layers():
    skin = skin_disc()
    out, info = layers(skin)
    lev = info["levels"]
    assert len(lev) == 4 and lev[0] > 300 and all(a > 2.5 * b for a, b in zip(lev[:-1], lev[1:]) if b), lev
    # about one splat per square spacing in the first layer, as large as its cell
    area = np.pi * 14.0 ** 2
    assert 0.6 < lev[0] / area < 1.4, lev[0] / area
    first = out.take(np.arange(lev[0]))
    big = np.sort(first.sigma, 1)[:, 2] / SP
    assert 0.45 < float(np.median(big)) < 0.75, float(np.median(big))
    # flat, facing as the skin does, under it, opaque
    assert (np.argmin(out.scale, 1) == 2).all() and (rotmat(out.quat)[:, 2, 2] > 0.99).all()
    assert float(out.xyz[:, 2].max()) < -0.3 * SP and float(out.alpha.min()) > 0.9
    # each carries the colour of the skin it stands for
    x = out.xyz[:, 0]
    assert (out.rgb[x < -4 * SP][:, 0] > 0.7).all() and (out.rgb[x > 4 * SP][:, 2] > 0.7).all()
    print("    layers      %s splats on %d fine ones, first layer sigma %.2f spacings" % (lev, skin.n, np.median(big)))


def test_it_never_reaches_past_the_skins_rim():
    """Close up the fine skin must be above all of a coarse splat: 2.5 sigma of it
    lie inside the skin (the rim is known to a spacing or so)."""
    skin = skin_disc()
    out, _ = layers(skin)
    reach = np.hypot(out.xyz[:, 0], out.xyz[:, 1]) + 2.5 * out.sigma.max(1)
    assert float(reach.max()) < (14.0 + 1.0) * SP, float(reach.max()) / SP
    # ... and the large ones in the middle are not all given up for it
    assert (out.sigma.max(1) > 2.0 * SP).sum() >= 3
    print("    rim         widest reach %.1f of 14 spacings" % (reach.max() / SP))


def test_it_lies_under_the_relief():
    """A groove in the skin: no coarse splat may stand in it."""
    groove = lambda x, y: np.where(np.abs(x) < 1.5 * SP, -1.5 * SP, 0.0)      # noqa: E731
    skin = skin_disc(height=groove)
    out, _ = layers(skin)
    assert out.n > 300
    R = rotmat(out.quat)
    X = skin.xyz.astype(np.float64)
    worst = np.inf
    for k in range(0, out.n, 7):
        d = X - out.xyz[k].astype(np.float64)
        h = d @ R[k][:, 2]
        lat = np.linalg.norm(d - np.outer(h, R[k][:, 2]), axis=1)
        m = lat < 2.0 * float(out.sigma[k].max())
        if m.sum() > 5:
            worst = min(worst, float(np.percentile(h[m], 5)))
    assert worst > 0.2 * SP, worst / SP
    print("    relief      the fine skin lies at least %.2f spacings above it everywhere" % (worst / SP))


def test_a_large_splat_lies_deep_under_the_skin():
    """A viewer draws splats in the order of their centres. Seen from a slant, a
    large flat splat just under small ones comes before those over its far
    half and shows through them: it has to lie about its own sigma deep."""
    out, _ = layers(skin_disc())
    depth, big = -out.xyz[:, 2].astype(np.float64), out.sigma.max(1)
    assert (depth > 1.24 * big - 1e-6).all() and float(depth.min()) > 0.39 * SP, \
        (float((depth / big).min()), float(depth.min()) / SP)
    assert float(depth.max()) > 3.0 * SP                    # (the largest ones really are down there)
    print("    depth       %.2f to %.1f spacings under the skin, at least %.2f sigma"
          % (depth.min() / SP, depth.max() / SP, (depth / big).min()))


def test_it_stays_above_what_lies_below():
    """The far side of a thin part three spacings under the skin: no coarse splat
    may lie in it or beyond it. There is room for small ones only."""
    from splatcore.skin import Cells
    skin = skin_disc()
    nrm = normals(skin, VIEW)
    grid = Cells(skin, nrm, SP)
    out, info = backing(skin, nrm, SP, grid=grid, floor=np.full(len(grid.centres), 3.0 * SP))
    depth = -out.xyz[:, 2].astype(np.float64)
    assert out.n > 300 and float(depth.max()) < (3.0 - 0.74) * SP, (out.n, float(depth.max()) / SP)
    assert float(out.sigma.max()) < 2.0 * SP
    free, _ = backing(skin, nrm, SP, grid=grid)
    assert free.n >= out.n and float(free.sigma.max()) > 2.5 * SP    # (with nothing below they are as large as their cells)
    print("    floor       with a back side at 3 spacings: deepest splat %.2f, largest %.2f spacings"
          % (depth.max() / SP, out.sigma.max() / SP))


def test_scan_sized_splats_need_none():
    out, info = layers(skin_disc(per_sp=1.0, sigma=0.9))
    assert out.n == 0, info
    print("    scan size   no layer")


def test_a_coarse_fill_gets_only_the_coarser_layers():
    """A large hole is lifted from a coarse picture: its splats are half a spacing
    already, and a layer of the same size would add nothing."""
    out, info = layers(skin_disc(radius=30.0, per_sp=1.16))
    lev = info["levels"]
    assert lev[0] == 0 and lev[1] > 100, lev
    print("    coarse skin layers %s" % lev)


def test_from_afar_the_first_layer_closes_the_surface():
    """What a viewer shows once it has dropped the fine splats: the first layer
    alone must cover the middle of the skin."""
    skin = skin_disc()
    out, info = layers(skin)
    first = out.take(np.arange(info["levels"][0]))
    rng = np.random.default_rng(1)
    a, r = rng.uniform(0, 2 * np.pi, 400), 9.0 * SP * np.sqrt(rng.uniform(0, 1, 400))
    P = np.stack([r * np.cos(a), r * np.sin(a)], 1)
    R = rotmat(first.quat)
    S = first.sigma
    d = P[:, None, :] - first.xyz[None, :, :2].astype(np.float64)
    u = d[..., 0] * R[None, :, 0, 0] + d[..., 1] * R[None, :, 1, 0]
    v = d[..., 0] * R[None, :, 0, 1] + d[..., 1] * R[None, :, 1, 1]
    alpha = first.alpha[None] * np.exp(-0.5 * ((u / S[None, :, 0]) ** 2 + (v / S[None, :, 1]) ** 2))
    cover = 1.0 - np.prod(1.0 - alpha, axis=1)
    assert float(np.percentile(cover, 5)) > 0.9, float(np.percentile(cover, 5))
    print("    from afar   coverage of the first layer: 5 %% of the spots below %.2f" % np.percentile(cover, 5))


def test_nothing_without_a_direction():
    skin = skin_disc(radius=6.0)
    out, info = backing(skin, None, SP)
    assert out.n == 0 and "why" in info
    print("    no view     no layer: %s" % info["why"])


def test_document_lays_the_layer_with_the_fill():
    """Through the picture fill: the layer comes with the fill, in its undo step;
    switched off, the fill is the engine's own."""
    if not gpu.available()[0]:
        print("    (skipped: %s)" % gpu.available()[1])
        return
    from test_imfill import surface, disc
    s = surface()
    idx = disc(s, 0.5)
    d = Document(s)
    d.clear_under = d.backing_layer = False
    e0 = d.repair(idx, RepairParams(seed=0, view=VIEW), objects=False)
    assert "lod" not in e0.info and "under" not in e0.info
    own = e0.added
    d.undo()
    d.backing_layer = True
    e = d.repair(idx, RepairParams(seed=0, view=VIEW), objects=False)
    lod = e.info["lod"]
    assert lod["added"] > 100 and e.added == own + lod["added"] == e.info["added"], (e.added, own, lod)
    assert len(e.removed) == len(e0.removed)
    # the engine's own splats come first, the layer after them
    tail = d.generated.take(np.arange(d.generated.n - lod["added"], d.generated.n))
    assert float(np.sort(tail.sigma, 1)[:, 2].min()) > 1.4 * float(np.median(d.generated.sigma[:own].max(1)))
    before = d.generated.xyz.copy()
    d.undo()
    assert d.n_alive == s.n and d.n_total == s.n
    d.redo()
    assert d.n_total == s.n + e.added and np.array_equal(d.generated.xyz, before)
    print("    document    %d + %d splats, one step back and forward again" % (own, lod["added"]))


def test_refining_a_quick_fill_keeps_its_layer():
    """`Document.refine_last` swaps the quick fill's splats for recoloured ones.
    The layer behind them stays the same splats, in the new colours, and the
    edit stays one step with every index where it was. (The copy engine's
    splats have the scan's sizes and get no layer; here they are given one
    by force, and the deep fill is stood in for.)"""
    from test_imfill import surface, disc
    from splatcore import skin as SK, viewfill
    from splatcore.repair import RepairResult
    real_finish, real_fill = SK.finish, viewfill.view_fill
    green = ((np.array([0.1, 0.9, 0.1]) - 0.5) / SH_C0).astype(np.float32)

    def stand_in(live, local, params, source="lama", deep=None, base=None, on_progress=None, **kw):
        added = base.added.copy()
        added.f_dc[:] = green
        return RepairResult(removed=base.removed, added=added, backing=base.backing,
                            info={**base.info, "deep": True}, context=base.context)
    SK.finish = lambda *a, **k: real_finish(*a, **{**k, "lod_params": LodParams(small=50.0, larger=0.2, inside=0.5)})
    viewfill.view_fill = stand_in
    try:
        s = surface()
        d = Document(s)
        d.fill_engine = "copy"
        e = d.repair(disc(s, 0.5), RepairParams(seed=0, view=VIEW), objects=False)
        nb = e.info["lod"]["added"]
        own = e.added - nb
        assert nb > 5 and own > 50, e.info["lod"]
        total, alive, pos = d.n_total, d.n_alive, d.generated.xyz.copy()
        e2 = d.refine_last(judge=False)
        assert e2 is e and e.info["refined"] is True, e.info.get("refine")
        assert (d.n_total, d.n_alive, e.added) == (total, alive, own + nb)
        assert np.array_equal(d.generated.xyz, pos), "the refine moved splats"
        assert np.allclose(d.generated.f_dc[:own], green) and np.allclose(d.generated.f_dc[own:], green, atol=1e-4), \
            "the layer kept the quick fill's colours"
        d.undo()
        assert d.n_total == s.n and d.n_alive == s.n
    finally:
        SK.finish, viewfill.view_fill = real_finish, real_fill
    print("    refine      %d + %d splats stay, the layer takes the new colors" % (own, nb))


if __name__ == "__main__":
    fails = 0
    for name, fn in list(globals().items()):
        if name.startswith("test_") and callable(fn):
            print(name)
            try:
                fn()
            except Exception as e:                          # noqa: BLE001
                import traceback
                traceback.print_exc()
                fails += 1
    print("all lod tests passed" if not fails else "%d FAILURES" % fails)
    sys.exit(1 if fails else 0)
