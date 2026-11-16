"""Tests for the picture fill (`splatcore/imfill.py`). Run as a script.

Needs CUDA + gsplat; without them every test is skipped (the app then falls
back to the copy engine, which has its own tests).
"""
import os
import sys

import numpy as np

HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, os.path.join(HERE, ".."))

from splatcore.io import SplatSet, SH_C0                      # noqa: E402
from splatcore import gpu, imfill                             # noqa: E402
from splatcore.document import Document                       # noqa: E402
from splatcore.repair import RepairParams, as_volume          # noqa: E402


def surface(n=110, step=0.05, dome=0.0, stripes=True, seed=0):
    """A square patch of surface: flat or a dome, striped along x, facing +z."""
    rng = np.random.default_rng(seed)
    g = (np.arange(n) - n / 2) * step
    x, y = [a.ravel() for a in np.meshgrid(g, g)]
    x = x + rng.normal(0, 0.1 * step, x.shape)
    y = y + rng.normal(0, 0.1 * step, y.shape)
    half = n * step / 2
    z = dome * (1.0 - (x * x + y * y) / (half * half))
    N = len(x)
    col = np.full((N, 3), 0.6)
    if stripes:
        dark = (np.floor(x / (6 * step)) % 2) == 0
        col[dark] = (0.25, 0.3, 0.2)
    q = np.zeros((N, 4), np.float32)
    q[:, 0] = 1.0
    return SplatSet(
        xyz=np.stack([x, y, z], 1).astype(np.float32),
        f_dc=((col - 0.5) / SH_C0).astype(np.float32),
        f_rest=np.zeros((N, 15, 3), np.float32),
        opacity=np.full(N, 4.0, np.float32),
        scale=np.log(np.stack([np.full(N, 0.7 * step), np.full(N, 0.7 * step),
                               np.full(N, 0.15 * step)], 1)).astype(np.float32),
        rot=q)


def disc(s, r, c=(0.0, 0.0)):
    return np.where(np.hypot(s.xyz[:, 0] - c[0], s.xyz[:, 1] - c[1]) < r)[0]


VIEW = [0.0, 0.0, -1.0]                 # looking down on the patch


def test_hole_is_closed_at_the_surfaces_height():
    s = surface()
    idx = disc(s, 0.6)
    r = imfill.fill(s, idx, VIEW, imfill.ImFillParams(lama=False))
    assert r.added.n > 500, r.info
    z = r.added.xyz[:, 2]
    assert abs(float(np.median(z))) < 0.03 and float(np.percentile(np.abs(z), 95)) < 0.06, \
        (np.median(z), np.percentile(np.abs(z), 95))
    # the hole's footprint is covered
    rad = np.hypot(r.added.xyz[:, 0], r.added.xyz[:, 1])
    assert rad.max() > 0.55 and (rad < 0.3).sum() > 50
    print("    flat        %d new splats, median height %+.4f (spacing 0.05)" % (r.added.n, np.median(z)))


def test_dome_stays_a_dome():
    """The base leaves the rim with its slope: the fill bulges with the dome."""
    s = surface(dome=0.5)
    idx = disc(s, 0.7)
    r = imfill.fill(s, idx, VIEW, imfill.ImFillParams(lama=False, shape_prior=False))
    half = 110 * 0.05 / 2
    x, y, z = r.added.xyz.T
    truth = 0.5 * (1.0 - (x * x + y * y) / (half * half))
    err = np.abs(z - truth)
    assert float(np.median(err)) < 0.02, float(np.median(err))
    flat = np.abs(z - 0.5 * (1.0 - 0.7 ** 2 / half ** 2))
    assert float(np.median(err)) < 0.5 * float(np.median(flat)) + 1e-3
    print("    dome        median error %.4f (filled flat it would be %.4f)"
          % (np.median(err), np.median(flat)))


def test_stripes_run_through():
    s = surface()
    idx = disc(s, 0.6, (0.15, 0.0))
    r = imfill.fill(s, idx, VIEW, imfill.ImFillParams(lama=gpu_lama()))
    x = r.added.xyz[:, 0]
    lum = 0.5 + SH_C0 * r.added.f_dc.astype(np.float64).mean(1)
    dark = (np.floor(x / 0.3) % 2) == 0
    inner = np.abs((x / 0.3) % 1.0 - 0.5) < 0.3            # away from the stripe edges
    d, l = lum[dark & inner], lum[~dark & inner]
    assert len(d) > 50 and len(l) > 50
    assert float(np.median(l)) - float(np.median(d)) > 0.15, (np.median(l), np.median(d))
    print("    stripes     light %.2f, dark %.2f" % (np.median(l), np.median(d)))


def test_hole_without_splats_is_filled_from_the_volume():
    s0 = surface()
    cut = disc(s0, 0.5)
    keep = np.ones(s0.n, bool)
    keep[cut] = False
    s = s0.take(keep)
    vol = as_volume({"capsules": [[0, 0, 0, 0, 0, 0, 0.6, 0.3, 0, 0, 0, -1]], "subtract": [],
                     "view": [0, 0, -1]})
    sel = np.where(vol.inside(s.xyz))[0]
    r = imfill.fill(s, sel, VIEW, imfill.ImFillParams(lama=False), volume=vol)
    assert r.added.n > 500 and r.info["holePainted"] > 0, r.info
    rad = np.hypot(r.added.xyz[:, 0], r.added.xyz[:, 1])
    assert (rad < 0.25).sum() > 50, "the middle of the hole stayed empty"
    assert abs(float(np.median(r.added.xyz[:, 2]))) < 0.03
    print("    hole        without splats: %d new, %d painted pixels" % (r.added.n, r.info["holePainted"]))


def test_a_spot_unpainted_in_the_middle_stays():
    # A thing lies on the surface. The user paints over all of it and then
    # takes the middle out of the stroke again ("Weg"). The middle of the
    # thing must stay: nothing new on it, nothing of it removed. (On a bare
    # surface the fill left such an island alone anyway, as skin that is
    # whole; under a thing it filled across and took a quarter of it away.)
    s0 = surface()
    rng = np.random.default_rng(2)
    n = 2600
    ang, rr = rng.uniform(0, 2 * np.pi, n), 0.5 * np.sqrt(rng.uniform(0, 1, n))
    thing = SplatSet(
        xyz=np.stack([rr * np.cos(ang), rr * np.sin(ang), 0.3 + rng.normal(0, 0.004, n)], 1).astype(np.float32),
        f_dc=np.tile(((np.array([0.15, 0.15, 0.2]) - 0.5) / SH_C0).astype(np.float32), (n, 1)),
        f_rest=np.zeros((n, 15, 3), np.float32), opacity=np.full(n, 4.0, np.float32),
        scale=np.log(np.full((n, 3), 0.03)).astype(np.float32),
        rot=np.tile(np.array([1, 0, 0, 0], np.float32), (n, 1)))
    s = s0.concat(thing)
    vol = as_volume({"capsules": [[0, 0, 0.3, 0, 0, 0.3, 0.62, 0.6, 0, 0, 0, -1]],
                     "subtract": [[0, 0, 0.3, 0, 0, 0.3, 0.25, 1.2, 1, 0, 0, -1]], "view": [0, 0, -1]})
    sel = np.where(vol.inside(s.xyz))[0]
    rad_s = np.hypot(s.xyz[:, 0], s.xyz[:, 1])
    assert (rad_s[sel] < 0.2).sum() == 0 and len(sel) > 1000, "the test's ring is not a ring"
    r = imfill.fill(s, sel, VIEW, imfill.ImFillParams(lama=False), volume=vol)
    rad = np.hypot(r.added.xyz[:, 0], r.added.xyz[:, 1])
    inside = int((rad < 0.15).sum())
    gone = int((rad_s[r.removed] < 0.15).sum())
    assert r.added.n > 300 and ((rad > 0.32) & (rad < 0.55)).sum() > 200, r.info
    assert inside == 0, "%d new splats on the un-painted island" % inside
    assert gone == 0, "%d of the island's own splats were removed" % gone
    print("    island      ring filled (%d new), the deselected middle untouched" % r.added.n)


def test_thing_before_a_whole_surface_is_only_deleted():
    s = surface()
    rng = np.random.default_rng(1)
    n = 400
    pole = SplatSet(
        xyz=np.stack([rng.normal(0, 0.05, n), rng.uniform(-0.8, 0.8, n), rng.uniform(0.8, 1.0, n)], 1).astype(np.float32),
        f_dc=np.tile(((np.array([0.8, 0.1, 0.1]) - 0.5) / SH_C0).astype(np.float32), (n, 1)),
        f_rest=np.zeros((n, 15, 3), np.float32), opacity=np.full(n, 4.0, np.float32),
        scale=np.log(np.full((n, 3), 0.04)).astype(np.float32),
        rot=np.tile(np.array([1, 0, 0, 0], np.float32), (n, 1)))
    both = s.concat(pole)
    idx = np.arange(s.n, both.n)
    r = imfill.fill(both, idx, VIEW, imfill.ImFillParams(lama=False))
    assert r.added.n == 0 and r.info["mode"] == "delete", r.info
    assert len(r.removed) == n
    print("    pole        deleted only (%s)" % r.info.get("why"))


def test_everything_selected_is_refused_in_words():
    # Ctrl+A, then "Repair": nothing is left to continue. This used to
    # end the engine's process (gsplat divides by the number of splats in
    # native code); it has to be a refusal the document can answer.
    s = surface(n=40)
    try:
        imfill.fill(s, np.arange(s.n), VIEW, imfill.ImFillParams(lama=False))
    except ValueError as ex:
        print("    everything  refused: %s" % ex)
    else:
        raise AssertionError("everything selected was filled")
    # ... and a picture of no splats at all is black, not a crash
    sc = imfill._Scene(s)
    n_, t1, t2 = imfill._frame(-np.asarray(VIEW, float))
    cam = imfill.Ortho(np.zeros(3), n_, t1, t2, 0.05, 64, 64, 10.0)
    rgb, alpha, _ = sc.picture(cam, sc.torch.zeros(0, dtype=sc.torch.long, device=sc.dev))
    assert rgb.shape == (64, 64, 3) and float(alpha.max()) == 0.0


def test_document_uses_the_picture_fill_and_undoes_in_one_step():
    s = surface()
    idx = disc(s, 0.5)
    d = Document(s)
    e = d.repair(idx, RepairParams(seed=0, view=VIEW), objects=False)
    assert e.info.get("engine") == "image" and e.added > 300, e.info
    assert d.n_alive == s.n - len(e.removed) + e.added
    d.undo()
    assert d.n_alive == s.n and not d.can_undo
    d.fill_engine = "copy"
    e = d.repair(idx, RepairParams(seed=0, view=VIEW), objects=False)
    assert e.info.get("engine") != "image", e.info
    print("    document    image fill as the default, one step back, copy selectable")


def _per_area(xyz, r, step):
    """New splats per unit of surface, counted in the middle of the hole."""
    m = np.hypot(xyz[:, 0], xyz[:, 1]) < 0.6 * r
    return m.sum() / (np.pi * (0.6 * r) ** 2) * step * step


def test_flank_gets_a_picture_of_its_own():
    """Looked at from 62 degrees off, one picture squeezes the surface to under
    half; lifted from there the fill is stretched. A side picture makes it as
    dense along the slope as across, and the stripes still run through."""
    s = surface(n=130)
    idx = disc(s, 0.55)
    t = np.radians(62.0)
    view = [0.0, -np.sin(t), -np.cos(t)]                   # the eye stands towards +y
    one = imfill.fill(s, idx, view, imfill.ImFillParams(lama=gpu_lama(), passes=1))
    r = imfill.fill(s, idx, view, imfill.ImFillParams(lama=gpu_lama()))
    assert r.info.get("passes") == 2, r.info
    d1, d2 = _per_area(one.added.xyz, 0.55, 0.05), _per_area(r.added.xyz, 0.55, 0.05)
    assert d2 > 1.6 * d1, (d1, d2)
    z = r.added.xyz[:, 2]
    assert abs(float(np.median(z))) < 0.03 and float(np.percentile(np.abs(z), 95)) < 0.08, \
        (np.median(z), np.percentile(np.abs(z), 95))
    x = r.added.xyz[:, 0]
    lum = 0.5 + SH_C0 * r.added.f_dc.astype(np.float64).mean(1)
    dark = (np.floor(x / 0.3) % 2) == 0
    inner = np.abs((x / 0.3) % 1.0 - 0.5) < 0.3
    assert float(np.median(lum[~dark & inner])) - float(np.median(lum[dark & inner])) > 0.15
    print("    flank       62 degrees: %d images, splats per area %.1f -> %.1f, median height %+.4f"
          % (r.info["passes"], d1, d2, np.median(z)))


def test_wall_past_an_edge_is_closed_from_the_side():
    """A table edge, painted from above: the top is seen well, the wall at 65
    degrees. One picture leaves the wall thin; with the side picture what was
    removed from the wall has new surface where it was."""
    from scipy.spatial import cKDTree
    step = 0.05
    top = surface(n=110, step=step)
    keep = top.xyz[:, 1] < 0.0
    top = top.take(keep)
    wall = surface(n=110, step=step, seed=1)
    wall = wall.take(wall.xyz[:, 1] < 0.0)
    w = wall.xyz.copy()
    wall.xyz[:, 1], wall.xyz[:, 2] = 0.0 * w[:, 1], w[:, 1]            # y in the patch runs down the wall
    q = np.zeros((wall.n, 4), np.float32)
    q[:, 0], q[:, 1] = np.cos(-np.pi / 4), np.sin(-np.pi / 4)        # flat splats: thin axis z -> y
    wall.rot[:] = q
    s = top.concat(wall)
    t = np.radians(25.0)
    eye = np.array([0.0, np.sin(t), np.cos(t)])
    view = (-eye).tolist()
    vol = as_volume({"capsules": [[0, 0, 0, 0, 0, 0, 0.5, 0.6, 0, *view]], "subtract": [], "view": view})
    idx = np.where(vol.inside(s.xyz))[0]
    on_wall = idx[s.xyz[idx, 2] < -2 * step]
    assert len(on_wall) > 60, len(on_wall)
    tree_of = lambda r_: cKDTree(r_.added.xyz.astype(np.float64))
    one = imfill.fill(s, idx, view, imfill.ImFillParams(lama=False, passes=1), volume=vol)
    r = imfill.fill(s, idx, view, imfill.ImFillParams(lama=False), volume=vol)
    assert r.info.get("passes", 1) >= 2, r.info
    d1 = tree_of(one).query(s.xyz[on_wall].astype(np.float64))[0] / step
    d2 = tree_of(r).query(s.xyz[on_wall].astype(np.float64))[0] / step
    assert float(np.percentile(d2, 90)) < 1.5, np.percentile(d2, 90)
    # and the new wall is as dense as the top: no more than a spacing between neighbours
    A = r.added.xyz.astype(np.float64)
    wl = A[(A[:, 2] < -3 * step) & (np.abs(A[:, 1]) < 3 * step)]
    assert len(wl) > 200, len(wl)
    nn = cKDTree(wl).query(wl, k=2)[0][:, 1] / step
    assert float(np.percentile(nn, 95)) < 0.6, np.percentile(nn, 95)
    print("    edge        wall: removed -> new p90 %.2f (one image %.2f) spacings, %d images"
          % (np.percentile(d2, 90), np.percentile(d1, 90), r.info.get("passes", 1)))


def test_thing_over_an_edge_leaves_no_ramp():
    """A thing sat over the edge of a table and hid a hole in it; far below
    lies the floor. Taken away, the picture shows table on one side and floor
    past the edge on the other. The fill continues the table and stops: it
    does not run down to the floor as a ramp."""
    step = 0.05
    table = surface(n=110, step=step)
    table = table.take(table.xyz[:, 1] < 0.0)                         # the table ends at y = 0
    hidden = np.hypot(table.xyz[:, 0], table.xyz[:, 1] + 0.1) < 0.4   # the hole the thing hides
    gone = table.xyz[hidden].copy()
    table = table.take(~hidden)
    floor = surface(n=110, step=step, stripes=False, seed=3)
    floor.xyz[:, 2] -= 2.0                                             # forty spacings down
    rng = np.random.default_rng(5)
    n = 1500
    thing = SplatSet(
        xyz=np.stack([rng.uniform(-0.5, 0.5, n), rng.uniform(-0.65, 0.35, n), rng.uniform(0.1, 0.4, n)], 1).astype(np.float32),
        f_dc=np.tile(((np.array([0.1, 0.1, 0.12]) - 0.5) / SH_C0).astype(np.float32), (n, 1)),
        f_rest=np.zeros((n, 15, 3), np.float32), opacity=np.full(n, 4.0, np.float32),
        scale=np.log(np.full((n, 3), 0.05)).astype(np.float32),
        rot=np.tile(np.array([1, 0, 0, 0], np.float32), (n, 1)))
    s = table.concat(floor).concat(thing)
    idx = np.arange(s.n - n, s.n)
    r = imfill.fill(s, idx, VIEW, imfill.ImFillParams(lama=gpu_lama()))
    assert r.info["mode"] == "image" and r.added.n > 300, r.info
    z = r.added.xyz[:, 2]
    between = (z < -0.2) & (z > -1.8)
    assert between.mean() < 0.02, "a ramp between table and floor: %.0f%% of the fill" % (100 * between.mean())
    from scipy.spatial import cKDTree
    on_table = r.added.xyz[np.abs(z) < 0.15].astype(np.float64)
    d = cKDTree(on_table).query(gone.astype(np.float64))[0] / step
    assert float(np.percentile(d, 90)) < 1.5, np.percentile(d, 90)
    print("    front edge  %d new splats, %.1f%% between table and floor, hole in the table p90 %.2f spacings"
          % (r.added.n, 100 * between.mean(), np.percentile(d, 90)))


def gpu_lama():
    try:
        import simple_lama_inpainting                       # noqa: F401
        return True
    except Exception:                                      # noqa: BLE001
        return False


if __name__ == "__main__":
    if not gpu.available()[0]:
        print("(skipped: %s)" % gpu.available()[1])
        print("all imfill tests passed")
        sys.exit(0)
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
    print("all imfill tests passed" if not fails else "%d FAILURES" % fails)
    sys.exit(1 if fails else 0)
