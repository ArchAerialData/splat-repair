"""The fill with a painted volume, and several views filled as one sheet.

Run as a script. Never starts Photoshop: every Photoshop path here runs
against a stand-in for ``photoshop.fill``.
"""
import os
import sys

import numpy as np

HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, os.path.join(HERE, ".."))
sys.path.insert(0, HERE)
from splatcore import repair as rp                              # noqa: E402
from splatcore import viewfill as vf, deep as dp, photoshop as ps  # noqa: E402
from test_repair import make_surface                            # noqa: E402

FLAT = staticmethod(lambda x, y: np.zeros_like(x))


def _plane(n=30000):
    return make_surface(lambda x, y: np.zeros_like(x), n=n)


def _disc_volume(cx, cy, r, depth=0.6):
    return {"capsules": [[cx, cy, 0.0, cx, cy, 0.0, r, depth]], "view": [0, 0, -1]}


def _stroke_volume(a, b, r, depth=0.6):
    return {"capsules": [[*a, *b, r, depth]], "view": [0, 0, -1]}


def _covered(res, want_xy, d=0.12):
    from scipy.spatial import cKDTree
    if res.added.n == 0:
        return 0.0
    return float((cKDTree(res.added.xyz[:, :2]).query(want_xy)[0] < d).mean())


def _grid_in(fn, step=0.1, ext=6.0):
    xs = np.arange(-ext, ext, step) + step / 2
    X, Y = np.meshgrid(xs, xs)
    m = fn(X, Y)
    return np.stack([X[m], Y[m]], 1)


def test_volume_fills_a_hole_the_scan_never_had():
    """A notch at the surface's edge and a crack: no splats to select there."""
    base = _plane()
    xy = base.xyz[:, :2].astype(float)
    cases = [
        ("notch", np.hypot(xy[:, 0] - 5.4, xy[:, 1]) < 1.2, _disc_volume(5.4, 0, 1.5),
         lambda X, Y: (np.hypot(X - 5.4, Y) < 1.1) & (X < 5.9), lambda x, y: x > 6.3),
        ("crack", np.abs(xy[:, 1]) < 0.35, _stroke_volume((-2, 0, 0), (2, 0, 0), 0.7),
         lambda X, Y: (np.abs(Y) < 0.3) & (np.abs(X) < 1.8), lambda x, y: np.abs(x) > 3.2),
    ]
    for name, hole, vol, want, past in cases:
        s = base.take(~hole)
        sel = rp.as_volume(vol).inside(s.xyz)
        p = rp.RepairParams(seed=1, view=[0, 0, -1])
        without = rp.repair(s, sel, p)
        p.volume = vol
        res = rp.repair(s, sel, p)
        c0, c1 = _covered(without, _grid_in(want)), _covered(res, _grid_in(want))
        assert c1 > 0.9, "%s: hole only %.2f covered with the volume" % (name, c1)
        assert c1 > c0 + 0.5, "%s: the volume changed nothing (%.2f vs %.2f)" % (name, c1, c0)
        assert not past(res.added.xyz[:, 0], res.added.xyz[:, 1]).any(), \
            "%s: fill laid past the surface's edge" % name
        assert np.abs(res.added.xyz[:, 2]).max() < 0.15, "%s: fill off the surface" % name
        assert res.info["volume_cells"] > 0
        print("    %-6s      covered %.2f without a volume, %.2f with" % (name, c0, c1))


def test_enclosed_hole_inside_the_outline_is_kept():
    """A real hole well inside the surface: the edge rule must not cut it."""
    base = _plane()
    xy = base.xyz[:, :2].astype(float)
    s = base.take(np.hypot(xy[:, 0] - 1.0, xy[:, 1]) >= 1.5)
    vol = _disc_volume(1.0, 0, 1.9)
    sel = rp.as_volume(vol).inside(s.xyz)
    res = rp.repair(s, sel, rp.RepairParams(seed=1, view=[0, 0, -1], volume=vol))
    c = _covered(res, _grid_in(lambda X, Y: np.hypot(X - 1.0, Y) < 1.4))
    assert c > 0.95 and res.info["beyond_edge"] == 0, (c, res.info["beyond_edge"])
    print("    inside      covered %.2f, nothing cut off" % c)


def test_volume_over_nothing_selected():
    """Painted wholly inside a hole: stand-ins carry the fill and leave with it."""
    base = _plane()
    xy = base.xyz[:, :2].astype(float)
    s = base.take(np.hypot(xy[:, 0], xy[:, 1]) >= 2.0)
    vol = _disc_volume(0, 0, 1.0)
    sel = rp.as_volume(vol).inside(s.xyz)
    assert sel.sum() == 0
    res = rp.repair(s, np.zeros(0, np.int64), rp.RepairParams(seed=1, view=[0, 0, -1], volume=vol))
    assert res.info.get("standins", 0) > 0
    assert (np.asarray(res.removed) < s.n).all(), "a stand-in was reported as removed"
    c = _covered(res, _grid_in(lambda X, Y: np.hypot(X, Y) < 0.9))
    assert c > 0.9, c
    print("    empty       %d stand-ins, covered %.2f" % (res.info["standins"], c))


def test_no_volume_is_unchanged():
    s = make_surface(lambda x, y: np.zeros_like(x), n=12000)
    sel = np.hypot(s.xyz[:, 0], s.xyz[:, 1]) < 1.2
    a = rp.repair(s, sel, rp.RepairParams(seed=3))
    b = rp.repair(s, sel, rp.RepairParams(seed=3, volume=None))
    assert np.array_equal(a.removed, b.removed) and np.allclose(a.added.xyz, b.added.xyz)
    assert a.context.get("volume") is None
    assert rp.as_volume(None) is None
    assert rp.as_volume({"capsules": [], "view": [0, 0, 1]}) is None, "an empty volume is none"
    print("    none        without a volume the same as before")


def test_volume_mask_is_what_was_painted_on_the_surface():
    """``deep.volume_mask``: the disc on the plane; nothing where the surface is elsewhere."""
    from splatcore import gpu
    vol = rp.as_volume(_disc_volume(0, 0, 1.0, depth=0.3))
    cam = gpu.look_at_camera(np.array([0, 0, 10.0]), np.zeros(3), np.array([0, 1.0, 0]), 200, 200, 30)
    depth = np.full((200, 200), 10.0)            # a plane at z = 0, seen from z = 10
    m = dp.volume_mask(vol, cam, 200, 200, depth)
    yy, xx = np.mgrid[0:200, 0:200]
    px = cam.K[0, 0] / 10.0                       # pixels per unit at the plane
    want = np.hypot(xx + 0.5 - 100, yy + 0.5 - 100) < 1.0 * px
    iou = (m & want).sum() / max((m | want).sum(), 1)
    assert iou > 0.9, iou
    far = dp.volume_mask(vol, cam, 200, 200, np.full((200, 200), 12.0))   # the surface 2 behind
    assert far.sum() == 0, "a surface outside the volume counted as painted"
    nothing = dp.volume_mask(vol, cam, 200, 200, np.full((200, 200), np.inf))
    assert nothing.sum() == 0, "background counted as painted"
    print("    mask        IoU %.3f, nothing behind the volume" % iou)


def test_sheet_round_trip():
    rng = np.random.default_rng(0)
    pics = [rng.integers(0, 255, (40, 40, 3), dtype=np.uint8) for _ in range(3)]
    masks = [np.zeros((40, 40), bool) for _ in range(3)]
    for k, m in enumerate(masks):
        m[10:20, 5 + k * 5:15 + k * 5] = True
    sheet, smask, boxes = vf.make_sheet(pics, masks)
    assert sheet.shape == (40, 3 * 40 + 2 * vf.SHEET_GAP, 3)
    assert smask.sum() == sum(m.sum() for m in masks)
    back = vf.split_sheet(sheet, boxes)
    assert all(np.array_equal(a, b) for a, b in zip(pics, back))
    # the gap is grey and never masked
    assert not smask[:, 40:40 + vf.SHEET_GAP].any()
    print("    sheet       assembled and split without loss")


def test_fill_views_one_call_per_sheet():
    calls = []
    saved = vf.fill_picture

    def fake(rgb, mask, source, say=None, sharpen=0.5, tile=None):
        calls.append((rgb.shape, tile))
        out = rgb.copy()
        out[mask] = (len(calls) * 40) % 255
        return out, "Stand-in", None
    pics = [np.zeros((32, 32, 3), np.uint8) for _ in range(3)]
    masks = [np.pad(np.ones((8, 8), bool), 12) for _ in range(3)]
    try:
        vf.fill_picture = fake
        f1, n1, _ = vf.fill_views(pics, masks, "lama", sheet=True)
        assert len(calls) == 1 and calls[0][1] == 32, calls
        assert all(int(f[16, 16, 0]) == 40 for f in f1), "views of one sheet must agree"
        calls.clear()
        f2, _, _ = vf.fill_views(pics, masks, "lama", sheet=False)
        assert len(calls) == 3, calls
        assert len({int(f[16, 16, 0]) for f in f2}) == 3
    finally:
        vf.fill_picture = saved
    print("    sheet       one call for all views, three when separate")


def test_photoshop_sheet_with_a_stand_in():
    """The Photoshop path over a sheet: one call to ``photoshop.fill``, split back."""
    seen = []
    saved = ps.fill

    def fake_fill(rgb, mask, mode="content_aware", **kw):
        seen.append((rgb.shape, int(mask.sum()), mode))
        out = rgb.copy()
        out[mask > 0] = (200, 10, 10)
        return out
    pics = [np.full((48, 48, 3), 90, np.uint8) for _ in range(3)]
    masks = [np.pad(np.ones((10, 10), bool), 19) for _ in range(3)]
    try:
        ps.fill = fake_fill
        fills, names, whys = vf.fill_views(pics, masks, "photoshop", sheet=True)
    finally:
        ps.fill = saved
    assert len(seen) == 1 and seen[0][2] == "generative", seen
    assert seen[0][0][1] == 3 * 48 + 2 * vf.SHEET_GAP
    for f, m in zip(fills, masks):
        assert (f[m] == (200, 10, 10)).all() and (f[~m] == 90).all()
    assert names == ["Photoshop (generative)"] and not whys
    print("    photoshop   one Generative Fill call over the sheet (stand-in)")


def test_view_fill_hands_every_view_to_the_refit():
    s = _plane(12000)
    sel = np.where(np.hypot(s.xyz[:, 0], s.xyz[:, 1]) < 1.0)[0]
    base = rp.repair(s, sel, rp.RepairParams(seed=0, view=[0, 0, -1]))
    got = {}
    saved = (vf.render_view, vf.view_mask, vf.fill_picture, dp.deep_repair)
    try:
        vf.render_view = lambda shown, e, t, u, W, H, f: (np.zeros((H, W, 3), np.uint8), e, None)
        vf.view_mask = lambda *a, **k: np.pad(np.ones((20, 20), bool), 22)

        def fill(rgb, mask, source, say=None, sharpen=0.5, tile=None):
            got.setdefault("fills", []).append(rgb.shape)
            return rgb, "Stand-in", None
        vf.fill_picture = fill

        def deep(s_, sel_, params, d, base=None, on_progress=None, reference=None, references=None):
            got["refs"] = references
            return rp.RepairResult(removed=base.removed, added=base.added, backing=base.backing,
                                   info={"deep": True, "fit": 0.01}, context=dict(base.context))
        dp.deep_repair = deep
        r = vf.view_fill(s, sel, base=base, source="lama", size=64, views=3, sheet=True)
    finally:
        vf.render_view, vf.view_mask, vf.fill_picture, dp.deep_repair = saved
    assert len(got["fills"]) == 1 and got["fills"][0][1] == 3 * 64 + 2 * vf.SHEET_GAP
    refs = got["refs"]
    assert len(refs) == 3 and r.info["fill_views"] == 3 and r.info["fill_sheet"]
    eyes = np.array([np.asarray(x["eye"]) for x in refs])
    assert np.linalg.norm(eyes[1] - eyes[0]) > 1e-3 and np.linalg.norm(eyes[2] - eyes[1]) > 1e-3
    d0 = np.linalg.norm(eyes - np.asarray(refs[0]["target"]), axis=1)
    assert np.allclose(d0, d0[0]), "the tilted views stand at the same distance"
    print("    views       three views, one call, all passed to the fit")


def test_deep_repair_takes_several_pictures():
    """On the GPU: a second picture from a tilted eye becomes a train view."""
    from splatcore import gpu
    if not gpu.available()[0]:
        print("    (no GPU, skipped)")
        return
    s = _plane(30000)
    sel = np.where(np.hypot(s.xyz[:, 0], s.xyz[:, 1]) < 1.2)[0]
    base = rp.repair(s, sel, rp.RepairParams(seed=0, view=[0, 0, -1]))
    frames = vf.view_frames(base.context, 3.0, views=2, tilt=25.0)
    truth = s
    refs = []
    for e, t, u, f in frames:
        img, _, _ = vf.render_view(truth, e, t, u, 256, 256, f)
        refs.append({"image": img, "eye": e, "target": t, "up": u, "fov": f})
    r = dp.deep_repair(s, sel, None, dp.DeepParams(steps=60), base=base, references=refs)
    assert r.info["views"] == 2, r.info
    assert r.info["fitExtra"] is not None and r.info["fitExtra"] <= r.info["fitExtraGeo"] + 1e-3, r.info
    print("    deep        2 images, side view %.4f -> %.4f" % (r.info["fitExtraGeo"], r.info["fitExtra"]))


def test_server_accepts_a_volume():
    """The sidecar: ``volume`` on repair, an empty selection allowed with it."""
    import tempfile
    from test_server import Client
    from splatcore.io import save_ply
    s = _plane(16000)
    xy = s.xyz[:, :2].astype(float)
    s = s.take((np.hypot(xy[:, 0] - 5.4, xy[:, 1]) >= 1.2) & (np.hypot(xy[:, 0] + 2, xy[:, 1]) >= 1.2))
    vol = _disc_volume(5.4, 0, 1.5)
    sel = np.where(rp.as_volume(vol).inside(s.xyz))[0].astype("<u4")
    with tempfile.TemporaryDirectory() as d:
        path = os.path.join(d, "notch.ply")
        save_ply(path, s)
        c = Client()
        try:
            h, _ = c.call("open", path=path)
            assert h.get("n") == s.n, h
            h, _ = c.call("repair", sel.tobytes(), params={"seed": 1, "view": [0, 0, -1]}, volume=vol,
                          engine="copy")
            assert "error" not in h, h
            assert h["info"]["volume_cells"] > 0, h["info"]
            h2, _ = c.call("undo")
            # absent volume: as before
            h, _ = c.call("repair", sel.tobytes(), params={"seed": 1, "view": [0, 0, -1]}, engine="copy")
            assert "error" not in h and h["info"]["volume_cells"] == 0, h
            c.call("undo")
            # nothing selected, only painted (inside the second, enclosed hole)
            h, _ = c.call("repair", b"", params={"seed": 1, "view": [0, 0, -1]},
                          volume=_disc_volume(-2.0, 0, 0.6))
            assert "error" not in h, h
            assert h["addedCount"] > 0 and h["info"].get("standins", 0) > 0, h["info"]
            c.call("undo")
            # a malformed volume is refused with a reason
            h, _ = c.call("repair", sel.tobytes(), volume={"capsules": [[1, 2, 3]]})
            assert "error" in h, h
        finally:
            c.close()
    print("    server      volume accepted, empty selection with a volume, error explained")


if __name__ == "__main__":
    fails = 0
    for name, fn in sorted(globals().items()):
        if name.startswith("test_") and callable(fn):
            try:
                print("  " + name)
                fn()
            except AssertionError as e:
                fails += 1
                print("    FAIL:", e)
    print("all volume fill tests passed" if not fails else "%d FAILURES" % fails)
    sys.exit(1 if fails else 0)
