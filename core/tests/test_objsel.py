"""Object selection (objsel.py): the stroke grown to the object, on synthetic scenes.

Run as a script: ``python test_objsel.py``.
"""
import os
import sys

import numpy as np

sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))
sys.path.insert(0, os.path.dirname(__file__))
from splatcore.io import SplatSet, SH_C0                      # noqa: E402
from splatcore import objsel                                  # noqa: E402
from splatcore import repair as rp                            # noqa: E402
from splatcore.document import Document                       # noqa: E402
from test_repair import make_surface                          # noqa: E402

VIEW = [0.0, 0.0, -1.0]          # looking down -z at the wall (z = 0, front +z)


def _like(s, xyz, rgb, alpha=None, scale=None):
    """New splats at ``xyz`` in colour ``rgb``, the rest taken from ``s``."""
    m = len(xyz)
    t = s.take(np.arange(m) % s.n)
    t.xyz = np.asarray(xyz, np.float32)
    t.f_dc = ((np.broadcast_to(np.asarray(rgb, float), (m, 3)) - 0.5) / SH_C0).astype(np.float32)
    t.f_rest = np.zeros_like(t.f_rest)
    if alpha is not None:
        a = np.clip(np.broadcast_to(alpha, (m,)), 1e-3, 0.999)
        t.opacity = np.log(a / (1 - a)).astype(np.float32)
    if scale is not None:
        t.scale = np.full((m, 3), np.log(scale), np.float32)
    return t


def _wall(seed=3, n=80000, plain=False, extent=3.0):
    s = make_surface(lambda x, y: np.zeros_like(x), n=n, extent=extent, seed=seed)
    if plain:
        # a painted wall: one colour, a little noise
        rng = np.random.default_rng(seed)
        rgb = np.array([0.85, 0.8, 0.7]) + rng.normal(0, 0.02, (s.n, 3))
        s.f_dc = ((rgb - 0.5) / SH_C0).astype(np.float32)
    return s


def _sign(wall, rgb=(0.8, 0.1, 0.1), depth=0.08, box=(0.5, 0.15), seed=1, shadow=False):
    """A raised box sign on the wall; the wall under it never scanned."""
    rng = np.random.default_rng(seed)
    bx, by = box
    under = (np.abs(wall.xyz[:, 0]) < bx) & (np.abs(wall.xyz[:, 1]) < by)
    keep = ~under
    if shadow:
        # a drop shadow below the sign: the wall darker there
        sh = (np.abs(wall.xyz[:, 0]) < bx + 0.02) & (wall.xyz[:, 1] < -by) \
            & (wall.xyz[:, 1] > -by - 0.05)
        rgb_w = 0.5 + SH_C0 * wall.f_dc.astype(float)
        rgb_w[sh] *= 0.55
        wall = wall.take(np.arange(wall.n))
        wall.f_dc = ((rgb_w - 0.5) / SH_C0).astype(np.float32)
    w = wall.take(keep)
    m_top = 5000
    top = np.column_stack([rng.uniform(-bx, bx, m_top), rng.uniform(-by, by, m_top),
                           np.full(m_top, depth)])
    m_side = 2500
    t = rng.uniform(0, 1, m_side)
    side = []
    for k in range(m_side):                      # the box's four sides
        z = rng.uniform(0, depth)
        e = k % 4
        if e == 0:
            side.append([-bx, (2 * t[k] - 1) * by, z])
        elif e == 1:
            side.append([bx, (2 * t[k] - 1) * by, z])
        elif e == 2:
            side.append([(2 * t[k] - 1) * bx, -by, z])
        else:
            side.append([(2 * t[k] - 1) * bx, by, z])
    side = np.asarray(side)
    # faint, large splats of the sign's colour round its edge
    m_f = 300
    ang = rng.uniform(0, 2 * np.pi, m_f)
    fx = np.clip(np.cos(ang) * bx * 1.08, -bx * 1.05, bx * 1.05)
    fy = np.clip(np.sin(ang) * by * 1.3, -by * 1.1, by * 1.1)
    faint = np.column_stack([fx, fy, rng.uniform(0.005, 0.02, m_f)])
    parts = [_like(wall, top, rgb), _like(wall, side, np.asarray(rgb) * 0.7),
             _like(wall, faint, 0.5 * np.asarray(rgb) + 0.25, alpha=0.15, scale=0.03)]
    s = w
    first = s.n
    for p_ in parts:
        s = s.concat(p_)
    n_top = m_top
    return s, np.arange(first, first + n_top), np.arange(first, s.n)


def test_pole_lifts_and_keeps_the_wall():
    """A pole standing clear of a whole wall, painted with a wide stroke."""
    wall = _wall()
    rng = np.random.default_rng(0)
    m = 600
    pole = _like(wall, np.column_stack([rng.normal(0, 0.012, m), rng.uniform(-1.2, 1.2, m),
                                        0.25 + rng.normal(0, 0.012, m)]), (0.85, 0.85, 0.85))
    s = wall.concat(pole)
    pole_idx = np.arange(wall.n, s.n)
    # the stroke: the pole between |y| < 0.8 and the wall beside it
    on_wall = np.where((np.abs(s.xyz[:, 0]) < 0.1) & (np.abs(s.xyz[:, 1]) < 0.8)
                       & (s.xyz[:, 2] < 0.05))[0]
    stroke_pole = pole_idx[np.abs(s.xyz[pole_idx, 1]) < 0.8]
    sel = np.concatenate([on_wall, stroke_pole])
    got = objsel.find_object(s, sel, VIEW)
    assert got.mode == "lift", got.info
    assert np.isin(stroke_pole, got.indices).all(), "the pole in the stroke is not all taken"
    assert not np.isin(on_wall, got.indices).any(), "wall splats were taken with the pole"
    d = Document(s)
    e = d.repair(sel, rp.RepairParams(seed=0, view=VIEW))
    assert e.kind == "delete" and e.info.get("mode") == "delete", (e.kind, e.info)
    assert d.alive[on_wall].all(), "the wall behind the pole lost splats"
    assert not d.alive[stroke_pole].any()
    d.undo()
    assert d.n_alive == s.n
    print("    pole        deleted %d, wall %d splats untouched, standoff %.0f"
          % (len(e.removed), len(on_wall), got.info["standoff"]))


def test_raised_sign_grows_to_its_sides():
    """Only the sign's face was painted; its sides and faint edge go with it."""
    s, top, sign = _sign(_wall(extent=1.5))
    got = objsel.find_object(s, top, VIEW)
    assert got.mode == "fill", got.info
    took = np.isin(sign, got.indices)
    assert took.mean() > 0.97, "only %.2f of the sign taken" % took.mean()
    # and not the wall far from it
    far = np.where((np.abs(s.xyz[:, 0]) > 0.75) | (np.abs(s.xyz[:, 1]) > 0.45))[0]
    assert not np.isin(far, got.indices).any(), "the wall far from the sign was taken"
    print("    sign        stroke %d -> %d, %.0f %% of the sign captured"
          % (len(top), len(got.indices), 100 * took.mean()))


def test_drop_shadow_goes_with_the_sign():
    s, top, sign = _sign(_wall(plain=True, extent=1.5), shadow=True)
    got = objsel.find_object(s, top, VIEW)
    assert got.mode == "fill", got.info
    shadow = np.where((np.abs(s.xyz[:, 0]) < 0.45) & (s.xyz[:, 1] < -0.17)
                      & (s.xyz[:, 1] > -0.20) & (np.abs(s.xyz[:, 2]) < 0.01))[0]
    assert got.info["shadowCells"] > 0, got.info
    share = np.isin(shadow, got.indices).mean()
    assert share > 0.8, "only %.2f of the drop shadow taken (%s)" % (share, got.info)
    print("    shadow      %.0f %% of the drop shadow taken along, %d cells"
          % (100 * share, got.info["shadowCells"]))


def test_print_is_grown_to_the_whole_print():
    """A flat green print; the stroke caught most of it, the rest is grown."""
    wall = _wall()
    rgb = 0.5 + SH_C0 * wall.f_dc.astype(float)
    r = np.hypot(wall.xyz[:, 0], wall.xyz[:, 1])
    green = r < 0.3
    rgb[green] = [0.2, 0.7, 0.4]
    s = wall.take(np.arange(wall.n))
    s.f_dc = ((rgb - 0.5) / SH_C0).astype(np.float32)
    sel = np.where(r < 0.26)[0]                 # the stroke missed the print's edge
    got = objsel.find_object(s, sel, VIEW)
    assert got.mode == "fill", got.info
    gi = np.where(green)[0]
    assert np.isin(gi, got.indices).all(), "%d green splats left" % (~np.isin(gi, got.indices)).sum()
    assert not np.isin(np.where(r > 0.45)[0], got.indices).any()
    d = Document(s)
    e = d.repair(sel, rp.RepairParams(seed=0, view=VIEW))
    assert e.kind == "repair" and e.info.get("object", {}).get("objsel") == "fill", e.info
    left = d.alive[:wall.n] & green
    assert not left.any(), "%d green splats survive the repair" % left.sum()
    print("    print       %d green splats, all gone; stroke %d -> %d"
          % (green.sum(), len(sel), len(got.indices)))


def test_plain_stroke_is_left_alone():
    """No object in the stroke: the selection stays exactly as painted."""
    wall = make_surface(lambda x, y: 0.05 * np.sin(x) * np.cos(y), n=60000, extent=3.0, seed=5)
    sel = np.where(np.hypot(wall.xyz[:, 0], wall.xyz[:, 1]) < 0.5)[0]
    got = objsel.find_object(wall, sel, VIEW)
    assert got.mode == "none", got.info
    assert np.array_equal(np.sort(got.indices), np.sort(sel))
    d = Document(wall)
    e = d.repair(sel, rp.RepairParams(seed=0, view=VIEW))
    assert "object" not in e.info
    print("    nothing     %s" % got.info.get("why"))


def test_switch_off_repairs_as_painted():
    s, top, sign = _sign(_wall(extent=1.5))
    d = Document(s)
    e = d.repair(top, rp.RepairParams(seed=0, view=VIEW), objects=False)
    assert "object" not in e.info
    assert d.alive[np.setdiff1d(sign, top)].any(), "with the switch off the sides went too"
    d.undo()
    d.object_select = False
    e = d.repair(top, rp.RepairParams(seed=0, view=VIEW))
    assert "object" not in e.info


def test_document_preview_is_in_document_indices():
    """object_selection names document splats, and skips dead ones."""
    s, top, sign = _sign(_wall(extent=1.5))
    d = Document(s)
    dead = np.arange(0, 5000)
    d.delete(dead)                         # shifts every live position
    r = d.object_selection(top, view=VIEW)
    assert r["mode"] == "fill", r["info"]
    assert d.alive[r["indices"]].all()
    assert np.isin(sign, r["indices"]).mean() > 0.97
    assert np.isin(r["object"], r["indices"]).all()


# Optional real scan for test_hut_pole_with_the_apps_reach: the path to the
# hut scan (.spz), given in the environment. Unset or missing: the test skips.
HUT = os.environ.get("SPLAT_REPAIR_TEST_SPLAT", "")
REF = os.path.join(os.path.dirname(os.path.abspath(__file__)), "..", "..", "reference", "objsel")


def test_hut_pole_with_the_apps_reach():
    """The hut's antenna pole, painted with the app's reach: deleted, wall kept.

    The stroke crosses the roof's edge and, being wider than the pole, takes
    a strip of wall beside it. The repair refused that with "beyond the
    edge" -- a RepairError, so the delete fallback never ran. Now the
    repair says "nothing to fill" for a stroke standing clear of what is
    left, and the object selection deletes the pole alone. Skipped unless
    SPLAT_REPAIR_TEST_SPLAT names the scan and the reference selection exists.
    """
    f = os.path.join(REF, "hut_stange_app_idx.npy")
    if not HUT:
        print("    (skipped: set SPLAT_REPAIR_TEST_SPLAT to the path of the hut scan, a .spz file)")
        return
    if not os.path.isfile(HUT):
        print("    (skipped: SPLAT_REPAIR_TEST_SPLAT points to a missing file: %s)" % HUT)
        return
    if not os.path.exists(f):
        print("    (skipped: the reference selection is missing: %s)" % f)
        return
    from splatcore.formats import load_any
    s = load_any(HUT)
    sel = np.load(f)
    view = [float(x) for x in np.load(os.path.join(REF, "hut_stange_app_view.npy"))]
    try:
        rp.repair(s, sel, rp.RepairParams(seed=0, view=view))
        raise AssertionError("the pole was filled")
    except rp.NothingToFill as e:
        assert e.standoff >= 2.0, e.standoff
    d = Document(s)
    d.fill_engine = "copy"            # the copy engine's fallback is what is tested here
    e = d.repair(sel, rp.RepairParams(seed=0, view=view), objects=False)
    assert e.info.get("mode") == "delete", e.info
    d.undo()
    d.fill_engine = "auto"
    e = d.repair(sel, rp.RepairParams(seed=0, view=view))
    assert e.info.get("mode") == "delete" and e.info["object"]["objsel"] == "lift", e.info
    kept = int(d.alive[sel].sum())
    assert kept > 800, "only %d of the wall strip in the stroke kept" % kept
    print("    hut         pole deleted (%d), wall strip kept %d splats"
          % (len(e.removed), kept))


def test_server_object_select():
    import io
    import server
    s, top, sign = _sign(_wall(extent=1.5))
    sess = server.Session(io.BytesIO())
    sess.doc = Document(s)
    payload = np.asarray(top, "<u4").tobytes()
    rep, _ = sess.cmd_object_select({"view": VIEW}, payload)
    assert rep["mode"] == "fill" and len(rep["indices"]) > len(top), rep["info"]
    import json
    json.dumps(rep)
    rep, _ = sess.cmd_repair({"params": {"seed": 0, "view": VIEW}, "objects": False}, payload)
    assert "object" not in rep["info"]


if __name__ == "__main__":
    import time
    fails = 0
    for name, fn in sorted(globals().items()):
        if name.startswith("test_"):
            t = time.time()
            try:
                print("  " + name)
                fn()
                print("    %.1f s" % (time.time() - t))
            except Exception as e:
                fails += 1
                print("    FAIL:", type(e).__name__, e)
    print("all objsel tests passed" if not fails else "%d FAILURES" % fails)
    sys.exit(1 if fails else 0)
