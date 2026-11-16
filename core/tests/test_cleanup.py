"""The debris finder on a scene where junk and structure are known.

A hut on a beach, built from splats: dense ground and walls, a shutter of a
few big faint splats, an antenna of small ones far apart -- the two things the
first version took for floaters -- plus floaters in the air and a haze layer
under the ground. World up is -y, as in the scans.
"""
import os
import sys

import numpy as np

sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))
from splatcore.io import SplatSet, SH_C0                     # noqa: E402
from splatcore import cleanup, gpu                          # noqa: E402


def _set(xyz, scale, alpha, rgb, seed=0):
    n = len(xyz)
    rng = np.random.default_rng(seed)
    q = np.zeros((n, 4)); q[:, 0] = 1.0
    if np.ndim(scale) == 1:
        scale = np.tile(scale, (n, 1))
    return SplatSet(
        xyz=np.asarray(xyz, np.float32),
        f_dc=np.tile((np.asarray(rgb) - 0.5) / SH_C0, (n, 1)).astype(np.float32),
        f_rest=np.zeros((n, 0, 3), np.float32),
        opacity=np.full(n, np.log(alpha / (1 - alpha)), np.float32),
        scale=np.log(np.asarray(scale, np.float64)).astype(np.float32),
        rot=(q + rng.normal(0, 1e-3, q.shape)).astype(np.float32))


def scene():
    parts = {}
    g = np.arange(-3.0, 3.0, 0.05)
    X, Z = np.meshgrid(g, g)
    top = np.column_stack([X.ravel(), np.zeros(X.size), Z.ravel()])
    parts["ground"] = _set(np.vstack([top, top + [0, 0.03, 0]]),
                           [0.035, 0.006, 0.035], 0.95, [0.8, 0.7, 0.5])
    w = np.arange(-1.0, 1.0, 0.05)
    A, B = np.meshgrid(w, w - 1.0)                     # B: y from -2 to 0
    walls = [np.column_stack([np.full(A.size, s), B.ravel(), A.ravel()]) for s in (-1.0, 1.0)]
    # a window in the +x wall, closed by the shutter below
    win = (B.ravel() > -1.75) & (B.ravel() < -0.45) & (np.abs(A.ravel()) < 0.75)
    walls[1] = walls[1][~win]
    walls += [np.column_stack([A.ravel(), B.ravel(), np.full(A.size, s)]) for s in (-1.0, 1.0)]
    R1, R2 = np.meshgrid(w, w)
    walls.append(np.column_stack([R1.ravel(), np.full(R1.size, -2.0), R2.ravel()]))
    parts["walls"] = _set(np.vstack(walls), [0.03, 0.03, 0.03], 0.9, [0.9, 0.9, 0.9])
    # a shutter: big, faint, far apart -- but touching each other
    sy, sz = np.meshgrid(np.arange(-1.6, -0.55, 0.25), np.arange(-0.6, 0.61, 0.25))
    sh = np.column_stack([np.full(sy.size, 1.0), sy.ravel(), sz.ravel()])
    parts["shutter"] = _set(sh, [0.01, 0.16, 0.16], 0.15, [0.5, 0.5, 0.55])
    # an antenna: small opaque splats, 0.15 apart, standing on the roof
    ay = np.arange(-2.05, -3.5, -0.15)
    parts["antenna"] = _set(np.column_stack([np.full(ay.size, 0.5), ay, np.full(ay.size, 0.5)]),
                            [0.02, 0.05, 0.02], 0.9, [0.3, 0.3, 0.3])
    rng = np.random.default_rng(1)
    fl = np.column_stack([rng.uniform(-3, 3, 30), rng.uniform(-5, -4, 30), rng.uniform(-3, 3, 30)])
    parts["floaters"] = _set(fl, [0.08, 0.08, 0.08], 0.2, [0.6, 0.6, 0.7])
    cl = np.array([2.5, -3.0, 2.5]) + rng.normal(0, 0.03, (15, 3))
    parts["cluster"] = _set(cl, [0.03, 0.03, 0.03], 0.5, [0.2, 0.8, 0.2])
    hz = np.column_stack([rng.uniform(-2.5, 2.5, 300), rng.uniform(0.1, 0.5, 300),
                          rng.uniform(-2.5, 2.5, 300)])
    parts["haze"] = _set(hz, [0.15, 0.04, 0.15], 0.12, [0.7, 0.6, 0.4])
    s, labels = None, []
    for k, p in parts.items():
        s = p if s is None else s.concat(p)
        labels += [k] * p.n
    return s, np.array(labels)


S, LAB = scene()
_ST = {}


def stats(use_gpu):
    if use_gpu not in _ST:
        _ST[use_gpu] = cleanup.analyse(S, use_gpu=use_gpu)
    return _ST[use_gpu]


def _report(st):
    for k in ("ground", "walls", "shutter", "antenna", "floaters", "cluster", "haze"):
        m = LAB == k
        print("    %-9s n=%5d  score max %.2f  >=0.3: %d" % (
            k, m.sum(), st.score[m].max(), (st.score[m] >= 0.3).sum()))


def test_old_measure_alone_takes_structure():
    # The reason for the rework: the isolation/opacity/size score on its own
    # convicts the shutter and the antenna.
    st = stats(False)
    for k in ("shutter", "antenna"):
        m = LAB == k
        assert (st.base[m] >= 0.3).mean() > 0.5, "%s: base %.2f" % (k, np.median(st.base[m]))
    print("    the old score alone flags shutter and antenna")


def test_floaters_found_structure_kept_cpu():
    st = stats(False)
    _report(st)
    for k in ("floaters", "cluster"):
        assert (st.score[LAB == k] >= 0.9).all(), k
    for k in ("ground", "walls", "shutter", "antenna"):
        assert (st.score[LAB == k] < 0.3).all(), k


def test_haze_found_structure_kept_gpu():
    ok, why = gpu.available()
    if not ok:
        print("    skipped: " + why)
        return
    st = stats(True)
    _report(st)
    assert st.gpu
    for k in ("floaters", "cluster"):
        assert (st.score[LAB == k] >= 0.9).all(), k
    for k in ("ground", "walls", "shutter", "antenna"):
        assert (st.score[LAB == k] < 0.3).all(), k
    assert (st.score[LAB == "haze"] >= 0.3).mean() > 0.6, \
        (st.score[LAB == "haze"] >= 0.3).mean()


def _mirrored():
    """The hut in a +y-up file: what the app's "Oben +Y" toggle declares."""
    F = S.take(np.arange(S.n))
    F.xyz = (S.xyz * np.array([1, -1, 1], np.float32)).astype(np.float32)
    F.rot = S.rot.copy()
    return F


def test_up_axis_is_respected_gpu():
    # The haze test splits its cameras into "from above" and "from below" by
    # the up axis. Measured with the default -y on a +y-up file it looked at
    # the sand from above and found none of the haze (0 of 300); told +y it
    # finds what it finds in the original.
    ok, why = gpu.available()
    if not ok:
        print("    skipped: " + why)
        return
    F = _mirrored()
    right = cleanup.analyse(F, use_gpu=True, up=(0.0, 1.0, 0.0))
    wrong = cleanup.analyse(F, use_gpu=True)
    orig = stats(True)
    for k in ("ground", "walls", "shutter", "antenna"):
        assert (right.score[LAB == k] < 0.3).all(), k
    for k in ("floaters", "cluster"):
        assert (right.score[LAB == k] >= 0.9).all(), k
    h_r = int((right.score[LAB == "haze"] >= 0.3).sum())
    h_o = int((orig.score[LAB == "haze"] >= 0.3).sum())
    h_w = int((wrong.score[LAB == "haze"] >= 0.3).sum())
    assert abs(h_r - h_o) <= 0.05 * h_o + 3, (h_r, h_o)
    assert h_w < 0.5 * h_r, "the wrong up found the haze too: %d vs %d" % (h_w, h_r)
    print("    up +y       haze %d (like the original %d), with -y only %d" % (h_r, h_o, h_w))


def test_up_axis_reaches_the_detector():
    # The app sends its up axis with "floaters"; the server passes it to the
    # document, the document to the detector, and the cache keys on it.
    from splatcore.document import Document
    import splatcore.cleanup as cl
    sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))
    import server
    seen = []
    orig = cl.analyse

    def spy(s, *a, up=(0.0, -1.0, 0.0), **k):
        seen.append(tuple(up))
        return orig(s, *a, use_gpu=False, up=up)
    try:
        cl.analyse = spy
        sess = server.Session()
        sess.doc = Document(S)
        sess.cmd_floaters({"threshold": 0.3, "up": [0, 1, 0]}, b"")
        sess.cmd_floaters({"threshold": 0.5, "up": [0, 1, 0]}, b"")      # cached
        sess.cmd_floaters({"threshold": 0.3}, b"")                        # default, measured again
        sess.cmd_floaters({"threshold": 0.3, "up": [0, 0, 0]}, b"")       # unusable -> default
    finally:
        cl.analyse = orig
    assert seen == [(0.0, 1.0, 0.0), (0.0, -1.0, 0.0)], seen
    print("    up          from the command to the detector, measured once per axis")


def tree_scene():
    """A tree at the edge of the ground, its crown hanging out over nothing,
    and under the crown the fringe the castle scan has: a ring of big,
    opaque, purplish blobs far from anything. Next to it on the ground a wall
    of big, sparse panels, seen from the side like the blobs, but with the
    ground between them and every view from below."""
    parts = {}
    g = np.arange(-3.0, 3.0, 0.05)
    X, Z = np.meshgrid(g, g)
    top = np.column_stack([X.ravel(), np.zeros(X.size), Z.ravel()])
    parts["ground"] = _set(np.vstack([top, top + [0, 0.03, 0]]),
                           [0.035, 0.006, 0.035], 0.95, [0.5, 0.45, 0.3])
    ty = np.arange(-0.02, -1.6, -0.04)
    parts["trunk"] = _set(np.column_stack([np.full(ty.size, 2.8), ty, np.zeros(ty.size)]),
                          [0.04, 0.04, 0.04], 0.9, [0.35, 0.25, 0.15])
    rng = np.random.default_rng(3)
    v = rng.normal(size=(6000, 3))
    v /= np.linalg.norm(v, axis=1, keepdims=True)
    r = 0.7 * rng.uniform(0.75, 1.0, 6000) ** 0.5
    crown = np.array([3.3, -1.9, 0.0]) + v * r[:, None] * [1.0, 0.6, 1.0]
    parts["crown"] = _set(crown, [0.04, 0.04, 0.04], 0.8, [0.2, 0.55, 0.15])
    a = np.linspace(0, 2 * np.pi, 12, endpoint=False)
    fr = np.column_stack([3.6 + 0.4 * np.cos(a), np.full(a.size, -1.0), 0.4 * np.sin(a)])
    parts["fringe"] = _set(fr, [0.25, 0.25, 0.25], 0.9, [0.4, 0.3, 0.55])
    wy, wz = np.meshgrid(np.arange(-0.15, -1.0, -0.25), np.arange(-1.0, 1.01, 0.25))
    parts["panels"] = _set(np.column_stack([np.zeros(wy.size), wy.ravel(), wz.ravel()]),
                           [0.01, 0.3, 0.3], 0.9, [0.9, 0.9, 0.9])
    s, labels = None, []
    for k, p in parts.items():
        s = p if s is None else s.concat(p)
        labels += [k] * p.n
    return s, np.array(labels)


def test_fringe_under_crown_gpu():
    # The castle's purple fringe: big, opaque blobs far from anything under
    # a tree crown. From the side they are in front of everything, so the
    # haze test never had them (base x hidden <= 0.05 here); nothing from
    # below covers them and the steep views from above hardly see them.
    # The panels are just as big and sparse and also face sideways, but the
    # ground covers them from below: they stay, as do crown and trunk.
    ok, why = gpu.available()
    if not ok:
        print("    skipped: " + why)
        return
    T, TL = tree_scene()
    st = cleanup.analyse(T, use_gpu=True)
    assert st.gpu and st.fringe is not None
    m = TL == "fringe"
    found = int((st.score[m] >= 0.3).sum())
    old = int((np.maximum(np.where(st.detached, 0.95, 0.0), st.base * st.hidden)[m] >= 0.3).sum())
    assert found >= 9, "fringe: %d of 12" % found
    assert old <= 2, "the old score already had %d of the fringe" % old
    for k in ("ground", "trunk", "crown", "panels"):
        sk = st.score[TL == k]
        assert (sk < 0.3).all(), "%s: %d >= 0.3, max %.2f" % (k, (sk >= 0.3).sum(), sk.max())
    assert st.info["fringe"] >= found - old
    print("    fringe      %d of 12 (before %d), crown, trunk, floor, panels intact" % (found, old))


def test_fringe_needs_gpu():
    # Without the views there is no fringe test, and the score is as before.
    st = stats(False)
    assert st.fringe is None and st.steep is None
    assert set(np.unique(st.score)) <= {0.0, 0.95}


def test_server_says_why():
    # Every pick has one reason: cut off, hidden, or fringe. On the GPU the
    # tree scene's fringe is reported as such.
    from splatcore.document import Document
    import server
    ok, _ = gpu.available()
    T, TL = tree_scene()
    sess = server.Session()
    sess.doc = Document(T)
    r, _ = sess.cmd_floaters({"threshold": 0.3}, b"")
    assert r["detached"] + r["hidden"] + r["fringe"] == r["count"], r
    if ok:
        assert r["fringe"] >= 9, r
    print("    reasons     %d detached, %d hidden, %d fringe" % (
        r["detached"], r["hidden"], r["fringe"]))


def stilt_scene():
    """A hut on short posts: the ground has no splats under it, since no photo saw
    there, only haze filling the gap -- the beach hut scan. Its floor is
    seen from nowhere: not from above (the hut is on it), not from below
    (the haze is in front). Elsewhere the usual haze under the sand."""
    parts = {}
    g = np.arange(-3.0, 3.0, 0.05)
    X, Z = np.meshgrid(g, g)
    top = np.column_stack([X.ravel(), np.zeros(X.size), Z.ravel()])
    top = top[(np.abs(top[:, 0]) > 1.05) | (np.abs(top[:, 2]) > 1.05)]
    parts["ground"] = _set(np.vstack([top, top + [0, 0.03, 0]]),
                           [0.035, 0.006, 0.035], 0.95, [0.8, 0.7, 0.5])
    w = np.arange(-1.0, 1.0, 0.05)
    A, B = np.meshgrid(w, np.arange(-2.0, -0.15, 0.05))
    walls = [np.column_stack([np.full(A.size, s), B.ravel(), A.ravel()]) for s in (-1.0, 1.0)]
    walls += [np.column_stack([A.ravel(), B.ravel(), np.full(A.size, s)]) for s in (-1.0, 1.0)]
    R1, R2 = np.meshgrid(w, w)
    walls.append(np.column_stack([R1.ravel(), np.full(R1.size, -2.0), R2.ravel()]))
    parts["walls"] = _set(np.vstack(walls), [0.03, 0.03, 0.03], 0.9, [0.9, 0.9, 0.9])
    floor = np.column_stack([R1.ravel(), np.full(R1.size, -0.2), R2.ravel()])
    parts["floor"] = _set(floor, [0.03, 0.006, 0.03], 0.9, [0.2, 0.2, 0.25])
    rng = np.random.default_rng(5)
    hz = np.column_stack([rng.uniform(-0.95, 0.95, 300), rng.uniform(-0.05, 0.3, 300),
                          rng.uniform(-0.95, 0.95, 300)])
    parts["gap"] = _set(hz, [0.15, 0.04, 0.15], 0.12, [0.7, 0.6, 0.4])
    ring = rng.uniform(-2.6, 2.6, (2000, 2))
    ring = ring[(np.abs(ring[:, 0]) > 1.4) | (np.abs(ring[:, 1]) > 1.4)][:300]
    oh = np.column_stack([ring[:, 0], rng.uniform(0.1, 0.5, len(ring)), ring[:, 1]])
    parts["haze"] = _set(oh, [0.15, 0.04, 0.15], 0.12, [0.7, 0.6, 0.4])
    s, labels = None, []
    for k, p in parts.items():
        s = p if s is None else s.concat(p)
        labels += [k] * p.n
    return s, np.array(labels)


def test_haze_over_an_unseen_floor_stays_gpu():
    # Taking the haze from under a hut on posts opened a hole in the ground,
    # and from below the hut's never-seen floor showed through it. That haze
    # now stays; the haze under the open sand still goes.
    ok, why = gpu.available()
    if not ok:
        print("    skipped: " + why)
        return
    H, HL = stilt_scene()
    st = cleanup.analyse(H, use_gpu=True)
    gap = st.score[HL == "gap"] >= 0.3
    raw = (st.base * st.hidden_raw)[HL == "gap"] >= 0.3
    haze = st.score[HL == "haze"] >= 0.3
    for k in ("ground", "walls", "floor"):
        assert (st.score[HL == k] < 0.3).all(), k
    assert raw.mean() > 0.5, "the gap haze was never a candidate: %.2f" % raw.mean()
    assert gap.mean() < 0.2, "gap haze still taken: %.2f" % gap.mean()
    assert haze.mean() > 0.5, "open haze no longer found: %.2f" % haze.mean()
    assert st.kept(0.3) >= raw.sum() - gap.sum()
    print("    hut         gap: %d of %d stay, haze in the open: %d of %d gone" % (
        raw.sum() - gap.sum(), raw.sum(), haze.sum(), haze.size))


def test_select_and_histogram():
    st = stats(False)
    idx = st.select(0.5)
    assert set(LAB[idx]) <= {"floaters", "cluster"}
    assert sum(st.histogram()) == S.n


if __name__ == "__main__":
    fails = 0
    for name, fn in sorted(globals().items()):
        if name.startswith("test_"):
            print("  " + name)
            try:
                fn()
            except AssertionError as e:
                fails += 1
                print("    FAIL:", e)
    print("all cleanup tests passed" if not fails else "%d FAILURES" % fails)
    sys.exit(1 if fails else 0)
