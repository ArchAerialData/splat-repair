"""Fill over an edge by unrolling it (`unfold`), on a synthetic corner.

Two walls meet at a right angle along y, with boards across the edge (stripes
in y) and a dark corner board along it. A stroke over the corner must come
back with the boards in phase on both walls and the corner board running on.
"""
import os
import sys

import numpy as np

sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))
from splatcore.io import SplatSet, SH_C0                      # noqa: E402
from splatcore import repair as rp                            # noqa: E402
from splatcore import unfold as uf                            # noqa: E402

PITCH = 0.4          # board pitch along the edge
BOARD = 0.12         # corner board half-width, measured on each wall


def expected_tone(y, d_edge):
    """Tone at height ``y`` and distance ``d_edge`` from the edge along a wall."""
    stripe = np.where(np.sin(2 * np.pi * y / PITCH) > 0, 0.75, 0.45)
    return np.where(d_edge < BOARD, 0.15, stripe)


def make_corner(n=60000, seed=0):
    rng = np.random.default_rng(seed)
    half = n // 2
    y = rng.uniform(-3, 3, n)
    d = rng.uniform(0, 3, n)                 # distance from the edge on the wall
    xyz = np.zeros((n, 3))
    xyz[:half, 0] = d[:half]                 # wall A: z = 0, x >= 0
    xyz[half:, 2] = d[half:]                 # wall B: x = 0, z >= 0
    xyz[:, 1] = y
    tone = expected_tone(y, d)
    rgb = np.column_stack([tone, tone * 0.9, tone * 0.8])
    return SplatSet(
        xyz=xyz.astype(np.float32),
        f_dc=((rgb - 0.5) / SH_C0).astype(np.float32),
        f_rest=np.zeros((n, 3, 3), np.float32),
        opacity=rng.uniform(1.5, 3.5, n).astype(np.float32),
        scale=rng.uniform(np.log(0.015), np.log(0.03), (n, 3)).astype(np.float32),
        rot=(lambda q: q / np.linalg.norm(q, axis=1, keepdims=True))(
            rng.normal(size=(n, 4))).astype(np.float32),
    )


def _stroke(s, r=0.7):
    return np.where(np.linalg.norm(s.xyz - np.array([0.0, 0.2, 0.0]), axis=1) < r)[0]


def test_corner_is_found():
    s = make_corner()
    sel = np.zeros(s.n, bool)
    sel[_stroke(s)] = True
    p = rp.RepairParams(view=[0.7071, 0.0, 0.7071])
    band = rp._rim_band(s.xyz.astype(np.float64), sel, s.alpha, p, np.random.default_rng(0))
    folds = uf.find_folds(s, sel, p, band)
    assert len(folds) == 1, "one edge, found %d" % len(folds)
    t = np.cross(folds[0]["nA"], folds[0]["nB"])
    t /= np.linalg.norm(t)
    assert abs(t[1]) > 0.95, "edge should run along y: %s" % np.round(t, 3)
    assert folds[0]["convex"], "the corner is convex"


def test_boards_and_corner_board_run_on():
    s = make_corner()
    idx = _stroke(s)
    res = rp.repair(s, idx, rp.RepairParams(view=[0.7071, 0.0, 0.7071], seed=0))
    info = res.info.get("unfold")
    assert info and info.get("unfold") == "rows", "the edge was not unrolled: %s" % info
    a = res.added
    frozen = np.asarray(res.context["unfold_frozen"], bool)
    assert frozen.sum() > 0.5 * a.n, "most of the fill should be copies along the edge"
    P = a.xyz[frozen].astype(np.float64)
    on_a = np.abs(P[:, 2]) < np.abs(P[:, 0])
    d_edge = np.where(on_a, P[:, 0], P[:, 2])
    want = expected_tone(P[:, 1], d_edge)
    got = a.rgb[frozen][:, 0]
    # away from stripe boundaries, the copy must be in phase
    ph = (P[:, 1] / PITCH) % 1.0
    clear = (np.abs(ph - 0.25) < 0.15) | (np.abs(ph - 0.75) < 0.15) | (d_edge < 0.8 * BOARD)
    clear &= (d_edge < 0.8 * BOARD) | (d_edge > 1.2 * BOARD)
    agree = np.mean(np.abs(got[clear] - want[clear]) < 0.1)
    print("    in phase: %.2f of %d copies" % (agree, clear.sum()))
    assert agree > 0.9, "boards or corner board out of phase (%.2f)" % agree
    # the copies sit on the walls
    off = np.minimum(np.abs(P[:, 0]), np.abs(P[:, 2]))
    assert np.median(off) < 0.02, "copies off the walls (%.3f)" % np.median(off)


def test_flat_stroke_is_left_alone():
    s = make_corner()
    far = np.where((np.linalg.norm(s.xyz - np.array([2.0, 0.0, 0.0]), axis=1) < 0.5))[0]
    res = rp.repair(s, far, rp.RepairParams(view=[0.0, 0.0, 1.0], seed=0))
    assert res.info.get("unfold") is None, "a stroke on one wall has no edge to unroll"


def test_a_stitch_is_a_one_off_a_joint_is_not():
    """Worst splat against its column and its row: a dark dot stands out in
    both, a dark joint running across the edge only in its column."""
    rng = np.random.default_rng(1)
    n_s, n_a = 40, 30
    I_ = rng.integers(0, n_s, 20000)
    J_ = rng.integers(0, n_a, 20000)
    lum = 0.7 + 0.02 * rng.normal(size=len(I_))
    joint = I_ % 8 == 0                       # a joint every eighth row
    lum[joint] = 0.3
    dot = (I_ == 13) & (J_ == 11)             # one stitch
    lum[dot] = 0.25
    colz, rowz = uf._line_worst(lum, I_, J_, n_s, n_a)
    odd = (colz > 4.0) & (rowz > 4.0)
    assert odd[13, 11], "the stitch should be a one-off"
    assert not odd[8].any() and not odd[16].any(), "a joint is the pattern, not a one-off"
    assert odd.sum() == 1, "only the stitch: %d cells" % odd.sum()


def test_source_inside_the_hole_is_not_taken():
    """With the edge followed only a little past the hole, no hole cell may
    copy from a cell that is itself in the hole (it holds no surface)."""
    s = make_corner()
    s = s.take(np.where(np.abs(s.xyz[:, 1]) < 1.6)[0])      # a short edge
    sel = np.zeros(s.n, bool)
    sel[np.where(np.linalg.norm(s.xyz, axis=1) < 0.7)[0]] = True
    p = rp.RepairParams(view=[0.7071, 0.0, 0.7071], seed=0)
    band = rp._rim_band(s.xyz.astype(np.float64), sel, s.alpha, p, np.random.default_rng(0))
    folds = uf.find_folds(s, sel, p, band)
    assert folds, "the edge should be found"
    D = {}
    got = uf.fill_fold(s, sel, p, band, folds[0], debug=D)
    assert got is not None, "not unrolled: %s" % D.get("why")
    hole, dcell = D["hole"], D["dcell"]
    i, j = np.where(D["src_ok"])
    assert not hole[i + dcell[i, j], j].any(), "a hole cell copies from the hole"
    assert D["src_ok"][hole].mean() > 0.95, "hole cells left without a source"


def test_a_bent_edge_copies_each_half_from_its_own_end():
    """With a reach, a row copies from no further past the nearer end of the
    hole than that: the first half from before the hole, the second after."""
    rng = np.random.default_rng(2)
    n_s, n_a = 120, 12
    f = rng.normal(size=(n_s, n_a, 7)) * 0.1
    f[..., 4] = 0.5
    hole = np.zeros((n_s, n_a), bool)
    hole[50:71, 3:9] = True
    known = ~hole
    f[hole] = 0.0
    rows = np.arange(50, 71)
    sh, _ = uf._choose_shifts(f, known, hole, rows, reach=8)
    assert sh is not None
    dl, dr = rows - 50 + 1, 70 - rows + 1
    near = np.minimum(dl, dr) + 8
    assert (np.abs(sh) <= near).all(), "a row reached past its own end: %s" % sh
    src = rows + sh
    assert ((src < 50) | (src > 70)).all(), "a row copied from the hole"
    assert (sh[:5] < 0).all() and (sh[-5:] > 0).all(), "each half from its own end: %s" % sh


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
    print("all unfold tests passed" if not fails else "%d FAILURES" % fails)
    sys.exit(1 if fails else 0)
