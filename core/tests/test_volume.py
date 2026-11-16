"""The painted selection volume (splatcore.volume): membership, order, wire,
projection into other cameras, and the app's own answer for the same stroke.

Run as a script:  .venv\\Scripts\\python.exe tests\\test_volume.py
With SPLAT_VOLUME_DUMP=<dir> it also checks the dumps the app's autotest
writes (volume-*.json): the engine's reference test must select exactly the
splats the viewport selected.
"""
import glob
import json
import os
import sys

import numpy as np

sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))
from splatcore import volume as vol  # noqa: E402
from splatcore.gpu import look_at_camera  # noqa: E402


def _row(a, b, r, depth, seq=0, view=(0.0, 0.0, 1.0)):
    return [*a, *b, r, depth, seq, *view]


def _cam(eye, target, W=160, H=120, fov=40.0, up=(0.0, -1.0, 0.0)):
    return look_at_camera(np.array(eye, float), np.array(target, float),
                          np.array(up, float), W, H, fov)


def _pc_vp(cam, near=0.01, far=1000.0):
    """The PlayCanvas view-projection (column-major, as the app sends it)
    for the same pinhole as an OpenCV ``gpu.Camera``."""
    R = cam.view[:3, :3]
    t = cam.view[:3, 3]
    eye = -R.T @ t
    r, d, f = R[0], R[1], R[2]
    V = np.eye(4)
    V[0, :3], V[0, 3] = r, -r @ eye
    V[1, :3], V[1, 3] = -d, d @ eye
    V[2, :3], V[2, 3] = -f, f @ eye
    fy = cam.K[1, 1]
    W, H = cam.width, cam.height
    ft = 2 * fy / H                       # 1 / tan(fov/2)
    P = np.zeros((4, 4))
    P[0, 0] = ft * H / W
    P[1, 1] = ft
    P[2, 2] = (far + near) / (near - far)
    P[2, 3] = 2 * far * near / (near - far)
    P[3, 2] = -1.0
    M = P @ V
    return [float(x) for x in M.T.reshape(16)]


# ------------------------------------------------------------------ capsules

def test_single_dab_is_a_flattened_ball():
    v = vol.Volume.from_wire({"capsules": [_row((0, 0, 0), (0, 0, 0), 1.0, 0.25)],
                              "view": [0, 0, 1]})
    P = np.array([[0.9, 0, 0], [1.1, 0, 0], [0, 0.9, 0],
                  [0, 0, 0.24], [0, 0, 0.26], [0, 0, -0.24], [0, 0, -0.26]])
    assert v.inside(P).tolist() == [True, False, True, True, False, True, False]


def test_segment_end_is_inside_its_own_capsule():
    # The old selector measured the squash from `a` only, so with a depth
    # step between the dabs the far end `b` fell outside its own capsule.
    a, b = (0.0, 0.0, 0.0), (1.0, 0.0, 0.3)
    v = vol.Volume.from_wire({"capsules": [_row(a, b, 0.2, 0.1)], "view": [0, 0, 1]})
    assert v.inside([b, a, (0.5, 0.0, 0.15)]).all()
    assert not v.inside([(0.5, 0.0, 0.15 + 0.2)]).any()


def test_eight_number_rows_are_accepted():
    v = vol.Volume.from_wire({"capsules": [[0, 0, 0, 1, 0, 0, 0.3, 0.1]],
                              "subtract": [[0.5, 0, 0, 0.5, 0, 0, 0.2, 0.2]],
                              "view": [0, 0, 2]})
    # union minus subtract when no order is given
    assert v.inside([(0.1, 0, 0), (0.5, 0, 0), (0.9, 0, 0)]).tolist() == [True, False, True]


def test_newest_wins():
    add = _row((0, 0, 0), (2, 0, 0), 0.3, 0.1, seq=1)
    sub = _row((1, 0, 0), (1, 0, 0), 0.25, 0.2, seq=2)
    again = _row((1, 0, 0), (1, 0, 0), 0.1, 0.1, seq=3)
    v = vol.Volume.from_wire({"capsules": [add, again], "subtract": [sub], "view": [0, 0, 1]})
    got = v.inside([(0.2, 0, 0), (1.0, 0, 0), (1.18, 0, 0), (1.8, 0, 0)]).tolist()
    assert got == [True, True, False, True], got


def test_wire_round_trip():
    rng = np.random.default_rng(3)
    caps = [_row(rng.normal(size=3), rng.normal(size=3), 0.4, 0.15, seq=k % 3,
                 view=tuple(rng.normal(size=3))) for k in range(6)]
    v = vol.Volume.from_wire({"capsules": caps[:4], "subtract": caps[4:], "view": [0, 1, 0]})
    w = vol.Volume.from_wire(json.loads(json.dumps(v.to_wire())))
    P = rng.normal(size=(4000, 3))
    assert np.array_equal(v.inside(P), w.inside(P))
    assert v.inside(P).any()


# ------------------------------------------------------------------- regions

def _square_region(cam, d0, x0=40, y0=30, x1=120, y1=90, mode="add", seq=1, hole=False):
    W, H = cam.width, cam.height
    gx0, gy0 = x0 // 6, y0 // 6
    gw, gh = (x1 // 6) - gx0 + 1, (y1 // 6) - gy0 + 1
    depth = [float(d0)] * (gw * gh)
    if hole:                      # no surface in the middle -- filled by the app
        depth[(gh // 2) * gw + gw // 2] = None
    return {"mode": mode, "seq": seq, "view": [0, 0, 1], "vp": _pc_vp(cam),
            "size": [W, H], "polygon": [[x0, y0], [x1, y0], [x1, y1], [x0, y1]],
            "through": False, "cell": 6, "grid": [gx0, gy0, gw, gh], "depth": depth,
            "back_per_depth": 0.02, "front_scale": 1.2,
            "near": d0 * 0.9, "far": d0 * 1.1}


def test_region_window_around_the_surface():
    cam = _cam((0, 0, -5), (0, 0, 0))
    v = vol.Volume.from_wire({"regions": [_square_region(cam, 5.0)], "view": [0, 0, 1]})
    # straight ahead: camera depth = 5 + z. back = 0.02*5 = 0.1, front 0.12
    P = np.array([[0, 0, 0], [0, 0, 0.09], [0, 0, 0.11], [0, 0, -0.11], [0, 0, -0.13],
                  [3, 0, 0]])
    assert v.inside(P).tolist() == [True, True, False, True, False, False]


def test_region_projects_back_onto_its_polygon():
    cam = _cam((0, 0, -5), (0, 0, 0))
    g = _square_region(cam, 5.0)
    v = vol.Volume.from_wire({"regions": [g], "view": [0, 0, 1]})
    m = v.mask_in_view(cam, cam.width, cam.height)
    want = vol.fill_polygon(np.array(g["polygon"], float), cam.width, cam.height)
    diff = (m != want).sum()
    assert diff <= 0.03 * want.sum(), (diff, want.sum())
    # and from the side the slab is thin
    side = _cam((5, 0, 0), (0, 0, 0))
    ms = v.mask_in_view(side, side.width, side.height)
    assert 0 < ms.sum() < 0.5 * want.sum()


def test_surface_points_cover_a_hole():
    cam = _cam((0, 0, -5), (0, 0, 0))
    g = _square_region(cam, 5.0)
    v = vol.Volume.from_wire({"regions": [g], "view": [0, 0, 1]})
    P = v.surface_points()
    assert len(P) > 50 and np.allclose(P[:, 2], 0.0, atol=1e-6)


# ---------------------------------------------------------------- projection

def _brute_mask(v, cam, pix, zmax=20.0, n=6000):
    R, t, K = cam.view[:3, :3], cam.view[:3, 3], cam.K
    eye = -R.T @ t
    out = []
    z = np.linspace(1e-3, zmax, n)
    for (y, x) in pix:
        d = np.array([(x + 0.5 - K[0, 2]) / K[0, 0], (y + 0.5 - K[1, 2]) / K[1, 1], 1.0]) @ R
        out.append(v.inside(eye[None, :] + z[:, None] * d[None, :]).any())
    return np.array(out)


def test_mask_matches_brute_force_from_other_views():
    rng = np.random.default_rng(11)
    pts = np.cumsum(rng.normal(scale=0.15, size=(12, 3)), 0)
    caps = [_row(pts[k], pts[k + 1], 0.25, 0.08, seq=1, view=(0, 0, 1)) for k in range(11)]
    sub = [_row(pts[5], pts[5], 0.2, 0.16, seq=2, view=(0, 0, 1))]
    v = vol.Volume.from_wire({"capsules": caps, "subtract": sub, "view": [0, 0, 1]})
    c = pts.mean(0)
    for eye in ((0, 0, -6), (6, 0, 0), (3, -4, -3)):
        cam = _cam(c + np.array(eye, float), c, W=96, H=72)
        m = v.mask_in_view(cam, 96, 72)
        assert m.sum() > 20
        ys, xs = np.mgrid[0:72, 0:96]
        # every pixel near the outline plus a random sample elsewhere
        pick = rng.choice(96 * 72, 500, replace=False)
        pix = list(zip(ys.ravel()[pick], xs.ravel()[pick]))
        want = _brute_mask(v, cam, pix)
        got = m[tuple(np.array(pix).T)]
        assert (want != got).sum() <= 3, (eye, (want != got).sum())


def test_unpainting_from_the_paint_view_clears_the_mask():
    # Subtract with twice the window, as the app sends it: no sliver is left
    # behind for the mask to keep.
    add = _row((0, 0, 0), (1, 0, 0.05), 0.3, 0.1, seq=1)
    sub = _row((0.5, 0, 0.02), (0.5, 0, 0.02), 0.25, 0.2, seq=2)
    v = vol.Volume.from_wire({"capsules": [add], "subtract": [sub], "view": [0, 0, 1]})
    cam = _cam((0.5, 0, -4), (0.5, 0, 0), W=120, H=90)
    m = v.mask_in_view(cam, 120, 90)
    assert not m[45, 60] and m[45, 47] and m[45, 73], (m[45, 40:80].astype(int))


def test_mask_without_splats_is_the_painted_area():
    # nothing in the scene at all: the mask is still what was painted
    v = vol.Volume.from_wire({"capsules": [_row((0, 0, 0), (0, 0, 0), 0.5, 0.2)],
                              "view": [0, 0, 1]})
    cam = _cam((0, 0, -5), (0, 0, 0), W=100, H=100, fov=30)
    m = v.mask_in_view(cam, 100, 100)
    fy = cam.K[1, 1]
    want_r = 0.5 * fy / 5.0
    area = m.sum()
    assert abs(area - np.pi * want_r ** 2) < 0.12 * np.pi * want_r ** 2, (area, want_r)


# ----------------------------------------------------------- the app's answer

def test_app_dumps_agree():
    root = os.environ.get("SPLAT_VOLUME_DUMP")
    if not root:
        return
    from splatcore.formats import load_any
    files = sorted(glob.glob(os.path.join(root, "volume-*.json")))
    assert files, "no volume dumps in " + root
    cache = {}
    for f in files:
        with open(f, encoding="utf-8") as h:
            d = json.load(h)
        src = d["file"]
        if src not in cache:
            cache[src] = load_any(src).xyz
        xyz = cache[src]
        sel = np.asarray(d["selected"], np.int64)
        hidden = np.asarray(d.get("hidden") or [], np.int64)
        v = vol.Volume.from_wire(d["volume"])
        ins = v.inside(xyz[: d["nBase"]])
        ins[hidden[hidden < len(ins)]] = False
        got = np.where(ins)[0]
        a, b = set(got.tolist()), set(sel[sel < d["nBase"]].tolist())
        diff = len(a ^ b)
        print(f"  {os.path.basename(f)}: app {len(b)}, engine {len(a)}, differ {diff}")
        assert diff <= max(2, 0.0005 * len(b)), (f, diff)


if __name__ == "__main__":
    for name, fn in sorted(globals().items()):
        if name.startswith("test_"):
            fn()
            print("PASS", name)
    print("all volume tests passed")
