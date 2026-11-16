"""Tests for clearing what lies under a new skin (`splatcore/under.py`). Run as a script.

The scenes are made by hand: an old surface with a hole, a fine new skin over
the hole as the picture engines lay it, and under it what a scan leaves
there. Only the last test needs CUDA + gsplat (it goes through the picture
fill); without them it is skipped.
"""
import os
import sys

import numpy as np

HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, os.path.join(HERE, ".."))

from splatcore.io import SplatSet, SH_C0                      # noqa: E402
from splatcore import gpu                                     # noqa: E402
from splatcore.skin import finish, normals                    # noqa: E402
from splatcore.under import clear_under, UnderParams          # noqa: E402
from splatcore.document import Document                       # noqa: E402
from splatcore.repair import RepairParams                     # noqa: E402

SP = 0.05                               # the scan's splat spacing
R = 12 * SP                             # the hole's radius
VIEW = [0.0, 0.0, -1.0]                 # looking down on the surface


def splats(xyz, col, sig, logit=4.0):
    """Axis-aligned splats at ``xyz`` (sigma per axis ``sig``), one colour."""
    xyz = np.asarray(xyz, np.float64).reshape(-1, 3)
    n = len(xyz)
    q = np.zeros((n, 4), np.float32)
    q[:, 0] = 1.0
    return SplatSet(
        xyz=xyz.astype(np.float32),
        f_dc=np.tile(((np.asarray(col, float) - 0.5) / SH_C0).astype(np.float32), (n, 1)),
        f_rest=np.zeros((n, 15, 3), np.float32), opacity=np.full(n, logit, np.float32),
        scale=np.log(np.tile(np.asarray(sig, float), (n, 1))).astype(np.float32), rot=q)


def grid(half, pitch, z=0.0, seed=0):
    rng = np.random.default_rng(seed)
    g = np.arange(-half, half + 1e-9, pitch)
    x, y = [a.ravel() for a in np.meshgrid(g, g)]
    x = x + rng.normal(0, 0.08 * pitch, x.shape)
    y = y + rng.normal(0, 0.08 * pitch, y.shape)
    return np.stack([x, y, np.full_like(x, z)], 1)


def scene(extra=()):
    """The old surface (two layers), holed by the engine, and the new skin.

    Returns (scene, removed, skin, {name: indices into the scene}).
    ``extra``: [(name, SplatSet)] laid into the scene as well.
    """
    top = grid(40 * SP, SP, 0.0)
    low = grid(40 * SP, SP, -1.5 * SP, seed=1)             # the old skin's lower layer
    parts = [("top", splats(top, (0.6, 0.5, 0.4), (0.7 * SP, 0.7 * SP, 0.15 * SP))),
             ("low", splats(low, (0.55, 0.45, 0.35), (0.7 * SP, 0.7 * SP, 0.15 * SP)))] + list(extra)
    s, where, at = parts[0][1], {}, 0
    for k, (name, part) in enumerate(parts):
        if k:
            s = s.concat(part)
        where[name] = np.arange(at, at + part.n)
        at += part.n
    removed = where["top"][np.hypot(top[:, 0], top[:, 1]) < R]       # what the engine takes out
    fine = grid(R, SP / 4.0, 0.0, seed=2)
    fine = fine[np.hypot(fine[:, 0], fine[:, 1]) < R]
    skin = splats(fine, (0.6, 0.5, 0.4), (0.62 * SP / 4, 0.62 * SP / 4, 0.3 * SP / 4), logit=3.5)
    return s, removed, skin, where


def mush(n, r, z0, z1, seed=3):
    """Blobs under the surface: within ``r`` of the middle, ``z0``..``z1`` spacings deep."""
    rng = np.random.default_rng(seed)
    a, rr = rng.uniform(0, 2 * np.pi, n), r * np.sqrt(rng.uniform(0, 1, n))
    xyz = np.stack([rr * np.cos(a), rr * np.sin(a), -rng.uniform(z0, z1, n) * SP], 1)
    return splats(xyz, (0.12, 0.08, 0.06), (1.5 * SP,) * 3, logit=0.5)


def run(s, removed, skin, **kw):
    dbg = {}
    extra, info = clear_under(s, removed, skin, normals(skin, VIEW), SP, UnderParams(**kw), debug=dbg)
    gone = np.zeros(s.n, bool)
    gone[extra] = True
    assert not gone[removed].any(), "something the engine removed came back as cleared"
    assert info["cleared"] == len(extra) == info["candidates"] - info["kept"], info
    return gone, info


def radius(s, idx):
    return np.hypot(s.xyz[idx, 0], s.xyz[idx, 1])


def test_mush_under_the_skin_goes_and_what_lies_beside_it_stays():
    s, removed, skin, w = scene([("mush", mush(900, 20 * SP, 2.5, 7.0)),
                                 ("above", splats([[0.1, 0.0, 2 * SP], [-0.2, 0.1, 3 * SP]], (0.9, 0.1, 0.1), (SP,) * 3))])
    gone, info = run(s, removed, skin)
    r = radius(s, w["mush"])
    inner, outer = w["mush"][r < R - 3.0 * SP], w["mush"][r > R + 0.5 * SP]
    assert len(inner) > 150 and len(outer) > 150
    assert gone[inner].mean() > 0.9, gone[inner].mean()
    assert not gone[outer].any(), "mush beside the new skin was cleared"
    # the old skin's own lower layer: gone under the skin, whole beside it
    rl = radius(s, w["low"])
    assert gone[w["low"][rl < R - 3.0 * SP]].mean() > 0.95
    assert not gone[w["low"][rl > R]].any()
    # ... and nothing at the skin's rim, nothing above it, nothing of the surface around
    assert not gone[w["top"]].any() and not gone[w["above"]].any()
    print("    mush        %d cleared (under the skin %.0f %%), nothing beside it" % (info["cleared"], 100 * gone[inner].mean()))


def test_only_down_to_the_depth_limit():
    s, removed, skin, w = scene([("deep", mush(400, 8 * SP, 10.0, 14.0))])
    gone, _ = run(s, removed, skin)
    assert not gone[w["deep"]].any(), "cleared deeper than the limit"
    gone, _ = run(s, removed, skin, depth=16.0)
    assert gone[w["deep"]].mean() > 0.9
    print("    depth       below the limit everything stays, with depth=16 it goes")


def wall(tilt_deg=60.0):
    """A surface that enters the tube under the skin at a slant and carries on beyond it."""
    t = np.radians(tilt_deg)
    u, v = np.meshgrid(np.arange(-26, 26.01, 1.0) * SP, np.arange(0.6, 16.01, 1.0) * SP)
    rng = np.random.default_rng(5)
    u, v = u.ravel() + rng.normal(0, 0.05 * SP, u.size), v.ravel() + rng.normal(0, 0.05 * SP, v.size)
    xyz = np.stack([-0.1 + v * np.cos(t), u, -v * np.sin(t)], 1)
    return splats(xyz, (0.85, 0.82, 0.7), (0.6 * SP,) * 3)


def test_a_surface_that_carries_on_stays():
    """The shoe's side wall under the sole's edge: in the tube, but no mush."""
    s, removed, skin, w = scene([("wall", wall()), ("mush", mush(700, 20 * SP, 2.5, 7.0))])
    gone, info = run(s, removed, skin)
    # (where it passes through the old skin's lower layers, three spacings under the
    # new skin, a strip of it may go with them; below that it is whole)
    below = w["wall"][s.xyz[w["wall"], 2] < -4.0 * SP]
    assert gone[w["wall"]].mean() < 0.1 and gone[below].mean() < 0.01, \
        "the wall was cleared: %.2f, below the old skin %.2f" % (gone[w["wall"]].mean(), gone[below].mean())
    # the mush away from the wall still goes
    X = s.xyz[w["mush"]].astype(np.float64)
    t = np.radians(60.0)
    off = np.abs((X[:, 0] + 0.1) * np.sin(t) + X[:, 2] * np.cos(t))      # distance from the wall's plane
    far = w["mush"][(off > 2.5 * SP) & (radius(s, w["mush"]) < R - 3.0 * SP)]
    assert len(far) > 80 and gone[far].mean() > 0.9, (len(far), gone[far].mean())
    assert info["kept"] > 50
    print("    wall        stays (%d candidates held), %.0f %% of the mush beside it goes"
          % (info["kept"], 100 * gone[far].mean()))


def test_far_side_of_a_thin_part_stays():
    """A second sheet five spacings behind the skin (lining, the other side of a
    tongue): it runs on beyond the skin, and it is not the old skin's own layer."""
    back = splats(grid(40 * SP, SP, -5.0 * SP, seed=7), (0.8, 0.8, 0.75), (0.7 * SP, 0.7 * SP, 0.15 * SP))
    s, removed, skin, w = scene([("back", back)])
    gone, info = run(s, removed, skin)
    assert gone[w["back"]].mean() < 0.01, gone[w["back"]].mean()
    rl = radius(s, w["low"])
    assert gone[w["low"][rl < R - 3.0 * SP]].mean() > 0.95, "the old skin's lower layer stayed"
    print("    back side   stays whole, the lower layer of the old skin goes")


def test_the_coarse_layer_keeps_clear_of_the_far_side():
    """Both steps together: the sheet five spacings behind the skin stays, and the
    coarse layer (`lod.py`) lies above it, however large its splats would like to be."""
    back = splats(grid(40 * SP, SP, -5.0 * SP, seed=7), (0.8, 0.8, 0.75), (0.7 * SP, 0.7 * SP, 0.15 * SP))
    s, removed, skin, w = scene([("back", back)])
    rem, added, got = finish(s, removed, skin, VIEW, SP)
    gone = np.zeros(s.n, bool)
    gone[rem] = True
    assert gone[w["back"]].mean() < 0.01 and got["lod"]["added"] > 100
    layer = added.take(np.arange(skin.n, added.n))
    low = float(layer.xyz[:, 2].min()) / SP
    assert low > -5.0 + 0.7 + 0.7, low           # the sheet's own splats reach 0.7 up; 0.75 are kept free
    print("    layer       %d splats, the deepest %.2f spacings under the skin, the back side at 5" % (layer.n, -low))


def test_nothing_without_a_direction():
    s, removed, skin, w = scene([("mush", mush(300, 8 * SP, 2.5, 7.0))])
    extra, info = clear_under(s, removed, skin, None, SP)
    assert len(extra) == 0 and "why" in info
    rem, added, got = finish(s, removed, skin, None, SP, signed=False)
    assert len(rem) == len(removed) and added.n == skin.n and "why" in got["under"]
    print("    no view     nothing cleared: %s" % info["why"])


def test_finish_only_adds_to_what_the_engine_removed():
    s, removed, skin, w = scene([("mush", mush(600, 12 * SP, 2.5, 7.0))])
    rem, added, got = finish(s, removed, skin, VIEW, SP, back=False)
    assert set(removed.tolist()) <= set(rem.tolist()) and len(rem) == len(removed) + got["under"]["cleared"]
    assert got["lod"] is None and added.n == skin.n
    rem2, _, got2 = finish(s, removed, skin, VIEW, SP, clear=False, back=False)
    assert len(rem2) == len(removed) and got2["under"] is None
    print("    finish      %d more removed, none back" % got["under"]["cleared"])


def test_document_clears_in_the_same_undo_step():
    """Through the picture fill: the mush under the hole goes with the fill and
    comes back with one undo; switched off, it stays."""
    if not gpu.available()[0]:
        print("    (skipped: %s)" % gpu.available()[1])
        return
    from test_imfill import surface, disc
    base = surface()
    blobs = mush(500, 6 * SP, 2.5, 5.5)
    s = base.concat(blobs)
    idx = disc(base, 0.5)
    m = np.arange(base.n, s.n)
    d = Document(s)
    d.backing_layer = False
    e = d.repair(idx, RepairParams(seed=0, view=VIEW), objects=False)
    assert e.info.get("engine") == "image" and "under" in e.info, e.info
    gone = ~d.alive[m]
    assert gone.mean() > 0.9 and e.info["under"]["cleared"] >= gone.sum(), (gone.mean(), e.info["under"])
    assert e.info["removed"] == len(e.removed)
    d.undo()
    assert d.n_alive == s.n and d.alive[m].all()
    d.redo()
    assert (~d.alive[m]).mean() > 0.9
    d.undo()
    d.clear_under = False
    e = d.repair(idx, RepairParams(seed=0, view=VIEW), objects=False)
    assert "under" not in e.info and d.alive[m].all()
    print("    document    %d cleared with the fill, one step back, can be switched off" % int(gone.sum()))


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
    print("all under tests passed" if not fails else "%d FAILURES" % fails)
    sys.exit(1 if fails else 0)
