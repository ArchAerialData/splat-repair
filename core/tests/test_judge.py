"""The refine judge (splatcore/judge.py): keep the refine only when it looks better.

Synthetic surfaces where the answer is plain: a fill that copies the ring's
pattern beats a washed-out one, and a plain fill beats one with dark streaks
the ring does not have. Then the wiring in `Document.refine_last`: a refine
the judge turns down leaves the quick fill's colours, says so in ``info``,
and the repair stays one undo step either way.

Needs the GPU (gsplat); without it every test says so and passes.
"""
import os
import sys

import numpy as np

sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))
sys.path.insert(0, os.path.dirname(__file__))
from splatcore.io import SplatSet, SH_C0                      # noqa: E402
from splatcore import gpu                                     # noqa: E402
from splatcore import judge as J                              # noqa: E402


def _no_gpu():
    ok, why = gpu.available()
    if not ok:
        print("    skipped: " + why)
    return not ok


def plane(colour_fn, step=0.035, extent=3.0):
    """Flat splats on z = 0, a grid ``step`` apart, coloured by ``colour_fn(x, y)``."""
    g = np.arange(-extent, extent + 1e-9, step)
    x, y = np.meshgrid(g, g)
    x, y = x.ravel(), y.ravel()
    n = len(x)
    rgb = np.clip(colour_fn(x, y), 0.0, 1.0)
    return SplatSet(
        xyz=np.column_stack([x, y, np.zeros(n)]).astype(np.float32),
        f_dc=((rgb - 0.5) / SH_C0).astype(np.float32),
        f_rest=np.zeros((n, 15, 3), np.float32),
        opacity=np.full(n, 4.0, np.float32),
        scale=np.log(np.tile([0.6 * step, 0.6 * step, 0.1 * step], (n, 1))).astype(np.float32),
        rot=np.tile([1.0, 0.0, 0.0, 0.0], (n, 1)).astype(np.float32),
    )


def lattice(x, y):
    """A diamond lattice, as on the shoe's sole: brown, strong thin ridges."""
    u = np.abs(np.sin((x + y) * np.pi / 0.35)) < 0.25
    v = np.abs(np.sin((x - y) * np.pi / 0.35)) < 0.25
    t = 0.42 + 0.22 * (u | v)
    return np.column_stack([t, 0.8 * t, 0.6 * t])


def planks(x, y):
    """A cream wall with faint horizontal plank joints, as on the hut."""
    j = np.abs(np.sin(y * np.pi / 0.3)) < 0.12
    t = 0.85 - 0.08 * j
    return np.column_stack([t, 0.95 * t, 0.85 * t])


def split(s, r=1.0):
    inside = np.hypot(s.xyz[:, 0], s.xyz[:, 1]) < r
    return s.take(~inside), s.take(inside)


def recolour(tail, rgb):
    out = tail.copy()
    out.f_dc = ((np.clip(rgb, 0, 1) - 0.5) / SH_C0).astype(np.float32)
    return out


def verdict(world_fn, quick_rgb_fn, refined_rgb_fn):
    s = plane(world_fn)
    rest, tail = split(s)
    x, y = tail.xyz[:, 0].astype(float), tail.xyz[:, 1].astype(float)
    quick = recolour(tail, quick_rgb_fn(x, y))
    refined = recolour(tail, refined_rgb_fn(x, y))
    whole = rest.concat(tail)
    removed = np.arange(rest.n, whole.n)
    return J.judge(whole, removed, quick, refined, view=[0.0, 0.0, -1.0])


def test_washed_out_refine_loses_to_a_copied_lattice():
    # the shoe's stand: the quick fill keeps the lattice, the refine smears it
    if _no_gpu():
        return
    mean = np.array([0.48, 0.384, 0.288])          # the lattice's own average
    v = verdict(lattice, lattice, lambda x, y: np.tile(mean, (len(x), 1)))
    print("   ", {k: v.get(k) for k in ("keep", "quick", "refined", "why", "seconds")})
    assert v["keep"] is False, v
    assert v["refined"] > v["quick"], v
    assert v["terms"]["refined"]["missing"] > v["terms"]["quick"]["missing"], v
    assert v["why"] == "it was sharper", v
    # the soft refine has the right colours: the blob term stays quiet
    assert v["terms"]["refined"]["blobs"] < 0.5, v


def test_streaked_quick_fill_loses_to_a_clean_refine():
    # the EHBO letters: dark vertical streaks across the planks in the quick fill
    if _no_gpu():
        return

    def streaks(x, y):
        c = planks(x, y)
        bad = np.abs(np.sin(x * np.pi / 0.45)) < 0.1
        c[bad] *= 0.55
        return c

    v = verdict(planks, streaks, planks)
    print("   ", {k: v.get(k) for k in ("keep", "quick", "refined", "why", "seconds")})
    assert v["keep"] is True, v
    assert v["terms"]["quick"]["foreign"] > v["terms"]["refined"]["foreign"], v


def test_dark_blob_refine_loses():
    # the castle's meadow: the refine came back as a dark blurred blob
    if _no_gpu():
        return

    def blob(x, y):
        return lattice(x, y) * 0.6 + 0.4 * np.array([0.48, 0.384, 0.288]) * 0.6

    v = verdict(lattice, lattice, blob)
    assert v["keep"] is False, v
    assert v["terms"]["refined"]["tone"] > v["terms"]["quick"]["tone"], v


def test_grey_patch_on_stone_by_grass_loses():
    # the castle floor: the ring is half pink stone, half grass; the quick fill
    # put a grey-green patch on the stone side, the refine is a little soft
    # but in the right colours. Neither colour is foreign to the ring as a
    # whole -- only to the stone pixels next to the patch.
    if _no_gpu():
        return
    stone = np.array([0.80, 0.62, 0.55])
    grass = np.array([0.35, 0.55, 0.20])

    def floor(x, y):
        c = np.where((y > 0)[:, None], stone, grass) * (1.0 + 0.06 * np.sin(x * 9.0)[:, None])
        return c

    def grey_patch(x, y):
        c = floor(x, y)
        bad = (y > 0.1) & (np.hypot(x, y - 0.45) < 0.4)
        c[bad] = np.array([0.55, 0.57, 0.48])
        return c

    def soft(x, y):
        t = np.clip(0.5 + y / 0.3, 0.0, 1.0)[:, None]
        return t * stone + (1 - t) * grass

    v = verdict(floor, grey_patch, soft)
    print("   ", {k: v.get(k) for k in ("keep", "quick", "refined", "why")})
    assert v["keep"] is True, v
    # (a soft edge between two materials mixes their colours: that counts a little)
    assert v["terms"]["quick"]["blobs"] > 2.0 * v["terms"]["refined"]["blobs"], v


def test_same_fill_is_not_swapped():
    # nothing to gain: the quick fill stays (a tie never swaps)
    if _no_gpu():
        return
    v = verdict(lattice, lattice, lattice)
    assert v["keep"] is False, v
    assert abs(v["quick"] - v["refined"]) < 1e-6, v


def test_patch_frame_follows_the_view_and_the_long_axis():
    s = plane(planks, extent=1.0)
    tail = s.take((np.abs(s.xyz[:, 0]) < 0.8) & (np.abs(s.xyz[:, 1]) < 0.2))
    fr = J.patch_frame(tail, None, view=[0.0, 0.0, 1.0])
    assert fr["n"] @ np.array([0.0, 0.0, -1.0]) > 0.99, fr["n"]       # toward the camera
    assert abs(fr["t1"][0]) > 0.99, fr["t1"]                           # long axis along x
    assert fr["a"] > 2.5 * fr["b"], (fr["a"], fr["b"])


def test_refine_last_keeps_quick_fill_when_judge_says_so():
    """The wiring: a refine the judge turns down changes nothing, one undo step."""
    if _no_gpu():
        return
    import splatcore.viewfill as vf
    from splatcore.document import Document
    from splatcore.repair import RepairParams, RepairResult
    from test_repair import make_surface, disc

    s = make_surface(lambda x, y: 0.0 * x, n=60000, extent=4.0, period=0.6)
    doc = Document(s)
    doc.fill_engine = "copy"       # refine_last builds on the copy engine's fill
    sel = np.where(disc(s, r=1.0))[0]
    e = doc.repair(sel, RepairParams(view=[0.0, 0.0, -1.0], seed=0), objects=False)
    assert e.added > 0, e.info
    quick_dc = doc.generated.f_dc.copy()
    steps = len(doc.undo_stack)

    def flat_refine(live, local, params, source="lama", deep=None, base=None, **kw):
        # what a washed-out deep fill hands back: same splats, one flat colour
        added = base.added.copy()
        added.f_dc[:] = added.f_dc.mean(0)
        added.f_rest[:] = 0.0
        info = dict(base.info)
        info.update({"deep": True, "fill_source": "LaMa", "fit": 0.01})
        return RepairResult(removed=base.removed, added=added, backing=base.backing,
                            info=info, context=dict(base.context))

    real = vf.view_fill
    vf.view_fill = flat_refine
    try:
        e2 = doc.refine_last()
        assert e2 is e
        print("   ", e.info.get("judge", {}).get("why"), e.info["judge"].get("quick"),
              e.info["judge"].get("refined"))
        assert e.info["refineKept"] is False, e.info.get("judge")
        assert e.info["refined"] is False
        assert np.array_equal(doc.generated.f_dc, quick_dc)       # the quick fill stayed
        assert len(doc.undo_stack) == steps
        doc.undo()
        assert doc.n_alive == s.n and not doc.can_undo

        # without the judge the same refine swaps in, still one step
        e = doc.repair(sel, RepairParams(view=[0.0, 0.0, -1.0], seed=0), objects=False)
        doc.refine_last(judge=False)
        assert e.info["refined"] is True and e.info["refineKept"] is None
        assert not np.array_equal(doc.generated.f_dc, quick_dc)
        assert len(doc.undo_stack) == 1
        doc.undo()
        assert doc.n_alive == s.n
    finally:
        vf.view_fill = real


if __name__ == "__main__":
    for name, fn in sorted(globals().items()):
        if name.startswith("test_"):
            fn()
            print("PASS", name)
    print("all judge tests passed")
