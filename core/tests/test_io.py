"""Round-trip and layout checks for the PLY reader/writer."""
import os
import sys
import tempfile

import numpy as np

sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))
from splatcore.io import SplatSet, load_ply, save_ply, degree_for_rest  # noqa: E402


def _synthetic(n=97, deg=3):
    rng = np.random.default_rng(7)
    k = (deg + 1) ** 2 - 1
    return SplatSet(
        xyz=rng.normal(size=(n, 3)).astype(np.float32),
        f_dc=rng.normal(size=(n, 3)).astype(np.float32),
        f_rest=rng.normal(size=(n, k, 3)).astype(np.float32),
        opacity=rng.normal(size=n).astype(np.float32),
        scale=rng.normal(size=(n, 3)).astype(np.float32),
        rot=rng.normal(size=(n, 4)).astype(np.float32),
    )


def test_degree_for_rest():
    assert degree_for_rest(0) == 0
    assert degree_for_rest(9) == 1
    assert degree_for_rest(24) == 2
    assert degree_for_rest(45) == 3


def test_roundtrip_bit_exact():
    s = _synthetic()
    with tempfile.TemporaryDirectory() as d:
        p = os.path.join(d, "t.ply")
        save_ply(p, s)
        b = load_ply(p)
    for f in ("xyz", "f_dc", "f_rest", "opacity", "scale", "rot"):
        assert np.array_equal(getattr(s, f), getattr(b, f)), f
    assert b.sh_degree == 3


def test_channel_major_on_disk():
    """f_rest must be written channel-major: f_rest_0..K-1 are all red."""
    s = _synthetic(n=4, deg=1)          # K = 3 coeffs per channel
    s.f_rest[:] = 0.0
    s.f_rest[:, 0, 0] = 1.0             # coeff 0, red  -> f_rest_0
    s.f_rest[:, 0, 1] = 2.0             # coeff 0, green-> f_rest_3
    s.f_rest[:, 2, 2] = 3.0             # coeff 2, blue -> f_rest_8
    with tempfile.TemporaryDirectory() as d:
        p = os.path.join(d, "t.ply")
        save_ply(p, s, write_normals=False)
        raw = np.fromfile(p, dtype="<f4", offset=_header_len(p))
    row = raw.reshape(4, -1)[0]
    # layout: x y z | f_dc 0..2 | f_rest 0..8 | opacity | scale 0..2 | rot 0..3
    rest = row[6:15]
    assert rest[0] == 1.0 and rest[3] == 2.0 and rest[8] == 3.0
    assert rest[[1, 2, 4, 5, 6, 7]].sum() == 0.0


def test_copy_is_a_copy():
    """copy() must not share storage: slicing a numpy array gives a view."""
    s = _synthetic()
    c = s.copy()
    c.xyz[0, 0] += 1.0
    c.f_rest[1, 2, 0] += 1.0
    c.opacity[3] += 1.0
    assert s.xyz[0, 0] != c.xyz[0, 0], "copy() shares positions with the original"
    assert s.f_rest[1, 2, 0] != c.f_rest[1, 2, 0], "copy() shares SH"
    assert s.opacity[3] != c.opacity[3], "copy() shares opacity"


def test_subset_and_concat():
    s = _synthetic(n=50)
    a, b = s.take(np.arange(20)), s.take(np.arange(20, 50))
    j = a.concat(b)
    assert j.n == 50
    assert np.array_equal(j.xyz, s.xyz)


def _header_len(p):
    with open(p, "rb") as f:
        buf = f.read(8192)
    return buf.index(b"end_header") + len("end_header\n")


if __name__ == "__main__":
    for name, fn in sorted(globals().items()):
        if name.startswith("test_"):
            fn()
            print("PASS", name)
    print("all io tests passed")
