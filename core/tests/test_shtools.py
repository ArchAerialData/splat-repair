"""Checks that SH rotation is actually a rotation of the radiance field."""
import os
import sys

import numpy as np

sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))
from splatcore import shtools as sh  # noqa: E402


def _rand_rot(rng):
    A = rng.normal(size=(3, 3))
    Q, R = np.linalg.qr(A)
    return Q * np.sign(np.diag(R))


def test_basis_orthonormal():
    """Real SH are orthonormal over the sphere; Monte-Carlo integrate to check."""
    rng = np.random.default_rng(0)
    d = rng.normal(size=(400000, 3))
    d /= np.linalg.norm(d, axis=1, keepdims=True)
    B = sh.sh_basis(d)
    G = (B.T @ B) / len(d) * 4.0 * np.pi
    assert np.abs(G - np.eye(16)).max() < 0.02, np.abs(G - np.eye(16)).max()


def test_rotation_matches_resampled_field():
    """Rotating coefficients must equal evaluating the field at rotated dirs."""
    rng = np.random.default_rng(1)
    R = _rand_rot(rng)
    coeffs = rng.normal(size=(5, 15, 3))
    rot = sh.rotate_sh(coeffs, R)

    d = rng.normal(size=(2000, 3))
    d /= np.linalg.norm(d, axis=1, keepdims=True)
    basis = sh.sh_basis(d)[:, 1:]
    # field of the rotated splat at d  ==  field of the original at R^T d
    got = np.einsum("sk,nkc->snc", basis, rot)
    want = np.einsum("sk,nkc->snc", sh.sh_basis(d @ R)[:, 1:], coeffs)
    assert np.abs(got - want).max() < 1e-8, np.abs(got - want).max()


def test_identity_rotation_is_noop():
    rng = np.random.default_rng(2)
    c = rng.normal(size=(7, 15, 3))
    assert np.abs(sh.rotate_sh(c, np.eye(3)) - c).max() < 1e-9


def test_rotation_preserves_band_energy():
    """Rotation is orthogonal per band, so per-band power is invariant."""
    rng = np.random.default_rng(3)
    c = rng.normal(size=(20, 15, 3))
    r = sh.rotate_sh(c, _rand_rot(rng))
    for lo, hi in sh.BANDS:
        a = (c[:, lo:hi] ** 2).sum(axis=(1, 2))
        b = (r[:, lo:hi] ** 2).sum(axis=(1, 2))
        assert np.allclose(a, b, rtol=1e-6), (lo, hi)


def test_quat_matrix_roundtrip():
    rng = np.random.default_rng(4)
    q = rng.normal(size=(200, 4))
    q /= np.linalg.norm(q, axis=1, keepdims=True)
    q *= np.sign(q[:, :1])                      # canonical hemisphere
    back = sh.matrix_to_quat(sh.quat_to_matrix(q))
    back *= np.sign(back[:, :1])
    assert np.abs(back - q).max() < 1e-6, np.abs(back - q).max()


def test_rotation_between():
    rng = np.random.default_rng(5)
    for _ in range(50):
        a = rng.normal(size=3); a /= np.linalg.norm(a)
        b = rng.normal(size=3); b /= np.linalg.norm(b)
        R = sh.rotation_between(a, b)
        assert np.abs(R @ a - b).max() < 1e-9
        assert np.abs(R.T @ R - np.eye(3)).max() < 1e-9
    # degenerate cases
    a = np.array([0.0, 0.0, 1.0])
    assert np.abs(sh.rotation_between(a, a) - np.eye(3)).max() < 1e-12
    R = sh.rotation_between(a, -a)
    assert np.abs(R @ a + a).max() < 1e-9


if __name__ == "__main__":
    for name, fn in sorted(globals().items()):
        if name.startswith("test_"):
            fn()
            print("PASS", name)
    print("all shtools tests passed")
