"""Moving, turning and scaling a selection.

A Gaussian is not a point, so a rigid transform touches four things, and
leaving any of them out is visible:

* **position** -- the obvious one.
* **orientation** -- the quaternion must be pre-rotated, or the anisotropic
  discs keep their old alignment and a rotated patch shears.
* **scale** -- under a uniform scale the splats must grow with the geometry,
  or a scaled-up patch turns into a sparse cloud of unchanged-size dots.
* **spherical harmonics** -- the view-dependent lobes live in world axes.
  Rotate the geometry and leave them, and the patch keeps lighting itself as
  if it were still facing the old way.

Only uniform scale is offered. A non-uniform one does not map a Gaussian to a
Gaussian unless its axes happen to line up with the scaling axes, so a
general-purpose version would quietly shear splats; the honest options are
uniform scaling or a full covariance rebuild, and the first is what a user
reaching for a scale handle almost always means.
"""
from __future__ import annotations

import numpy as np

from .io import SplatSet
from . import shtools


def quat_to_matrix(q) -> np.ndarray:
    """A single (w, x, y, z) quaternion as a 3x3 rotation matrix."""
    return shtools.quat_to_matrix(np.asarray(q, float).reshape(1, 4))[0]


def apply(s: SplatSet, indices, rotation=(1.0, 0.0, 0.0, 0.0),
          translation=(0.0, 0.0, 0.0), scale: float = 1.0,
          pivot=None, rotate_sh: bool = True) -> dict:
    """Transform the selected splats in place. Returns a short report."""
    idx = np.asarray(indices, np.int64)
    if len(idx) == 0:
        raise ValueError("the selection is empty")

    q = np.asarray(rotation, float)
    n = float(np.linalg.norm(q))
    if n < 1e-9:
        raise ValueError("the rotation is degenerate (zero quaternion)")
    q = q / n
    R = quat_to_matrix(q)
    t = np.asarray(translation, float)
    sc = float(scale)
    if not np.isfinite(sc) or sc <= 0:
        raise ValueError("the scale must be positive")

    p = s.xyz[idx].astype(np.float64)
    c = np.asarray(pivot, float) if pivot is not None else p.mean(0)

    # A pure move or scale leaves orientations and radiance alone. Skipping
    # them is not only faster: round-tripping a quaternion through a matrix
    # flips signs, so "rotate by nothing" would still rewrite every splat.
    turning = float(np.abs(R - np.eye(3)).max()) > 1e-7

    s.xyz[idx] = (c + (p - c) * sc @ R.T + t).astype(np.float32)
    if turning:
        s.rot[idx] = shtools.rotate_quats(s.quat[idx], R).astype(np.float32)
    if sc != 1.0:
        s.scale[idx] = (s.scale[idx].astype(np.float64) + np.log(sc)).astype(np.float32)
    if turning and rotate_sh and s.f_rest.shape[1]:
        s.f_rest[idx] = shtools.rotate_sh(s.f_rest[idx].astype(np.float64), R) \
            .astype(np.float32)

    angle = float(np.degrees(np.arccos(np.clip((np.trace(R) - 1.0) / 2.0, -1, 1))))
    return {
        "count": int(len(idx)),
        "rotation": round(angle, 2),
        "translation": [round(float(v), 4) for v in t],
        "scale": round(sc, 4),
        "pivot": [round(float(v), 4) for v in c],
        "shRotated": bool(turning and rotate_sh and s.f_rest.shape[1] > 0),
    }


def invert(rotation, translation, scale: float, pivot):
    """The transform that undoes ``apply`` with these arguments."""
    q = np.asarray(rotation, float)
    q = q / max(float(np.linalg.norm(q)), 1e-9)
    q_inv = np.array([q[0], -q[1], -q[2], -q[3]])
    R = quat_to_matrix(q)
    c = np.asarray(pivot, float)
    t = np.asarray(translation, float)
    # forward: x' = c + R @ (s (x - c)) + t
    # inverse: x  = c + (1/s) R^T (x' - c - t)
    # expressed in the same (rotate, scale about pivot, then translate) form,
    # the translation that remains is what the pivot picks up
    inv_s = 1.0 / float(scale)
    t_inv = c - (c + R.T @ (c - c - t) * inv_s) + (c - c)
    t_inv = -(R.T @ t) * inv_s
    return {"rotation": q_inv.tolist(), "translation": t_inv.tolist(),
            "scale": inv_s, "pivot": c.tolist()}
