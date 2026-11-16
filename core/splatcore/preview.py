"""A small CPU reference rasteriser for splats.

This is not the interactive viewport -- that lives in the app and runs on the
GPU. This exists so the engine can be validated headlessly, so repairs can be
checked from angles the user never looked at, and so the future ML path has a
trusted renderer to compare against. Correctness over speed throughout:
anisotropic EWA splatting with full spherical-harmonics evaluation, which is
what the real renderer does too.
"""
from __future__ import annotations

import struct
import zlib

import numpy as np

from .io import SplatSet
from .shtools import sh_basis, quat_to_matrix


def look_at(eye, target, up=(0.0, 1.0, 0.0)):
    """World-to-camera rotation (rows = right, up, forward) and eye position."""
    eye = np.asarray(eye, float)
    f = np.asarray(target, float) - eye
    f /= max(np.linalg.norm(f), 1e-12)
    u0 = np.asarray(up, float)
    r = np.cross(f, u0)
    if np.linalg.norm(r) < 1e-8:
        r = np.cross(f, np.array([1.0, 0.0, 0.0]))
        if np.linalg.norm(r) < 1e-8:
            r = np.cross(f, np.array([0.0, 0.0, 1.0]))
    r /= np.linalg.norm(r)
    u = np.cross(r, f)
    return np.stack([r, u, f]), eye


def _project(s: SplatSet, R, eye, W, H, fov):
    cam = (s.xyz.astype(np.float64) - eye) @ R.T
    z = cam[:, 2]
    f = 0.5 * H / np.tan(np.radians(fov) * 0.5)
    with np.errstate(divide="ignore", invalid="ignore"):
        sx = cam[:, 0] * f / z + W * 0.5
        sy = -cam[:, 1] * f / z + H * 0.5
    return cam, z, f, sx, sy


def _conics(s: SplatSet, R, cam, z, f, idx):
    """2D screen-space inverse covariances and footprint radii."""
    Rm = quat_to_matrix(s.quat[idx])
    M = Rm * s.sigma[idx][:, None, :]
    Sig = M @ np.transpose(M, (0, 2, 1))
    Sc = R @ Sig @ R.T
    J = np.zeros((len(idx), 2, 3))
    J[:, 0, 0] = f / z
    J[:, 0, 2] = -f * cam[:, 0] / z ** 2
    J[:, 1, 1] = -f / z
    J[:, 1, 2] = f * cam[:, 1] / z ** 2
    C2 = J @ Sc @ np.transpose(J, (0, 2, 1))
    C2[:, 0, 0] += 0.3                     # low-pass, as in the reference impl
    C2[:, 1, 1] += 0.3
    a, b, c = C2[:, 0, 0], C2[:, 0, 1], C2[:, 1, 1]
    det = a * c - b * b
    ok = det > 1e-9
    inv = np.zeros_like(C2)
    d = np.where(ok, det, 1.0)
    inv[:, 0, 0] = c / d
    inv[:, 1, 1] = a / d
    inv[:, 0, 1] = inv[:, 1, 0] = -b / d
    lam = 0.5 * (a + c) + np.sqrt(np.maximum(((a - c) * 0.5) ** 2 + b * b, 0.0))
    return inv, np.sqrt(np.maximum(lam, 1e-9)), ok


def _colour(s: SplatSet, idx, dirs, sh_degree):
    rgb = 0.5 + 0.28209479177387814 * s.f_dc[idx].astype(np.float64)
    K = s.f_rest.shape[1]
    if K and sh_degree > 0:
        deg = min(sh_degree, s.sh_degree)
        basis = sh_basis(dirs, deg)[:, 1:]
        n = basis.shape[1]
        rgb = rgb + np.einsum("nk,nkc->nc", basis, s.f_rest[idx][:, :n].astype(np.float64))
    return np.clip(rgb, 0.0, 1.0)


def render(s: SplatSet, eye, target, up=(0, 1, 0), W=640, H=640, fov=50.0,
           bg=(0.04, 0.04, 0.05), sh_degree=3, cutoff=3.0, max_radius=64):
    """Render to ``(rgb uint8 HxWx3, alpha HxW float)``."""
    R, eye = look_at(eye, target, up)
    cam, z, f, sx, sy = _project(s, R, eye, W, H, fov)
    idx = np.where(z > 1e-3)[0]
    if len(idx) == 0:
        return (np.full((H, W, 3), 0, np.uint8), np.zeros((H, W)))
    cam, z, sx, sy = cam[idx], z[idx], sx[idx], sy[idx]
    inv, rad, ok = _conics(s, R, cam, z, f, idx)
    rad = np.minimum(cutoff * rad, max_radius)
    a = s.alpha[idx]
    keep = ok & (a > 0.012) & (rad > 0.3) & (sx > -rad) & (sx < W + rad) \
        & (sy > -rad) & (sy < H + rad)
    idx, sx, sy, z, inv, rad, a = (idx[keep], sx[keep], sy[keep], z[keep],
                                   inv[keep], rad[keep], a[keep])
    order = np.argsort(z)
    idx, sx, sy, inv, rad, a = (idx[order], sx[order], sy[order], inv[order],
                                rad[order], a[order])
    dirs = s.xyz[idx].astype(np.float64) - eye
    dirs /= np.maximum(np.linalg.norm(dirs, axis=1, keepdims=True), 1e-12)
    col = _colour(s, idx, dirs, sh_degree)

    out = np.zeros((H, W, 3))
    T = np.ones((H, W))
    ri = np.ceil(rad).astype(np.int32)
    lim = cutoff * cutoff * 0.5
    for k in range(len(idx)):
        r = ri[k]
        x0, x1 = max(0, int(sx[k]) - r), min(W, int(sx[k]) + r + 1)
        y0, y1 = max(0, int(sy[k]) - r), min(H, int(sy[k]) + r + 1)
        if x0 >= x1 or y0 >= y1:
            continue
        Tl = T[y0:y1, x0:x1]
        if Tl.max() < 0.004:
            continue
        yy, xx = np.mgrid[y0:y1, x0:x1]
        dx, dy = xx - sx[k], yy - sy[k]
        I = inv[k]
        p = 0.5 * (I[0, 0] * dx * dx + 2 * I[0, 1] * dx * dy + I[1, 1] * dy * dy)
        al = np.where(p > lim, 0.0, np.minimum(a[k] * np.exp(-p), 0.995))
        w = al * Tl
        out[y0:y1, x0:x1] += w[..., None] * col[k]
        T[y0:y1, x0:x1] = Tl * (1.0 - al)
    img = out + T[..., None] * np.asarray(bg, float)
    return (np.clip(img, 0, 1) * 255).astype(np.uint8), 1.0 - T


def pick_buffer(s: SplatSet, eye, target, up=(0, 1, 0), W=320, H=320, fov=50.0,
                cutoff=3.0, max_radius=64):
    """Per-pixel dominant visible splat index and its depth.

    "Dominant" means largest ``alpha * transmittance`` -- the splat a user would
    say they are looking at, which is what a brush must agree with.
    """
    R, eye = look_at(eye, target, up)
    cam, z, f, sx, sy = _project(s, R, eye, W, H, fov)
    idx = np.where(z > 1e-3)[0]
    cam, z, sx, sy = cam[idx], z[idx], sx[idx], sy[idx]
    inv, rad, ok = _conics(s, R, cam, z, f, idx)
    rad = np.minimum(cutoff * rad, max_radius)
    a = s.alpha[idx]
    keep = ok & (a > 0.012) & (rad > 0.3) & (sx > -rad) & (sx < W + rad) \
        & (sy > -rad) & (sy < H + rad)
    idx, sx, sy, z, inv, rad, a = (idx[keep], sx[keep], sy[keep], z[keep],
                                   inv[keep], rad[keep], a[keep])
    order = np.argsort(z)
    idx, sx, sy, z, inv, rad, a = (idx[order], sx[order], sy[order], z[order],
                                   inv[order], rad[order], a[order])
    T = np.ones((H, W))
    best = np.zeros((H, W))
    pid = np.full((H, W), -1, np.int64)
    depth = np.full((H, W), np.inf)
    ri = np.ceil(rad).astype(np.int32)
    lim = cutoff * cutoff * 0.5
    for k in range(len(idx)):
        r = ri[k]
        x0, x1 = max(0, int(sx[k]) - r), min(W, int(sx[k]) + r + 1)
        y0, y1 = max(0, int(sy[k]) - r), min(H, int(sy[k]) + r + 1)
        if x0 >= x1 or y0 >= y1:
            continue
        Tl = T[y0:y1, x0:x1]
        if Tl.max() < 0.004:
            continue
        yy, xx = np.mgrid[y0:y1, x0:x1]
        dx, dy = xx - sx[k], yy - sy[k]
        I = inv[k]
        p = 0.5 * (I[0, 0] * dx * dx + 2 * I[0, 1] * dx * dy + I[1, 1] * dy * dy)
        al = np.where(p > lim, 0.0, np.minimum(a[k] * np.exp(-p), 0.995))
        w = al * Tl
        better = w > best[y0:y1, x0:x1]
        if better.any():
            bl = best[y0:y1, x0:x1]; bl[better] = w[better]; best[y0:y1, x0:x1] = bl
            pl = pid[y0:y1, x0:x1]; pl[better] = idx[k]; pid[y0:y1, x0:x1] = pl
            dl = depth[y0:y1, x0:x1]; dl[better] = z[k]; depth[y0:y1, x0:x1] = dl
        T[y0:y1, x0:x1] = Tl * (1.0 - al)
    return pid, depth, 1.0 - T


def brush_select(s: SplatSet, eye, target, up, W, H, fov, mask, depth,
                 front=1.0, back=1.0):
    """Splats whose centre falls under ``mask`` and near the visible surface.

    The depth window is what makes a 3D brush feel like a 2D one: without it,
    painting the front of an object also selects everything behind it.
    """
    R, eye = look_at(eye, target, up)
    _, z, _, sx, sy = _project(s, R, eye, W, H, fov)
    ix = np.floor(sx).astype(np.int64)
    iy = np.floor(sy).astype(np.int64)
    inb = (z > 0) & (ix >= 0) & (ix < W) & (iy >= 0) & (iy < H)
    out = np.zeros(s.n, bool)
    cand = np.where(inb)[0]
    cand = cand[mask[iy[cand], ix[cand]]]
    if len(cand) == 0:
        return out
    d = depth[iy[cand], ix[cand]]
    good = np.isfinite(d) & (z[cand] > d - front) & (z[cand] < d + back)
    out[cand[good]] = True
    return out


def write_png(path, img):
    """Minimal PNG writer (RGB8), so tests need no image library."""
    h, w, _ = img.shape
    rows = b"".join(b"\x00" + img[y].tobytes() for y in range(h))

    def chunk(tag, data):
        c = tag + data
        return (struct.pack(">I", len(data)) + c
                + struct.pack(">I", zlib.crc32(c) & 0xFFFFFFFF))

    with open(path, "wb") as fh:
        fh.write(b"\x89PNG\r\n\x1a\n")
        fh.write(chunk(b"IHDR", struct.pack(">IIBBBBB", w, h, 8, 2, 0, 0, 0)))
        fh.write(chunk(b"IDAT", zlib.compress(rows, 6)))
        fh.write(chunk(b"IEND", b""))
