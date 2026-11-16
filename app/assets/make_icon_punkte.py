"""Builds the program icon. Run it when the design changes, not at build time.

The icon is drawn rather than designed in a vector tool, because the subject
is Gaussians and a Gaussian is easier to evaluate than to approximate with
blur filters: every dot here is a real 2D Gaussian with its own sigma,
opacity and colour, composited the same way the renderer composites splats.

What it shows: a patch of scanned surface in creme, with a piece of it filled
in by the repair brush, in the scene's leading colour. The scene is
Kaminzimmer/Abend, the same one the application window runs in.

Two densities, because an icon is read at two distances. At 32 pixels and
below a hundred dots turn to porridge, so the small sizes get about thirty
large ones -- the usual practice of simplifying a mark for small sizes, done
with numbers instead of a second drawing.

Outputs ``icon.ico`` (Windows window and taskbar), ``icon.png`` at 1024 for
anything that wants a bitmap, and ``icon-256.png`` for the readme.
"""
from __future__ import annotations

import os

import numpy as np
from PIL import Image

HERE = os.path.dirname(os.path.abspath(__file__))

# The "fireside room" scene, evening -- the palette the application itself uses.
BG = (0x10, 0x0C, 0x0A)          # wood black
INK = (0xF1, 0xE9, 0xDC)         # cream
QUIET = (0xB9, 0xAA, 0x95)       # quiet text
LEAD = (0xE5, 0xA2, 0x62)        # light whisky

# Geometry, all as fractions of the icon's edge so every size matches.
DISC_R = 0.345                   # the scanned patch
PATCH_C = (0.085, 0.10)          # where the repair sits, from the centre
PATCH_R = 0.145
CORNER = 0.22                    # rounded-square radius


def _rounded_box_mask(size: int, radius: float, samples: int = 4) -> np.ndarray:
    """Coverage of a rounded square, supersampled so the edge is clean."""
    n = size * samples
    t = (np.arange(n) + 0.5) / n
    x, y = np.meshgrid(t, t, indexing="xy")
    # distance to the rounded box, in the same 0..1 coordinates
    q = np.abs(np.stack([x - 0.5, y - 0.5])) - (0.5 - radius)
    q = np.maximum(q, 0.0)
    d = np.linalg.norm(q, axis=0) - radius
    inside = (d <= 0.0).astype(np.float64)
    inner = np.maximum(np.abs(np.stack([x - 0.5, y - 0.5])), 0).max(0)
    inside = np.where(inner <= 0.5 - radius, 1.0, inside)
    return inside.reshape(size, samples, size, samples).mean((1, 3))


def _dots(size: int, big: bool, seed: int = 20261103):
    """Jittered grid of anisotropic Gaussians over the disc.

    Anisotropic on purpose: a Gaussian splat is an ellipse with its own
    orientation, and a field of round dots would be a picture of a different
    technique. Sparse on purpose too -- at roughly 0.45 sigma per grid step
    the dots touch without merging, and the background still shows between
    them, which is what keeps it a cloud instead of a ball of wool.
    """
    rng = np.random.default_rng(seed)
    step = 0.138 if big else 0.0575
    sigma0 = 0.060 if big else 0.0252

    g = np.arange(-DISC_R - step, DISC_R + step + 1e-9, step)
    gx, gy = np.meshgrid(g, g, indexing="xy")
    pts = np.stack([gx.ravel(), gy.ravel()], 1)
    pts += rng.uniform(-0.34, 0.34, pts.shape) * step
    keep = np.linalg.norm(pts, axis=1) <= DISC_R
    pts = pts[keep]

    # Which dots belong to the repaired patch.
    off = pts - np.asarray(PATCH_C)
    repaired = np.linalg.norm(off, axis=1) <= PATCH_R

    n = len(pts)
    sa = sigma0 * rng.uniform(0.80, 1.30, n)
    ratio = rng.uniform(1.0, 1.30 if big else 1.62, n)
    ang = rng.uniform(0.0, np.pi, n)
    # The repair lays down a tighter, rounder, more even patch -- which is
    # what the real one does, and it keeps the mark legible when small.
    sa[repaired] = sigma0 * rng.uniform(0.82, 1.02, int(repaired.sum()))
    ratio[repaired] = rng.uniform(1.0, 1.22, int(repaired.sum()))

    mix = rng.uniform(0.0, 1.0, n)[:, None]
    col = np.where(repaired[:, None],
                   np.asarray(LEAD, float),
                   np.asarray(INK, float) * (1 - mix * 0.34)
                   + np.asarray(QUIET, float) * (mix * 0.34))
    alpha = np.where(repaired, rng.uniform(0.80, 0.96, n),
                     rng.uniform(0.56, 0.93, n))

    o = rng.permutation(n)
    return ((pts[o] + 0.5) * size, sa[o] * size, (sa[o] / ratio[o]) * size,
            ang[o], col[o], alpha[o])


def render(size: int, big: bool) -> Image.Image:
    img = np.tile(np.asarray(BG, float), (size, size, 1))

    # Light from the lower right in the scene's colour. Light, not decoration:
    # it never becomes an edge or a shape of its own.
    yy, xx = np.mgrid[0:size, 0:size].astype(np.float64) + 0.5
    d = np.hypot(xx - 0.80 * size, yy - 0.84 * size) / (0.85 * size)
    img += np.asarray(LEAD, float) * (0.085 * np.clip(1.0 - d, 0, 1) ** 2)[..., None]

    px, py = (0.5 + PATCH_C[0]) * size, (0.5 + PATCH_C[1]) * size

    if big:
        # At 32 pixels and below, individual splats are smaller than a pixel
        # and drawing them anyway produces noise with a lumpy outline. The
        # small sizes get the same mark with the cloud as one soft disc --
        # the ordinary practice of simplifying a mark, not a second design.
        def disc(cx, cy, r, colour, a, soft):
            t = np.clip((r - np.hypot(xx - cx, yy - cy)) / soft, 0.0, 1.0)
            w = (a * t * t * (3.0 - 2.0 * t))[..., None]
            return img * (1.0 - w) + np.asarray(colour, float) * w

        img = disc(0.5 * size, 0.5 * size, DISC_R * size, INK, 0.90, 0.075 * size)
        img = disc(px, py, PATCH_R * size, LEAD, 0.96, 0.035 * size)
    else:
        centres, sa, sb, ang, cols, alphas = _dots(size, big)
        for (cx, cy), s1, s2, th, col, a in zip(centres, sa, sb, ang, cols, alphas):
            r = int(np.ceil(3.0 * max(s1, s2)))
            x0, x1 = max(0, int(cx) - r), min(size, int(cx) + r + 1)
            y0, y1 = max(0, int(cy) - r), min(size, int(cy) + r + 1)
            if x0 >= x1 or y0 >= y1:
                continue
            dx = (np.arange(x0, x1) + 0.5 - cx)[None, :]
            dy = (np.arange(y0, y1) + 0.5 - cy)[:, None]
            c, s = np.cos(th), np.sin(th)
            u = (dx * c + dy * s) / s1          # along the major axis
            v = (-dx * s + dy * c) / s2
            w = a * np.exp(-0.5 * (u ** 2 + v ** 2))
            win = img[y0:y1, x0:x1]
            img[y0:y1, x0:x1] = win * (1.0 - w[..., None]) + col * w[..., None]

    # The brush, drawn the way the application draws its own cursor: a ring
    # in the leading colour. It is the one hard-edged thing in the picture,
    # which is what gives the mark a rank order -- tool over material -- and
    # what still reads when the icon is sixteen pixels wide.
    ring_r = PATCH_R * 1.26 * size
    stroke = max(1.25, (0.042 if big else 0.021) * size)
    d = np.abs(np.hypot(xx - px, yy - py) - ring_r)
    # one pixel of softness, so the circle is smooth without being blurred
    w = np.clip(0.5 * stroke + 0.5 - d, 0.0, 1.0)[..., None]
    img = img * (1.0 - w) + np.asarray(LEAD, float) * w

    rgb = np.clip(np.rint(img), 0, 255).astype(np.uint8)
    cov = _rounded_box_mask(size, CORNER)
    out = np.dstack([rgb, np.rint(cov * 255).astype(np.uint8)])
    return Image.fromarray(out, "RGBA")


def main():
    master = render(1024, big=False)
    master.save(os.path.join(HERE, "icon.png"))
    master.resize((256, 256), Image.LANCZOS).save(os.path.join(HERE, "icon-256.png"))

    # Windows reads whichever size it needs out of the .ico, so each one is
    # produced at the density that still reads at that size.
    layers = []
    for s in (16, 24, 32, 48, 64, 128, 256):
        if s <= 32:
            layers.append(render(s, big=True))
        else:
            layers.append(master.resize((s, s), Image.LANCZOS))
    layers[-1].save(os.path.join(HERE, "icon.ico"),
                    sizes=[(im.width, im.height) for im in layers],
                    append_images=layers[:-1])
    print("icon.png 1024, icon-256.png, icon.ico %s"
          % ", ".join(str(im.width) for im in layers))


if __name__ == "__main__":
    main()
