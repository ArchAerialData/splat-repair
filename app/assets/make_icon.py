"""Builds the program icon: "Gs" in the manner of Photoshop's "Ps".

A rounded square in the style guide's creme (#F1E9DC), a thin inset border
and the letters in its softer focus green (#6C8A5F), Prompt 700 -- the face that carries every
heading in the style guide. G for Gaussian, s for splat, in the family
look the app borrows its idea from.

Run it when the design changes, not at build time. Outputs ``icon.ico``
(Windows window and taskbar, 16-256 px), ``icon.png`` at 1024,
``icon-256.png`` for the readme and ``logo.svg`` for the app's header.
The earlier dot icon is kept in ``make_icon_punkte.py``.
"""
from __future__ import annotations

import os

from PIL import Image, ImageDraw, ImageFont

HERE = os.path.dirname(os.path.abspath(__file__))
FONT = os.path.join(HERE, "prompt-700.ttf")      # from ../public/fonts/prompt-700.woff2

GREEN = (0xF1, 0xE9, 0xDC)       # tile: cream, the warm white of the style guide
WHITE = (0x6C, 0x8A, 0x5F)       # letters and outline: focus green, softer than hedge green

CORNER = 0.18                    # rounded-square radius, as Adobe's tiles
BORDER = 0.035                   # inset outline width
INSET = 0.075                    # outline distance from the edge
TEXT_H = 0.43                    # cap height share of the tile
TRACK = -0.03                    # letter spacing, as the headings use


def draw(size: int) -> Image.Image:
    ss = 4 if size < 256 else 2              # supersample small sizes harder
    n = size * ss
    im = Image.new("RGBA", (n, n), (0, 0, 0, 0))
    d = ImageDraw.Draw(im)
    d.rounded_rectangle((0, 0, n - 1, n - 1), radius=CORNER * n, fill=GREEN + (255,))
    if size >= 48:                           # the outline turns to noise below
        i = INSET * n
        w = max(1, round(BORDER * n))
        d.rounded_rectangle((i, i, n - 1 - i, n - 1 - i), radius=(CORNER - INSET * 0.7) * n,
                            outline=WHITE + (255,), width=w)
    # letters: size the font so the cap height of G is TEXT_H of the tile
    probe = ImageFont.truetype(FONT, 1000)
    gb = probe.getbbox("G")
    cap = gb[3] - gb[1]
    font = ImageFont.truetype(FONT, int(1000 * TEXT_H * n / cap))
    glyphs = ["G", "s"]
    boxes = [font.getbbox(g) for g in glyphs]
    adv = [font.getlength(g) for g in glyphs]
    track = TRACK * font.size
    total = adv[0] + track + (boxes[1][2])
    x0 = (n - total) / 2.0 - boxes[0][0] * 0.5
    gtop, gbot = boxes[0][1], boxes[0][3]
    y0 = (n - (gbot - gtop)) / 2.0 - gtop + 0.02 * n   # optical centre a hair low
    d.text((x0, y0), "G", font=font, fill=WHITE + (255,))
    d.text((x0 + adv[0] + track, y0), "s", font=font, fill=WHITE + (255,))
    return im.resize((size, size), Image.LANCZOS)


def svg() -> str:
    """The header mark: the same tile as vector, text in the app's Prompt."""
    r = CORNER * 64
    i = INSET * 64
    return (
        '<svg xmlns="http://www.w3.org/2000/svg" viewBox="0 0 64 64" role="img" aria-label="Gs">'
        '<rect width="64" height="64" rx="%.2f" fill="#F1E9DC"/>'
        '<rect x="%.2f" y="%.2f" width="%.2f" height="%.2f" rx="%.2f" fill="none" '
        'stroke="#6C8A5F" stroke-width="%.2f"/>'
        '<text x="32" y="42.4" text-anchor="middle" font-family="Prompt, system-ui, sans-serif" '
        'font-weight="700" font-size="31" letter-spacing="-1" fill="#6C8A5F">Gs</text></svg>'
        % (r, i, i, 64 - 2 * i, 64 - 2 * i, (CORNER - INSET * 0.7) * 64, BORDER * 64))


def main():
    big = draw(1024)
    big.save(os.path.join(HERE, "icon.png"))
    draw(256).save(os.path.join(HERE, "icon-256.png"))
    sizes = [16, 24, 32, 48, 64, 128, 256]
    frames = [draw(s) for s in sizes]
    frames[-1].save(os.path.join(HERE, "icon.ico"), sizes=[(s, s) for s in sizes],
                    append_images=frames[:-1])
    with open(os.path.join(HERE, "logo.svg"), "w", encoding="utf-8") as f:
        f.write(svg())
    print("icon.png, icon-256.png, icon.ico, logo.svg")


if __name__ == "__main__":
    main()
