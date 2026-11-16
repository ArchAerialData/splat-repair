"""Fill a hole from a wide view: render, fill the picture in 2-D, project it back.

The deep fill used to hand LaMa a view framed tight on the patch: the hole
filled 55 % of it and the stand's remnants most of the rest. On the shoe's
stand hole LaMa came back with mush and the fill was rejected. Given the
whole sole from below instead -- the same view a person would take into
Photoshop -- LaMa at half resolution continued the plaque's arc, the smooth
plaque and the lattice around it, close to Photoshop's own result, in 2.8 s.
What it lacked was context, not ability.

So the view is wide: straight down the patch normal, the frame ``context``
times the hole's radius. The picture is filled -- by LaMa, or by the
user's own Photoshop (Generative Fill, falling back to Content-Aware Fill)
-- and handed to ``deep_repair`` as a reference. Its camera stands at the
same eye, so the picture maps onto the patch exactly (``deep._reproject``).
"""
from __future__ import annotations

import time
from typing import Optional

import numpy as np

from .io import SplatSet
from .repair import RepairError, RepairParams, RepairResult, repair

SOURCES = ("lama", "photoshop")


def view_frame(ctx, context: float = 3.0, fov: float = 30.0):
    """Eye, target, up and fov of the wide view for a repair's context."""
    origin, basis = ctx["origin"], ctx["basis"]
    cu0, cv0 = ctx["uv_centre"]
    hf = ctx["surface"]
    radius = float(ctx["radius"])
    centre = origin + np.array([cu0, cv0, float(hf(0.0, 0.0))]) @ basis
    normal = basis[2] * float(ctx["outward"])
    normal = normal / max(np.linalg.norm(normal), 1e-12)
    half = radius * context
    dist = half / np.tan(np.radians(fov) * 0.5)
    return centre + normal * dist, centre, np.asarray(basis[1], float), fov


def view_subset(s: SplatSet, centre, normal, half, selection=None, above=0.5):
    """The splats the wide view may show: the surface and what sits on it.

    The eye stands about eleven hole radii out, so anything between it and
    the surface -- a strap, a lace, a floater, another part of the object --
    was in the picture, and LaMa continued it across the hole: in a test with
    a red strap halfway to the eye, 64 % of the hole came back red and the fit
    accepted it, because it checks against that same picture. Kept: splats
    within the frame's reach of the centre and no further than ``above`` of
    the frame's half-width in front of the surface, plus everything selected
    (the stand being removed may stand taller than that).
    """
    xyz = s.xyz.astype(np.float64)
    d = xyz - np.asarray(centre, float)
    n = np.asarray(normal, float)
    n = n / max(np.linalg.norm(n), 1e-12)
    keep = (np.linalg.norm(d, axis=1) < 2.2 * half) & (d @ n < above * half)
    if selection is not None and len(selection):
        keep[np.asarray(selection, np.int64)] = True
    return np.where(keep)[0]


def render_view(s: SplatSet, eye, target, up, W, H, fov):
    import torch
    from . import gpu
    cam = gpu.look_at_camera(np.asarray(eye, float), np.asarray(target, float),
                             np.asarray(up, float), W, H, fov)
    t = gpu.to_tensors(s, torch.device("cuda"))
    with torch.no_grad():
        img, alpha, dep = gpu.render(t, [cam], sh_degree=s.sh_degree, antialiased=True,
                                     with_depth=True)
    depth = None
    if dep is not None:
        a = alpha[0].detach().cpu().numpy().squeeze()
        depth = dep[0].detach().cpu().numpy().squeeze()      # expected depth, already per alpha
        depth[a < 0.5] = np.inf
    return gpu.to_uint8(img[0]), cam, depth


def view_mask(s: SplatSet, selection, ctx, cam, W, H, depth=None, todo_ctx=None):
    """What the 2-D fill may repaint: the selected splats and the hole, grown.

    With ``depth`` (the view's rendered depth), selected splats lying well
    behind the visible surface -- selected through a thin part -- are left
    out: they would grow the mask over intact surface and take the fill's
    context away.

    With a painted volume in ``ctx`` the mask is what was painted instead:
    the volume on the surface this view shows (`deep.volume_mask`), and the
    cells the quick fill filled, lightly grown. Splat centres made the mask
    too small: a splat reaches past its centre, the tails of the stand
    stayed outside, and the fill continued them. ``todo_ctx`` (face by face)
    keeps the volume to the faces still to be filled.
    """
    from .deep import _hole_points, _mask_from_points, _project, volume_mask
    vol = ctx.get("volume")
    if vol is not None:
        from scipy import ndimage
        grow = max(2, int(round(W / 250)))
        holes = _mask_from_points(np.vstack([_hole_points(c) for c in (todo_ctx or [ctx])]),
                                  cam, W, H, grow)
        m = volume_mask(vol, cam, W, H, depth, grow=grow)
        if todo_ctx is not None:
            sel = np.asarray(selection, np.int64)
            near = holes | (_mask_from_points(np.asarray(s.xyz[sel], float), cam, W, H, grow)
                            if len(sel) else holes)
            m &= ndimage.binary_dilation(near, np.ones((3, 3)), iterations=3 * grow)
        return m | holes
    pts = [_hole_points(ctx)]
    sel = np.asarray(selection, np.int64)
    if len(sel):
        p = np.asarray(s.xyz[sel], float)
        if depth is not None:
            x, y, z = _project(p, cam)
            ok = np.isfinite(x) & (z > 0) & (x >= 0) & (x < W) & (y >= 0) & (y < H)
            dz = np.full(len(p), np.inf)
            dz[ok] = depth[y[ok].astype(np.int64), x[ok].astype(np.int64)]
            tol = 0.3 * float(ctx["radius"])
            p = p[~ok | ~np.isfinite(dz) | (z <= dz + tol)]
        pts.append(p)
    grow = max(3, int(round(W / 170)))
    return _mask_from_points(np.vstack(pts), cam, W, H, grow)


def default_fill() -> str:
    """The deep fill to use when the caller names none: what the app offers first.

    The wide view filled by LaMa where this machine can run it (a GPU and the
    LaMa package, as ``capabilities`` reports ``viewFill``), else the tight
    view around the patch. The app's own default is LaMa; the engine used to
    default to the tight view, so the same request meant two different fills.
    """
    import importlib.util
    from . import gpu
    if gpu.available()[0] and importlib.util.find_spec("simple_lama_inpainting") is not None:
        return "lama"
    return "patch"


def fill_picture(rgb, mask, source: str, say=lambda *_: None, sharpen: float = 0.5,
                 tile: Optional[int] = None):
    """Return ``(filled picture, name of what filled it, fallback reason)``.

    ``tile``: the size of one view when ``rgb`` is a sheet of several. LaMa
    then runs at about a thousand pixels per view, not per sheet, and the
    sharpening is sized by the view.

    The reason is None unless Photoshop's Generative Fill failed and
    Content-Aware Fill stood in for it. It used to be left on the function as
    an attribute: missing until the first fill, stale from an earlier call on
    the next, and overwritten face by face on a fill over a fold.

    ``sharpen``: unsharp mask on LaMa's fill, 2 px per 1000 px of view. LaMa
    comes back a little soft -- the plaque's arc, the lattice ridges -- and
    the splat fit softens it once more. Measured on five hold-out discs
    (reference/kombi/bench_view.py) 0.5 lifted the local contrast against the
    truth from 0.76 to 0.80 head-on and 0.71 to 0.74 at 45 degrees at the
    same RMS and structure; 1.0 went on to speckle the plaque. 0 = off.
    """
    from PIL import Image
    if source == "photoshop":
        from . import photoshop
        try:
            say("Photoshop: Generative Fill", 0.12)
            return photoshop.fill(rgb, mask, "generative"), "Photoshop (generative)", None
        except photoshop.PhotoshopFillFailed as e:
            # No credits, not signed in, offline: Content-Aware Fill runs
            # locally in the same Photoshop. Weaker on structure, but it is
            # Photoshop, and it says so, with the reason. Not after a timeout
            # or when Photoshop is missing (plain PhotoshopUnavailable): a
            # second run against a Photoshop still busy with the first would
            # only block again.
            say("Photoshop: Content-Aware Fill", 0.14)
            return (photoshop.fill(rgb, mask, "content_aware"), "Photoshop (content-aware)",
                    e.reason)
    if source != "lama":
        raise ValueError("the source must be one of %s" % (SOURCES,))
    from .deep import load_inpainter
    fn, name = load_inpainter("lama")
    # LaMa at about a thousand pixels: at full size it smeared the lattice
    # above the plaque, at half size it continued it.
    H, W = mask.shape
    view_px = int(tile) if tile else max(W, H)
    k = min(1.0, 1000.0 / view_px)
    if k < 1.0:
        w2, h2 = int(round(W * k)) // 8 * 8, int(round(H * k)) // 8 * 8
        small = np.asarray(Image.fromarray(rgb).resize((w2, h2), Image.LANCZOS))
        sm = np.asarray(Image.fromarray((mask * 255).astype(np.uint8)).resize((w2, h2), Image.NEAREST))
        f = np.asarray(Image.fromarray(fn(small, sm)).resize((W, H), Image.LANCZOS))
    else:
        f = fn(rgb, (mask * 255).astype(np.uint8))
    if sharpen > 0:
        from scipy import ndimage
        sig = 2.0 * view_px / 1000.0
        ff = f.astype(np.float32)
        ff = ff + sharpen * (ff - ndimage.gaussian_filter(ff, (sig, sig, 0)))
        f = np.clip(ff + 0.5, 0, 255).astype(np.uint8)
    out = rgb.copy()
    out[mask] = f[mask]
    return out, name, None


def grain_from_copy(filled, shown, mask, sigma, ring=12):
    """LaMa's layout, the quick fill's grain: ``(picture, amount)``.

    On a wide hole LaMa keeps the layout -- the plaque's arc, the tone, where
    the lattice runs -- and returns its texture as a local mean: the stand's
    tread came back as brown felt. The quick fill under the mask (``shown``,
    the view with the quick fill in place) has the texture, copied from the
    surface itself and, since the motif price, no circle or lettering in it.
    So everything finer than ``sigma`` pixels comes from the copy, in the
    share LaMa lost: twice the missing fine energy against the ring round
    the mask, capped at 1, as `deep._detail_split` measures it. A small hole
    LaMa filled well loses nothing and keeps its picture. Outside the mask
    nothing changes. Earlier mixes of copy and LaMa brought ghost rings: the
    copy had the heel's circle in it then.
    """
    from scipy import ndimage
    f = np.asarray(filled, np.float64) / 255.0
    g = np.asarray(shown, np.float64)[..., :3] / 255.0
    m = np.asarray(mask, bool)
    if m.sum() < 64:
        return filled, 0.0

    def fine(a):
        return a - np.stack([ndimage.gaussian_filter(a[..., c], sigma) for c in range(3)], -1)

    def energy(a, where):
        hi = fine(a).mean(-1)
        return float(np.sqrt((hi[where] ** 2).mean())) if where.any() else 0.0

    rg = ndimage.binary_dilation(m, np.ones((3, 3)), iterations=ring) & ~m
    want = energy(f, rg)
    have = energy(f, m)
    amount = float(np.clip(2.0 * (1.0 - have / max(want, 1e-6)), 0.0, 1.0))
    if amount <= 0.05:
        return filled, 0.0
    # the copy's own grain, scaled to the ring's where it is stronger
    gc = energy(g, m)
    k = min(1.0, want / max(gc, 1e-6)) if gc > 0 else 0.0
    w = ndimage.gaussian_filter(m.astype(np.float64), 1.5)[..., None] * m[..., None]
    out = f + w * amount * (k * fine(g) - fine(f))
    out = np.where(m[..., None], np.clip(out, 0.0, 1.0), f)
    return (out * 255.0 + 0.5).astype(np.uint8), amount


def _fill(rgb, mask, source, say, sharpen, tile=None):
    """``fill_picture``, read as ``(picture, name, reason)``.

    A stand-in that returns only the first two (the benches swap their own
    fills in) still works, and one that takes no ``tile`` too.
    """
    kw = {"tile": tile} if tile else {}
    out = fill_picture(rgb, mask, source, say, sharpen=sharpen, **kw)
    return out[0], out[1], (out[2] if len(out) > 2 else None)


# ------------------------------------------------------- several views

SHEET_GAP = 24          # pixels of neutral grey between two views of a sheet


def make_sheet(pictures, masks, gap: int = SHEET_GAP):
    """Several views side by side in one picture: ``(sheet, mask, boxes)``.

    One Generative Fill over all of them at once is meant to give views that
    agree with each other, where one call per view invents something new
    each time. The views keep their own size; between two runs a band of
    grey that is not masked, so neither view's border reads as the other's
    context. ``boxes`` are ``(y0, x0, h, w)`` per view, for `split_sheet`.
    """
    hs = [p.shape[0] for p in pictures]
    ws = [p.shape[1] for p in pictures]
    Hs = max(hs)
    Ws = sum(ws) + gap * (len(pictures) - 1)
    sheet = np.full((Hs, Ws, 3), 128, np.uint8)
    smask = np.zeros((Hs, Ws), bool)
    boxes, x = [], 0
    for p, m, h, w in zip(pictures, masks, hs, ws):
        sheet[:h, x:x + w] = np.asarray(p)[..., :3]
        smask[:h, x:x + w] = m
        boxes.append((0, x, h, w))
        x += w + gap
    return sheet, smask, boxes


def split_sheet(sheet, boxes):
    """The views of a filled sheet, as `make_sheet` laid them out."""
    return [np.ascontiguousarray(sheet[y:y + h, x:x + w]) for y, x, h, w in boxes]


def fill_views(pictures, masks, source, say=lambda *_: None, sharpen=0.5, sheet=True):
    """Fill several views: ``(filled pictures, names, fallback reasons)``.

    ``sheet``: all in one call over `make_sheet`, else one call per view.
    """
    if len(pictures) == 1 or not sheet:
        out = [_fill(p, m, source, say, sharpen) for p, m in zip(pictures, masks)]
        return [o[0] for o in out], _uniq(o[1] for o in out), _uniq(o[2] for o in out)
    tile = max(max(p.shape[:2]) for p in pictures)
    sh, sm, boxes = make_sheet(pictures, masks)
    filled, name, why = _fill(sh, sm, source, say, sharpen, tile=tile)
    return split_sheet(filled, boxes), _uniq([name]), _uniq([why])


def view_frames(ctx, context: float = 3.0, fov: float = 30.0, views: int = 1,
                tilt: float = 30.0):
    """The straight wide view and ``views - 1`` tilted ones about the patch.

    The tilted views swing about the patch's ``v`` axis (to either side of
    the straight one, then further), the same axis the deep fit's own tilted
    train views use; its held-out checks swing about the other axis and stay
    unseen.
    """
    eye, target, up, fov = view_frame(ctx, context, fov)
    out = [(eye, target, up, fov)]
    d = eye - target
    side = np.asarray(ctx["basis"][0], float)
    side = side - up * float(side @ up)
    side /= max(np.linalg.norm(side), 1e-12)
    angles = []
    for k in range(1, views):
        a = tilt * ((k + 1) // 2) * (1 if k % 2 else -1)
        angles.append(a)
    for a in angles:
        t = np.radians(a)
        dist = float(np.linalg.norm(d))
        n = d / dist
        e = target + (n * np.cos(t) + side * np.sin(t)) * dist
        out.append((e, target, up, fov))
    return out


def _keep_base(base: RepairResult, why: str, **extra) -> RepairResult:
    """The quick fill as it stands, with why the deep part did not happen.

    Once the geometric fill exists, a deep step that cannot run -- the patch
    too small or too large in its view, a picture that does not show it --
    used to throw the fill away with it, and the user was left with an error
    and nothing done. Now the quick fill is kept and the status line says
    "Deep Fill rejected" with the reason, as it does when the guard rejects a fit.
    """
    info = dict(base.info)
    info.update({"deep": False, "rejected": why})
    info.update(extra)
    return RepairResult(removed=base.removed, added=base.added, backing=base.backing,
                        info=info, context=dict(base.context or {}))


def _uniq(xs):
    return list(dict.fromkeys(x for x in xs if x))


def _view_fill_faces(s, params, source, deep, base, on_progress, size, context,
                     sharpen, t0, reference=None, keep_picture=False, sheet=False,
                     scene="before", grain=0.0):
    """A fill made face by face (repair across a fold), deep-filled face by face.

    One view straight down one normal saw the other face at a grazing angle
    or not at all, and LaMa continued the sole's lattice up the wall. Each
    face gets its own wide view, picture and refit, in the order the quick
    fill made them. In each picture this face and the ones still to come are
    repainted; the ones done stand as they came out, so the next face
    continues them. The refit sees only this face's splats as new.

    ``source`` "patch" runs the tight-view deep fill per face instead, and a
    ``reference`` (a picture made in Photoshop, with its camera) is mapped
    onto each face in turn; a face the picture does not show keeps its quick
    fill. Both used to run once over the whole fold, from the first face's
    camera.

    What filled each face, and why Generative Fill fell back where it did,
    is reported per face in ``deep_charts``; ``fill_source`` and
    ``fill_fallback`` list them all. A Photoshop that stops answering after
    the first face keeps the faces already done.

    ``sheet``: every face's view is rendered first and all are filled in
    one call over one sheet, so the faces' pictures come from one fill;
    a face then no longer sees the deep fill of the one before. ``scene``
    "after" shows the survivors and every face's quick fill around the
    mask instead of what is being removed (see `view_fill`).
    """
    from .deep import DeepParams, deep_repair
    from .photoshop import PhotoshopUnavailable
    say = on_progress or (lambda *_: None)
    ctx = base.context
    spans, sels = ctx["chart_spans"], ctx["chart_sel"]
    back = (np.asarray(base.backing, bool) if base.backing is not None
            and len(base.backing) == base.added.n else np.zeros(base.added.n, bool))
    cur = [base.added.take(np.arange(a, b)) for a, b in spans]
    cur_back = [back[a:b] for a, b in spans]
    wide = source in SOURCES and reference is None
    seed = int(getattr(deep, "seed", 0) or 0) if deep is not None else 0
    if seed and source == "lama" and wide:
        context = context * (1.0 + 0.15 * (((seed * 7) % 5) - 2) / 2.0)
    keep = np.ones(s.n, bool)
    keep[base.removed] = False
    survivors = s.take(keep)
    infos, pictures, t_fill = [], [], 0.0
    W = H = int(size or (2000 if source == "photoshop" else 1000))
    charts = ctx["charts"]

    def face_picture(k):
        """Face ``k``'s wide view, rendered and masked: ``(rgb, mask, frame)``."""
        ck = charts[k]
        if scene == "after":
            # what will be left: survivors and every face's fill as it stands
            world = survivors
            for a in cur:
                world = world.concat(a)
            sel_w = None
        else:
            # what survives, the faces' fills as they stand, and this face's
            # painted splats (masked, as on a single chart)
            own = np.zeros(s.n, bool)
            own[sels[k]] = True
            world = s.take(keep | own)
            for j, a in enumerate(cur):
                if j != k:
                    world = world.concat(a)
            pos = np.full(s.n, -1, np.int64)
            pos[np.where(keep | own)[0]] = np.arange(int((keep | own).sum()))
            sel_w = pos[sels[k]]
        eye, target, up, fov = view_frame(ck, context)
        half = float(ck["radius"]) * context
        normal = (eye - target) / max(np.linalg.norm(eye - target), 1e-12)
        sub = view_subset(world, target, normal, half, sel_w)
        shown = world.take(np.isin(np.arange(world.n), sub)) if len(sub) < world.n else world
        say("rendering view (face %d)" % (k + 1), 0.06)
        rgb, cam, depth = render_view(shown, eye, target, up, W, H, fov)
        todo = np.concatenate([sels[j] for j in range(k, len(sels))])
        mask = view_mask(s, todo, ck, cam, W, H, depth,
                         todo_ctx=[charts[j] for j in range(k, len(charts))])
        if mask.sum() < 64 or mask.mean() > 0.6:
            raise RepairError("The face cannot be filled in its view")
        return rgb, mask, (eye, target, up, fov)

    pre = {}
    if wide and sheet:
        # every face's view first, then one fill over all of them
        got = []
        for k in range(len(charts)):
            if cur[k].n == 0:
                continue
            try:
                got.append((k,) + face_picture(k))
            except RepairError as e:
                infos.append({"chart": k, "deep": False, "rejected": str(e)})
        if got:
            t1 = time.time()
            fills, names, whys = fill_views([g[1] for g in got], [g[2] for g in got],
                                            source, say, sharpen, sheet=True)
            t_fill += time.time() - t1
            for g, f in zip(got, fills):
                pre[g[0]] = (g[1], g[2], f, g[3], " / ".join(names), "; ".join(whys))
    for k, ck in enumerate(charts):
        if cur[k].n == 0 or (wide and sheet and k not in pre):
            continue
        frozen = survivors
        for j, a in enumerate(cur):
            if j != k:
                frozen = frozen.concat(a)
        face = {"chart": k}
        try:
            ref = reference
            if wide:
                if sheet:
                    rgb, mask, filled, (eye, target, up, fov), name, why = pre[k]
                else:
                    rgb, mask, (eye, target, up, fov) = face_picture(k)
                    t1 = time.time()
                    filled, name, why = _fill(rgb, mask, source, say, sharpen)
                    t_fill += time.time() - t1
                if grain > 0 and source == "lama" and scene == "after":
                    px_sp = float(ck["spacing"]) * W / (2.0 * float(ck["radius"]) * context)
                    filled, face["grain_from_copy"] = grain_from_copy(
                        filled, rgb, mask, max(grain * px_sp, 1.5))
                face["fill_source"] = name
                if why:
                    face["fill_fallback"] = why
                if keep_picture:
                    pictures.append((rgb, mask, filled))
                ref = {"image": filled, "eye": eye, "target": target, "up": up, "fov": fov,
                       "grain": float(face.get("grain_from_copy") or 0.0)}
            base_k = RepairResult(removed=np.zeros(0, np.int64), added=cur[k],
                                  backing=cur_back[k].copy(), info=dict(base.info),
                                  context=dict(ck))
            dp = deep or DeepParams()
            if k > 0:
                # a later, smaller face fits in fewer steps: two faces in
                # about the time budget of one and a half
                import dataclasses
                share = len(sels[k]) / max(len(sels[0]), 1)
                dp = dataclasses.replace(dp, steps=max(100, int(round(dp.steps * min(1.0, share)))))
            r = deep_repair(frozen, None, params, dp, base=base_k,
                            on_progress=on_progress, reference=ref)
        except RepairError as e:
            face.update({"deep": False, "rejected": str(e)})
            infos.append(face)
            continue
        except PhotoshopUnavailable as e:
            # Photoshop gone, or out of time, after an earlier face was
            # filled: keep what is done, leave this face and the rest as the
            # quick fill made them. On the first face there is nothing to
            # keep, and the caller says why.
            if not infos:
                raise
            face.update({"deep": False, "rejected": str(e)})
            infos.append(face)
            break
        if r.info.get("deep"):
            cur[k] = r.added
            cur_back[k] = (np.asarray(r.backing, bool) if r.backing is not None
                           and len(r.backing) == r.added.n else np.zeros(r.added.n, bool))
        face.update({k2: r.info.get(k2) for k2 in ("deep", "fit", "offAxisGeo", "offAxisDeep", "keptCopy",
                                                    "rejected", "inpainter")})
        if not wide:
            face["fill_source"] = r.info.get("inpainter")
        infos.append(face)
    added = cur[0]
    for a in cur[1:]:
        added = added.concat(a)
    done = [i for i in infos if i.get("deep")]
    sources = _uniq(i.get("fill_source") for i in infos)
    whys = _uniq(i.get("fill_fallback") for i in infos)
    info = dict(base.info)
    info.update({"deep": bool(done), "deep_charts": infos,
                 "inpainter": " / ".join(sources) if sources else None,
                 "fill_source": " / ".join(sources) if sources else None,
                 "fit": round(float(np.mean([i["fit"] for i in done])), 4) if done else None,
                 "offAxisGeo": done[0]["offAxisGeo"] if done else None,
                 "offAxisDeep": done[0]["offAxisDeep"] if done else None,
                 "rejected": None if done else (infos[0].get("rejected") if infos else "no improvement"),
                 "view_context": round(context, 2), "fill_seconds": round(t_fill, 2),
                 "view_seconds": round(time.time() - t0, 2)})
    if whys:
        info["fill_fallback"] = "; ".join(whys)
    if keep_picture and pictures:
        info["picture"] = pictures[0]
        info["pictures"] = pictures
    spans2, st = [], 0
    for a in cur:
        spans2.append((st, st + a.n))
        st += a.n
    c2 = dict(ctx)
    c2["chart_spans"] = spans2
    return RepairResult(removed=base.removed, added=added, backing=np.concatenate(cur_back),
                        info=info, context=c2)


def deep_patch(s: SplatSet, selection, params: Optional[RepairParams] = None,
               deep=None, base: Optional[RepairResult] = None, on_progress=None,
               reference: Optional[dict] = None) -> RepairResult:
    """The deep fill that is not a wide view: the tight view, or a picture given.

    A quick fill made face by face goes face by face here too (see
    `_view_fill_faces`); before, the tight view and a Photoshop picture were
    fitted once, from the first face's camera, over every face's splats.
    Once the quick fill exists, a deep step that cannot run keeps it.
    """
    from .deep import deep_repair
    base = base or repair(s, selection, params)
    if not base.context:
        raise RepairError("The geometric pass returned no frame")
    if len(base.context.get("charts") or ()) > 1:
        return _view_fill_faces(s, params, "patch", deep, base, on_progress, None, 3.0,
                                0.0, time.time(), reference=reference)
    try:
        return deep_repair(s, selection, params, deep, base=base, on_progress=on_progress,
                           reference=reference)
    except RepairError as e:
        return _keep_base(base, str(e))


def view_fill(s: SplatSet, selection, params: Optional[RepairParams] = None,
              source: str = "lama", deep=None, base: Optional[RepairResult] = None,
              on_progress=None, size: Optional[int] = None, context: float = 3.0,
              keep_picture: bool = False, sharpen: float = 0.5, views: int = 1,
              sheet: bool = True, tilt: float = 30.0,
              scene: Optional[str] = None, grain: float = 3.0) -> RepairResult:
    """Geometric repair, then a wide-view 2-D fill projected onto the patch.

    ``views``: how many views to fill -- the straight one, then tilted ones
    (`view_frames`); all of them go to the refit as pictures of their own.
    ``sheet``: fill them in one call over one sheet (`make_sheet`), else one
    call each. On a fill made face by face each face is one view, and
    ``sheet`` fills all faces' views in one call.

    ``scene``: what the pictures show around the mask. "before" is the scan
    as it is, with what is being removed still in it; "after" is what will
    be left -- the survivors and the quick fill -- so that nothing that is
    about to go, a splat's tail past the mask, can be continued by the fill.
    None: "after" with a painted volume, else "before" (as it always was).

    ``grain``: with LaMa and the scene "after", detail finer than this many
    splat spacings comes from the quick fill under the mask where LaMa lost
    it (`grain_from_copy`); 0 = LaMa's picture as it is.
    """
    from .deep import deep_repair
    say = on_progress or (lambda *_: None)
    t0 = time.time()
    say("geometry", 0.03)
    base = base or repair(s, selection, params)
    if not base.context:
        raise RepairError("The geometric pass returned no frame")
    if scene is None:
        scene = "after" if base.context.get("volume") is not None else "before"
    if len(base.context.get("charts") or ()) > 1:
        return _view_fill_faces(s, params, source, deep, base, on_progress, size,
                                context, sharpen, t0, keep_picture=keep_picture,
                                sheet=sheet and views > 1, scene=scene, grain=grain)
    # "Nochmal anders": LaMa gives the same picture for the same input, so a
    # reroll frames the view a little differently -- the context, and so
    # what LaMa sees, moves by up to 15 %.
    seed = int(getattr(deep, "seed", 0) or 0) if deep is not None else 0
    if seed and source == "lama":
        context = context * (1.0 + 0.15 * (((seed * 7) % 5) - 2) / 2.0)
    name = why = None
    t_fill = 0.0
    try:
        frames = view_frames(base.context, context, views=max(int(views), 1), tilt=tilt)
        eye, target, up, fov = frames[0]
        W = H = int(size or (2000 if source == "photoshop" else 1000))
        say("rendering view", 0.06)
        half = float(base.context["radius"]) * context
        normal = (eye - target) / max(np.linalg.norm(eye - target), 1e-12)
        if scene == "after":
            world = base.apply_to(s)
            sub = view_subset(world, target, normal, half)
        else:
            world = s
            sub = view_subset(s, target, normal, half, selection)
        shown = world.take(np.isin(np.arange(world.n), sub)) if len(sub) < world.n else world
        pics, masks, cams = [], [], []
        for k, (e_k, t_k, u_k, f_k) in enumerate(frames):
            rgb_k, cam_k, depth_k = render_view(shown, e_k, t_k, u_k, W, H, f_k)
            m_k = view_mask(s, selection, base.context, cam_k, W, H, depth_k)
            if k == 0:
                if m_k.sum() < 64:
                    raise RepairError("The spot is too small in the view")
                if m_k.mean() > 0.6:
                    raise RepairError("The spot fills almost the whole view")
            elif m_k.sum() < 64 or m_k.mean() > 0.6:
                continue                       # a tilted view that does not show it
            pics.append(rgb_k)
            masks.append(m_k)
            cams.append((e_k, t_k, u_k, f_k))
        rgb, mask = pics[0], masks[0]
        t1 = time.time()
        fills, names, whys = fill_views(pics, masks, source, say, sharpen, sheet=sheet)
        t_fill = time.time() - t1
        grain_used = None
        grains = [0.0] * len(fills)
        if grain > 0 and source == "lama" and scene == "after":
            px_sp = float(base.context["spacing"]) * W / (2.0 * half)
            got = [grain_from_copy(f_k, p_k, m_k, max(grain * px_sp, 1.5))
                   for f_k, p_k, m_k in zip(fills, pics, masks)]
            fills = [g[0] for g in got]
            grains = [float(g[1]) for g in got]
            grain_used = round(got[0][1], 2)
        filled = fills[0]
        name = " / ".join(names) if names else None
        why = "; ".join(whys) if whys else None
        # ``grain``: how much of the picture's grain came from the copy; the
        # fit then keeps the copy's structure (`deep._tone_onto_copy`)
        refs = [{"image": f_k, "eye": c[0], "target": c[1], "up": c[2], "fov": c[3], "grain": g_k}
                for f_k, c, g_k in zip(fills, cams, grains)]
        res = deep_repair(s, selection, params, deep, base=base, on_progress=on_progress,
                          references=refs)
    except RepairError as e:
        # The quick fill is made; a deep step that cannot run keeps it.
        res = _keep_base(base, str(e), view_context=round(context, 2),
                         view_seconds=round(time.time() - t0, 2))
        if name:
            res.info["fill_source"] = name
        if why:
            res.info["fill_fallback"] = why
        return res
    res.info["fill_source"] = name
    if why:
        res.info["fill_fallback"] = why
    res.info["view_context"] = round(context, 2)
    res.info["fill_seconds"] = round(t_fill, 2)
    res.info["view_seconds"] = round(time.time() - t0, 2)
    res.info["fill_views"] = len(pics)
    res.info["fill_sheet"] = bool(sheet and len(pics) > 1)
    res.info["view_scene"] = scene
    res.info["grain_from_copy"] = grain_used
    if keep_picture:
        res.info["picture"] = (rgb, mask, filled)
        res.info["pictures"] = list(zip(pics, masks, fills))
    return res
