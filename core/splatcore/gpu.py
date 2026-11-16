"""GPU rasterisation via gsplat, and the camera bookkeeping around it.

Kept apart from the rest of the engine so the editor runs with nothing but
NumPy installed. Nothing here is imported until a deep repair is actually
asked for, and :func:`available` reports why if it cannot be.

Conventions, because mixing them up produces pictures that look almost right:

* gsplat takes OpenCV cameras -- world-to-camera matrices, +X right, +Y **down**,
  +Z forward into the screen. Our own preview renderer uses +Y up, so the two
  differ by a sign on rows 1 and 2.
* scales are world units, not logs; opacities are 0..1, not logits.
* quaternions are (w, x, y, z), which is what the PLY stores.
* SH coefficients are [N, K, 3] with the DC term at index 0, and gsplat adds the
  0.5 offset itself -- passing colours that already include it doubles it.
"""
from __future__ import annotations

from dataclasses import dataclass
from typing import Optional

import numpy as np

from .io import SplatSet

_CACHE: dict = {}
_PROBE_LOCK = __import__("threading").Lock()


def probe_in_background() -> None:
    """Start ``available()`` on a thread, so the engine stays responsive.

    Importing torch and loading gsplat's compiled extension take 3 s on an
    idle machine and were measured at 79-107 s under load. Run inline, the
    single-threaded engine could not even open a file meanwhile.
    """
    import threading
    if "status" in _CACHE or _CACHE.get("probing"):
        return
    _CACHE["probing"] = True
    threading.Thread(target=available, name="gpu-probe", daemon=True).start()


def peek():
    """``available()``'s answer if it is known, else None. Never blocks."""
    return _CACHE.get("status")


def _ensure_build_env() -> None:
    """Make gsplat's build check work when launched outside an activated venv.

    gsplat verifies its compiled extension on first use and needs ``ninja`` on
    PATH to do so, even when nothing has to be rebuilt. Running the interpreter
    by full path -- which is how the editor spawns it -- does not put the venv's
    Scripts directory on PATH, so it is added here rather than demanding the
    caller remember.
    """
    import os
    import sys

    scripts = os.path.join(sys.prefix, "Scripts" if os.name == "nt" else "bin")
    if os.path.isdir(scripts) and scripts not in os.environ.get("PATH", ""):
        os.environ["PATH"] = scripts + os.pathsep + os.environ.get("PATH", "")
    if os.name != "nt":
        return
    if not os.environ.get("CUDA_HOME"):
        root = r"C:\Program Files\NVIDIA GPU Computing Toolkit\CUDA"
        if os.path.isdir(root):
            vs = sorted(os.listdir(root), reverse=True)
            if vs:
                os.environ["CUDA_HOME"] = os.path.join(root, vs[0])
    _add_msvc_to_path()


def _add_msvc_to_path() -> None:
    """Put cl.exe on PATH if it is installed but not visible.

    torch's extension loader runs ``where cl`` on every load, even when the
    extension is already compiled and nothing needs building. Without it the
    load fails with a compiler-version error that has nothing to do with the
    actual problem.
    """
    import glob
    import os
    import shutil
    import subprocess

    if shutil.which("cl"):
        return
    vswhere = os.path.join(
        os.environ.get("ProgramFiles(x86)", r"C:\Program Files (x86)"),
        "Microsoft Visual Studio", "Installer", "vswhere.exe")
    roots = []
    if os.path.exists(vswhere):
        try:
            out = subprocess.run(
                [vswhere, "-latest", "-products", "*", "-property", "installationPath"],
                capture_output=True, text=True, timeout=20)
            roots = [p for p in out.stdout.splitlines() if p.strip()]
        except Exception:                        # noqa: BLE001
            roots = []
    for root in roots:
        hits = sorted(glob.glob(os.path.join(
            root, "VC", "Tools", "MSVC", "*", "bin", "Hostx64", "x64", "cl.exe")))
        if hits:
            os.environ["PATH"] = os.path.dirname(hits[-1]) + os.pathsep + os.environ["PATH"]
            return


def available() -> tuple[bool, str]:
    """(usable, explanation). Safe to call when nothing is installed."""
    if "status" in _CACHE:
        return _CACHE["status"]
    with _PROBE_LOCK:
        if "status" in _CACHE:
            return _CACHE["status"]
        return _available()


def _available() -> tuple[bool, str]:
    try:
        import torch
    except Exception as e:                       # noqa: BLE001
        _CACHE["status"] = (False, "PyTorch is missing (%s)" % type(e).__name__)
        return _CACHE["status"]
    if not torch.cuda.is_available():
        _CACHE["status"] = (False, "CUDA is not available")
        return _CACHE["status"]
    try:
        import gsplat                            # noqa: F401
    except Exception as e:                       # noqa: BLE001
        _CACHE["status"] = (False, "gsplat is missing (%s)" % type(e).__name__)
        return _CACHE["status"]
    _ensure_build_env()
    # Importing gsplat proves nothing: its CUDA extension loads lazily on the
    # first real call, and that is where a missing toolchain shows up. Force it
    # now with a trivial render so the editor can say so before the user waits.
    try:
        import gsplat as _g
        m = torch.zeros((1, 3), device="cuda")
        q = torch.tensor([[1.0, 0.0, 0.0, 0.0]], device="cuda")
        sc = torch.full((1, 3), 0.1, device="cuda")
        op = torch.ones(1, device="cuda")
        col = torch.ones((1, 1, 3), device="cuda")
        view = torch.eye(4, device="cuda")[None].clone()
        view[0, 2, 3] = 3.0
        K = torch.tensor([[[8.0, 0, 8.0], [0, 8.0, 8.0], [0, 0, 1.0]]], device="cuda")
        _g.rasterization(means=m, quats=q, scales=sc, opacities=op, colors=col,
                         viewmats=view, Ks=K, width=16, height=16, sh_degree=0)
    except Exception as e:                       # noqa: BLE001
        msg = str(e)
        if "Ninja" in msg or "ninja" in msg:
            why = "gsplat is not built (ninja is missing)"
        elif "cl.exe" in msg or "D8021" in msg or "Error building" in msg:
            why = "gsplat needs to be compiled once (requires a Visual Studio environment)"
        else:
            why = "gsplat kernel could not be loaded: %s" % msg.strip().splitlines()[-1][:120]
        _CACHE["status"] = (False, why)
        return _CACHE["status"]
    name = torch.cuda.get_device_name(0)
    _CACHE["status"] = (True, "%s, torch %s" % (name, torch.__version__))
    return _CACHE["status"]


def require():
    ok, why = available()
    if not ok:
        raise RuntimeError("Deep Fill is not available: " + why)
    import torch
    import gsplat
    return torch, gsplat


@dataclass
class Camera:
    """A pinhole view. ``view`` is world-to-camera in OpenCV convention."""

    view: np.ndarray      # (4, 4)
    K: np.ndarray         # (3, 3)
    width: int
    height: int

    @property
    def position(self) -> np.ndarray:
        R, t = self.view[:3, :3], self.view[:3, 3]
        return -R.T @ t

    @property
    def forward(self) -> np.ndarray:
        return self.view[2, :3]


def look_at_camera(eye, target, up, width, height, fov_deg) -> Camera:
    """Build an OpenCV-convention camera looking from ``eye`` at ``target``."""
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
    d = np.cross(f, r)                 # +Y down, so "down" = f x right
    R = np.stack([r, d, f])            # world -> camera rows
    view = np.eye(4)
    view[:3, :3] = R
    view[:3, 3] = -R @ eye
    fy = 0.5 * height / np.tan(np.radians(fov_deg) * 0.5)
    K = np.array([[fy, 0.0, width * 0.5],
                  [0.0, fy, height * 0.5],
                  [0.0, 0.0, 1.0]])
    return Camera(view=view, K=K, width=width, height=height)


def orbit_cameras(centre, normal, distance, width, height, fov_deg,
                  n: int = 8, tilt_deg: float = 18.0) -> list[Camera]:
    """A reference view down ``normal``, plus ``n`` views tilted around it.

    The tilt is deliberately modest. These exist to stop the patch from being
    correct in exactly one direction and wrong everywhere else; swinging far
    enough to see geometry the reference never saw would ask the optimiser to
    invent, which is the failure mode this whole approach is trying to avoid.
    """
    centre = np.asarray(centre, float)
    nrm = np.asarray(normal, float)
    nrm = nrm / max(np.linalg.norm(nrm), 1e-12)
    tmp = np.array([0.0, 0.0, 1.0])
    if abs(float(nrm @ tmp)) > 0.9:
        tmp = np.array([1.0, 0.0, 0.0])
    e0 = np.cross(nrm, tmp); e0 /= np.linalg.norm(e0)
    e1 = np.cross(nrm, e0)

    cams = [look_at_camera(centre + nrm * distance, centre, e1,
                           width, height, fov_deg)]
    t = np.radians(tilt_deg)
    for i in range(n):
        a = 2.0 * np.pi * i / max(n, 1)
        offset = (e0 * np.cos(a) + e1 * np.sin(a)) * np.tan(t)
        d = nrm + offset
        d /= np.linalg.norm(d)
        cams.append(look_at_camera(centre + d * distance, centre, e1,
                                   width, height, fov_deg))
    return cams


def to_tensors(s: SplatSet, device, requires_grad: bool = False):
    """SplatSet -> the activated tensors gsplat wants."""
    torch, _ = require()
    t = lambda a, dt=None: torch.tensor(np.ascontiguousarray(a), device=device,
                                        dtype=dt or torch.float32)
    means = t(s.xyz)
    quats = t(s.quat)
    scales = t(s.sigma)
    opac = t(s.alpha)
    sh = np.concatenate([s.f_dc[:, None, :], s.f_rest], axis=1)
    colors = t(sh)
    if requires_grad:
        for x in (means, quats, scales, opac, colors):
            x.requires_grad_(True)
    return {"means": means, "quats": quats, "scales": scales,
            "opacities": opac, "colors": colors}


def render(params: dict, cams: list[Camera], sh_degree: int = 3,
           background=None, with_depth: bool = True,
           antialiased: bool = True):
    """Rasterise ``params`` from ``cams``.

    Returns ``(rgb [C,H,W,3], alpha [C,H,W,1], depth [C,H,W] or None)``, all
    torch tensors with gradients intact where the inputs had them.
    """
    torch, gsplat = require()
    dev = params["means"].device
    viewmats = torch.tensor(np.stack([c.view for c in cams]), device=dev,
                            dtype=torch.float32)
    Ks = torch.tensor(np.stack([c.K for c in cams]), device=dev,
                      dtype=torch.float32)
    W, H = cams[0].width, cams[0].height
    # Left to gsplat unless asked for: with a depth channel in play it expects
    # the background to carry one too, and getting that wrong is an assertion
    # deep inside the rasteriser rather than anything the caller can read.
    bg = None
    if background is not None:
        ch = 4 if with_depth else 3
        v = np.zeros((len(cams), ch), np.float32)
        v[:, :3] = np.asarray(background, np.float32)
        bg = torch.tensor(v, device=dev, dtype=torch.float32)
    out, alphas, _ = gsplat.rasterization(
        means=params["means"], quats=params["quats"], scales=params["scales"],
        opacities=params["opacities"], colors=params["colors"],
        viewmats=viewmats, Ks=Ks, width=W, height=H,
        sh_degree=sh_degree,
        render_mode="RGB+ED" if with_depth else "RGB",
        rasterize_mode="antialiased" if antialiased else "classic",
        backgrounds=bg,
    )
    if with_depth:
        return out[..., :3], alphas, out[..., 3]
    return out, alphas, None


def to_uint8(img) -> np.ndarray:
    """Torch image tensor (H,W,3) in 0..1 -> uint8 numpy."""
    a = img.detach().float().clamp(0.0, 1.0).cpu().numpy()
    return (a * 255.0 + 0.5).astype(np.uint8)
