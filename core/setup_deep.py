"""Install and exercise the pinned Deep Fill environment.

    core/.venv/Scripts/python.exe core/setup_deep.py

The editor and its taskbar shortcut use this same virtual environment.

LaMa's old package pins conflict with current NumPy/Pillow/Torch, so install
its tested runtime dependencies explicitly. gsplat compiles its official
source with the existing MSVC tools and an app-local NVIDIA compiler. Any
failed download, installation, render or model inference fails this script.
"""
from __future__ import annotations

import os
from pathlib import Path
import re
import subprocess
import sys

TORCH = "2.9.1"
TORCHVISION = "0.24.1"
GSPLAT = "1.5.3"
LAMA = "0.1.2"
CUDA = "cu128"
INDEX = "https://download.pytorch.org/whl/" + CUDA
CORE = Path(__file__).resolve().parent
REPOSITORY = CORE.parent


def run(args, **kw):
    print("   $", " ".join(args[:6]), "…" if len(args) > 6 else "", flush=True)
    return subprocess.run(args, check=True, **kw)


def pip(*args):
    return run([sys.executable, "-m", "pip", *args])


def step(n, title):
    print("\n[%d] %s" % (n, title), flush=True)


def main():
    print("Python:", sys.version.split()[0], "|", sys.executable)
    if sys.version_info[:2] not in ((3, 10), (3, 11), (3, 12), (3, 13)):
        raise RuntimeError("Deep Fill expects Python 3.10-3.13.")
    if sys.prefix == sys.base_prefix:
        raise RuntimeError("Use core/.venv/Scripts/python.exe so setup changes only the editor environment.")
    os.environ.setdefault("PIP_CACHE_DIR", str(REPOSITORY / ".cache" / "pip"))
    os.environ.setdefault("TORCH_HOME", str(CORE / "models" / "torch"))
    if sys.platform == "win32":
        step(0, "App-local CUDA compiler (no driver or system changes)")
        from setup_cuda import ensure_cuda
        compiler = ensure_cuda()
        os.environ["CUDA_HOME"] = str(compiler)
        os.environ["PATH"] = str(compiler / "bin") + os.pathsep + os.environ.get("PATH", "")

    step(1, "PyTorch with CUDA")
    pip("install", "torch==" + TORCH, "torchvision==" + TORCHVISION,
        "--index-url", INDEX)

    step(2, "Build tools for gsplat")
    pip("install", "ninja", "packaging", "jaxtyping", "rich",
        "typing_extensions")

    step(3, "gsplat (official package, compiles on first run)")
    pip("install", "gsplat==" + GSPLAT)

    step(4, "LaMa, without touching PyTorch")
    # Install the explicitly selected, tested dependencies instead of LaMa's
    # obsolete upper bounds. Do not let its resolver replace the CUDA stack.
    print("   (pip may now report dependency conflicts for simple-lama-inpainting.")
    print("    That is expected: it is installed without its pinned versions on purpose.)")
    pip("install", "--no-deps", "simple-lama-inpainting==" + LAMA)
    pip("install", "pillow", "opencv-python-headless==4.13.0.92", "fire==0.5.0")

    step(5, "Remove the flag MSVC rejects from gsplat")
    patch_gsplat()

    step(6, "Verify")
    verify()


def patch_gsplat():
    if os.name != "nt":
        return
    import importlib.util
    spec = importlib.util.find_spec("gsplat")
    if spec is None or spec.origin is None:
        raise RuntimeError("gsplat was not installed.")
    path = os.path.join(os.path.dirname(spec.origin), "cuda", "_backend.py")
    try:
        src = open(path, encoding="utf-8").read()
    except OSError as e:
        raise RuntimeError("Cannot read gsplat compiler flags") from e
    if "os.name != \"nt\"" in src and "-Wno-attributes" in src:
        print("   already patched")
        return
    new = re.sub(
        r'extra_cflags\s*=\s*\[opt_level,\s*"-Wno-attributes"\]',
        'extra_cflags = [opt_level]\n'
        '        # MSVC reads -Wno-attributes as /Wno-attributes and refuses it\n'
        '        # with "error D8021"; the flag is GCC/Clang only.\n'
        '        if os.name != "nt":\n'
        '            extra_cflags.append("-Wno-attributes")',
        src)
    if new == src:
        print("   compiler flags differ; the real GPU render will verify compatibility")
        return
    open(path, "w", encoding="utf-8").write(new)
    print("   patched:", path)


def verify():
    code = r'''
import numpy as np
from splatcore import gpu
good, why = gpu.available()
print("gsplat:", good, "-", why, flush=True)
if not good:
    raise RuntimeError("GPU render failed: " + why)
from splatcore.deep import load_inpainter
fill, name = load_inpainter("lama")
y, x = np.indices((128, 128))
image = np.stack([x * 2, y * 2, (x + y)], axis=-1).astype(np.uint8)
mask = np.zeros((128, 128), np.uint8)
mask[48:80, 48:80] = 255
damaged = image.copy()
damaged[mask != 0] = 0
result = fill(damaged, mask)
if result.shape != image.shape or result.dtype != np.uint8 or not np.any(result[mask != 0]):
    raise RuntimeError("LaMa did not produce the expected filled image.")
print("LaMa inference passed:", result.shape, result.dtype, flush=True)
print("\nDeep Fill is ready", flush=True)
'''
    run([sys.executable, "-c", code], cwd=CORE)


if __name__ == "__main__":
    try:
        main()
    except (OSError, RuntimeError, subprocess.CalledProcessError) as error:
        print(f"\nDeep Fill setup failed: {error}", file=sys.stderr)
        raise SystemExit(1)
