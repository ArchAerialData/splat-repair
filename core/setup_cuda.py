"""Install the small, official CUDA 12.8 compiler needed by gsplat on Windows.

The three NVIDIA redistributable archives live inside the active virtual
environment. No driver, registry entry or system PATH is changed. Full toolkit
installers are unnecessary, and the similarly named pip nvcc wheel does not
contain nvcc.exe.
"""
from __future__ import annotations

from dataclasses import dataclass
import hashlib
import json
import os
from pathlib import Path, PurePosixPath
import shutil
import stat
import sys
import tempfile
import urllib.request
import zipfile


BASE_URL = "https://developer.download.nvidia.com/compute/cuda/redist/"
MANIFEST_URL = BASE_URL + "redistrib_12.8.1.json"
MARKER = "splat-repair-cuda.json"


@dataclass(frozen=True)
class Component:
    name: str
    version: str
    size: int
    sha256: str

    @property
    def archive_root(self):
        return f"{self.name}-windows-x86_64-{self.version}-archive"

    @property
    def filename(self):
        return self.archive_root + ".zip"

    @property
    def url(self):
        return BASE_URL + f"{self.name}/windows-x86_64/{self.filename}"


# Exact versions, sizes and checksums from NVIDIA's CUDA 12.8.1 manifest.
COMPONENTS = (
    Component("cuda_nvcc", "12.8.93", 120912792,
              "9fdc70b4271ed9aad4d64cd7076a7d96ec36512d074b9995fe638de669197391"),
    Component("cuda_cudart", "12.8.90", 3037735,
              "4a39058fd8519444a81cfc7ae055d136f48d1a31ffa41ae255b35b2edd61e13b"),
    Component("cuda_cccl", "12.8.90", 2911311,
              "bd8548fa1ae82f92910bebc3079e14bd58c5a92aa64596d46bd610a478cb39d7"),
)
REQUIRED = ("bin/nvcc.exe", "bin/ptxas.exe", "nvvm/bin/cicc.exe",
            "include/cuda_runtime.h", "include/crt/host_config.h",
            "include/cub/cub.cuh", "lib/x64/cudart.lib")


def _digest(path: Path) -> str:
    with path.open("rb") as stream:
        return _stream_digest(stream)


def _stream_digest(stream) -> str:
    digest = hashlib.sha256()
    while chunk := stream.read(1024 * 1024):
        digest.update(chunk)
    return digest.hexdigest()


def _download(component: Component, cache: Path) -> Path:
    """Reuse a verified archive, or download it without touching its final name."""
    cache.mkdir(parents=True, exist_ok=True)
    target = cache / component.filename
    if target.is_file() and target.stat().st_size == component.size and _digest(target) == component.sha256:
        return target
    print(f"Downloading {component.name} ({component.size / 1e6:.1f} MB)", flush=True)
    partial = None
    try:
        with tempfile.NamedTemporaryFile(dir=cache, prefix=".cuda-download-", delete=False) as output:
            partial = Path(output.name)
            received = 0
            with urllib.request.urlopen(component.url, timeout=60) as response:
                while chunk := response.read(1024 * 1024):
                    received += len(chunk)
                    if received > component.size:
                        raise RuntimeError(f"Unexpected download size: {component.name}")
                    output.write(chunk)
        if received != component.size or _digest(partial) != component.sha256:
            raise RuntimeError(f"Download integrity check failed: {component.name}")
        partial.replace(target)
        return target
    finally:
        if partial is not None:
            partial.unlink(missing_ok=True)


def _extract(component: Component, archive: Path, stage: Path) -> None:
    """Merge a component after validating all names, sizes and file types."""
    with zipfile.ZipFile(archive) as source:
        entries = []
        expanded = 0
        for info in source.infolist():
            name = info.filename
            parts = PurePosixPath(name).parts
            if ("\\" in name or ":" in name or not parts
                    or parts[0] != component.archive_root
                    or any(p in (".", "..") for p in parts)):
                raise RuntimeError(f"Unsafe archive path in {component.name}: {name}")
            mode = info.external_attr >> 16
            if stat.S_ISLNK(mode):
                raise RuntimeError(f"Archive links are not supported: {name}")
            if len(parts) == 1 or info.is_dir():
                continue
            expanded += info.file_size
            if expanded > 1024 * 1024 * 1024:
                raise RuntimeError(f"Archive expands beyond its expected size: {component.name}")
            relative = Path(*parts[1:])
            if relative.parts[0].upper().startswith("LICENSE"):
                relative = Path("licenses", component.name) / relative
            destination = stage / relative
            if not destination.resolve().is_relative_to(stage.resolve()):
                raise RuntimeError(f"Archive escaped installation directory: {name}")
            entries.append((info, destination))
        for info, destination in entries:
            destination.parent.mkdir(parents=True, exist_ok=True)
            if destination.exists():
                # Never silently replace a conflicting header from another component.
                with source.open(info) as stream:
                    incoming = _stream_digest(stream)
                if _digest(destination) != incoming:
                    raise RuntimeError(f"Conflicting CUDA component file: {destination}")
                continue
            with source.open(info) as stream, destination.open("wb") as output:
                shutil.copyfileobj(stream, output)


def ensure_cuda(destination: Path | None = None, cache: Path | None = None) -> Path:
    if sys.platform != "win32":
        raise RuntimeError("The bundled CUDA compiler setup currently supports Windows only.")
    if destination is None:
        if sys.prefix == sys.base_prefix:
            raise RuntimeError("Run this with core/.venv/Scripts/python.exe, not a system Python.")
        destination = Path(sys.prefix) / "cuda" / "v12.8"
    destination = destination.resolve()
    cache = cache or Path(__file__).resolve().parents[1] / ".cache" / "cuda"
    expected = {c.name: c.sha256 for c in COMPONENTS}
    if destination.exists():
        try:
            saved = json.loads((destination / MARKER).read_text(encoding="utf-8"))
        except (OSError, ValueError):
            saved = {}
        if saved.get("components") == expected and all((destination / p).is_file() for p in REQUIRED):
            print(f"CUDA compiler already installed: {destination}", flush=True)
            return destination
        raise RuntimeError(f"Refusing to overwrite an incomplete or unmanaged CUDA directory: {destination}")
    destination.parent.mkdir(parents=True, exist_ok=True)
    # Only an entirely extracted and validated toolchain receives the final name.
    with tempfile.TemporaryDirectory(dir=destination.parent, prefix=".cuda-stage-") as folder:
        stage = Path(folder) / "toolkit"
        stage.mkdir()
        for component in COMPONENTS:
            _extract(component, _download(component, cache), stage)
        missing = [p for p in REQUIRED if not (stage / p).is_file()]
        if missing:
            raise RuntimeError("CUDA installation is incomplete: " + ", ".join(missing))
        (stage / MARKER).write_text(json.dumps({"manifest": MANIFEST_URL,
                                              "components": expected}, indent=2) + "\n", encoding="utf-8")
        stage.rename(destination)
    print(f"CUDA compiler installed: {destination}", flush=True)
    return destination


if __name__ == "__main__":
    try:
        ensure_cuda()
    except Exception as error:
        print(f"CUDA setup failed: {error}", file=sys.stderr)
        raise SystemExit(1)
