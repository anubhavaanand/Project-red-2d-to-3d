"""Provenance checks for model weights fetched implicitly at runtime.

Two libraries download pretrained weights on first use with no integrity check:

* ``rembg`` fetches the segmentation model into ``~/.u2net`` on the first
  ``new_session(...)`` call.
* ``lpips`` fetches VGG16 ImageNet weights via ``torchvision`` into the torch
  hub cache.

A compromised or substituted CDN would silently change background removal and
perceptual metrics — exactly the numbers the benchmark reports.

This module pins the expected SHA-256 of each artefact. A cached copy is
verified in place; otherwise the file is downloaded to the path the upstream
library expects and verified before use. A mismatch is fatal: we never fall
back to unverified bytes.
"""

from __future__ import annotations

import hashlib
import os
import shutil
import urllib.request
from pathlib import Path
from typing import NamedTuple

_CHUNK = 1024 * 1024


class ChecksumMismatchError(RuntimeError):
    """Raised when a cached or downloaded artefact fails its pinned hash."""


class Asset(NamedTuple):
    path: Path
    url: str
    sha256: str


def _sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with open(path, "rb") as fh:
        for chunk in iter(lambda: fh.read(_CHUNK), b""):
            digest.update(chunk)
    return digest.hexdigest()


def _download(url: str, dest: Path, expected: str) -> None:
    """Stream ``url`` to ``dest``, verify, then move into place."""
    dest.parent.mkdir(parents=True, exist_ok=True)
    part = dest.with_name(dest.name + ".part")
    try:
        with urllib.request.urlopen(url) as response, open(part, "wb") as fh:
            shutil.copyfileobj(response, fh, _CHUNK)
        digest = _sha256(part)
        if digest != expected:
            raise ChecksumMismatchError(f"{dest.name}: expected sha256 {expected}, got {digest}")
        os.replace(part, dest)
    finally:
        part.unlink(missing_ok=True)


def ensure(asset: Asset) -> Path:
    """Return a verified local path for ``asset``, downloading it if absent."""
    if asset.path.exists():
        digest = _sha256(asset.path)
        if digest != asset.sha256:
            raise ChecksumMismatchError(
                f"{asset.path}: expected sha256 {asset.sha256}, got {digest}"
            )
        return asset.path
    _download(asset.url, asset.path, asset.sha256)
    return asset.path


def _u2net_home() -> Path:
    return Path(os.environ.get("U2NET_HOME", Path.home() / ".u2net"))


def _torch_checkpoint_dir() -> Path:
    torch_home = Path(os.environ.get("TORCH_HOME", Path.home() / ".cache" / "torch"))
    return torch_home / "hub" / "checkpoints"


# name -> (url, sha256). Keep in sync with requirements.txt
# (rembg[gpu]==2.0.50, lpips==0.1.4 / torchvision==0.17.0).
# Hashes were computed by downloading each file from the URL below.
_PINS: dict[str, tuple[str, str]] = {
    "isnet-general-use.onnx": (
        "https://github.com/danielgatis/rembg/releases/download/v0.0.0/isnet-general-use.onnx",
        "60920e99c45464f2ba57bee2ad08c919a52bbf852739e96947fbb4358c0d964a",
    ),
    "vgg16-397923af.pth": (
        "https://download.pytorch.org/models/vgg16-397923af.pth",
        "397923af8e79cdbb6a7127f12361acd7a2f83e06b05044ddf496e83de57a5bf0",
    ),
}


def asset(name: str) -> Asset:
    """Build the pinned :class:`Asset` record for a known weight file."""
    try:
        url, sha256 = _PINS[name]
    except KeyError:
        raise KeyError(f"no pinned weights for {name!r}") from None
    directory = _u2net_home() if name.endswith(".onnx") else _torch_checkpoint_dir()
    return Asset(path=directory / name, url=url, sha256=sha256)
