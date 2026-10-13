"""Binary assets (figure images, video keyframes) stored next to the index.

Chunks reference assets by a relative path ``<doc_id>/<name>``; the API serves them so a UI
can show the actual figure behind a citation (visual citations).
"""

from __future__ import annotations

import hashlib
import io
import shutil
from pathlib import Path


class AssetStore:
    def __init__(self, root: Path) -> None:
        self.root = root
        root.mkdir(parents=True, exist_ok=True)

    def _resolve(self, rel: str) -> Path:
        path = (self.root / rel).resolve()
        if self.root.resolve() not in path.parents:
            raise ValueError(f"Invalid asset path: {rel!r}")
        return path

    def save(self, doc_id: str, data: bytes, suffix: str = ".png", name: str | None = None) -> str:
        name = name or hashlib.sha256(data).hexdigest()[:16] + suffix
        rel = f"{doc_id}/{name}"
        path = self._resolve(rel)
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_bytes(data)
        return rel

    def read(self, rel: str) -> bytes:
        return self._resolve(rel).read_bytes()

    def path(self, rel: str) -> Path:
        return self._resolve(rel)

    def exists(self, rel: str) -> bool:
        try:
            return self._resolve(rel).exists()
        except ValueError:
            return False

    def delete_document(self, doc_id: str) -> None:
        shutil.rmtree(self.root / doc_id, ignore_errors=True)


def to_png(data: bytes, max_side: int = 1600) -> tuple[bytes, int, int]:
    """Normalise any image to PNG (vision APIs reject JPX/CMYK/TIFF) and cap its size."""
    from PIL import Image

    with Image.open(io.BytesIO(data)) as img:
        img.load()
        if img.mode not in ("RGB", "RGBA", "L"):
            img = img.convert("RGB")
        if max(img.size) > max_side:
            img.thumbnail((max_side, max_side))
        out = io.BytesIO()
        img.save(out, format="PNG", optimize=True)
        return out.getvalue(), img.width, img.height
