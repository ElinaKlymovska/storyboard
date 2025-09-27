"""Local mock of Supabase storage for offline runs.

The real pipeline expects a ``SupabaseStorageManager`` with a method
``upload_image_from_pil`` that returns a mapping containing a
``public_url`` key.  In the absence of real Supabase credentials we save
the image to ``output/supabase_mock`` and return a ``file://`` URL so the
rest of the pipeline can continue working.
"""

from __future__ import annotations

from pathlib import Path
from typing import Any, Dict

from PIL import Image


class SupabaseStorageManager:
    """Minimal stub that mirrors the API used by the pipeline."""

    def __init__(self, base_dir: Path | None = None) -> None:
        self.base_dir = base_dir or Path.cwd() / "output" / "supabase_mock"
        self.base_dir.mkdir(parents=True, exist_ok=True)

    def upload_image_from_pil(
        self,
        image: Image.Image,
        filename: str,
        folder: str | Path | None = None,
        **_: Any,
    ) -> Dict[str, str]:
        target_dir = self.base_dir
        if folder:
            target_dir = target_dir / Path(folder)
        target_dir.mkdir(parents=True, exist_ok=True)

        file_path = target_dir / filename
        image.save(file_path)

        return {"public_url": file_path.resolve().as_uri()}

