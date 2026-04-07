"""
KAIR pretrained DnCNN weights (RGB color blind denoising).

If the .pth file is missing locally, we download it once from the official KAIR v1.0 release.
Weights are gitignored via *.pth; keep a copy under weights/ after first run.
"""

from __future__ import annotations

import urllib.request
from pathlib import Path

# Same URL as cszn/KAIR main_download_pretrained_models.py (DnCNN group).
DEFAULT_DNCNN_COLOR_BLIND_URL = (
    "https://github.com/cszn/KAIR/releases/download/v1.0/dncnn_color_blind.pth"
)


def default_weights_path(project_root: Path | None = None) -> Path:
    """Default location: <repo>/weights/dncnn_color_blind.pth (next to data/, src/)."""
    root = project_root if project_root is not None else Path(__file__).resolve().parent.parent
    return root / "weights" / "dncnn_color_blind.pth"


def ensure_weights(
    path: Path | str | None = None,
    url: str = DEFAULT_DNCNN_COLOR_BLIND_URL,
) -> Path:
    """
    Return an absolute path to the weights file. If it does not exist, download from `url`.
    Pass a custom `path` if you stored the checkpoint elsewhere.
    """
    path = Path(path) if path is not None else default_weights_path()
    if path.exists():
        return path

    path.parent.mkdir(parents=True, exist_ok=True)
    print(f"Downloading DnCNN weights to {path} ...")
    urllib.request.urlretrieve(url, path)
    print("Done.")
    return path
