"""
Pretrained DnCNN weights manager.

Downloads dncnn_color_blind.pth from the official KAIR v1.0 release
if not already present locally.

Weights are for RGB color blind denoising (channels=3, 20 layers).
Source: https://github.com/cszn/KAIR/releases/download/v1.0/dncnn_color_blind.pth
"""

from __future__ import annotations
import urllib.request
from pathlib import Path

DEFAULT_URL = "https://github.com/cszn/KAIR/releases/download/v1.0/dncnn_color_blind.pth"


def default_weights_path(project_root: Path | None = None) -> Path:
    """
    Default location: <repo>/model/weights/dncnn_color_blind.pth
    """
    root = project_root if project_root is not None else Path(__file__).resolve().parent.parent
    return root / "model" / "weights" / "dncnn_color_blind.pth"


def ensure_weights(
    path: Path | str | None = None,
    url: str = DEFAULT_URL,
) -> Path:
    """
    Return path to weights file.
    If the file does not exist, download it from the KAIR release URL.

    Args:
        path: custom path to weights file (default: model/weights/dncnn_color_blind.pth)
        url:  download URL (default: KAIR v1.0 release)

    Returns:
        Path to the weights file
    """
    path = Path(path) if path is not None else default_weights_path()

    if path.exists():
        print(f"Weights already exist at {path}")
        return path

    path.parent.mkdir(parents=True, exist_ok=True)
    print(f"Downloading DnCNN weights to {path} ...")
    urllib.request.urlretrieve(url, path)
    print("Download complete.")

    return path
