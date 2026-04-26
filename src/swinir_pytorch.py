"""
SwinIR model setup and tile-based inference for RGB color image denoising.

SwinIR (Swin Transformer for Image Restoration) is a transformer-based model
that outperforms DnCNN on complex images.

Pretrained weights source:
    https://github.com/JingyunLiang/SwinIR/releases/download/v0.0/005_colorDN_DFWB_s128w8_SwinIR-M_noise50.pth

Architecture parameters (must match pretrained weight file exactly):
    - upscale=1          → denoising only (no super-resolution)
    - in_chans=3         → RGB input (3 channels)
    - img_size=128       → training patch size
    - window_size=8      → transformer window size
    - embed_dim=180      → embedding dimension
    - depths=[6,6,6,6,6,6]     → 6 transformer stages
    - num_heads=[6,6,6,6,6,6]  → 6 attention heads per stage

Usage:
    from src.swinir_pytorch import load_swinir_model, swinir_inference

    model, device = load_swinir_model()
    denoised = swinir_inference(model, device, degraded_rgb)
"""

from __future__ import annotations #Import annotations library to work with type hints
import shutil #Import shutil library to remove directories and files
import subprocess #Import subprocess library to run commands in the terminal
import sys #Import sys library to work with the system
import urllib.request #Import urllib.request library to download files from the internet
from pathlib import Path #Import pathlib library to work with file paths

import numpy as np #Import numpy library to work with arrays
import torch #Import torch library to work with tensors
import torch.nn.functional as F #Import torch.nn.functional library to work with neural networks


# ── Constants ─────────────────────────────────────────────────────────────────

SWINIR_REPO_URL    = "https://github.com/JingyunLiang/SwinIR.git"
SWINIR_WEIGHTS_URL = (
    "https://github.com/JingyunLiang/SwinIR/releases/download/v0.0/"
    "005_colorDN_DFWB_s128w8_SwinIR-M_noise50.pth"
)
DEFAULT_WEIGHTS_PATH = Path(__file__).resolve().parent.parent / "model" / "weights" / "swinir_color_dn50.pth"
SWINIR_REPO_PATH     = Path(__file__).resolve().parent.parent / "SwinIR"

WINDOW_SIZE  = 8    # SwinIR transformer window size — image dims must be divisible by this
TILE_SIZE    = 256  # process image in 256x256 tiles to avoid GPU out of memory
TILE_OVERLAP = 32   # overlap between tiles to avoid visible seam artifacts


def _swinir_repo_has_architecture() -> bool:
    """True if cloned upstream repo contains the SwinIR network module we import."""
    # load_swinir_model does "from models.network_swinir import SwinIR" — this file must exist or import fails.
    return (SWINIR_REPO_PATH / "models" / "network_swinir.py").is_file()


def _clone_swinir_repo() -> None:
    """
    Clone JingyunLiang/SwinIR into SWINIR_REPO_PATH.

    Must use subprocess with a argv list (not os.system + unquoted paths): project
    paths often contain spaces (e.g. 'Code Projects'), which breaks shell parsing
    and yields an incomplete clone — then 'import models' fails.
    """
    SWINIR_REPO_PATH.parent.mkdir(parents=True, exist_ok=True)
    # argv list (no shell): each argument is one token — paths like "...\Code Projects\..." stay a single path.
    proc = subprocess.run(
        [
            "git",
            "clone",
            "--depth",
            "1",  # shallow clone — enough for models/ code, faster first run
            SWINIR_REPO_URL,
            str(SWINIR_REPO_PATH),  # destination; quoted implicitly because it is one list element, not split on spaces
        ],
        capture_output=True,  # capture git errors for the RuntimeError below
        text=True,
        cwd=str(SWINIR_REPO_PATH.parent),  # run git from project parent so clone target resolves cleanly
    )
    if proc.returncode != 0:
        raise RuntimeError(
            "Could not clone SwinIR (git failed). Install Git and ensure "
            "'git' is on your PATH, then try again.\n"
            f"git stdout: {proc.stdout.strip()}\n"
            f"git stderr: {proc.stderr.strip()}"
        )


# ── Setup: Clone SwinIR repo and download weights ─────────────────────────────

def setup_swinir(weights_path: Path | str | None = None) -> Path:
    """
    Clone SwinIR repo (for model architecture) and download pretrained weights.

    The SwinIR model architecture lives in the SwinIR GitHub repo.
    We need to clone it so we can import models.network_swinir.SwinIR.

    Args:
        weights_path: custom path to weights file
                      (default: model/weights/swinir_color_dn50.pth)

    Returns:
        Path to the weights file
    """
    # Step 1: Ensure a valid SwinIR checkout (models/network_swinir.py must exist).
    # Old code used os.system("git clone ... " + path): spaces in path broke the shell → bad/missing clone → ModuleNotFoundError: models.
    if SWINIR_REPO_PATH.exists() and not _swinir_repo_has_architecture():
        print("SwinIR folder exists but is incomplete; removing and re-cloning...")
        shutil.rmtree(SWINIR_REPO_PATH, ignore_errors=True)  # drop broken folder so _clone_swinir_repo() can run fresh

    if not SWINIR_REPO_PATH.exists():
        print("Cloning SwinIR repo (first run; may take a minute)...")
        _clone_swinir_repo()  # uses subprocess list form — fixes clone under paths with spaces
        print("SwinIR repo cloned.")
    else:
        print("SwinIR repo already present.")

    if not _swinir_repo_has_architecture():
        raise RuntimeError(
            f"SwinIR checkout at {SWINIR_REPO_PATH} is missing models/network_swinir.py. "
            "Delete the SwinIR folder and try again, or check that git clone completed."
        )  # last line of defense if git failed silently or disk was interrupted mid-clone

    # Step 2: Add SwinIR to Python path so we can import from it
    swinir_str = str(SWINIR_REPO_PATH)
    if swinir_str not in sys.path:
        sys.path.insert(0, swinir_str)

    # Step 3: Download weights if not present
    weights_path = Path(weights_path) if weights_path is not None else DEFAULT_WEIGHTS_PATH

    if not weights_path.exists():
        weights_path.parent.mkdir(parents=True, exist_ok=True)
        print(f"Downloading SwinIR weights to {weights_path} ...")
        urllib.request.urlretrieve(SWINIR_WEIGHTS_URL, weights_path)
        print("Download complete.")
    else:
        print(f"SwinIR weights already exist at {weights_path}")

    return weights_path


# ── Load Model ────────────────────────────────────────────────────────────────

def load_swinir_model(weights_path: Path | str | None = None):
    """
    Load pretrained SwinIR model onto GPU if available, else CPU.

    Architecture parameters match the pretrained weight file exactly:
        upscale=1, in_chans=3, img_size=128, window_size=8,
        embed_dim=180, depths=[6,6,6,6,6,6], num_heads=[6,6,6,6,6,6]

    Args:
        weights_path: custom path to weights file
                      (default: model/weights/swinir_color_dn50.pth)

    Returns:
        (model, device) tuple
    """
    # detect GPU
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Using device: {device}")

    # setup repo and weights
    weights_path = setup_swinir(weights_path)

    # import SwinIR architecture from cloned repo
    from models.network_swinir import SwinIR

    # build model — parameters must match weight file exactly
    model = SwinIR(
        upscale=1,                   # 1 = denoising only (no upscaling)
        in_chans=3,                  # 3 = RGB input channels
        img_size=128,                # training patch size
        window_size=WINDOW_SIZE,     # transformer window size
        img_range=1.0,               # pixel range [0, 1]
        depths=[6, 6, 6, 6, 6, 6],  # depth of each transformer stage
        embed_dim=180,               # embedding dimension
        num_heads=[6, 6, 6, 6, 6, 6],  # attention heads per stage
        mlp_ratio=2,                 # MLP expansion ratio
        upsampler='',                # no upsampler (denoising only)
        resi_connection='1conv'      # residual connection type
    )

    # load pretrained weights
    pretrained = torch.load(weights_path, map_location=device)

    # unwrap 'params' key if present (same pattern as DnCNN)
    if 'params' in pretrained:
        pretrained = pretrained['params']

    model.load_state_dict(pretrained, strict=True)
    model = model.to(device)
    model.eval()

    print("SwinIR model loaded successfully on", device)
    return model, device


# ── Tile-Based Inference ──────────────────────────────────────────────────────

def tile_inference(model, img_tensor, device, *, return_stats: bool = False):
    """
    Split large image into overlapping tiles, run SwinIR on each tile,
    then stitch results back together.

    Why tiles are needed:
        SwinIR is a large transformer model. Processing a full 2040x1356 image
        at once requires ~5.5GB GPU memory which causes OutOfMemoryError.
        Each 256x256 tile only needs ~0.5GB — fits easily.

    Tile overlap (32 pixels):
        Tiles overlap at edges to prevent visible seam lines.
        Overlapping regions are averaged together for smooth transitions.

    Args:
        model:      loaded SwinIR model (on GPU)
        img_tensor: input tensor of shape (1, C, H, W) on GPU
        device:     torch device

    Returns:
        If return_stats is False:
            output tensor of shape (1, C, H, W) on GPU
        If return_stats is True:
            (output, stats) where stats is a dict with tiling information
    """
    b, c, h, w = img_tensor.shape

    # accumulate output and count for averaging overlapping regions
    output = torch.zeros_like(img_tensor)
    count  = torch.zeros_like(img_tensor)

    stride = TILE_SIZE - TILE_OVERLAP
    tiles_processed = 0

    for y in range(0, h, stride):
        for x in range(0, w, stride):
            tiles_processed += 1

            # tile boundaries
            y_end = min(y + TILE_SIZE, h)
            x_end = min(x + TILE_SIZE, w)
            # Align last partial tile to the bottom/right with a full TILE_SIZE crop when possible.
            # Must clamp to 0: if h (or w) < TILE_SIZE, y_end - TILE_SIZE is negative; negative
            # indices in PyTorch slice from the end, so writes miss the top/left → count stays 0
            # there → inf/NaN after divide → black bands (common when short side < 256 px).
            y_start = max(0, y_end - TILE_SIZE) if y_end - y < TILE_SIZE else y
            x_start = max(0, x_end - TILE_SIZE) if x_end - x < TILE_SIZE else x

            # extract tile
            tile = img_tensor[:, :, y_start:y_end, x_start:x_end]

            # pad tile so dims are divisible by window_size (8)
            # SwinIR uses 8x8 transformer windows
            _, _, th, tw = tile.shape
            th_pad = (WINDOW_SIZE - th % WINDOW_SIZE) % WINDOW_SIZE
            tw_pad = (WINDOW_SIZE - tw % WINDOW_SIZE) % WINDOW_SIZE
            if th_pad > 0 or tw_pad > 0:
                tile = F.pad(tile, (0, tw_pad, 0, th_pad), mode='reflect')

            # run model on tile
            with torch.no_grad():
                tile_out = model(tile)

            # crop back to original tile size (remove padding)
            tile_out = tile_out[:, :, :th, :tw]

            # accumulate output (overlapping regions will be averaged)
            output[:, :, y_start:y_end, x_start:x_end] += tile_out
            count[:, :, y_start:y_end, x_start:x_end]  += 1

    # average overlapping regions for smooth transitions
    safe_count = count.clamp(min=1)  # avoid div-by-zero if a pixel were never covered (should not happen after correct y_start/x_start)
    output = output / safe_count
    output = torch.nan_to_num(output, nan=0.0, posinf=1.0, neginf=0.0)
    if not return_stats:
        return output

    return output, {
        "tile_size": TILE_SIZE,
        "tile_overlap": TILE_OVERLAP,
        "tiles_processed": tiles_processed,
        "window_size": WINDOW_SIZE,
    }


# ── Full Inference Pipeline ───────────────────────────────────────────────────

def swinir_inference(model, device, degraded_rgb: np.ndarray, *, return_stats: bool = False):
    """
    Run SwinIR denoising on a single RGB image.

    Args:
        model:       loaded SwinIR model
        device:      torch device
        degraded_rgb: degraded image as uint8 RGB NumPy array (H, W, 3)

    Returns:
        If return_stats is False:
            denoised image as uint8 RGB NumPy array (H, W, 3)
        If return_stats is True:
            (denoised_rgb, stats) where stats includes tiling info

    Pipeline:
        degraded_rgb (uint8)
            ↓ / 255.0
        normalized [0, 1]
            ↓ permute + unsqueeze
        tensor (1, C, H, W) on GPU
            ↓ tile_inference(swinir_model)
        output tensor (1, C, H, W)
            ↓ squeeze + permute + cpu + numpy
        denoised float [0, 1]
            ↓ * 255 + clip + uint8
        denoised_rgb (uint8)
    """
    # normalize to [0, 1]
    img_norm = degraded_rgb / 255.0

    # NumPy (H, W, C) → PyTorch tensor (1, C, H, W) on GPU
    img_tensor = (
        torch.from_numpy(img_norm)
        .permute(2, 0, 1)
        .unsqueeze(0)
        .float()
        .to(device)
    )

    # run tile-based inference
    if return_stats:
        output, stats = tile_inference(model, img_tensor, device, return_stats=True)
    else:
        output = tile_inference(model, img_tensor, device, return_stats=False)

    # tensor (1, C, H, W) → NumPy (H, W, C)
    denoised = output.squeeze().cpu().permute(1, 2, 0).numpy()

    # undo normalization: [0, 1] → [0, 255]
    denoised = np.clip(denoised * 255, 0, 255).astype(np.uint8)

    # free GPU memory
    torch.cuda.empty_cache()

    if return_stats:
        return denoised, stats
    return denoised
