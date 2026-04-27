"""
SwinIR model setup and tile-based inference for RGB color image denoising.

This module mirrors what photo_enhancement_pipeline_final.ipynb does in cells
13 and 15 (Check-in 4: SwinIR). The notebook implements this inline; this file
exposes the same logic as importable functions so the same pipeline can be run
from the command line or other scripts.

Pretrained weights:
    https://github.com/JingyunLiang/SwinIR/releases/download/v0.0/
    005_colorDN_DFWB_s128w8_SwinIR-M_noise50.pth

Architecture (must match the pretrained weight file exactly):
    upscale=1            denoising only (no super-resolution)
    in_chans=3           RGB
    img_size=128         training patch size
    window_size=8        transformer window size
    embed_dim=180
    depths=[6,6,6,6,6,6]
    num_heads=[6,6,6,6,6,6]
    mlp_ratio=2
    upsampler=''
    resi_connection='1conv'

Usage:
    from src.swinir_pytorch import setup_swinir, tile_inference

    swinir_model, device = setup_swinir()
    output = tile_inference(swinir_model, img_tensor,
                            tile_size=256, tile_overlap=32,
                            window_size=8, device=device)
"""

import os
import subprocess
import sys
import urllib.request

import torch
import torch.nn.functional as F


SWINIR_REPO_URL    = "https://github.com/JingyunLiang/SwinIR.git"
SWINIR_WEIGHTS_URL = (
    "https://github.com/JingyunLiang/SwinIR/releases/download/v0.0/"
    "005_colorDN_DFWB_s128w8_SwinIR-M_noise50.pth"
)
SWINIR_WEIGHTS_PATH = "swinir_color_dn50.pth"
SWINIR_REPO_PATH    = "SwinIR"

# tile-inference defaults (chosen to fit on a Colab T4 GPU)
WINDOW_SIZE  = 8     # SwinIR transformer window — image dims must be divisible
TILE_SIZE    = 256   # process image in 256x256 tiles to avoid GPU OOM
TILE_OVERLAP = 32    # overlap between tiles to avoid visible seam artifacts


# ─── Setup ──────────────────────────────────────────────────────────────────

def _ensure_swinir_repo():
    """Clone the SwinIR repo if it isn't already on disk, and add to sys.path."""
    if not os.path.isdir(SWINIR_REPO_PATH):
        subprocess.run(["git", "clone", SWINIR_REPO_URL, SWINIR_REPO_PATH], check=True)
    if SWINIR_REPO_PATH not in sys.path:
        sys.path.insert(0, SWINIR_REPO_PATH)


def _ensure_swinir_weights():
    """Download the pretrained SwinIR weights file if it isn't already on disk."""
    if not os.path.isfile(SWINIR_WEIGHTS_PATH):
        print(f"Downloading SwinIR weights to {SWINIR_WEIGHTS_PATH}...")
        urllib.request.urlretrieve(SWINIR_WEIGHTS_URL, SWINIR_WEIGHTS_PATH)


def setup_swinir(device=None):
    """
    Clone the SwinIR repo, download weights, build the model, and load weights.
    Returns (model, device). The model is moved to GPU if available and put in eval mode.
    """
    _ensure_swinir_repo()
    _ensure_swinir_weights()

    # imported here because it depends on the cloned repo being on sys.path
    from models.network_swinir import SwinIR

    if device is None:
        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Using device: {device}")

    # build model — these parameters MUST match the pretrained weight file exactly
    model = SwinIR(
        upscale=1,                          # 1 = denoising only (no super-resolution)
        in_chans=3,                         # RGB input
        img_size=128,                       # training patch size
        window_size=WINDOW_SIZE,            # transformer window size
        img_range=1.0,                      # pixel range [0, 1]
        depths=[6, 6, 6, 6, 6, 6],          # 6 transformer stages
        embed_dim=180,                      # embedding dimension
        num_heads=[6, 6, 6, 6, 6, 6],       # 6 attention heads per stage
        mlp_ratio=2,                        # MLP expansion ratio
        upsampler='',                       # no upsampler
        resi_connection='1conv',            # residual connection type
    )

    # load weights
    pretrained = torch.load(SWINIR_WEIGHTS_PATH, map_location=device)
    if 'params' in pretrained:               # unwrap if needed
        pretrained = pretrained['params']

    model.load_state_dict(pretrained, strict=True)
    model = model.to(device)
    model.eval()

    print("SwinIR model loaded successfully on", device)
    return model, device


# ─── Inference ──────────────────────────────────────────────────────────────

def tile_inference(model, img_tensor,
                   tile_size=TILE_SIZE,
                   tile_overlap=TILE_OVERLAP,
                   window_size=WINDOW_SIZE,
                   device=None):
    """
    Split a large image into overlapping tiles, run model on each tile,
    then stitch tiles back together by averaging the overlapping regions.

    Avoids GPU out of memory on large images (e.g. 2040x1356 Flickr2K images
    require ~5.5 GB, which exceeds many GPUs' available memory).

    Parameters
    ----------
    model : torch.nn.Module
        SwinIR model (already in eval mode and moved to `device`).
    img_tensor : torch.Tensor
        Input tensor of shape (1, C, H, W), values in [0, 1], on `device`.
    tile_size : int
        Tile edge length. Default 256.
    tile_overlap : int
        Pixels of overlap between adjacent tiles. Default 32.
    window_size : int
        SwinIR window size. Tile dims will be padded to a multiple of this.
    device : torch.device or None
        Compute device (only used for empty buffer creation).

    Returns
    -------
    torch.Tensor
        Output tensor of the same shape as `img_tensor`, on the same device.
    """
    b, c, h, w = img_tensor.shape
    output = torch.zeros_like(img_tensor)
    count  = torch.zeros_like(img_tensor)

    stride = tile_size - tile_overlap

    for y in range(0, h, stride):
        for x in range(0, w, stride):
            # tile boundaries
            y_end = min(y + tile_size, h)
            x_end = min(x + tile_size, w)
            y_start = y_end - tile_size if y_end - y < tile_size else y
            x_start = x_end - tile_size if x_end - x < tile_size else x

            # extract tile
            tile = img_tensor[:, :, y_start:y_end, x_start:x_end]

            # pad so dimensions are divisible by window_size
            _, _, th, tw = tile.shape
            th_pad = (window_size - th % window_size) % window_size
            tw_pad = (window_size - tw % window_size) % window_size
            if th_pad > 0 or tw_pad > 0:
                tile = F.pad(tile, (0, tw_pad, 0, th_pad), mode='reflect')

            # run model
            with torch.no_grad():
                tile_out = model(tile)

            # crop back to original tile size
            tile_out = tile_out[:, :, :th, :tw]

            # accumulate
            output[:, :, y_start:y_end, x_start:x_end] += tile_out
            count[:, :, y_start:y_end, x_start:x_end]  += 1

    # average overlapping regions
    return output / count


if __name__ == "__main__":
    # quick smoke test — load model, run a dummy 256x256 RGB image through it
    model, device = setup_swinir()
    dummy = torch.rand(1, 3, 256, 256, device=device)
    out = tile_inference(model, dummy, device=device)
    print(f"Smoke test OK — output shape: {tuple(out.shape)}")