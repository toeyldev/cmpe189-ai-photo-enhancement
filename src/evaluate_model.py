"""
Run pretrained DnCNN on degraded images and measure quality vs ground truth.

Expects paired PNGs in data/clean and data/degraded (same filenames). Writes
denoised images to data/enhanced/ and a metrics table to results_table.csv.

Run from the project root (or use run_pipeline.py), so data/ paths resolve correctly.
"""

import os
import cv2
import numpy as np
import pandas as pd
import torch
from skimage.metrics import peak_signal_noise_ratio as psnr
from skimage.metrics import structural_similarity as ssim

from dncnn_pytorch import DnCNN
from dncnn_weights import ensure_weights


def compute_metrics(clean_img, compare_img):
    """PSNR / SSIM between ground truth (clean) and another RGB image (degraded or denoised)."""
    psnr_val = psnr(clean_img, compare_img)
    ssim_val = ssim(clean_img, compare_img, channel_axis=2)
    return psnr_val, ssim_val


def load_model(weights_path=None):
    """
    Load KAIR dncnn_color_blind weights (RGB). Architecture must match the checkpoint
    (nb=20, act_mode='R'). Runs on CPU; paths come from ensure_weights().
    """
    weights_file = ensure_weights(weights_path)
    model = DnCNN(in_nc=3, out_nc=3, nc=64, nb=20, act_mode="R")

    state_dict = torch.load(weights_file, map_location="cpu")
    model.load_state_dict(state_dict, strict=True)
    model.eval()
    return model


def run_evaluation(weights_path=None, results_csv="results_table.csv"):
    """
    Loop over all clean/degraded pairs: denoise, save to data/enhanced/, build CSV.

    weights_path: optional path to .pth; None uses weights/dncnn_color_blind.pth
    (downloaded automatically if missing via dncnn_weights).
    """
    # Paths are relative to current working directory — run_pipeline.py sets cwd to repo root.
    clean_dir = "data/clean"
    degraded_dir = "data/degraded"
    enhanced_dir = "data/enhanced"

    if not os.path.isdir(clean_dir):
        raise FileNotFoundError("Missing data/clean. Run download_data.py first.")
    if not os.path.isdir(degraded_dir):
        raise FileNotFoundError("Missing data/degraded. Run degrade_images.py first.")

    os.makedirs(enhanced_dir, exist_ok=True)

    model = load_model(weights_path)
    results = []

    for fileName in sorted(os.listdir(clean_dir)):
        clean_path = os.path.join(clean_dir, fileName)
        degraded_path = os.path.join(degraded_dir, fileName)

        clean = cv2.imread(clean_path)
        degraded = cv2.imread(degraded_path)

        if clean is None or degraded is None:
            print(f"Skipping {fileName} - could not read file")
            continue

        # OpenCV loads BGR; model and skimage metrics expect RGB.
        clean_rgb = cv2.cvtColor(clean, cv2.COLOR_BGR2RGB)
        degraded_rgb = cv2.cvtColor(degraded, cv2.COLOR_BGR2RGB)

        # PyTorch expects NCHW float in [0, 1] (batch, 3, H, W).
        img_norm = degraded_rgb / 255.0
        img_tensor = torch.from_numpy(img_norm).permute(2, 0, 1).unsqueeze(0).float()

        with torch.no_grad():
            output = model(img_tensor)  # denoised tensor (DnCNN forward: x - noise)

        denoised = output.squeeze(0).permute(1, 2, 0).cpu().numpy()
        denoised = np.clip(denoised * 255, 0, 255).astype(np.uint8)

        # Save enhanced image (convert back to BGR for cv2.imwrite).
        save_path = os.path.join(enhanced_dir, fileName)
        denoised_bgr = cv2.cvtColor(denoised, cv2.COLOR_RGB2BGR)
        cv2.imwrite(save_path, denoised_bgr)

        # Baseline: degraded vs clean; model: denoised vs clean.
        psnr_degraded, ssim_degraded = compute_metrics(clean_rgb, degraded_rgb)
        psnr_denoised, ssim_denoised = compute_metrics(clean_rgb, denoised)

        results.append({
            "image": fileName,
            "PSNR_degraded": round(psnr_degraded, 2),
            "PSNR_denoised": round(psnr_denoised, 2),
            "SSIM_degraded": round(ssim_degraded, 4),
            "SSIM_denoised": round(ssim_denoised, 4),
        })

    df = pd.DataFrame(results)

    if not df.empty:
        # Summary row for the report (mean over numeric columns).
        avg_row = df.mean(numeric_only=True).round(4)
        avg_row["image"] = "AVERAGE"
        df = pd.concat([df, avg_row.to_frame().T], ignore_index=True)

    print(df.to_string(index=False))
    df.to_csv(results_csv, index=False)
    print(f"Saved results to {results_csv}")


if __name__ == "__main__":
    run_evaluation()
