"""
Run SwinIR inference on Flickr2K degraded images and save PSNR/SSIM results.

Mirrors photo_enhancement_pipeline_final.ipynb Cell 15 (Check-in 4).
The notebook implements this inline; this file exposes the same logic as a
reusable function and a __main__ entry point.

Inputs  (read from disk):
    data/clean/        — ground truth clean images
    data/degraded/     — degraded versions (downsample → upsample → noise σ=50)

Outputs (written to disk):
    data/swinir_enhanced/  — SwinIR-restored images
    swinir_results_table.csv — per-image + AVERAGE row of PSNR / SSIM

Usage:
    python src/evaluate_swinir.py
"""

import os

import cv2
import numpy as np
import pandas as pd
import torch
from skimage.metrics import peak_signal_noise_ratio as psnr_metric
from skimage.metrics import structural_similarity as ssim_metric

from src.swinir_pytorch import setup_swinir, tile_inference, TILE_SIZE, TILE_OVERLAP, WINDOW_SIZE


def evaluate_swinir(clean_dir="data/clean",
                    degraded_dir="data/degraded",
                    enhanced_dir="data/swinir_enhanced",
                    csv_out="swinir_results_table.csv",
                    tile_size=TILE_SIZE,
                    tile_overlap=TILE_OVERLAP,
                    window_size=WINDOW_SIZE):
    """
    Run SwinIR (tile-based) on every image in `degraded_dir`, saving outputs
    to `enhanced_dir` and a results CSV to `csv_out`.

    Returns the results DataFrame (per-image rows + final AVERAGE row).
    """
    os.makedirs(enhanced_dir, exist_ok=True)

    # load model once, reuse for all images
    model, device = setup_swinir()

    results = []

    for filename in sorted(os.listdir(clean_dir)):
        clean    = cv2.imread(os.path.join(clean_dir,    filename))
        degraded = cv2.imread(os.path.join(degraded_dir, filename))

        if clean is None or degraded is None:
            print(f"Skipping {filename} - could not read file")
            continue

        # BGR → RGB
        clean_rgb    = cv2.cvtColor(clean,    cv2.COLOR_BGR2RGB)
        degraded_rgb = cv2.cvtColor(degraded, cv2.COLOR_BGR2RGB)

        # normalize to [0, 1]
        img_norm = degraded_rgb / 255.0

        # NumPy (H, W, C) → PyTorch (1, C, H, W) → GPU
        img_tensor = (
            torch.from_numpy(img_norm)
            .permute(2, 0, 1)
            .unsqueeze(0)
            .float()
            .to(device)
        )

        # tile-based inference
        output = tile_inference(model, img_tensor,
                                tile_size, tile_overlap, window_size, device)

        # back to NumPy uint8
        denoised = output.squeeze().cpu().permute(1, 2, 0).numpy()
        denoised = np.clip(denoised * 255, 0, 255).astype(np.uint8)

        # save (RGB → BGR for cv2.imwrite)
        cv2.imwrite(os.path.join(enhanced_dir, filename),
                    cv2.cvtColor(denoised, cv2.COLOR_RGB2BGR))

        # metrics
        psnr_degraded = psnr_metric(clean_rgb, degraded_rgb)
        ssim_degraded = ssim_metric(clean_rgb, degraded_rgb, channel_axis=2)
        psnr_swinir   = psnr_metric(clean_rgb, denoised)
        ssim_swinir   = ssim_metric(clean_rgb, denoised, channel_axis=2)

        results.append({
            "image"         : filename,
            "PSNR_degraded" : round(psnr_degraded, 2),
            "PSNR_swinir"   : round(psnr_swinir,   2),
            "SSIM_degraded" : round(ssim_degraded, 4),
            "SSIM_swinir"   : round(ssim_swinir,   4),
        })

        print(f"{filename}  PSNR: {psnr_degraded:.2f} → {psnr_swinir:.2f}  "
              f"SSIM: {ssim_degraded:.4f} → {ssim_swinir:.4f}")

        # release GPU memory between images
        torch.cuda.empty_cache()

    # build dataframe + AVERAGE row
    df = pd.DataFrame(results)
    avg_row = df.mean(numeric_only=True).round(4)
    avg_row["image"] = "AVERAGE"
    df = pd.concat([df, avg_row.to_frame().T], ignore_index=True)

    print("\n" + df.to_string(index=False))

    df.to_csv(csv_out, index=False)
    print(f"\nSaved to {csv_out}")
    return df


if __name__ == "__main__":
    evaluate_swinir()
