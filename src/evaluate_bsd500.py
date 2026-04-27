"""
Run BOTH DnCNN and SwinIR on BSD500 degraded images, save outputs and metrics.

Mirrors photo_enhancement_pipeline_final.ipynb Cell 22 (Check-in 5).
Reuses pretrained weights — no retraining, pure cross-dataset evaluation.

Inputs:  data_bsd500/clean/, data_bsd500/degraded/
Outputs:
    data_bsd500/enhanced/         — DnCNN-restored images
    data_bsd500/swinir_enhanced/  — SwinIR-restored images
    results_table_bsd500.csv      — DnCNN per-image PSNR / SSIM + AVERAGE
    swinir_results_table_bsd500.csv — SwinIR per-image PSNR / SSIM + AVERAGE

Usage:
    python src/evaluate_bsd500.py
"""

import os
import sys

import cv2
import numpy as np
import pandas as pd
import torch
from skimage.metrics import peak_signal_noise_ratio as psnr_metric
from skimage.metrics import structural_similarity as ssim_metric

from src.dncnn_pytorch  import DnCNN
from src.dncnn_weights  import download_dncnn_weights
from src.swinir_pytorch import setup_swinir, tile_inference, TILE_SIZE, TILE_OVERLAP, WINDOW_SIZE


def _add_avg_row(df):
    """Append an AVERAGE row to a metrics DataFrame."""
    avg = df.mean(numeric_only=True).round(4)
    avg["image"] = "AVERAGE"
    return pd.concat([df, avg.to_frame().T], ignore_index=True)


def _setup_dncnn(weights_path="dncnn_color_blind.pth"):
    """Build DnCNN, download weights if missing, load them, return (model, device)."""
    if not os.path.isfile(weights_path):
        download_dncnn_weights(weights_path)

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model = DnCNN(channels=3, num_of_layers=20, features=64)
    state = torch.load(weights_path, map_location=device)
    model.load_state_dict(state)
    model = model.to(device).eval()
    return model, device


def evaluate_bsd500(clean_dir="data_bsd500/clean",
                    degraded_dir="data_bsd500/degraded",
                    dncnn_dir="data_bsd500/enhanced",
                    swinir_dir="data_bsd500/swinir_enhanced",
                    dncnn_csv="results_table_bsd500.csv",
                    swinir_csv="swinir_results_table_bsd500.csv"):
    """
    Run DnCNN + SwinIR on every image in `degraded_dir`. Saves restored images
    and per-image PSNR/SSIM CSVs for each model.
    """
    os.makedirs(dncnn_dir,  exist_ok=True)
    os.makedirs(swinir_dir, exist_ok=True)

    # load both models once
    dncnn_model,  device = _setup_dncnn()
    swinir_model, _      = setup_swinir(device=device)

    dncnn_results, swinir_results = [], []

    for filename in sorted(os.listdir(clean_dir)):
        clean    = cv2.imread(os.path.join(clean_dir,    filename))
        degraded = cv2.imread(os.path.join(degraded_dir, filename))

        if clean is None or degraded is None:
            print(f"Skipping {filename} - could not read file")
            continue

        clean_rgb    = cv2.cvtColor(clean,    cv2.COLOR_BGR2RGB)
        degraded_rgb = cv2.cvtColor(degraded, cv2.COLOR_BGR2RGB)

        # normalize + tensorize
        img_norm   = degraded_rgb / 255.0
        img_tensor = (
            torch.from_numpy(img_norm)
            .permute(2, 0, 1)
            .unsqueeze(0)
            .float()
            .to(device)
        )

        # ─── DnCNN ──────────────────────────────────────────────
        # BSD500 images are small (~481x321) — no tiling needed for DnCNN
        with torch.no_grad():
            dncnn_out = dncnn_model(img_tensor)

        dncnn_img = dncnn_out.squeeze().cpu().permute(1, 2, 0).numpy()
        dncnn_img = np.clip(dncnn_img * 255, 0, 255).astype(np.uint8)

        cv2.imwrite(os.path.join(dncnn_dir, filename),
                    cv2.cvtColor(dncnn_img, cv2.COLOR_RGB2BGR))

        # ─── SwinIR (tile-based, same as Flickr2K) ─────────────
        swinir_out = tile_inference(swinir_model, img_tensor,
                                    TILE_SIZE, TILE_OVERLAP, WINDOW_SIZE, device)

        swinir_img = swinir_out.squeeze().cpu().permute(1, 2, 0).numpy()
        swinir_img = np.clip(swinir_img * 255, 0, 255).astype(np.uint8)

        cv2.imwrite(os.path.join(swinir_dir, filename),
                    cv2.cvtColor(swinir_img, cv2.COLOR_RGB2BGR))

        # ─── Metrics ────────────────────────────────────────────
        psnr_degraded = psnr_metric(clean_rgb, degraded_rgb)
        ssim_degraded = ssim_metric(clean_rgb, degraded_rgb, channel_axis=2)

        psnr_dncnn = psnr_metric(clean_rgb, dncnn_img)
        ssim_dncnn = ssim_metric(clean_rgb, dncnn_img, channel_axis=2)

        psnr_swinir = psnr_metric(clean_rgb, swinir_img)
        ssim_swinir = ssim_metric(clean_rgb, swinir_img, channel_axis=2)

        dncnn_results.append({
            "image"         : filename,
            "PSNR_degraded" : round(psnr_degraded, 2),
            "PSNR_denoised" : round(psnr_dncnn,    2),
            "SSIM_degraded" : round(ssim_degraded, 4),
            "SSIM_denoised" : round(ssim_dncnn,    4),
        })
        swinir_results.append({
            "image"         : filename,
            "PSNR_degraded" : round(psnr_degraded, 2),
            "PSNR_swinir"   : round(psnr_swinir,   2),
            "SSIM_degraded" : round(ssim_degraded, 4),
            "SSIM_swinir"   : round(ssim_swinir,   4),
        })

        print(f"{filename}  "
              f"DnCNN: {psnr_degraded:.2f}→{psnr_dncnn:.2f}  "
              f"SwinIR: {psnr_degraded:.2f}→{psnr_swinir:.2f}")

        torch.cuda.empty_cache()

    # ─── Save CSVs ───────────────────────────────────────────
    df_dncnn  = _add_avg_row(pd.DataFrame(dncnn_results))
    df_swinir = _add_avg_row(pd.DataFrame(swinir_results))

    df_dncnn.to_csv(dncnn_csv,   index=False)
    df_swinir.to_csv(swinir_csv, index=False)

    print("\n── BSD500 · DnCNN ──")
    print(df_dncnn.tail(3).to_string(index=False))

    print("\n── BSD500 · SwinIR ──")
    print(df_swinir.tail(3).to_string(index=False))

    print(f"\nSaved: {dncnn_csv}, {swinir_csv}")
    return df_dncnn, df_swinir


if __name__ == "__main__":
    evaluate_bsd500()
