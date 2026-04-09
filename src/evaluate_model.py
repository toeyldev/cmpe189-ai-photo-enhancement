"""
Evaluate DnCNN model using PSNR and SSIM.
Also saves enhanced images for visualization.

Usage (from project root):
    python src/evaluate_model.py
    python src/evaluate_model.py --weights model/weights/dncnn_color_blind.pth
"""

import os
import sys
import argparse
from pathlib import Path

import cv2
import numpy as np
import pandas as pd
import torch

from skimage.metrics import peak_signal_noise_ratio as psnr
from skimage.metrics import structural_similarity as ssim

# allow imports from src/
SRC = Path(__file__).resolve().parent
sys.path.insert(0, str(SRC))

from dncnn_pytorch import DnCNN
from dncnn_weights import ensure_weights


# --- Compute metrics ---
def compute_metrics(clean_img, compare_img):
    """
    Compute PSNR and SSIM between two images.

    clean_img    → ground truth RGB uint8 NumPy array
    compare_img  → degraded OR denoised RGB uint8 NumPy array

    PSNR: measures pixel-level difference (higher is better)
    SSIM: measures structural similarity (range 0-1, higher is better)
    """
    psnr_val = psnr(clean_img, compare_img)
    ssim_val = ssim(clean_img, compare_img, channel_axis=2)
    return psnr_val, ssim_val


# --- Load model ---
def load_model(weights_path=None):
    """
    Load pretrained DnCNN model onto GPU if available, else CPU.

    Architecture: 20 layers, channels=3 (RGB), features=64, no BatchNorm
    Confirmed from dncnn_color_blind.pth weight file inspection.
    """
    # detect GPU
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Using device: {device}")

    # download weights if missing
    weights_path = ensure_weights(weights_path)

    # load state dict
    state_dict = torch.load(weights_path, map_location=device)
    if "params" in state_dict:
        state_dict = state_dict["params"]

    # build model matching exact weight file architecture
    model = DnCNN(channels=3, num_of_layers=20, features=64)
    model.load_state_dict(state_dict, strict=True)
    model = model.to(device)
    model.eval()

    print(f"DnCNN model loaded successfully on {device}")
    return model, device


# --- Run evaluation ---
def run_evaluation(weights_path=None, results_csv="results_table.csv"):
    """
    Run DnCNN inference on all images in data/degraded/.
    Saves enhanced images to data/enhanced/.
    Computes PSNR and SSIM for each image.
    Writes results to results_table.csv.
    """
    clean_dir    = "data/clean"
    degraded_dir = "data/degraded"
    enhanced_dir = "data/enhanced"

    os.makedirs(enhanced_dir, exist_ok=True)

    # load model
    model, device = load_model(weights_path)

    results = []

    for fileName in sorted(os.listdir(clean_dir)):

        # --- Load images ---
        clean_path    = os.path.join(clean_dir, fileName)
        degraded_path = os.path.join(degraded_dir, fileName)

        clean    = cv2.imread(clean_path)
        degraded = cv2.imread(degraded_path)

        if clean is None or degraded is None:
            print(f"Skipping {fileName} - could not read file")
            continue

        # --- Convert BGR → RGB ---
        # cv2.cvtColor() always returns a 3D NumPy array
        clean_rgb    = cv2.cvtColor(clean, cv2.COLOR_BGR2RGB)
        degraded_rgb = cv2.cvtColor(degraded, cv2.COLOR_BGR2RGB)

        # --- Run DnCNN inference ---
        # min-max normalization: [0, 255] → [0.0, 1.0]
        img_norm = degraded_rgb / 255.0

        # NumPy (H, W, C) → PyTorch tensor (1, C, H, W)
        img_tensor = torch.from_numpy(img_norm).permute(2, 0, 1).unsqueeze(0).float()

        # move tensor to GPU
        img_tensor = img_tensor.to(device)

        # inference (no gradient tracking needed)
        with torch.no_grad():
            output = model(img_tensor)

        # move output back to CPU and convert to NumPy
        # (1, C, H, W) → (H, W, C) → uint8
        denoised = output.squeeze().cpu().permute(1, 2, 0).numpy()
        denoised = np.clip(denoised * 255, 0, 255).astype(np.uint8)

        # --- Save enhanced image ---
        save_path = os.path.join(enhanced_dir, fileName)
        # OpenCV uses BGR, so convert RGB → BGR before saving
        denoised_bgr = cv2.cvtColor(denoised, cv2.COLOR_RGB2BGR)
        cv2.imwrite(save_path, denoised_bgr)

        # --- Compute metrics ---
        # baseline: how bad is degraded vs clean
        psnr_degraded, ssim_degraded = compute_metrics(clean_rgb, degraded_rgb)

        # model output: how good is denoised vs clean
        psnr_denoised, ssim_denoised = compute_metrics(clean_rgb, denoised)

        results.append({
            "image"         : fileName,
            "PSNR_degraded" : round(psnr_degraded, 2),
            "PSNR_denoised" : round(psnr_denoised, 2),
            "SSIM_degraded" : round(ssim_degraded, 4),
            "SSIM_denoised" : round(ssim_denoised, 4),
        })

        print(f"{fileName}  PSNR: {psnr_degraded:.2f} → {psnr_denoised:.2f}  SSIM: {ssim_degraded:.4f} → {ssim_denoised:.4f}")

    # --- Build results table ---
    df = pd.DataFrame(results)

    # compute average row
    avg_row = df.mean(numeric_only=True).round(4)
    avg_row["image"] = "AVERAGE"
    df = pd.concat([df, avg_row.to_frame().T], ignore_index=True)

    print("\n" + df.to_string(index=False))

    # save to CSV
    df.to_csv(results_csv, index=False)
    print(f"\nSaved results to {results_csv}")

    return df


# --- Direct execution ---
if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Evaluate DnCNN on degraded images.")
    parser.add_argument("--weights", type=str, default=None,
                        help="Path to dncnn_color_blind.pth (auto-download if missing)")
    parser.add_argument("--results", type=str, default="results_table.csv",
                        help="Output CSV path")
    args = parser.parse_args()

    run_evaluation(weights_path=args.weights, results_csv=args.results)
