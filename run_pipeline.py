#!/usr/bin/env python3
"""
End-to-end AI-Powered Photo Enhancement Pipeline.

Runs the full pipeline from start to finish:
  Step 1: Download clean images from Flickr2K (HuggingFace)
  Step 2: Generate degraded images (downsample + upsample + Gaussian noise)
  Step 3: Run DnCNN denoising inference
  Step 4: Compute PSNR and SSIM metrics
  Step 5: Save enhanced images + results CSV

Usage (run from project root):
    python run_pipeline.py
    python run_pipeline.py --limit 10
    python run_pipeline.py --skip-download --skip-degrade
    python run_pipeline.py --weights model/weights/dncnn_color_blind.pth

Input/output:
    data/clean/*.png    - ground truth images (from HuggingFace)
    data/degraded/*.png - degraded versions (same filenames)
    data/enhanced/*.png - DnCNN denoised outputs
    results_table.csv   - per-image PSNR/SSIM metrics + AVERAGE row
"""

from __future__ import annotations
import argparse
import os
import sys
from pathlib import Path

# project root = folder containing this script
ROOT = Path(__file__).resolve().parent
SRC  = ROOT / "src"

# allow imports from src/
sys.path.insert(0, str(SRC))


def main() -> None:
    # ensure all relative paths resolve from project root
    os.chdir(ROOT)

    parser = argparse.ArgumentParser(
        description="Download, degrade, denoise and evaluate images."
    )
    parser.add_argument(
        "--limit",
        type=int,
        default=50,
        help="Number of Flickr2K images to download (default: 50).",
    )
    parser.add_argument(
        "--skip-download",
        action="store_true",
        help="Skip download — reuse existing data/clean/.",
    )
    parser.add_argument(
        "--skip-degrade",
        action="store_true",
        help="Skip degradation — reuse existing data/degraded/.",
    )
    parser.add_argument(
        "--skip-eval",
        action="store_true",
        help="Skip DnCNN inference and evaluation.",
    )
    parser.add_argument(
        "--weights",
        type=str,
        default=None,
        help="Path to dncnn_color_blind.pth (auto-download if missing).",
    )
    parser.add_argument(
        "--results",
        type=str,
        default="results_table.csv",
        help="Output metrics CSV path (default: results_table.csv).",
    )
    args = parser.parse_args()

    # Step 1: Download clean images from Flickr2K
    if not args.skip_download:
        print("=" * 50)
        print("Step 1: Downloading clean images...")
        print("=" * 50)
        from download_data import download_and_save
        download_and_save(args.limit)

    # Step 2: Generate degraded images
    if not args.skip_degrade:
        print("=" * 50)
        print("Step 2: Generating degraded images...")
        print("=" * 50)
        from degrade_images import degrade_images
        degrade_images()

    # Step 3-5: DnCNN inference + evaluation + save CSV
    if not args.skip_eval:
        print("=" * 50)
        print("Step 3-5: Running DnCNN inference and evaluation...")
        print("=" * 50)
        from evaluate_model import run_evaluation
        run_evaluation(weights_path=args.weights, results_csv=args.results)

    print("=" * 50)
    print("Pipeline complete.")
    print("=" * 50)


if __name__ == "__main__":
    main()
