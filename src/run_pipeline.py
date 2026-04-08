#!/usr/bin/env python3
"""
End-to-end pipeline (Cody / integration):

  Clean images (Flickr2K) → degradation → DnCNN denoising → PSNR/SSIM + CSV

Run from the project root:

  python run_pipeline.py
  python run_pipeline.py --limit 10
  python run_pipeline.py --skip-download --skip-degrade   # only evaluate

Input/output contract:
  data/clean/*.png   — ground truth (from Hugging Face)
  data/degraded/*.png — same filenames, artificially degraded
  data/enhanced/*.png — DnCNN outputs
  results_table.csv  — per-image metrics + AVERAGE row
"""

from __future__ import annotations

import argparse
import os
import sys
from pathlib import Path

# Project root = folder that contains this script (so data/ and weights/ resolve correctly).
ROOT = Path(__file__).resolve().parent
SRC = ROOT / "src"
# Import helpers from src/ (download_data.py, degrade_images.py, evaluate_model.py) as plain modules.
sys.path.insert(0, str(SRC))


def main() -> None:
    # All relative paths in the pipeline (data/*, results_table.csv) assume cwd is the repo root.
    os.chdir(ROOT)

    parser = argparse.ArgumentParser(
        description="Download data, degrade images, run DnCNN evaluation."
    )
    parser.add_argument(
        "--limit",
        type=int,
        default=50,
        help="Number of Flickr2K training images to download (default: 50).",
    )
    parser.add_argument(
        "--skip-download",
        action="store_true",
        help="Reuse existing data/clean without calling Hugging Face.",
    )
    parser.add_argument(
        "--skip-degrade",
        action="store_true",
        help="Skip degradation; expects data/degraded to already exist.",
    )
    parser.add_argument(
        "--skip-eval",
        action="store_true",
        help="Only download/degrade; do not run DnCNN or write results CSV.",
    )
    parser.add_argument(
        "--weights",
        type=str,
        default=None,
        help="Path to dncnn_color_blind.pth (default: weights/dncnn_color_blind.pth, auto-download if missing).",
    )
    parser.add_argument(
        "--results",
        type=str,
        default="results_table.csv",
        help="Output metrics CSV path (relative to project root).",
    )
    args = parser.parse_args()

    # Step 1: Hugging Face → data/clean/image_*.png
    if not args.skip_download:
        from download_data import download_and_save

        download_and_save(args.limit)

    # Step 2: data/clean → data/degraded (same filenames; blur + noise)
    if not args.skip_degrade:
        from degrade_images import degrade_images

        degrade_images()

    # Step 3: DnCNN on degraded → data/enhanced + PSNR/SSIM CSV
    if not args.skip_eval:
        from evaluate_model import run_evaluation

        run_evaluation(weights_path=args.weights, results_csv=args.results)


if __name__ == "__main__":
    main()
