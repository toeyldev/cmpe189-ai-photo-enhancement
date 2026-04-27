"""
Apply the same degradation pipeline (downsample → upsample → noise σ=50) to
BSD500 clean images.

Mirrors photo_enhancement_pipeline_final.ipynb Cell 21 (Check-in 5).
Uses the EXACT SAME parameters as the Flickr2K degradation in src/degrade_images.py
so the cross-dataset comparison is apples-to-apples.

Inputs:  data_bsd500/clean/*.png
Outputs: data_bsd500/degraded/*.png

Usage:
    python src/degrade_bsd500.py
"""

import os

import cv2
import numpy as np


def degrade_bsd500(clean_dir="data_bsd500/clean",
                   degraded_dir="data_bsd500/degraded",
                   noise_sigma=50,
                   downscale=0.2):
    """
    For every image in `clean_dir`:
        1. Downsample to `downscale`× original (INTER_AREA — avoids aliasing)
        2. Upsample back to original (INTER_CUBIC — smooth 4×4 neighbourhood)
        3. Add Gaussian noise with std = `noise_sigma`
        4. Clip to [0, 255], save as uint8 to `degraded_dir`
    """
    os.makedirs(degraded_dir, exist_ok=True)

    for filename in os.listdir(clean_dir):
        path = os.path.join(clean_dir, filename)

        img = cv2.imread(path)
        if img is None:
            print(f"Could not read {path}")
            continue

        # 1. downsample (INTER_AREA — best for shrinking)
        downsized = cv2.resize(img, None,
                               fx=downscale, fy=downscale,
                               interpolation=cv2.INTER_AREA)

        # 2. upsample back (INTER_CUBIC — smooth, high-quality enlargement)
        upsized = cv2.resize(downsized,
                             (img.shape[1], img.shape[0]),
                             interpolation=cv2.INTER_CUBIC)

        # 3. add Gaussian noise — σ=50 matches SwinIR's training noise level
        noise = np.random.normal(0, noise_sigma, img.shape)
        noisy = np.clip(upsized + noise, 0, 255).astype(np.uint8)

        cv2.imwrite(os.path.join(degraded_dir, filename), noisy)

    print(f"Done — created {len(os.listdir(degraded_dir))} degraded BSD500 images "
          f"in {degraded_dir}/")


if __name__ == "__main__":
    degrade_bsd500()
