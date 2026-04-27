"""
Download the BSD500 dataset and save clean images for cross-dataset evaluation.

Mirrors photo_enhancement_pipeline_final.ipynb Cell 20 (Check-in 5).
Drop-in second dataset — uses the same load_dataset() pattern as Flickr2K
(see src/download_data.py), but writes to data_bsd500/clean/ so that nothing
in the existing Flickr2K pipeline is touched.

Source: https://huggingface.co/datasets/delta-prox/BSD500

Usage:
    python src/download_bsd500.py             # default: 50 images
    python src/download_bsd500.py 100         # custom: 100 images
"""

import os
import sys

from datasets import load_dataset


def download_bsd500(limit=50, save_dir="data_bsd500/clean"):
    """
    Download `limit` images from the BSD500 train split and save as PNG.

    The dataset schema is {"image": <PIL Image>}, identical to yangtao9009/Flickr2K,
    so the same iteration pattern works.
    """
    dataset = load_dataset("delta-prox/BSD500", split=f"train[:{limit}]")

    print(dataset)
    print(dataset[0])

    os.makedirs(save_dir, exist_ok=True)

    for i in range(len(dataset)):
        try:
            item = dataset[i]
            img  = item["image"]                 # PIL Image
            img.save(f"{save_dir}/image_{i}.png")
        except Exception as e:
            print(f"Skipping item {i}: {e}")

    print(f"Done — saved {len(os.listdir(save_dir))} BSD500 clean images to {save_dir}/")


if __name__ == "__main__":
    limit = int(sys.argv[1]) if len(sys.argv) > 1 else 50
    download_bsd500(limit)
