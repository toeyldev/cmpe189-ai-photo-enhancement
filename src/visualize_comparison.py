"""
Generate 4-panel comparison figures: Clean | Degraded | DnCNN | SwinIR.

Mirrors photo_enhancement_pipeline_final.ipynb Cells 18 (Flickr2K) and 24 (BSD500).
Generates one figure per dataset.

Usage:
    python src/visualize_comparison.py                # both datasets, image_0.png
    python src/visualize_comparison.py image_5.png    # both datasets, image_5.png
"""

import os
import sys

import cv2
import matplotlib.pyplot as plt


def visualize_4panel(name="image_0.png",
                     clean_dir="data/clean",
                     degraded_dir="data/degraded",
                     dncnn_dir="data/enhanced",
                     swinir_dir="data/swinir_enhanced",
                     suptitle="Model Comparison: DnCNN vs SwinIR",
                     out_path="comparison_figure.png"):
    """
    Build a 1x4 figure (Clean / Degraded / DnCNN / SwinIR) for `name`.
    """
    clean    = cv2.cvtColor(cv2.imread(os.path.join(clean_dir,    name)), cv2.COLOR_BGR2RGB)
    degraded = cv2.cvtColor(cv2.imread(os.path.join(degraded_dir, name)), cv2.COLOR_BGR2RGB)
    dncnn    = cv2.cvtColor(cv2.imread(os.path.join(dncnn_dir,    name)), cv2.COLOR_BGR2RGB)
    swinir   = cv2.cvtColor(cv2.imread(os.path.join(swinir_dir,   name)), cv2.COLOR_BGR2RGB)

    fig, axes = plt.subplots(1, 4, figsize=(20, 5))

    axes[0].imshow(clean);    axes[0].set_title("Clean (Ground Truth)");   axes[0].axis("off")
    axes[1].imshow(degraded); axes[1].set_title("Degraded (Model Input)"); axes[1].axis("off")
    axes[2].imshow(dncnn);    axes[2].set_title("DnCNN Output");           axes[2].axis("off")
    axes[3].imshow(swinir);   axes[3].set_title("SwinIR Output");          axes[3].axis("off")

    plt.suptitle(suptitle, fontsize=14, fontweight="bold")
    plt.tight_layout()
    plt.savefig(out_path, dpi=150, bbox_inches="tight")
    plt.close(fig)

    print(f"Figure saved to {out_path}")


if __name__ == "__main__":
    name = sys.argv[1] if len(sys.argv) > 1 else "image_0.png"

    # Flickr2K
    visualize_4panel(
        name=name,
        clean_dir="data/clean",
        degraded_dir="data/degraded",
        dncnn_dir="data/enhanced",
        swinir_dir="data/swinir_enhanced",
        suptitle="Model Comparison: DnCNN vs SwinIR (Flickr2K)",
        out_path="comparison_figure_checkin4.png",
    )

    # BSD500
    visualize_4panel(
        name=name,
        clean_dir="data_bsd500/clean",
        degraded_dir="data_bsd500/degraded",
        dncnn_dir="data_bsd500/enhanced",
        swinir_dir="data_bsd500/swinir_enhanced",
        suptitle=f"BSD500 — {name}: DnCNN vs SwinIR",
        out_path="comparison_figure_bsd500.png",
    )
