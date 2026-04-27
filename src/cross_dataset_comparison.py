"""
Build the headline cross-dataset comparison table — Flickr2K vs BSD500.

Mirrors photo_enhancement_pipeline_final.ipynb Cell 23 (Check-in 5).
Reads the AVERAGE rows from the four per-dataset CSV files, pivots them into
a 6-row × 5-column summary, and saves the result.

Inputs (must already exist):
    results_table.csv               (Flickr2K · DnCNN)
    swinir_results_table.csv        (Flickr2K · SwinIR)
    results_table_bsd500.csv        (BSD500   · DnCNN)
    swinir_results_table_bsd500.csv (BSD500   · SwinIR)

Output:
    cross_dataset_comparison.csv    — final headline table

Usage:
    python src/cross_dataset_comparison.py
"""

import pandas as pd


def _get_avg(csv_path, col_psnr, col_ssim):
    """Pull the AVERAGE row out of a per-dataset CSV and return a small dict."""
    df  = pd.read_csv(csv_path)
    row = df[df["image"] == "AVERAGE"].iloc[0]
    return {
        "PSNR_degraded": float(row["PSNR_degraded"]),
        "PSNR_model"   : float(row[col_psnr]),
        "SSIM_degraded": float(row["SSIM_degraded"]),
        "SSIM_model"   : float(row[col_ssim]),
    }


def build_cross_dataset_table(out_csv="cross_dataset_comparison.csv"):
    """
    Read the four per-dataset average rows, build the summary table,
    print a headline takeaway, and save to CSV.
    """
    flickr_dncnn  = _get_avg("results_table.csv",               "PSNR_denoised", "SSIM_denoised")
    flickr_swinir = _get_avg("swinir_results_table.csv",        "PSNR_swinir",   "SSIM_swinir")
    bsd_dncnn     = _get_avg("results_table_bsd500.csv",        "PSNR_denoised", "SSIM_denoised")
    bsd_swinir    = _get_avg("swinir_results_table_bsd500.csv", "PSNR_swinir",   "SSIM_swinir")

    summary = pd.DataFrame([
        {"Dataset": "Flickr2K", "Model": "Degraded",
         "PSNR (dB)": flickr_dncnn["PSNR_degraded"],
         "SSIM":      flickr_dncnn["SSIM_degraded"],
         "ΔPSNR":     0.0},
        {"Dataset": "Flickr2K", "Model": "DnCNN",
         "PSNR (dB)": flickr_dncnn["PSNR_model"],
         "SSIM":      flickr_dncnn["SSIM_model"],
         "ΔPSNR":     round(flickr_dncnn["PSNR_model"]  - flickr_dncnn["PSNR_degraded"],  2)},
        {"Dataset": "Flickr2K", "Model": "SwinIR",
         "PSNR (dB)": flickr_swinir["PSNR_model"],
         "SSIM":      flickr_swinir["SSIM_model"],
         "ΔPSNR":     round(flickr_swinir["PSNR_model"] - flickr_swinir["PSNR_degraded"], 2)},
        {"Dataset": "BSD500", "Model": "Degraded",
         "PSNR (dB)": bsd_dncnn["PSNR_degraded"],
         "SSIM":      bsd_dncnn["SSIM_degraded"],
         "ΔPSNR":     0.0},
        {"Dataset": "BSD500", "Model": "DnCNN",
         "PSNR (dB)": bsd_dncnn["PSNR_model"],
         "SSIM":      bsd_dncnn["SSIM_model"],
         "ΔPSNR":     round(bsd_dncnn["PSNR_model"]     - bsd_dncnn["PSNR_degraded"],     2)},
        {"Dataset": "BSD500", "Model": "SwinIR",
         "PSNR (dB)": bsd_swinir["PSNR_model"],
         "SSIM":      bsd_swinir["SSIM_model"],
         "ΔPSNR":     round(bsd_swinir["PSNR_model"]    - bsd_swinir["PSNR_degraded"],    2)},
    ])

    # round for display
    summary["PSNR (dB)"] = summary["PSNR (dB)"].round(2)
    summary["SSIM"]      = summary["SSIM"].round(4)

    print("═" * 70)
    print("  Cross-Dataset Evaluation — Flickr2K vs BSD500")
    print("═" * 70)
    print(summary.to_string(index=False))
    print("═" * 70)

    # ─── Headline takeaway ──────────────────────────────────────────
    print("\nPer-dataset winners (by average PSNR):")
    for ds, d, s in [("Flickr2K", flickr_dncnn, flickr_swinir),
                     ("BSD500",   bsd_dncnn,    bsd_swinir)]:
        winner = "SwinIR" if s["PSNR_model"] > d["PSNR_model"] else "DnCNN"
        gap    = abs(s["PSNR_model"] - d["PSNR_model"])
        print(f"  {ds:9s}  →  {winner}  (by {gap:.2f} dB)")

    print("\nGeneralization check (does each model behave consistently across datasets?):")
    print(f"  DnCNN  ΔPSNR — Flickr2K: +{flickr_dncnn['PSNR_model']  - flickr_dncnn['PSNR_degraded']:.2f} dB  |  "
          f"BSD500: +{bsd_dncnn['PSNR_model']     - bsd_dncnn['PSNR_degraded']:.2f} dB")
    print(f"  SwinIR ΔPSNR — Flickr2K: +{flickr_swinir['PSNR_model'] - flickr_swinir['PSNR_degraded']:.2f} dB  |  "
          f"BSD500: +{bsd_swinir['PSNR_model']    - bsd_swinir['PSNR_degraded']:.2f} dB")

    summary.to_csv(out_csv, index=False)
    print(f"\nSaved: {out_csv}")
    return summary


if __name__ == "__main__":
    build_cross_dataset_table()
