# AI-Powered Photo Enhancement

**CMPE 189-03 | Group #7 | Spring 2026**

Thao Huynh · Toey Lui · Cody Ambrosio · Ryan Darghous · Zahid Khan
*Department of Computer Engineering, San José State University*

---

## Overview

This project builds an end-to-end pipeline for AI-powered image enhancement using deep learning. We degrade high-quality images with Gaussian noise and downsampling, then restore them using two pretrained deep learning models (**DnCNN** and **SwinIR**). Performance is evaluated using PSNR and SSIM metrics on **two benchmark datasets** — Flickr2K and BSD500 — to test cross-dataset generalization.

---

## Results

### Flickr2K — DnCNN vs SwinIR (50 images, σ=50)

| Metric | Degraded | DnCNN | SwinIR | Winner |
|--------|----------|-------|--------|--------|
| PSNR (dB) | 14.49 | 24.08 | **24.27** | SwinIR (+0.19 dB) |
| SSIM      | 0.0832 | 0.6499 | **0.6644** | SwinIR (+0.0145) |

*SwinIR outperforms DnCNN on **all 50 / 50** individual test images.*

### BSD500 — Cross-Dataset Evaluation (50 images, σ=50)

| Metric | Degraded | DnCNN | SwinIR | Winner |
|--------|----------|-------|--------|--------|
| PSNR (dB) | 14.29 | 22.89 | **23.02** | SwinIR (+0.12 dB) |
| SSIM      | 0.0903 | 0.5620 | **0.5736** | SwinIR (+0.0116) |

*Both models generalize to BSD500 with only ~1 dB drop, confirming SwinIR's advantage is architectural rather than dataset-specific.*

### Noise Level Sensitivity (DnCNN)

| σ  | PSNR Degraded | PSNR Denoised | SSIM Degraded | SSIM Denoised |
|----|---------------|---------------|---------------|---------------|
| 15 | 21.73 | 21.71 | 0.3032 | 0.3038 |
| 25 | 19.00 | 18.99 | 0.1855 | 0.1859 |
| 50 | 14.49 | 14.50 | 0.0832 | 0.0834 |

*At lower σ, the dominant degradation is blur (not noise), so a dedicated denoiser provides limited improvement.*

---

## Project Structure

```
cmpe189-ai-photo-enhancement/
├── photo_enhancement_pipeline_final.ipynb   # main Colab notebook (recommended entry point)
├── run_pipeline.py                          # standalone DnCNN pipeline script
├── requirements.txt                         # Python dependencies
├── src/
│   ├── download_data.py             # downloads Flickr2K via HuggingFace
│   ├── degrade_images.py            # applies degradation (downsample + noise)
│   ├── dncnn_pytorch.py             # DnCNN model architecture (PyTorch)
│   ├── dncnn_weights.py             # pretrained DnCNN weight loader
│   ├── evaluate_model.py            # DnCNN inference + PSNR/SSIM (Check-in 3)
│   ├── swinir_pytorch.py            # SwinIR setup + tile-based inference (Check-in 4)
│   ├── evaluate_swinir.py           # SwinIR inference on Flickr2K (Check-in 4)
│   ├── download_bsd500.py           # downloads BSD500 dataset (Check-in 5)
│   ├── degrade_bsd500.py            # degrades BSD500 with same pipeline (Check-in 5)
│   ├── evaluate_bsd500.py           # runs DnCNN + SwinIR on BSD500 (Check-in 5)
│   ├── cross_dataset_comparison.py  # builds Flickr2K vs BSD500 summary table (Check-in 5)
│   └── visualize_comparison.py      # generates 4-panel comparison figures
├── notebooks/
│   ├── DnCNN_Inference_Color.ipynb              # color DnCNN exploration
│   ├── DnCNN_Inference_Grayscale.ipynb          # grayscale DnCNN exploration
│   └── DnCNN_Inference_NoiseLevelComparison.ipynb  # noise level study
└── model/
    └── weights/
        ├── dncnn_color_blind.pth        # pretrained DnCNN weights (RGB)
        └── dncnn_25.pth                 # pretrained DnCNN weights (grayscale)
```

> **Note:** The main Colab notebook (`photo_enhancement_pipeline_final.ipynb`) is the primary orchestrator for the full project. Each `src/*.py` file mirrors a corresponding section of the notebook so the pipeline can also be run from the command line. The notebook clones this repo's `DnCNN` branch in Cell 0 to access `src/download_data.py` and `src/degrade_images.py`.

---

## Quick Start

### Option 1 — Run in Google Colab (recommended)

Open `photo_enhancement_pipeline_final.ipynb` in Colab. Enable GPU under **Runtime → Change runtime type → T4 GPU**, then run all cells.

The notebook covers:
- Cells 0–7: DnCNN denoising on Flickr2K (Check-in 3)
- Cells 8–18: SwinIR denoising with tile-based inference + DnCNN vs. SwinIR comparison (Check-in 4)
- Cells 19–24: BSD500 cross-dataset evaluation (Check-in 5)
- Cell 25: Export all results into a `report_outputs.zip` for the final report

End-to-end runtime: **~10–13 minutes** on a Colab T4 GPU.

### Option 2 — Run locally (DnCNN only)

```bash
git clone https://github.com/toeyldev/cmpe189-ai-photo-enhancement.git
cd cmpe189-ai-photo-enhancement
pip install -r requirements.txt

python run_pipeline.py            # full pipeline, 50 images
python run_pipeline.py --limit 5  # quick test, 5 images
```

---

## Pipeline

```
Flickr2K / BSD500 dataset
        ↓  download_data.py
data/clean/*.png              (ground truth)
        ↓  degrade_images.py
data/degraded/*.png           (downsample → upsample → +noise σ=50)
        ↓                ↓
DnCNN inference    SwinIR inference (tile-based, 256×256 with 32px overlap)
        ↓                ↓
data/enhanced/     data/swinir_enhanced/
        ↓                ↓
results_table.csv  swinir_results_table.csv
        ↓
DnCNN vs SwinIR comparison + 4-panel visualization
```

---

## Models

### DnCNN (Check-in 3)
- **Weights:** `dncnn_color_blind.pth` from [cszn/KAIR](https://github.com/cszn/KAIR) releases v1.0
- **Architecture:** 20 Conv2d + ReLU layers, 3 RGB channels, 64 features per layer, no batch norm
- **Parameters:** ~670K
- **Task:** blind color image denoising via residual learning: ĉ = x − F(x)
- **Average PSNR improvement:** +9.59 dB on Flickr2K, +8.60 dB on BSD500

### SwinIR (Check-in 4)
- **Weights:** `005_colorDN_DFWB_s128w8_SwinIR-M_noise50.pth` from [JingyunLiang/SwinIR](https://github.com/JingyunLiang/SwinIR)
- **Architecture:** Swin Transformer backbone, 6 stages, embed_dim=180, window_size=8
- **Parameters:** ~11.8M (~10× larger than DnCNN)
- **Task:** color image denoising at noise level σ=50
- **Average PSNR improvement:** +9.78 dB on Flickr2K, +8.72 dB on BSD500
- **Note:** Uses tile-based inference (256×256 tiles, 32-pixel overlap) to handle large images without GPU OOM

---

## Degradation Pipeline

Each clean image is degraded in three steps:

1. **Downsample** to 20% using `INTER_AREA` (averages pixels, discards detail)
2. **Upsample** back to original size using `INTER_CUBIC` (estimates pixels, introduces blur)
3. **Add Gaussian noise** with σ = 50 (simulates severe sensor noise)

We chose σ = 50 because it (a) matches SwinIR's pretrained noise level for fair comparison, and (b) creates a meaningful restoration challenge.

---

## Evaluation Metrics

- **PSNR** (Peak Signal-to-Noise Ratio) — pixel-level reconstruction accuracy in decibels; higher is better
- **SSIM** (Structural Similarity Index) — perceptual similarity in [0, 1]; higher is better

Both are computed twice per image: degraded vs. clean (baseline) and model output vs. clean (restored). The difference (ΔPSNR) measures the model's net improvement.

---

## References

1. K. Zhang, W. Zuo, Y. Chen, D. Meng, and L. Zhang, "Beyond a Gaussian denoiser: Residual learning of deep CNN for image denoising," *IEEE Transactions on Image Processing*, vol. 26, no. 7, pp. 3142–3155, Jul. 2017.
2. J. Liang, J. Cao, G. Sun, K. Zhang, L. Van Gool, and R. Timofte, "SwinIR: Image restoration using Swin Transformer," in *Proc. IEEE/CVF International Conference on Computer Vision Workshops (ICCVW)*, Oct. 2021, pp. 1833–1844.
3. Z. Liu, Y. Lin, Y. Cao, et al., "Swin Transformer: Hierarchical vision transformer using shifted windows," in *Proc. IEEE/CVF ICCV*, Oct. 2021, pp. 10012–10022.
4. D. Martin, C. Fowlkes, D. Tal, and J. Malik, "A database of human segmented natural images and its application to evaluating segmentation algorithms and measuring ecological statistics," in *Proc. IEEE ICCV*, Jul. 2001, vol. 2, pp. 416–423.

See the full project report for the complete reference list (9 citations) and methodology details.

---

## Links

- **GitHub:** https://github.com/toeyldev/cmpe189-ai-photo-enhancement
- **Google Drive (outputs):** https://drive.google.com/drive/folders/1qHJwI9oF29m_OOoGmQRwfr1zE0cDUxLh
- **Google Colab:** https://colab.research.google.com/drive/1U0zSlriKwTqMPQBZiTRrYfh_XsS_W-cC

---

## Acknowledgments

This project uses pretrained weights from [cszn/KAIR](https://github.com/cszn/KAIR) (DnCNN) and [JingyunLiang/SwinIR](https://github.com/JingyunLiang/SwinIR) (SwinIR). Datasets are loaded via the [HuggingFace `datasets`](https://huggingface.co/docs/datasets/index) library.