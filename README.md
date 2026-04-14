# AI-Powered Photo Enhancement

**CMPE 189-03 | Group #7**

Thao Huynh · Toey Lui · Zahid Khan · Cody Ambrosio · Ryan Darghous

---

## Overview

This project builds an end-to-end pipeline for AI-powered image enhancement using deep learning. We degrade high-quality images with Gaussian noise and downsampling, then restore them using pretrained deep learning models (DnCNN and SwinIR). Performance is evaluated using PSNR and SSIM metrics, and results are compared across both models.

---

## Results

### DnCNN vs SwinIR — Average Results (50 images, Flickr2K)

| Metric | Degraded (Baseline) | DnCNN | SwinIR | Winner |
|--------|---------------------|-------|--------|--------|
| PSNR   | 14.49 dB            | 24.08 dB | 24.27 dB | SwinIR (+0.19 dB) |
| SSIM   | 0.0833              | 0.6499   | 0.6645   | SwinIR (+0.0146)  |

*SwinIR outperforms DnCNN on all 50 images.*

---

## Project Structure

```
cmpe189-ai-photo-enhancement/
├── run_pipeline.py          # main script — runs full DnCNN pipeline end-to-end
├── requirements.txt         # Python dependencies
├── src/
│   ├── download_data.py     # downloads Flickr2K images via HuggingFace
│   ├── degrade_images.py    # applies degradation (downsample + noise)
│   ├── evaluate_model.py    # runs DnCNN inference + computes PSNR/SSIM
│   ├── dncnn_pytorch.py     # DnCNN model architecture (PyTorch)
│   ├── dncnn_weights.py     # pretrained DnCNN weight downloader
│   └── swinir_pytorch.py    # SwinIR model setup + tile-based inference (Check-in 4)
├── notebooks/
│   ├── DnCNN_Inference_Color.ipynb      # color DnCNN inference notebook
│   └── DnCNN_Inference_Grayscale.ipynb  # grayscale DnCNN inference notebook
│   └── DnCNN_Inference_NoiseLevelComparison.ipynb  # noise level comparison notebook
├── model/
│   └── weights/
│       ├── dncnn_color_blind.pth        # pretrained DnCNN weights (RGB)
│       └── dncnn_25.pth                 # pretrained DnCNN weights (grayscale)
└── photo_enhancement_pipeline_final.ipynb  # main Colab notebook
```

---

## Quick Start

### Run full pipeline (DnCNN)

```bash
# clone the repo
git clone https://github.com/toeyldev/cmpe189-ai-photo-enhancement.git
cd cmpe189-ai-photo-enhancement

# install dependencies
pip install -r requirements.txt

# run full pipeline (downloads 50 images, degrades, denoise, evaluates)
python run_pipeline.py

# run with fewer images for quick testing
python run_pipeline.py --limit 5
```

### Run in Google Colab

Open `photo_enhancement_pipeline_final.ipynb` in Google Colab. Make sure to enable **GPU** under `Runtime → Change runtime type → T4 GPU`.

The notebook includes:
- DnCNN denoising pipeline (Check-in 3)
- SwinIR denoising pipeline with tile-based inference (Check-in 4)
- DnCNN vs SwinIR comparison table and 4-panel visualization

---

## Pipeline

```
Flickr2K Dataset
      ↓  download_data.py
data/clean/*.png              (ground truth images)
      ↓  degrade_images.py
data/degraded/*.png           (noisy + blurry images)
      ↓                    ↓
DnCNN inference         SwinIR inference
(evaluate_model.py)     (swinir_pytorch.py)
      ↓                    ↓
data/enhanced/*.png     data/swinir_enhanced/*.png
      ↓                    ↓
results_table.csv       swinir_results_table.csv
      ↓
comparison table + 4-panel figure
```

---

## Models

### DnCNN (Check-in 3)
- **Weights:** `dncnn_color_blind.pth` from [cszn/KAIR](https://github.com/cszn/KAIR) releases v1.0
- **Architecture:** 20 Conv2d layers, ReLU activations, 3 RGB channels, 64 features/layer
- **Task:** blind color image denoising
- **Average PSNR improvement:** +9.59 dB

### SwinIR (Check-in 4)
- **Weights:** `005_colorDN_DFWB_s128w8_SwinIR-M_noise50.pth` from [JingyunLiang/SwinIR](https://github.com/JingyunLiang/SwinIR)
- **Architecture:** Swin Transformer, 6 stages, embed_dim=180, window_size=8
- **Task:** color image denoising at noise level σ=50
- **Average PSNR improvement:** +9.78 dB
- **Note:** Uses tile-based inference (256×256 tiles) to handle large images without GPU out of memory

---

## Degradation Pipeline

Each clean image is degraded in 3 steps:

1. **Downsample** to 20% using `INTER_AREA` (loses detail)
2. **Upsample** back to original size using `INTER_CUBIC` (introduces blur)
3. **Add Gaussian noise** with σ = 50 (simulates sensor noise)

---

## Evaluation Metrics

- **PSNR** (Peak Signal-to-Noise Ratio) — pixel-level accuracy, higher is better (dB)
- **SSIM** (Structural Similarity Index) — perceptual similarity, range 0–1, higher is better

---

## References

1. Zhang, K. et al. (2017). Beyond a Gaussian Denoiser: Residual Learning of Deep CNN for Image Denoising. *IEEE Transactions on Image Processing*.
2. Wang, X. et al. (2019). ESRGAN: Enhanced Super-Resolution Generative Adversarial Networks. *ECCV 2018 Workshops*.
3. Liang, J. et al. (2021). SwinIR: Image Restoration Using Swin Transformer. *ICCV 2021 Workshops*.

---

## Links

- **GitHub:** https://github.com/toeyldev/cmpe189-ai-photo-enhancement
- **Google Drive:** https://drive.google.com/drive/folders/1qHJwI9oF29m_OOoGmQRwfr1zE0cDUxLh?usp=drive_link
- **Google Colab:** https://colab.research.google.com/drive/1U0zSlriKwTqMPQBZiTRrYfh_XsS_W-cC
