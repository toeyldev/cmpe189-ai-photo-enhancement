# AI-Powered Photo Enhancement

**CMPE 189-03 | Group #7**

Thao Huynh · Toey Lui · Zahid Khan · Cody Ambrosio · Ryan Darghous

---

## Overview

This project builds an end-to-end pipeline for AI-powered image enhancement using deep learning. We degrade high-quality images with Gaussian noise and downsampling, then restore them using a pretrained DnCNN model. Performance is evaluated using PSNR and SSIM metrics.

---

## Results

| Metric | Degraded (Baseline) | Denoised (DnCNN) | Improvement |
|--------|--------------------|--------------------|-------------|
| PSNR   | 14.49 dB           | 24.08 dB           | +9.59 dB    |
| SSIM   | 0.0833             | 0.6499             | +0.5666     |

*Evaluated on 50 images from the Flickr2K dataset.*

---

## Project Structure

```
cmpe189-ai-photo-enhancement/
├── run_pipeline.py          # main script — runs full pipeline end-to-end
├── requirements.txt         # Python dependencies
├── src/
│   ├── download_data.py     # downloads Flickr2K images via HuggingFace
│   ├── degrade_images.py    # applies degradation (downsample + noise)
│   ├── evaluate_model.py    # runs DnCNN inference + computes PSNR/SSIM
│   ├── dncnn_pytorch.py     # DnCNN model architecture (PyTorch)
│   └── dncnn_weights.py     # pretrained weight downloader
├── notebooks/
│   ├── DnCNN_Inference_Color.ipynb      # color DnCNN inference notebook
│   └── DnCNN_Inference_Grayscale.ipynb  # grayscale DnCNN inference notebook
├── model/
│   └── weights/
│       └── dncnn_color_blind.pth        # pretrained weights (RGB)
└── photo_enhancement_pipeline_final.ipynb  # main Colab notebook
```

---

## Quick Start

### Run full pipeline (recommended)

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

---

## Pipeline

```
Flickr2K Dataset
      ↓  download_data.py
data/clean/*.png          (ground truth images)
      ↓  degrade_images.py
data/degraded/*.png       (noisy + blurry images)
      ↓  evaluate_model.py (DnCNN inference)
data/enhanced/*.png       (denoised outputs)
      ↓
results_table.csv         (PSNR + SSIM metrics)
```

---

## Model

**DnCNN Color Blind** (`dncnn_color_blind.pth`)
- Source: [cszn/KAIR](https://github.com/cszn/KAIR) releases v1.0
- Architecture: 20 Conv2d layers, ReLU activations, 3 RGB channels, 64 features/layer
- Task: blind color image denoising (handles unknown noise levels)

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

---

## Links

- **GitHub:** https://github.com/toeyldev/cmpe189-ai-photo-enhancement
- **Google Drive:** https://drive.google.com/drive/folders/1qHJwI9oF29m_OOoGmQRwfr1zE0cDUxLh?usp=drive_link