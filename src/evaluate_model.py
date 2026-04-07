"""
Evaluate DnCNN model using PSNR and SSIM.
Also saves enhanced images for visualization.
"""

import os
import cv2
import numpy as np
import pandas as pd
import torch
from skimage.metrics import peak_signal_noise_ratio as psnr
from skimage.metrics import structural_similarity as ssim


# --- Compute metrics function ---
def compute_metrics(clean_img, compare_img):
    """
    clean_img    → ground truth image
    compare_img  → degraded OR denoised image

    PSNR: measures pixel-level difference (higher is better)
    SSIM: measures structural similarity (range 0–1, higher is better)
    """
    psnr_val = psnr(clean_img, compare_img)
    ssim_val = ssim(clean_img, compare_img, channel_axis=2)
    return psnr_val, ssim_val


# --- Direct execution (same as notebook) ---

clean_dir    = "data/clean"
degraded_dir = "data/degraded"
enhanced_dir = "data/enhanced"

os.makedirs(enhanced_dir, exist_ok=True)

#create a list (of dictionary later)
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
    # normalize → tensor → model → back to uint8 RGB

    """
    min-max formula:  (x - min) / (max - min)
                = (x - 0)   / (255 - 0)
                = x / 255

    Since min is always 0 for images, the formula simplifies to just / 255.0.
    """

    img_norm = degraded_rgb / 255.0   # min-max normalization to [0, 1]


    # NOW, img_tensor has 4 dimensions, still on CPU (default)
    img_tensor = torch.from_numpy(img_norm).permute(2, 0, 1).unsqueeze(0).float()

    # unsqueeze(0) → adds batch dimension at front
    # (C, H, W) → (1, C, H, W)
    # batch size is 1 since we process one image at a time

    """
    torch.from_numpy(img_norm)   # (H, W, C) → tensor (H, W, C)
    .permute(2, 0, 1)            # (H, W, C) → (C, H, W)
    .unsqueeze(0)                # (C, H, W) → (1, C, H, W)
    .float()                     # convert to float32
    """

    # torch.no_grad():
    # → do not compute gradients (faster, less memory)
    # gradients are only needed during training

    with torch.no_grad():
        output = model(img_tensor)  
        # model must be defined (Toey's pretrained DnCNN)


    # --- Convert output tensor back to image ---

    """
    output.squeeze()      # (1, C, H, W) → (C, H, W)
    .permute(1, 2, 0)     # (C, H, W) → (H, W, C)
    .numpy()              # tensor → NumPy array
    """

    """
    unsqueeze → adds dimension
    squeeze   → removes dimension (only if size = 1)
    """

    """
    Forward (NumPy → Tensor):          Backward (Tensor → NumPy):
    (H, W, C)                          (1, C, H, W)
      ↓ permute(2, 0, 1)                 ↓ squeeze()
    (C, H, W)                          (C, H, W)
      ↓ unsqueeze(0)                     ↓ permute(1, 2, 0)
    (1, C, H, W)                       (H, W, C)
      ↓ float()                          ↓ numpy()
    """

    denoised = output.squeeze().permute(1, 2, 0).numpy()

    # undo normalization: [0,1] → [0,255]
    denoised = np.clip(denoised * 255, 0, 255).astype(np.uint8)


    # --- Save enhanced image for Ryan ---
    save_path = os.path.join(enhanced_dir, fileName)

    # OpenCV uses BGR, so convert RGB → BGR before saving
    denoised_bgr = cv2.cvtColor(denoised, cv2.COLOR_RGB2BGR)
    cv2.imwrite(save_path, denoised_bgr)



    # --- Compute metrics ---

    # baseline: degraded vs clean
    psnr_degraded, ssim_degraded = compute_metrics(clean_rgb, degraded_rgb)

    # model output: denoised vs clean
    psnr_denoised, ssim_denoised = compute_metrics(clean_rgb, denoised)

    # results is a list of dictionaries (each loop = one row)
    # SSIM uses more decimals because range is 0–1

    results.append({
        "image"         : fileName,
        "PSNR_degraded" : round(psnr_degraded, 2),
        "PSNR_denoised" : round(psnr_denoised, 2),
        "SSIM_degraded" : round(ssim_degraded, 4),
        "SSIM_denoised" : round(ssim_denoised, 4),
    })


# --- Build table ---

# list of dict → pandas DataFrame
df = pd.DataFrame(results)

# compute averages (skip "image" column)
avg_row = df.mean(numeric_only=True).round(4)

# restore missing "image" label
avg_row["image"] = "AVERAGE"

"""
avg_row.to_frame()   → Series → DataFrame (column)
.T                   → transpose → row
pd.concat()          → append to table
"""

df = pd.concat([df, avg_row.to_frame().T], ignore_index=True)


# display table
print(df.to_string(index=False))


# save to CSV (for report)
df.to_csv("results_table.csv", index=False)

print("Saved results to results_table.csv")
