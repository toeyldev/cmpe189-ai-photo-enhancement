"""
Evaluate DnCNN model using PSNR and SSIM.
Also saves enhanced images for visualization.

Usage (from project root):
    python src/evaluate_model.py
    python src/evaluate_model.py --weights model/weights/dncnn_color_blind.pth
"""

import os #Imports Python tool to create directories and list files
import sys #Used to help Python find files in the src directory
import argparse #Lets user pass command line arguments like --weights
from pathlib import Path #Used for file paths and directories

import cv2 #Imports OpenCV library to read, resize and save images 
import numpy as np #Import numpy library for array/image math
import pandas as pd #Imports Pandas library to work with data frames
import torch #Imports PyTorch library to build and run neural networks

from skimage.metrics import peak_signal_noise_ratio as psnr #Imports PSNR metric from scikit-image
from skimage.metrics import structural_similarity as ssim #Imports SSIM metric from scikit-image

# allow imports from src/
SRC = Path(__file__).resolve().parent #Find the src directory by going up from the current file
sys.path.insert(0, str(SRC)) #Insert src directory at the front of the path so Python can find the files in the src directory

from dncnn_pytorch import DnCNN #Imports DnCNN model from dncnn_pytorch.py
from dncnn_weights import ensure_weights #Imports ensure_weights function from dncnn_weights.py


# --- Compute metrics ---
#Function to compute PSNR and SSIM between two images
def compute_metrics(clean_img, compare_img): 
    """
    Compute PSNR and SSIM between two images.

    clean_img    → ground truth RGB uint8 NumPy array
    compare_img  → degraded OR denoised RGB uint8 NumPy array

    PSNR: measures pixel-level difference (higher is better)
    SSIM: measures structural similarity (range 0-1, higher is better)
    """
    psnr_val = psnr(clean_img, compare_img)
    ssim_val = ssim(clean_img, compare_img, channel_axis=2)
    return psnr_val, ssim_val


# --- Load model ---
#Function to load the pretrained DnCNN model
def load_model(weights_path=None): 
    """
    Load pretrained DnCNN model onto GPU if available, else CPU.

    Architecture: 20 layers, channels=3 (RGB), features=64, no BatchNorm
    Confirmed from dncnn_color_blind.pth weight file inspection.
    """
    # detect GPU
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu") #Use GPU if available, else use CPU
    print(f"Using device: {device}") #Print the device being used

    weights_path = ensure_weights(weights_path) # download weights if missing

    # load state dict
    state_dict = torch.load(weights_path, map_location=device) #Load saved model weights from the file into memory 
    if "params" in state_dict: #If the weights are nested under a "params" key, unwrap them
        state_dict = state_dict["params"]

    # build model matching exact weight file architecture
    model = DnCNN(channels=3, num_of_layers=20, features=64) #Build model with correct architecture
    model.load_state_dict(state_dict, strict=True) #load weights into the model 
    model = model.to(device) #Move model to the device (GPU or CPU) 
    model.eval() #Set model to evaluation mode

    print(f"DnCNN model loaded successfully on {device}")
    return model, device


# --- Run evaluation ---
#Function to run the evaluation
def run_evaluation(weights_path=None, results_csv="results_table.csv"): 
    """
    Run DnCNN inference on all images in data/degraded/.
    Saves enhanced images to data/enhanced/.
    Computes PSNR and SSIM for each image.
    Writes results to results_table.csv.
    """
    clean_dir    = "data/clean" #Original images
    degraded_dir = "data/degraded" #Nosiy images 
    enhanced_dir = "data/enhanced" #Restored model outputs 

    os.makedirs(enhanced_dir, exist_ok=True) #Create directory called enhanced if it doesn't exist 

    # load model
    model, device = load_model(weights_path)

    results = [] #Empty list to store results 

    for fileName in sorted(os.listdir(clean_dir)): #Loop through all images in clean directory 

        # --- Load images ---
        clean_path    = os.path.join(clean_dir, fileName) #Join the clean directory with the image name to get the path to the image 
        degraded_path = os.path.join(degraded_dir, fileName) #Join the degraded directory with the image name to get the path to the image 

        clean    = cv2.imread(clean_path) #Read image from clean directory 
        degraded = cv2.imread(degraded_path) #Read image from degraded directory 

        if clean is None or degraded is None: #If the image is not found, print an error message and continue to the next image if avaliable. 
            print(f"Skipping {fileName} - could not read file")
            continue

        # --- Convert BGR → RGB ---
        # cv2.cvtColor() always returns a 3D NumPy array
        clean_rgb    = cv2.cvtColor(clean, cv2.COLOR_BGR2RGB)
        degraded_rgb = cv2.cvtColor(degraded, cv2.COLOR_BGR2RGB)

        # --- Run DnCNN inference ---
        # min-max normalization: [0, 255] → [0.0, 1.0] (Needed for neural network input)
        img_norm = degraded_rgb / 255.0

        # NumPy (H, W, C) → PyTorch tensor (1, C, H, W)
        img_tensor = torch.from_numpy(img_norm).permute(2, 0, 1).unsqueeze(0).float()

        # move tensor to (GPU or CPU)
        img_tensor = img_tensor.to(device)

        # inference (no gradient tracking needed)
        with torch.no_grad(): 
            output = model(img_tensor)

        # move output back to CPU and convert to NumPy
        # (1, C, H, W) → (H, W, C) → uint8
        denoised = output.squeeze().cpu().permute(1, 2, 0).numpy()
        denoised = np.clip(denoised * 255, 0, 255).astype(np.uint8)

        # --- Save enhanced image ---
        save_path = os.path.join(enhanced_dir, fileName)
        # OpenCV uses BGR, so convert RGB → BGR before saving
        denoised_bgr = cv2.cvtColor(denoised, cv2.COLOR_RGB2BGR)
        cv2.imwrite(save_path, denoised_bgr)

        # --- Compute metrics ---
        # baseline: how bad is degraded vs clean
        psnr_degraded, ssim_degraded = compute_metrics(clean_rgb, degraded_rgb)

        # model output: how good is denoised vs clean
        psnr_denoised, ssim_denoised = compute_metrics(clean_rgb, denoised)

        results.append({ #Append the results to the results list
            "image"         : fileName,
            "PSNR_degraded" : round(psnr_degraded, 2),
            "PSNR_denoised" : round(psnr_denoised, 2),
            "SSIM_degraded" : round(ssim_degraded, 4),
            "SSIM_denoised" : round(ssim_denoised, 4),
        })

        print(f"{fileName}  PSNR: {psnr_degraded:.2f} → {psnr_denoised:.2f}  SSIM: {ssim_degraded:.4f} → {ssim_denoised:.4f}")

    # --- Build results table ---
    df = pd.DataFrame(results) #Create a data frame from the results list

    # compute average row
    avg_row = df.mean(numeric_only=True).round(4) #Compute the average of the results
    avg_row["image"] = "AVERAGE" #Add a row for the average results
    df = pd.concat([df, avg_row.to_frame().T], ignore_index=True) #Concatenate the average row to the data frame

    print("\n" + df.to_string(index=False)) #Print the data frame to the console

    # save to CSV
    df.to_csv(results_csv, index=False) #Save the data frame to a CSV file
    print(f"\nSaved results to {results_csv}") #Print a message to the console

    return df #Return the data frame


# --- Direct execution ---
if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Evaluate DnCNN on degraded images.") #Create an argument parser for the command line arguments
    parser.add_argument("--weights", type=str, default=None, #Add an argument for the weights path
                        help="Path to dncnn_color_blind.pth (auto-download if missing)")
    parser.add_argument("--results", type=str, default="results_table.csv", #Add an argument for the results path
                        help="Output CSV path")
    args = parser.parse_args() #Parse the command line arguments

    run_evaluation(weights_path=args.weights, results_csv=args.results) #Run the evaluation
    #This will run the evaluation and save the results to a CSV file