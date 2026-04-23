#!/usr/bin/env python3
"""
End-to-end AI-Powered Photo Enhancement Pipeline.

Runs the full pipeline from start to finish:
  Step 1: Download clean images from Flickr2K (HuggingFace)
  Step 2: Generate degraded images (downsample + upsample + Gaussian noise)
  Step 3: Run DnCNN denoising inference
  Step 4: Compute PSNR and SSIM metrics
  Step 5: Save enhanced images + results CSV

Usage (run from project root):
    python run_pipeline.py
    python run_pipeline.py --limit 10
    python run_pipeline.py --skip-download --skip-degrade
    python run_pipeline.py --weights model/weights/dncnn_color_blind.pth

Input/output:
    data/clean/*.png    - ground truth images (from HuggingFace)
    data/degraded/*.png - degraded versions (same filenames)
    data/enhanced/*.png - DnCNN denoised outputs
    results_table.csv   - per-image PSNR/SSIM metrics + AVERAGE row
"""

from __future__ import annotations #Enables postponed evaluation of type annotations 
import argparse #module to read command line arguments 
import os #moduule for directory interactions 
import sys #module to import files to src folder
from pathlib import Path #module for handling filesystem paths 


ROOT = Path(__file__).resolve().parent #project root = foder containing this script 
SRC  = ROOT / "src" #Build path to src folder inside project 

sys.path.insert(0, str(SRC)) #Look in src when importing modules 

#Main functino for whole pipeline 
def main() -> None:
    
    os.chdir(ROOT) # change current working directory to project root 

    parser = argparse.ArgumentParser(description="Download, degrade, denoise and evaluate images." ) #Command-line argument parser 

    #argument of how many iimages should be downloaded 
    parser.add_argument(
        "--limit", 
        type=int,
        default=50,
        help="Number of Flickr2K images to download (default: 50).",
    )
    
    #argument for boolean flag to avoid redownlading data every time 
    parser.add_argument(
        "--skip-download",
        action="store_true",
        help="Skip download — reuse existing data/clean/.",
    )

    #argument for boolean flag to avoid repetition of generating degraded image 
    parser.add_argument(
        "--skip-degrade",
        action="store_true",
        help="Skip degradation — reuse existing data/degraded/.",
    )

    #argument for skipping the denoising and evaluation step
    parser.add_argument(
        "--skip-eval",
        action="store_true",
        help="Skip DnCNN inference and evaluation.",
    )

    #argument to let user provide custom model weights file 
    parser.add_argument(
        "--weights",
        type=str,
        default=None,
        help="Path to dncnn_color_blind.pth (auto-download if missing).",
    )
    
    #argument to let user choose ooutput CSV file name/path
    parser.add_argument(
        "--results",
        type=str,
        default="results_table.csv",
        help="Output metrics CSV path (default: results_table.csv).",
    )
    args = parser.parse_args() #read user command line input and store it in args 

    # Step 1: Download clean images from Flickr2K
    if not args.skip_download: #Run this if user did not ask to skip it 
        print("=" * 50)
        print("Step 1: Downloading clean images...")
        print("=" * 50)
        from download_data import download_and_save #imports download_data.py 
        download_and_save(args.limit) #perform download step 

    # Step 2: Generate degraded images
    if not args.skip_degrade: #Run this only if degrade step is not skipped 
        print("=" * 50)
        print("Step 2: Generating degraded images...")
        print("=" * 50)
        from degrade_images import degrade_images #import from degrade_images.py
        degrade_images() #create degraded images from clean ones 

    # Step 3-5: DnCNN inference + evaluation + save CSV
    if not args.skip_eval: #Run this only if evaluation is enabled
        print("=" * 50)
        print("Step 3-5: Running DnCNN inference and evaluation...")
        print("=" * 50)
        from evaluate_model import run_evaluation #import from evaluate_model.py
        run_evaluation(weights_path=args.weights, results_csv=args.results) #run evaluation function 

    print("=" * 50)
    print("Pipeline complete.")
    print("=" * 50)

#If the file is ran directly, call main() 
if __name__ == "__main__":
    main()
