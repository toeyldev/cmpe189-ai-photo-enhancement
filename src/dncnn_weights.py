"""
Pretrained DnCNN weights manager.

Downloads dncnn_color_blind.pth from the official KAIR v1.0 release
if not already present locally.

Weights are for RGB color blind denoising (channels=3, 20 layers).
Source: https://github.com/cszn/KAIR/releases/download/v1.0/dncnn_color_blind.pth
"""

from __future__ import annotations #This allows the file to be used in other files and projects
import urllib.request #Imports Python tool to download data from the internet 
from pathlib import Path #Imports Python tool to work with file paths and directories

DEFAULT_URL = "https://github.com/cszn/KAIR/releases/download/v1.0/dncnn_color_blind.pth" #Link of pretrained weight files to be downloaded 


def default_weights_path(project_root: Path | None = None) -> Path: #Function to get the default path for the pretrained weight files
    """
    Default location: <repo>/model/weights/dncnn_color_blind.pth
    """
    root = project_root if project_root is not None else Path(__file__).resolve().parent.parent #Find repo root by going up from current file 
    return root / "model" / "weights" / "dncnn_color_blind.pth" #Return the path to the pretrained weight files


"""
    Return path to weights file.
    If the file does not exist, download it from the KAIR release URL.

    Args:
        path: custom path to weights file (default: model/weights/dncnn_color_blind.pth)
        url:  download URL (default: KAIR v1.0 release)

    Returns:
        Path to the weights file
"""
def ensure_weights( #Function to ensure the pretrained weight files are present
    path: Path | str | None = None,
    url: str = DEFAULT_URL,
) -> Path: 

   
    path = Path(path) if path is not None else default_weights_path() #If a custom path is provided, use it. Otherwise, use the default path. 

    if path.exists():
        print(f"Weights already exist at {path}") #If the weights file already exists, print a message and return the path. 
        return path

    path.parent.mkdir(parents=True, exist_ok=True) #Check if parent directory exists. If not, create it. 
    print(f"Downloading DnCNN weights to {path} ...")
    urllib.request.urlretrieve(url, path) #Download the weights file from the KAIR release URL and save it to the path. 
    print("Download complete.")

    return path
