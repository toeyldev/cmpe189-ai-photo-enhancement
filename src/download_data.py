# -*- coding: utf-8 -*-

"""
Download dataset and save clean images
"""

# install (run manually in terminal, not here)
# pip install datasets

from datasets import load_dataset #Imports HuggingFace tool to load datasets from the internet
import os #Imports Python tool to create directories and list files
import sys #Imports Python tool to work with command line arguments

#dataset is a Dataset object
#Each item inside the dataset is a dictionary

"""
Ex:
dataset
   ↓
[ item, item, item, ... ]
     ↓
   {
     "image": <PIL Image>
   }
"""

#Function to download ans save dataset. Will download 50 images by default 
def download_and_save(limit=50):

    dataset = load_dataset("yangtao9009/Flickr2K", split=f"train[:{limit}]") #Download Flicker2K dataset. Loads first "limit" images from the trianing split.

    print(dataset) #Print the dataset to the console
    print(dataset[0]) #Print the first item in the dataset to the console
    #each item is a dictionary-like object: contains the image & metadata

    os.makedirs("data/clean", exist_ok = True) #exist_ok=True = avoid errors if folder exists

    for i in range(len(dataset)): #Loop through the dataset and save each image to the clean folder
        try:
            item = dataset[i] #Get the i-th item in the dataset
            img = item["image"] #key -> image as value on dict

            #save image into the folder
            img.save(f"data/clean/image_{i}.png") #Save image into clean folder
        except Exception as e:
            print(f"Skipping item {i}: {e}") #If an error occurs, print the error message and continue to the next image if avaliable. 

    print("Done saving clean images.")


if __name__ == "__main__": #If the script is run directly, call the download_and_save function
    limit = int(sys.argv[1]) if len(sys.argv) > 1 else 50 #Allow user to specify dataset size from command line
    download_and_save(limit) # allow user to specify dataset size from command line
