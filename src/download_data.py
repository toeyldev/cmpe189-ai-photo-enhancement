# -*- coding: utf-8 -*-

"""
Download dataset and save clean images
"""

# install (run manually in terminal, not here)
# pip install datasets

from datasets import load_dataset
import os
import sys

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

def download_and_save(limit=50):

    dataset = load_dataset("yangtao9009/Flickr2K", split=f"train[:{limit}]")

    print(dataset)
    print(dataset[0])
    #each item is a dictionary-like object: contains the image & metadata

    os.makedirs("data/clean", exist_ok = True) #exist_ok=True = avoid errors if folder exists

    for i in range(len(dataset)):
        try:
            item = dataset[i]
            img = item["image"] #key -> image as value on dict

            #save image into the folder
            img.save(f"data/clean/image_{i}.png")
        except Exception as e:
            print(f"Skipping item {i}: {e}")

    print("Done saving clean images.")


if __name__ == "__main__":
    limit = int(sys.argv[1]) if len(sys.argv) > 1 else 50
    download_and_save(limit) # allow user to specify dataset size from command line
