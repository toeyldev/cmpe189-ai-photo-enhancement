# -*- coding: utf-8 -*-

"""
Download dataset and save clean images
"""

# install (run manually in terminal, not here)
# pip install datasets

from datasets import load_dataset
import os

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

def download_and_save(limit=10):

    dataset = load_dataset("yangtao9009/Flickr2K", split=f"train[:{limit}]")

    print(dataset)
    print(dataset[0])
    #each item is a dictionary-like object: contains the image & metadata

    os.makedirs("data/clean", exist_ok = True) #exist_ok=True = avoid errors if folder exists

    for i, item in enumerate(dataset):
        img = item["image"] #key -> image as value on dict

        #save image into the folder
        img.save(f"data/clean/image_{i}.png")

    print("Done saving clean images.")


if __name__ == "__main__":
    download_and_save()
