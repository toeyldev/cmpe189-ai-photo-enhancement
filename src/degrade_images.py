"""
clean image
   ↓
downsample → lose detail
   ↓
upsample → blur
   ↓
add noise → realistic degradation
   ↓
save → degraded image
"""

"""
cv2: OpenCV (Open Source Computer Vision Library) in Python

It’s a library used for:
* image processing
* computer vision
* video processing
"""

import cv2 #Import OpenCV library to read, resize and save images
import numpy as np #Import NumPy library to generate noise
import os #Import os library to create directories and list files

#Function to degrade images
def degrade_images():

    os.makedirs("data/degraded", exist_ok= True) #Create degraded directory if it doesn't exist

    #return: ["image_0.png", "image_1.png", ...]
    #getFilename (Ex:image_0.png)
    for fileName in os.listdir("data/clean"): #Loop through the files in the clean directory

        #"data/clean" + "image_0.png"
        #builds: data/clean/image_0.png
        path = os.path.join("data/clean", fileName) #Build the path for the image

        img = cv2.imread(path) #Load the image into memory

        if img is None: #If OpenCV could not read the image, print an error message and continue to the next image if avaliable. 
            print(f"Could not read {path}")
            continue

        #We want: same size image, but lower quality

        #dsize → exact output size (width, height)
        #fx, fy → scale factors
        #dsize = None, OpenCV ignores dsize, Uses fx and fy instead
        downSized = cv2.resize(img, None, fx= 0.2, fy= 0.2,interpolation = cv2.INTER_AREA) #Shrink image to 20% of its original width and height using INTER_AREA interpolation
        
        #img.shape[0] = height
        #img.shape[1] = width
        upSized = cv2.resize(downSized, (img.shape[1], img.shape[0]), interpolation=cv2.INTER_CUBIC) #Resize image to original size. Image will now look blurry because of detail lost during shrinking

        #noise = the disturbance

        #np.random.normal(...): for centered, symmetric → rule applies
        #generate noise: mean = 0, std = 50 (controls noise strength), same shape as image
        noise = np.random.normal(0, 50, img.shape) #Random values generated for every pixel and channel. 

        """
        Original pixel: [100, 150, 200]
        Noise:          [+10,  -20,  +5]
        --------------------------------
        Result:         [110, 130, 205]
        """

        #0–255 = value of each pixel
        # astype(np.uint8): 123.7 → 123

        noisyImage = np.clip(upSized + noise, 0, 255).astype(np.uint8) #Add noise to blury image 

        savePath = os.path.join("data/degraded", fileName) #Output path for degraded folder

        cv2.imwrite(savePath, noisyImage) #Save the noisy image to the degraded folder

    print("Done creating degraded images.")


if __name__ == "__main__": #If the script is run directly, call the degrade_images function
    degrade_images()
