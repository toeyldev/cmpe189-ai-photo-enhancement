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

import cv2
import numpy as np
import os


def degrade_images():

    os.makedirs("data/degraded", exist_ok= True)

    #return: ["image_0.png", "image_1.png", ...]
    #getFilename (Ex:image_0.png)
    for fileName in os.listdir("data/clean"):

        #"data/clean" + "image_0.png"
        #builds: data/clean/image_0.png
        path= os.path.join("data/clean", fileName)

        img= cv2.imread(path)

        if img is None:
            print(f"Could not read {path}")
            continue

        #We want: same size image, but lower quality

        #dsize → exact output size (width, height)
        #fx, fy → scale factors

        #dsize = None, OpenCV ignores dsize, Uses fx and fy instead
        downSized = cv2.resize(img, None, fx= 0.2, fy= 0.2,interpolation = cv2.INTER_AREA)

        #Resize back to the original width and height
        #img.shape[0] = height
        #img.shape[1] = width

        upSized = cv2.resize(downSized, (img.shape[1], img.shape[0]), interpolation=cv2.INTER_CUBIC)

        #noise = the disturbance

        #np.random.normal(...): for centered, symmetric → rule applies
        #generate noise: mean = 0, std = 50 (controls noise strength), same shape as image
        noise = np.random.normal(0, 50, img.shape)

        """
        Original pixel: [100, 150, 200]
        Noise:          [+10,  -20,  +5]
        --------------------------------
        Result:         [110, 130, 205]
        """

        #0–255 = value of each pixel
        # astype(np.uint8): 123.7 → 123

        noisyImage = np.clip(upSized + noise, 0, 255).astype(np.uint8)

        savePath = os.path.join("data/degraded", fileName)

        cv2.imwrite(savePath, noisyImage)

    print("Done creating degraded images.")


if __name__ == "__main__":
    degrade_images()
