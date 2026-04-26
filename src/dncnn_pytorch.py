"""
DnCNN model architecture for RGB color image denoising.
Process:
    - Take noisy RGB image  
    - Predict the noise in the image 
    - Subtract noise 
    - Return a cleaner image 

Architecture confirmed from weight file inspection (dncnn_color_blind.pth):
- 20 convolutional layers (model.0 to model.38, step 2)
- 3 input/output channels (RGB)
- 64 feature maps per layer
- ReLU activations only (no BatchNorm)
- bias=True on all layers

Pretrained weights source: cszn/KAIR GitHub releases v1.0
"""

import torch.nn as nn #Import PyTorch's neural network tools so file can build layers like convolutions and ReLu activations


class DnCNN(nn.Module): #Define a class for the DnCNN model

    #__init__ is a constructor that initializes the model
    #self: The actual object being created 
    #channels = 3: The number of color channels in the image (RGB)
    #num_of_layers = 20: How deep the neural network is 
    #features = 64: how many features map (filters) each layer uses
    def __init__(self, channels=3, num_of_layers=20, features=64):

        super(DnCNN, self).__init__() #Call the constructor of the parent class (nn.Module)

        layers = [] #Empty list that will add layers and combine them into a full model later on

        #First layer: channels → features 
        layers.append(nn.Conv2d(channels, features, 3, padding=1, bias=True)) #Take RGB image and turn it into 64 learned feature maps 
        layers.append(nn.ReLU(inplace=True)) #Apply ReLU activation function to the feature map

        #Middle layers: features → features (no BatchNorm)
        #Keeps same shape while learnign more complex image patterns
        for _ in range(num_of_layers - 2): #Loop through the middle layers
            layers.append(nn.Conv2d(features, features, 3, padding=1, bias=True)) #Take 64 feature maps in and 64 feature maps out. 
            layers.append(nn.ReLU(inplace=True)) #Apply ReLU activation function to the feature map

        #last layer: features → channels
        #Convert hidden representation back into the original image channel count.
        layers.append(nn.Conv2d(features, channels, 3, padding=1, bias=True))

        # name must be "model" to match weight keys: model.0, model.2, etc.
        self.model = nn.Sequential(*layers) #Combine all layers in the list into a sequential neural network

    def forward(self, x): #Define what hapepsn when input image goes through the model 
        noise = self.model(x) #Send image through all the colvolution and ReLU layers 
        return x - noise  # residual learning: output = input - predicted noise