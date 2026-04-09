"""
DnCNN model architecture for RGB color image denoising.

Architecture confirmed from weight file inspection (dncnn_color_blind.pth):
- 20 convolutional layers (model.0 to model.38, step 2)
- 3 input/output channels (RGB)
- 64 feature maps per layer
- ReLU activations only (no BatchNorm)
- bias=True on all layers

Pretrained weights source: cszn/KAIR GitHub releases v1.0
"""

import torch.nn as nn


class DnCNN(nn.Module):
    def __init__(self, channels=3, num_of_layers=20, features=64):
        super(DnCNN, self).__init__()

        layers = []

        # first layer: channels → features
        layers.append(nn.Conv2d(channels, features, 3, padding=1, bias=True))
        layers.append(nn.ReLU(inplace=True))

        # middle layers: features → features (no BatchNorm)
        for _ in range(num_of_layers - 2):
            layers.append(nn.Conv2d(features, features, 3, padding=1, bias=True))
            layers.append(nn.ReLU(inplace=True))

        # last layer: features → channels
        layers.append(nn.Conv2d(features, channels, 3, padding=1, bias=True))

        # name must be "model" to match weight keys: model.0, model.2, etc.
        self.model = nn.Sequential(*layers)

    def forward(self, x):
        noise = self.model(x)
        return x - noise  # residual learning: output = input - predicted noise
