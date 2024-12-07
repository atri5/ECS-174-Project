"""
@brief Implements the modified CNN architecture but using spline-linear layers
    for a denser, more interpretable classifier head.
@author Arjun Ashok (arjun3.ashok@gmail.com)
"""


# --- Environment Setup --- #
# external modules
import torch
import torch.nn as nn
import torch.functional as F
import torch.optim as optim
import numpy as np
from deepkan import SplineLinearLayer

# built-in modules
## none for now...

# internal modules
from interface import *


# --- Model --- #
class CKAN(nn.Module, CVModel):
    # build model
    def __init__(self, **kwargs):
        # build up parents
        super(CKAN, self).__init__()
        
        # convolution & pooling layers
        self.conv = nn.ModuleList([
            nn.Conv2d(NUM_INPUT_CHANNELS, 8, 7),
            nn.Conv2d(8, 16, 7),
            nn.Conv2d(16, 32, 7)
        ])
        self.conv_norm = nn.ModuleList([
            nn.LayerNorm([8, *IMAGE_DIMS]),
            nn.LayerNorm([16, *IMAGE_DIMS]),
            nn.LayerNorm([32, *IMAGE_DIMS])
        ])
        self.pool = nn.MaxPool2d(2, 2)
        
        # activation fn
        self.conv_act = nn.GELU()
        self.fc_act = nn.ReLU()
        
        # classifier fn
        self.class_fn = nn.Softmax()
        
        # linear classifier layers
        self.fc = nn.ModuleList([
            nn.Linear(32 * 7 * 7, 256), # convert to linear first to make things efficient
            SplineLinearLayer(256, 32),
            SplineLinearLayer(32, NUM_OUTPUT_CLASSES)
        ])
        self.fc_norm = nn.ModuleList([
            nn.LayerNorm(256),
            nn.LayerNorm(32)
        ])

    def forward(self, x):
        # run through the convolutions & pools
        for i in range(len(self.conv)):
            # convolve and normalize
            x = self.conv_norm[i](self.conv[i](x))
            
            # activate and pool
            x = self.pool(self.conv_act(x))
        
        # reshape
        x = x.view(-1, 32 * 7 * 7)
        
        # linear classification & normalization
        for i in range(len(self.fc) - 1):
            x = self.fc_norm[i](self.fc[i](x))
            x = self.fc_act(x)
        x = self.fc[-1](x)
        
        # export forward pass
        return x
    
    # override methods
    def train(self, loader: torch.utils.data.DataLoader,
              optimizer: torch.optim.Optimizer, loss_fn: torch.nn.Loss,
              **kwargs):
        """Trains the CNN.

        Args:
            loader (torch.utils.data.DataLoader): _description_
            optimizer (torch.optim.Optimizer): _description_
            loss_fn (torch.nn.Loss): _description_
        """
        
        pass
    
    def validate(self, loader: torch.utils.data.DataLoader, loss_fn: Any, **kwargs):
        pass
    
    def test(self, loader: torch.utils.data.DataLoader, loss_fn: Any, **kwargs) -> torch.Tensor:
        pass
    
    def predict(self, x: torch.Tensor, **kwargs) -> torch.Tensor:
        """Predict wrapper method for generating the prediction we want.

        Args:
            x (torch.Tensor): input tensor

        Returns:
            torch.Tensor: model output
        """
        
        # wrap the forward pass w/ softmax
        return self.class_fn(self.forward(x))
    
    def save(path: Path | str, **kwargs) -> None:
        """Saves the model to a specified weights directory for easy caching.

        Args:
            path (Path | str): path to save to, relative or absolute.
        """
        pass
