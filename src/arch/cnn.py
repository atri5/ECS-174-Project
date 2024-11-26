"""
@brief Interface for the model architectures; all models used should follow the 
    same structure to ensure compatability.
@author Arjun Ashok (arjun3.ashok@gmail.com)
"""


# --- Environment Setup --- #
# external modules
import torch
import torch.nn as nn
import torch.functional as F
import torch.optim as optim
import numpy as np

# built-in modules
## none for now...

# internal modules
from interface import *


# --- Model --- #
class CNN(nn.Module, CVModel):
    # build model
    def __init__(self, **kwargs):
        # build up parents
        super(CNN, self).__init__()
        
        # convolution & pooling layers
        self.conv = nn.ModuleList([
            nn.Conv2d(NUM_INPUT_CHANNELS, 8, 7),
            nn.Conv2d(8, 16, 7),
            nn.Conv2d(16, 32, 7)
        ])
        self.pool = nn.MaxPool2d(2, 2)
        
        # activation fn
        self.conv_act = nn.GELU()
        self.fc_act = nn.ReLU()
        
        # classifier fn
        self.class_fn = nn.Softmax()
        
        # linear classifier layers
        self.fc = nn.ModuleList([
            nn.Linear(32 * 7 * 7, 256),
            nn.Linear(256, 64),
            nn.Linear(64, NUM_OUTPUT_CLASSES),
        ])

    def forward(self, x):
        # run through the convolutions & pools
        for i in range(len(self.conv)):
            x = self.pool(self.conv_act(self.conv[i](x)))
        
        # reshape
        x = x.view(-1, 32 * 7 * 7)
        
        # linear classification
        for i in range(len(self.fc) - 1):
            x = self.fc_act(self.fc[i](x))
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

