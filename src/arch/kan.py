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
from src.arch.interface import *


# --- Model --- #
class CKAN(nn.Module, CVModel):
    # build model
    def __init__(self,hyperparams, **kwargs):
        # build up parents
        super(CKAN, self).__init__()
        self.hyperparams = hyperparams
        
        # convolution & pooling layers
        self.conv = nn.ModuleList([
            nn.Conv2d(NUM_INPUT_CHANNELS, 8, 7),
            nn.Conv2d(8, 16, 7),
            nn.Conv2d(16, 32, 7)
        ])
        self.conv_norm = nn.ModuleList([
            nn.LayerNorm([8, 218, 218]),    # 224 - 6
            nn.LayerNorm([16, 103, 103]),   # 218 / 2 - 6
            nn.LayerNorm([32, 45, 45])      # 103 / 2 - 6 floored
        ])
        self.pool = nn.MaxPool2d(2, 2)
        
        # activation fn
        self.conv_act = nn.GELU()
        self.fc_act = nn.ReLU()
        
        # classifier fn
        self.class_fn = nn.Softmax()
        
        # linear classifier layers
        self.connect_n = 32 * 22 * 22
        self.fc = nn.ModuleList([
            nn.Linear(self.connect_n, 256), # convert to linear first to make things efficient
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
        x = x.view(-1, self.connect_n)
        
        # linear classification & normalization
        for i in range(len(self.fc) - 1):
            x = self.fc_norm[i](self.fc[i](x))
            x = self.fc_act(x)
        x = self.fc[-1](x)
        
        # export forward pass
        return x
    
    # override methods
    def train_model(self, train_loader: torch.utils.data.DataLoader,
              val_loader: torch.utils.data.DataLoader, **kwargs):
        """Wraps the trainer method for the Conv KAN training.

        Args:
            train_loader (torch.utils.data.DataLoader): dataloader for train set
            val_loader (torch.utils.data.DataLoader): dataloader for val set
        """
        
        # wrap trainer call
        return trainer(self.hyperparams, self, train_loader, val_loader, device="cpu")
    
    def validate_model(self, loader: torch.utils.data.DataLoader) -> dict[str, Any]:
        """Validation on the Conv KAN.

        Args:
            loader (torch.utils.data.DataLoader): loader for the validation set
        """
        
        # wrap validator call
        return validation(self.hyperparams, self, loader, device="cpu")
    
    def test_model(self, loader: torch.utils.data.DataLoader, **kwargs) -> dict[str, Any]:
        """Wrap the call to the tester function.

        Args:
            loader (torch.utils.data.DataLoader): test data loader

        Returns:
            dict[str, Any]: metrics
        """
        
        # wrap tester call
        return tester(self.hyperparams, self, loader, device="cpu")
    
    def predict(self, x: torch.Tensor, **kwargs) -> torch.Tensor:
        """Predict wrapper method for generating the prediction we want.

        Args:
            x (torch.Tensor): input tensor

        Returns:
            torch.Tensor: model output
        """
        
        # wrap the forward pass w/ softmax
        return self.class_fn(self.forward(x))
    
    def save(self, path: Path | str, **kwargs) -> None:
        """Saves the model to a specified weights directory for easy caching.

        Args:
            path (Path | str): path to save to, relative or absolute.
        """
        
        # wraps saver method
        return saver(self.hyperparams, self, path)
    
    def interpret(self, test_input: Any, **kwargs) -> None:
        """Inteprets the model by wrapping the CNN interpretation via Grad-Cam.

        Args:
            test_input (Any): test-loader.
        """
        
        # wrap call
        return cnn_interpreter(self, test_input, target_layer=self.conv[-1], device="cpu")

