"""
@brief Basic CNN as a baseline for experiments.
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
from src.arch.interface import *


# --- Model --- #
class CNN(nn.Module, CVModel):
    # build model
    def __init__(self, hyperparams, **kwargs):
        # build up parents
        super(CNN, self).__init__()
        self.hyperparams = hyperparams
        
        # convolution & pooling layers
        self.conv = nn.ModuleList([
            nn.Conv2d(NUM_INPUT_CHANNELS, 8, 7),
            nn.Conv2d(8, 16, 7),
            nn.Conv2d(16, 32, 7)
        ])
        self.pool = nn.MaxPool2d(2, 2)
        
        # activation fn
        self.conv_act = nn.ReLU()
        self.fc_act = nn.ReLU()
        
        # classifier fn
        self.class_fn = nn.Softmax()
        
        # linear classifier layers
        self.connect_n = 32 * 22 * 22
        self.fc = nn.ModuleList([
            nn.Linear(self.connect_n, 256),
            nn.Linear(256, 64),
            nn.Linear(64, NUM_OUTPUT_CLASSES)
        ])

    def forward(self, x):
        # run through the convolutions & pools
        for i in range(len(self.conv)):
            x = self.pool(self.conv_act(self.conv[i](x)))
        
        # reshape as vector
        x = x.view(-1, self.connect_n)
        
        # linear classification
        for i in range(len(self.fc) - 1):
            x = self.fc_act(self.fc[i](x))
        x = self.fc[-1](x)
        
        # export forward pass
        return x
    
    # override methods
    def train_model(self, train_loader: torch.utils.data.DataLoader,
              val_loader: torch.utils.data.DataLoader, **kwargs):
        """Wraps the trainer method for the CNN training.

        Args:
            train_loader (torch.utils.data.DataLoader): dataloader for train set
            val_loader (torch.utils.data.DataLoader): dataloader for val set
        """
        
        # wrap trainer call
        return trainer(self.hyperparams, self, train_loader, val_loader)
    
    def validate_model(self, loader: torch.utils.data.DataLoader) -> dict[str, Any]:
        """Validation on the CNN.

        Args:
            loader (torch.utils.data.DataLoader): loader for the validation set
        """
        
        # wrap validator call
        return validation(self.hyperparams, self, loader)
    
    def test_model(self, loader: torch.utils.data.DataLoader, **kwargs) -> dict[str, Any]:
        """Wrap the call to the tester function.

        Args:
            loader (torch.utils.data.DataLoader): test data loader

        Returns:
            dict[str, Any]: metrics
        """
        
        # wrap tester call
        return tester(self.hyperparams, self, loader)
    
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
        return cnn_interpreter(self, test_input)
    
    