"""
@brief U-Net architecture and double convolution class 
@author Ayush Tripathi (atripathi7783@gmail.com)

"""


# --- Environment Setup --- #
# external modules
import torch
import torch.nn as nn
import torchvision.transforms.functional as tf
import torch.functional as F
import torch.utils.checkpoint as checkpoint
import sys, os #for testing
import tqdm #progress bar
from torchvision import models
from torch.nn.functional import relu
from torch.profiler import profile, record_function, ProfilerActivity
from time import time

# built-in modules
from typing import Any

# internal modules
# from interface import *
from src.arch.interface import *


# --- Constants --- #
NUM_INPUT_CHANNELS = 1
NUM_OUTPUT_CLASSES = 3
DEVICE = "cuda" if torch.cuda.is_available else "cpu"


class UNet(nn.Module, CVModel):
    def __init__(self, hyperparams, **kwargs):
        super().__init__()
        
        # Extract hyperparameters
        self.hyperparams = hyperparams
        self.features = hyperparams.get("features", [64, 128, 256, 512])  # Default feature sizes
        input_channels = hyperparams.get("input_channels", NUM_INPUT_CHANNELS)  # Default input channels
        n_class = hyperparams.get("n_class", NUM_OUTPUT_CLASSES)  # Default output classes
        # Encoder
        self.e11 = nn.Conv2d(input_channels, 64, kernel_size=3, padding=1)
        self.e12 = nn.Conv2d(64, 64, kernel_size=3, padding=1)
        self.pool1 = nn.MaxPool2d(kernel_size=2, stride=2)

        self.e21 = nn.Conv2d(64, 128, kernel_size=3, padding=1)
        self.e22 = nn.Conv2d(128, 128, kernel_size=3, padding=1)
        self.pool2 = nn.MaxPool2d(kernel_size=2, stride=2)

        self.e31 = nn.Conv2d(128, 256, kernel_size=3, padding=1)
        self.e32 = nn.Conv2d(256, 256, kernel_size=3, padding=1)
        self.pool3 = nn.MaxPool2d(kernel_size=2, stride=2)

        self.e41 = nn.Conv2d(256, 512, kernel_size=3, padding=1)
        self.e42 = nn.Conv2d(512, 512, kernel_size=3, padding=1)
        self.pool4 = nn.MaxPool2d(kernel_size=2, stride=2)

        self.e51 = nn.Conv2d(512, 1024, kernel_size=3, padding=1)
        self.e52 = nn.Conv2d(1024, 1024, kernel_size=3, padding=1)

        # Decoder
        self.upconv1 = nn.ConvTranspose2d(1024, 512, kernel_size=2, stride=2)
        self.d11 = nn.Conv2d(1024, 512, kernel_size=3, padding=1)
        self.d12 = nn.Conv2d(512, 512, kernel_size=3, padding=1)

        self.upconv2 = nn.ConvTranspose2d(512, 256, kernel_size=2, stride=2)
        self.d21 = nn.Conv2d(512, 256, kernel_size=3, padding=1)
        self.d22 = nn.Conv2d(256, 256, kernel_size=3, padding=1)

        self.upconv3 = nn.ConvTranspose2d(256, 128, kernel_size=2, stride=2)
        self.d31 = nn.Conv2d(256, 128, kernel_size=3, padding=1)
        self.d32 = nn.Conv2d(128, 128, kernel_size=3, padding=1)

        self.upconv4 = nn.ConvTranspose2d(128, 64, kernel_size=2, stride=2)
        self.d41 = nn.Conv2d(128, 64, kernel_size=3, padding=1)
        self.d42 = nn.Conv2d(64, 64, kernel_size=3, padding=1)

        self.linear = nn.Linear(in_features=1, out_features=3) 
        self.fconv  = nn.Conv2d(3, 3, kernel_size=224, stride=224)


        # Output layer
        self.outconv = nn.Conv2d(64, n_class, kernel_size=1)

    def forward(self, x):
        # Encoder
        start_time = time()
        xe11 = relu(self.e11(x))
        xe12 = relu(self.e12(xe11))
        xp1 = self.pool1(xe12)

        xe21 = relu(self.e21(xp1))
        xe22 = relu(self.e22(xe21))
        xp2 = self.pool2(xe22)

        xe31 = relu(self.e31(xp2))
        xe32 = relu(self.e32(xe31))
        xp3 = self.pool3(xe32)

        xe41 = relu(self.e41(xp3))
        xe42 = relu(self.e42(xe41))
        xp4 = self.pool4(xe42)

        xe51 = relu(self.e51(xp4))
        xe52 = relu(self.e52(xe51))

        # Decoder
        xu1 = self.upconv1(xe52)
        xu11 = torch.cat([xu1, xe42], dim=1)
        xd11 = relu(self.d11(xu11))
        xd12 = relu(self.d12(xd11))

        xu2 = self.upconv2(xd12)
        xu22 = torch.cat([xu2, xe32], dim=1)
        xd21 = relu(self.d21(xu22))
        xd22 = relu(self.d22(xd21))

        xu3 = self.upconv3(xd22)
        xu33 = torch.cat([xu3, xe22], dim=1)
        xd31 = relu(self.d31(xu33))
        xd32 = relu(self.d32(xd31))

        xu4 = self.upconv4(xd32)
        xu44 = torch.cat([xu4, xe12], dim=1)
        xd41 = relu(self.d41(xu44))
        xd42 = relu(self.d42(xd41))

        # Output layer
        out = self.outconv(xd42)
        start_time = time()
        out = self.fconv(out)
        out = torch.flatten(out, 1)  
        end_time = time() - start_time
        print(f'time taken: {end_time}')
        shape(out)
        # out = out.view(-1, 32*3)
        # out = self.linear(out)
        return out
    def train_model(self, train_loader: torch.utils.data.DataLoader,
              val_loader: torch.utils.data.DataLoader, **kwargs):
        trainer(self.hyperparams, self, train_loader, val_loader)
                
    def validate_model(self, loader: torch.utils.data.DataLoader):
        validation(self.hyperparams, self, loader )
    
    def test_model(self, loader: torch.utils.data.DataLoader, loss_fn: Any, **kwargs):
        pass

    def predict(self, loader: torch.utils.data.DataLoader, **kwargs):
        pass

    def save(self, path: Path | str, **kwargs) -> torch.Tensor:
        pass

    def load(path: Path | str, model_class, **kwargs) -> None:
        pass    
    def interpret(test_input, **kwargs):
        return super().interpret_model(**kwargs)


if __name__ == "__main__":
    hyperparams = {"features": [64, 128, 256, 512], "input_channels": 3, "n_class": 2}
    model = UNet(hyperparams)

    # Check if GPU is available
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(device)
    model = model.to(device)  # Move model to GPU if available

    # Example input tensor
    input_tensor = torch.rand(1, 3, 224, 224).to(device)  # Move input tensor to GPU

    with profile(activities=[ProfilerActivity.CPU, ProfilerActivity.CUDA], record_shapes=True) as prof:
        with torch.no_grad():
            model(input_tensor)

    print(prof.key_averages().table(sort_by="cuda_time_total" if torch.cuda.is_available() else "cpu_time_total"))


def shape(x):
    print(f"SHAPE: {x.shape}")