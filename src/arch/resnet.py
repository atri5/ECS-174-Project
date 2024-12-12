"""
@brief ResNet from the classic model architecture, but with functionality for 
    fine-tuning.
@author Arjun Ashok (arjun3.ashok@gmail.com)
"""


# --- Environment Setup --- #
# external modules
import torch
import torch.nn as nn
import torch.functional as F
import torch.optim as optim
import torchvision.models as models

# built-in modules
## none for now...

# internal modules
from src.arch.interface import *


# --- Model --- #
class ResNet(nn.Module, CVModel):
    # build model
    def __init__(self, hyperparams, **kwargs):
        # build up parents
        super(ResNet, self).__init__()
        self.hyperparams = hyperparams
        
        # load the non-pre-trained backbone
        self.backbone = models.resnet18(weights=None)
        
        # modify the input layer
        self.backbone.conv1 = nn.Conv2d(
            in_channels=NUM_INPUT_CHANNELS,
            out_channels=self.backbone.conv1.out_channels,
            kernel_size=self.backbone.conv1.kernel_size,
            stride=self.backbone.conv1.stride,
            padding=self.backbone.conv1.padding,
            bias=self.backbone.conv1.bias
        )
        
        # modify the classification head
        self.backbone.fc = nn.Linear(
            self.backbone.fc.in_features, NUM_OUTPUT_CLASSES
        )
        self.class_fn = nn.Softmax()
        
    # override the pytorch methods
    def train(self):
        """Wrap the pytorch method to allow the interface to work.
        """
        
        # resnet to train mode
        self.backbone.train()
    
    def eval(self):
        """Wrap the pytorch method to allow the interface to work.
        """
        
        # resnet to eval mode
        self.backbone.eval()

    def forward(self, x):
        # wrap backbone output
        return self.backbone(x)
    
    # override methods
    def train_model(self, train_loader: torch.utils.data.DataLoader,
              val_loader: torch.utils.data.DataLoader, **kwargs):
        """Wraps the trainer method for the ResNet training.

        Args:
            train_loader (torch.utils.data.DataLoader): dataloader for train set
            val_loader (torch.utils.data.DataLoader): dataloader for val set
        """
        
        # wrap trainer call
        trainer(self.hyperparams, self, train_loader, val_loader)
    
    def validate_model(self, loader: torch.utils.data.DataLoader) -> dict[str, Any]:
        """Validation on the ResNet.

        Args:
            loader (torch.utils.data.DataLoader): loader for the validation set
        """
        
        # wrap validator call
        validation(self.hyperparams, self, loader)
    
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
        saver(self.hyperparams, self, path)
    
    def interpret(test_input: Any, **kwargs) -> None:
        pass
    
    