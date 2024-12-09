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
import sys, os #for testing
import tqdm #progress bar

# built-in modules
from typing import Any

# internal modules
# from interface import *
from src.arch.interface import *


# --- Constants --- #
NUM_INPUT_CHANNELS = 1
NUM_OUTPUT_CLASSES = 3
DEVICE = "cuda" if torch.cuda.is_available else "cpu"

# --- Double Convolution --- #
class DoubleConv(nn.Module):
    def __init__(self, in_channels, out_channels):
        super(DoubleConv, self).__init__()
        self.conv = nn.Sequential(
            nn.Conv2d(in_channels, out_channels, 3, 1, 1, bias = False), #5th parameter has image padding for simplicity, will change later
            nn.BatchNorm2d(out_channels),
            nn.ReLU(inplace = True),
            nn.Conv2d(out_channels, out_channels, 3, 1, 1, bias = False), #5th parameter has image padding for simplicity, will change later
            nn.BatchNorm2d(out_channels),
            nn.ReLU(inplace = True)    
        )
    def forward(self, x):
        return self.conv(x)





# --- Model --- #
class UNet(nn.Module, CVModel):
    #building model
    def __init__(self, hyperparams, **kwargs):
        super(UNet, self).__init__()

        
        self.features = [64,128,256,512]
        
        #set up model from hyperparam list
        self.hyperparams = hyperparams
        self.ups  = nn.ModuleList()    
        self.downs = nn.ModuleList()
        self.pool = nn.MaxPool2d(kernel_size=2, stride = 2)
        in_channels = NUM_INPUT_CHANNELS
        out_channels = NUM_OUTPUT_CLASSES

        #U-Net down portion
        for feature in self.features:
            self.downs.append(DoubleConv(in_channels, feature))
            in_channels = feature
        
        #U-Net Up portion
        for feature in reversed(self.features):
            self.ups.append(nn.ConvTranspose2d(feature*2, feature, kernel_size=2, stride=2 )) #double feature size for skip connections on each up layer
            self.ups.append(DoubleConv(feature*2, feature))
        
        #U-Net bottleneck layer
        self.bottleneck = DoubleConv(self.features[-1], self.features[-1] * 2)
        self.final_conv = nn.Conv2d(self.features[0], out_channels, kernel_size=1)

    def forward(self,x):
        skip_connections = []
        for down in self.downs:
            x = down(x)
            skip_connections.append(x)
            x = self.pool(x)
        
        #bottom bottleneck layer
        x = self.bottleneck(x)
        skip_connections = skip_connections[::-1]

        for i in range(0, len(self.ups), 2):
            x = self.ups[i](x) 
            skip_connection = skip_connections[i // 2] # accomodate for 2 steps in loop, linear ordering


            #if the x dim are different than skip dims before concatenation, pad and resize
            if(x.shape != skip_connection.shape):
                #for PIL images, use:
                #x = tf.resize(x, size =skip_connection.shape[2:] )
                #for tensors, use:
                x = torch.nn.functional.interpolate(x, size=skip_connection.shape[2:], mode="bilinear", align_corners=True)
                
            concat_skip = torch.cat((skip_connection, x), dim = 1) #along channel dimension 
            x = self.ups[i + 1](concat_skip)

        return self.final_conv(x)
    
        # function override

    def train(self, train_loader: torch.utils.data.DataLoader,
              val_loader: torch.utils.data.DataLoader, **kwargs):
        trainer(self.hyperparams, self, train_loader, val_loader)
                
    def validate(self, loader: torch.utils.data.DataLoader):
        validation(self.hyperparams, self, loader )
    
    def test(self, loader: torch.utils.data.DataLoader, loss_fn: Any, **kwargs):
        pass

    def predict(self, loader: torch.utils.data.DataLoader, **kwargs):
        pass

    def save(self, path: Path | str, **kwargs) -> torch.Tensor:
        pass

    def load(path: Path | str, model_class, **kwargs) -> None:
        pass    
    def interpret_model(test_input, **kwargs):
        return super().interpret_model(**kwargs)

def test():
    defaults = {
        "nearly_stop": 5,   # 10 epochs before early stopping
        "nepochs": 30,  # 30 epochs to train
        "batch_size": 32,  # 32 images per batch
        "lr": 0.0001,   # learning rate
        "optimizer": optim.Adam,    # optimizer to use
        "loss_fn": nn.CrossEntropyLoss, # loss fn to use
        "class_fn": nn.Softmax  # function to operate on the final outputs
    }
    x = torch.randn((3, 1, 160, 160))
    model = UNet(defaults)
    preds = model(x)
    print(preds.shape, x.shape)

if __name__ == "__main__":
    
    project_root = Path(os.getcwd()).resolve().parents[1]
    sys.path.append(str(project_root))
    print(project_root) #should be base file
    test()


    





'''# --- Interface --- #
class CVModel(ABCMeta):
    # methods we expect to be overridden
    @abstractmethod
    def train(loader: torch.utils.data.DataLoader,
              optimizer: torch.optim.Optimizer, loss_fn: torch.nn.Loss,
              **kwargs):
        pass
    
    @abstractmethod
    def validate(loader: torch.utils.data.DataLoader, loss_fn: Any, **kwargs):
        pass
    
    @abstractmethod
    def test(loader: torch.utils.data.DataLoader, loss_fn: Any, **kwargs) -> torch.Tensor:
        pass
    
    @abstractmethod
    def predict(loader: torch.utils.data.DataLoader, **kwargs) -> torch.Tensor:
        pass
'''