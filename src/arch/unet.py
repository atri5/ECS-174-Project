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

# built-in modules
from typing import Any

# internal modules
from interface import *


# --- Constants --- #
NUM_INPUT_CHANNELS = 3
NUM_OUTPUT_CLASSES = 3
DEVICE = "cuda" if torch.cuda.is_available else "cpu"

# --- Double Convolution --- #
class DoubleConv(nn.Module):
    def __init__(self, in_channels, out_channels):
        super(DoubleConv, self).init()
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
    def __init__(self, in_channels = NUM_INPUT_CHANNELS, out_channels = NUM_OUTPUT_CLASSES, features = [64,128,256,512]):
        super(UNet, self).__init__()
        
        #for model.eval for batch layers
        self.ups  = nn.ModuleList()    
        self.downs = nn.ModuleList()
        self.pool = nn.MaxPool2d(kernel_size=2, stride = 2)


        #U-Net down portion
        for feature in features:
            self.downs.append(DoubleConv(in_channels, feature))
            in_channels = feature
        
        #U-Net Up portion
        for feature in reversed(features):
            nn.ConvTranspose2d(feature*2, feature, kernel_size=2, stride=2 ) #double feature size for skip connections on each up layer
            self.ups.append(DoubleConv(feature*2, feature))
        
        #U-Net bottleneck layer
        self.bottleneck = DoubleConv(features[-1], features[-1] * 2)
        self.final_conv = nn.Conv2d(features[0], out_channels, kernel_size=1)

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
                x = tf.resize(x, size =skip_connection.shape[2:] )
 
            concat_skip = torch.cat((skip_connection, x), dim = 1) #along channel dimension 
            x = self.ups[i + 1](concat_skip)

        return self.final_conv(x)


def test():
    x = torch.randn((3, 1, 160, 160))
    model = UNet(in_channels = 1, out_channels= 1)
    preds = model(x)
    print(preds.shape, x.shape)

if __name__ == "__main__":
    test()


    


    # function override
    def train(loader: torch.utils.data.DataLoader,
              optimizer: torch.optim.Optimizer, loss_fn: torch.nn.Loss,
              **kwargs):
        pass
    def validate(loader: torch.utils.data.DataLoader, loss_fn: Any, **kwargs):
        pass
    
    def test(loader: torch.utils.data.DataLoader, loss_fn: Any, **kwargs):
        pass

    def predict(loader: torch.utils.data.DataLoader, **kwargs):
        pass    


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