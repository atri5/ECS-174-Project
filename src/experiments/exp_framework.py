'''
@brief: Testing framework for all models
@author: Ayush Tripathi(atripathi7783@gmail.com)
'''

#imports
import os
import time
import pydicom
from tqdm import tqdm
import pandas as pd
from PIL import Image
import numpy as np
from torch.utils.data import Dataset
import torch
from torch.utils.data import Dataset, DataLoader
from torchvision import transforms
import logging
import sys
from pathlib import Path
from src.etl.data_loading import LumbarSpineDataset

#model specific import
from src.arch.unet import *
from src.arch.cnn import *
from src.arch.mcnn import *
from src.arch.kan import *


'''
Goal: Load and store dataloader as an object for use throughout future models. 
'''

class test_framework():

    def init_dataloader(self, image_dir = "", metadata_dir = "", batch_size = 32, manual_seed = 110):
        
        torch.manual_seed(manual_seed)
        print(f"manual seed: {manual_seed}")

        # Initialize the dataset
        image_dir = r"C:\Users\atrip\Classes\ECS-174-Project\src\dataset\rsna-2024-lumbar-spine-degenerative-classification\train_images"
        metadata_dir = r"C:\Users\atrip\Classes\ECS-174-Project\src\dataset\rsna-2024-lumbar-spine-degenerative-classification" 

        transform = transforms.Compose([  
            transforms.Resize((224,224)),
            transforms.ToTensor(),
            transforms.Normalize((0.5), (0.5))])

        dataset = LumbarSpineDataset(image_dir=image_dir, metadata_dir=metadata_dir, transform=transform, load_fraction=1)

        # Create DataLoader with tqdm for progress bar
        dataloader = DataLoader(dataset, batch_size=32, shuffle=True)
        return dataloader
    
    def load_model(self, input_channels = 1, output_classes = 3, ):
        model = UNet(NUM_INPUT_CHANNELS, NUM_OUTPUT_CLASSES)
        return model
    
