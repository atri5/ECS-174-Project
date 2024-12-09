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
import torch.optim as optim
import torch.nn as nn
from torch.utils.data import Dataset, DataLoader
from torchvision import transforms
import logging
import sys
from pathlib import Path




project_root = Path(__file__).resolve().parents[2]
sys.path.append(str(project_root))
print(project_root) #should be base file
#model specific import
from src.etl.data_loading import LumbarSpineDataset
from src.arch.unet import *
# from src.arch.cnn import *
# from src.arch.mcnn import *
# from src.arch.kan import *



def load_model(input_channels = 1, output_classes = 3, ):
    model = UNet(NUM_INPUT_CHANNELS, NUM_OUTPUT_CLASSES)
    return model


#store all params in this format
base_params = {}

base_params['learning_rate'] = 0.001
base_params['batch_size'] = 32
base_params['num_epochs'] = 20
base_params['image_dir'] = r"C:\Users\atrip\Classes\ECS-174-Project\src\dataset\rsna-2024-lumbar-spine-degenerative-classification\train_images"
base_params['metadata_dir'] = r"C:\Users\atrip\Classes\ECS-174-Project\src\dataset\rsna-2024-lumbar-spine-degenerative-classification" 
base_params['optimizer'] = lambda model,lr: optim.Adam(model.parameters(), lr=lr)
base_params['loss_fn'] = nn.CrossEntropyLoss()



class TestFramework:

    def __init__(self, model, params = base_params):
        self.model = model
        self.device = "cuda" if torch.cuda.is_available() else "cpu"
        self.dataloader = None
        self.params = params
        self.params['optimizer'](self.model, self.params['learning_rate']) #set up optimizer 

    def init_dataloader(self, image_dir = None, metadata_dir = None, batch_size = None, manual_seed = 110):
        '''
        NOTE: leave image_dir and meta_dir as none unless you want to specify, will use the directories from the params defined
        '''

        torch.manual_seed(manual_seed)
        print(f"manual seed: {manual_seed}")

        #set based on params passed
        image_dir = image_dir or self.params["image_dir"]
        metadata_dir = metadata_dir or self.params["metadata_dir"]
        batch_size = batch_size or self.params["batch_size"]    
        
        transform = transforms.Compose([  
            transforms.Resize((224,224)),
            transforms.ToTensor(),
            transforms.Normalize((0.5), (0.5))])

        dataset = LumbarSpineDataset(image_dir=image_dir, metadata_dir=metadata_dir, transform=transform, load_fraction=1)

        # Create DataLoader, store within obj 
        dataloader = DataLoader(dataset, batch_size=self.params['batch_size'], shuffle=True)
        self.dataloader = dataloader
        return True
    

    
    def train_model(self, epochs = 10):
        #train for x epochs
        print("train model func")
        self.model.train(loader = self.dataloader, optimizer = self.params['optimizer'], loss_fn = self.params['loss_fn'], batch_size = self.params['batch_size'])

    def test_model(self):
        pass
    def save_model(self, path):
        pass
    
    def run_pipeline(self):
        pass



# --- Main Module --- #

if __name__ == "__main__":
    model = load_model()
    testing = TestFramework(model)
    testing.init_dataloader()
    
    #possible caching
    # torch.save(testing.dataloader, "dataloader.pt")

    # loaded_dataset = torch.load("dataset.pt")
    print("initialized dataloader")
    testing.train_model()
    print("completed training")
