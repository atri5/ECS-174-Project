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

# Set up logger
logging.basicConfig(filename='data_loading_errors.log', level=logging.ERROR, 
                    format='%(asctime)s - %(levelname)s - %(message)s')
logger = logging.getLogger()

class LumbarSpineDataset(Dataset):
    def __init__(self, image_dir, metadata_dir, transform=None, load_fraction=1):
        """
        Args:
            image_dir (string): Directory with images organized by study_id, series_id.
            metadata_dir (string): Directory containing the CSV files.
            transform (callable, optional): Optional transform to be applied on a sample.
            load_fraction (float, optional): Fraction of data to load for debugging (default 1).
        """
        self.image_dir = image_dir
        self.transform = transform
        self.load_fraction = load_fraction
        
        # Load the coordinates and severity data from CSV files
        self.coordinates = pd.read_csv(os.path.join(metadata_dir, 'train_label_coordinates.csv'))
        self.metadata = pd.read_csv(os.path.join(metadata_dir, 'train.csv'))

        # Define severity mapping for severity conditions
        self.severity_mapping = {
            'Normal/Mild': 0,
            'Moderate': 1,
            'Severe': 2
        }

        self.severity_columns = [
            'spinal_canal_stenosis_l1_l2', 'spinal_canal_stenosis_l2_l3', 'spinal_canal_stenosis_l3_l4', 
            'spinal_canal_stenosis_l4_l5', 'spinal_canal_stenosis_l5_s1',
            'left_neural_foraminal_narrowing_l1_l2', 'left_neural_foraminal_narrowing_l2_l3', 'left_neural_foraminal_narrowing_l3_l4', 
            'left_neural_foraminal_narrowing_l4_l5', 'left_neural_foraminal_narrowing_l5_s1',
            'right_neural_foraminal_narrowing_l1_l2', 'right_neural_foraminal_narrowing_l2_l3', 'right_neural_foraminal_narrowing_l3_l4', 
            'right_neural_foraminal_narrowing_l4_l5', 'right_neural_foraminal_narrowing_l5_s1',
            'left_subarticular_stenosis_l1_l2', 'left_subarticular_stenosis_l2_l3', 'left_subarticular_stenosis_l3_l4', 
            'left_subarticular_stenosis_l4_l5', 'left_subarticular_stenosis_l5_s1',
            'right_subarticular_stenosis_l1_l2', 'right_subarticular_stenosis_l2_l3', 'right_subarticular_stenosis_l3_l4', 
            'right_subarticular_stenosis_l4_l5', 'right_subarticular_stenosis_l5_s1'
        ]

        self.severity_levels = ['L1/L2', 'L2/L3', 'L3/L4', 'L4/L5', 'L5/S1']
        
        # Load only a fraction of the data for debugging
        self.data = self.load_data()

    def __len__(self):
        return len(self.data)

    def load_data(self):
        data = []
        start_time = time.perf_counter()

        # Calculate the number of items to load based on the load_fraction
        num_items = int(len(self.coordinates) * self.load_fraction)

        for idx, row in tqdm(self.coordinates.iterrows(), total=num_items, desc="Loading images"):
            if len(data) >= num_items:
                break

            study_id = row['study_id']
            series_id = row['series_id']
            instance_number = row['instance_number']
            condition_level = row['level']

            severity = self.get_severity_for_level(study_id, condition_level)
            
            img_path = os.path.join(self.image_dir, f"{study_id}/{series_id}/{instance_number}.dcm")

            coordinates = (row['x'], row['y'])
            sample = {
                'image_path': img_path,
                'severity': severity,
                'coordinates': coordinates
            }
            data.append(sample)

        end_time = time.perf_counter()
        print(f"Time taken to load data: {end_time - start_time:.2f} seconds")
        return data

    def get_severity_for_level(self, study_id, level):
        severity_column = self.severity_columns[self.severity_levels.index(level)]
        severity_row = self.metadata[self.metadata['study_id'] == study_id]
        
        if not severity_row.empty:
            severity_value = severity_row[severity_column].values[0]
            return self.severity_mapping.get(severity_value, -1)
        return -1

    def __getitem__(self, idx):
        sample = self.data[idx]
        img_path = sample["image_path"]
        dicom_image = pydicom.dcmread(img_path)
        image = dicom_image.pixel_array
        image = image.astype(np.float32) / np.max(image)
        image = Image.fromarray(image)
        image = self.transform(image)
        
        sample['image'] = image
        return sample

# Scripting + global vars
transform = transforms.Compose([ 
    transforms.Resize((224,224)),
    transforms.ToTensor(),
    transforms.Normalize((0.5), (0.5))
])

if __name__ == "__main__":
    manual_seed = 17
    torch.manual_seed(manual_seed)
    print(f"manual seed: {manual_seed}")
    # Initialize the dataset
    image_dir = Path().cwd() / "Project" / "train_images"
    metadata_dir = Path().cwd() / "Project"
    
    dataset = LumbarSpineDataset(image_dir=image_dir, metadata_dir=metadata_dir, transform=transform, load_fraction=1)
    total_size = len(dataset)
    print(f"total size of the dataset: {total_size}")
    
    # Create a 70/30 split of train and validation
    train_size = int(total_size*0.7)
    val_size = total_size - train_size
    print(f"train_size: {train_size}, val_size: {val_size}")
    train_dataset, val_dataset = random_split(dataset, [train_size, val_size])
    
    # Create DataLoader with tqdm for progress bar
    train_dataloader = DataLoader(train_dataset, batch_size=32, shuffle=True)
    val_dataloader = DataLoader(val_dataset, batch_size=32, shuffle=True)
