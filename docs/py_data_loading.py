import torch
from torch.utils.data import Dataset, DataLoader
from PIL import Image
import pydicom
import pandas as pd
import os
from torchvision import transforms
import numpy as np

class LumbarSpineDataset(Dataset):
    def __init__(self, image_dir, metadata_dir, transform=None):
        """
        Args:
            image_dir (string): Directory with images organized by study_id, series_id.
            metadata_dir (string): Directory containing the CSV files.
            transform (callable, optional): Optional transform to be applied on a sample.
        """
        self.image_dir = image_dir
        
        # Load the coordinates and severity data from CSV files
        self.coordinates = pd.read_csv(os.path.join(metadata_dir, 'train_label_coordinates.csv'))
        self.metadata = pd.read_csv(os.path.join(metadata_dir, 'train.csv'))

        # Define severity mapping for severity conditions
        self.severity_mapping = {
            'Normal/Mild': 0,
            'Moderate': 1,
            'Severe': 2
        }

        # Create a mapping of severity levels for each condition level (e.g., L1/L2, L2/L3, etc.)
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

        # Transpose severity levels to correspond to each condition level
        self.severity_levels = ['L1/L2', 'L2/L3', 'L3/L4', 'L4/L5', 'L5/S1']
        self.transform = transform

        # Cache to store images to avoid redundant loading
        self.image_cache = {}

        # Merge severity information for each level with coordinates
        self.data = self.merge_severity_with_coordinates()

    def __len__(self):
        return len(self.data)

    def merge_severity_with_coordinates(self):
        data = []
        # Iterate through each study and series_id in the coordinates dataframe
        for _, row in self.coordinates.iterrows():
            study_id = row['study_id']
            series_id = row['series_id']
            instance_number = row['instance_number']
            condition_level = row['level']  # e.g., L1/L2, L2/L3, etc.

            # Extract corresponding severity for the condition level from metadata
            severity = self.get_severity_for_level(study_id, condition_level)
            
            # Check if the image is already cached
            img_path = os.path.join(self.image_dir, f"{study_id}/{series_id}/{instance_number}.dcm")
            if instance_number in self.image_cache:
                image = self.image_cache[instance_number]
            else:
                dicom_image = pydicom.dcmread(img_path)
                image = dicom_image.pixel_array
                image = image.astype(np.float32) / np.max(image)  # Normalize image
                image = Image.fromarray(image)

                # Cache the image for future use
                self.image_cache[instance_number] = image

            # Prepare the sample dictionary
            coordinates = (row['x'], row['y'])  # Coordinates for the condition level
            sample = {
                'image': image,
                'severity': severity,
                'coordinates': coordinates
            }
            data.append(sample)

        return data

    def get_severity_for_level(self, study_id, level):
        """
        Get the severity condition for a specific level (e.g., L1/L2, L2/L3) for the given study_id.
        """
        # Find the relevant severity columns for this level
        severity_column = self.severity_columns[self.severity_levels.index(level)]
        # Find the corresponding severity value for this level
        severity_row = self.metadata[self.metadata['study_id'] == study_id]
        
        if not severity_row.empty:
            severity_value = severity_row[severity_column].values[0]
            return self.severity_mapping.get(severity_value, -1)  # Return the mapped severity value
        return -1  # Return -1 if no severity found for the given study_id

    def __getitem__(self, idx):
        sample = self.data[idx]
        
        # Apply transformations if any
        if self.transform:
            sample['image'] = self.transform(sample['image'])
        
        return sample


image_dir = r"Project\train_images"
metadata_dir = r"Project"
transform = transforms.Compose([
    transforms.Resize((224,224)),
    transforms.ToTensor(), 
    transforms.Normalize((0.5), (0.5))])
dataset = LumbarSpineDataset(image_dir=image_dir, metadata_dir=metadata_dir, transform=transform)

# Create DataLoader
dataloader = DataLoader(dataset, batch_size=1, shuffle=True)