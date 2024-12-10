'''
@brief Testing framework for all models
@author Arjun Ashok (arjun3.ashok@gmail.com),
        Ayush Tripathi(atripathi7783@gmail.com)
'''

#imports
import torch
from tqdm import tqdm
from PIL import Image
from torch.utils.data import Dataset, DataLoader, random_split
from torchvision import transforms
from pathlib import Path

#model specific import
from src.etl.data_loading import LumbarSpineDataset
from src.arch.interface import *
from src.arch.unet import *
from src.arch.cnn import *
from src.arch.mcnn import *
from src.arch.kan import *
from src.utils.visualization import *


# Helper Methods
def load_model(self, input_channels = 1, output_classes = 3, ):
    model = UNet(NUM_INPUT_CHANNELS, NUM_OUTPUT_CLASSES)
    return model

# Framework
class Pipeline(object):
    def __init__(self, model, model_descr: str):
        self.model = model
        self.device = DEVICE
        self.model_descr = model_descr


    def init_dataloader(self, image_dir: Path | str, metadata_dir: Path | str, batch_size = 32, manual_seed = 110):
        # set seed for reproducibility
        print("dataloader initialized")
        torch.manual_seed(RAND_SEED)

        # initialize data loader
        transform = transforms.Compose([ 
            transforms.Resize((224,224)),
            transforms.ToTensor(),
            transforms.Normalize((0.5), (0.5))
        ])
        dataset = LumbarSpineDataset(
            image_dir=image_dir, metadata_dir=metadata_dir, transform=transform,
            load_fraction=1
        )
        
        
        #split data into train, validate
        total_size = len(dataset)
        train_size = int(total_size * TTV_SPLIT[0])
        test_size = int(total_size * TTV_SPLIT[1])
        val_size = total_size - train_size - test_size 
        train_dataset, val_dataset, test_dataset = random_split(dataset, [train_size, val_size, test_size])
        
        # Create DataLoader with tqdm for progress bar
        self.train_loader = DataLoader(train_dataset, batch_size=32, shuffle=True)
        self.val_loader = DataLoader(val_dataset, batch_size=32, shuffle=True)
        self.test_loader = DataLoader(test_dataset, batch_size=batch_size, shuffle=False)
        
        # # dataloader w/ progress bar
        # dataloader = DataLoader(dataset, batch_size=32, shuffle=True)
    
    
    def split_loader(self, dataset: DataLoader) -> None:
        """Split the dataloader into a train, test, and validation set.
        """
        pass
    


    def run_pipeline(self) -> dict[str, Any]:
        """Wraps the full training pipeline.

        Returns:
            dict[str, Any]: metrics
        """
        #split ttv data


        # train & test
        train_metrics = self.model.train_model(
            self.train_loader, self.val_loader
        )
        test_metrics = self.model.test_model(
            self.test_loader
        )

        # plotting loss and accuracy on each epoch

        ## for training metrics
        plot_train_metrics(train_metrics, self.model_descr)
        
        ## for test metrics
        print(f'Accuracy: {test_metrics["acc"]} %, Loss: {test_metrics["loss"]}')       


        
        # saving
        self.model.save(path=self.model_descr)
        

# Testing
def main():
    # collect directories
    data_dir = Path().cwd() / "src" / "dataset" / "rsna-2024-lumbar-spine-degenerative-classification"
    img_dir = data_dir / "train_images"
    
    # initialize model
    hp = load_hyperparams()
    model = CNN(hp).to(DEVICE)
    
    # pipeline
    pipe = Pipeline(model=model, model_descr="baseline_CNN")
    dataset = pipe.init_dataloader(image_dir=img_dir, metadata_dir=data_dir)
    pipe.split_loader(dataset=dataset)
    pipe.run_pipeline()

if __name__ == "__main__":
    main()
    
