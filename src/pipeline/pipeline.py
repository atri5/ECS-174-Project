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
    def __init__(self, model_class: CVModel, hyperparams: dict[str, Any], 
                 model_descr: str, image_dir: Path | str,
                 metadata_dir: Path | str):
        self.device = DEVICE if model_class != CKAN else "cpu"
        self.model = model_class(hyperparams).to(self.device)
        self.hyperparams = hyperparams
        self.model_descr = model_descr
        self.image_dir = image_dir
        self.data_dir = metadata_dir
        
    def init_dataloader(self):
        # set seed for reproducibility + constants
        print("initializing dataloader...")
        torch.manual_seed(RAND_SEED)
        batch_size = self.hyperparams["batch_size"]

        # initialize data loader
        transform = transforms.Compose([ 
            transforms.Resize((224,224)),
            transforms.ToTensor(),
            transforms.Normalize((0.5), (0.5))
        ])
        dataset = LumbarSpineDataset(
            image_dir=self.image_dir, metadata_dir=self.data_dir,
            transform=transform, load_fraction=1
        )
        
        # split data into train, test, validate
        total_size = len(dataset)
        train_size = int(total_size * TTV_SPLIT[0])
        test_size = int(total_size * TTV_SPLIT[1])
        val_size = total_size - train_size - test_size
        
        train_dataset, val_dataset, test_dataset = random_split(
            dataset,
            [train_size, val_size, test_size]
        )
        
        # Create DataLoader with tqdm for progress bar
        self.train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True)
        self.val_loader = DataLoader(val_dataset, batch_size=batch_size, shuffle=True)
        self.test_loader = DataLoader(test_dataset, batch_size=batch_size, shuffle=False)
    
    def training(self) -> dict[str, Any]:
        """Wraps the full training pipeline.

        Returns:
            dict[str, Any]: metrics
        """

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
        print(f"Accuracy: {test_metrics['acc']} %, Loss: {test_metrics['loss']}")
        
        # saving
        self.model.save(path=self.model_descr)
        
        # export compiled metrics
        return {
            "train": train_metrics,
            "test": test_metrics
        }
        
    def pipeline(self) -> dict[str, Any]:
        """Runs the full pipeline for a given model

        Returns:
            dict[str, Any]: _description_
        """
        
        # load data
        self.init_dataloader()
        
        # training
        metrics = self.training()
        
        # save metrics
        metrics_path = Path().cwd() / "model-reports"
        
        with open(metrics_path / self.model_descr, "w") as f:
            dump(metrics, f, indent=4)
            
        return metrics
        
        # TODO @Ayush: visualizations
        
        

# Testing
def main():
    # collect directories
    data_dir = Path().cwd() / "data"
    img_dir = data_dir / "train_images"
    
    # initialize model
    hp = load_hyperparams()
    model_class = CKAN
    
    # pipeline
    res = Pipeline(
        model_class=model_class, hyperparams=hp, model_descr=f"trial_ckan",
        image_dir=img_dir, metadata_dir=data_dir
    ).pipeline()

if __name__ == "__main__":
    main()
    
