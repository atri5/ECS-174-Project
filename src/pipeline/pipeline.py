'''
@brief Testing framework for all models
@author Arjun Ashok (arjun3.ashok@gmail.com),
        Ayush Tripathi(atripathi7783@gmail.com)
'''

# imports
import torch
from tqdm import tqdm
from PIL import Image
from torch.utils.data import Dataset, DataLoader, random_split
from torchvision import transforms
from pathlib import Path

# model specific import
from src.etl.data_loading import LumbarSpineDataset
from src.arch.interface import *
from src.arch.cnn import *
from src.arch.mcnn import *
from src.arch.unet import *
from src.arch.resnet import *
from src.arch.transformer import *
from src.arch.kan import *
from src.utils.visualization import *

WEIGHTSDIR = Path().cwd() / "model-weights" 
DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")

# Framework
class Pipeline(object):
    def __init__(self, model_class: CVModel, hyperparams: dict[str, Any], 
                 model_descr: str, image_dir: Path | str,
                 metadata_dir: Path | str):
        # CKAN can only run on cpu
        self.device = DEVICE if model_class != CKAN else "cpu"
        
        # buildup the model
        self.model = model_class(hyperparams).to(self.device)
        self.hyperparams = hyperparams
        
        # meta data
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
            transforms.Resize((224, 224)),
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
        
        print(train_metrics)
        print(test_metrics)

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
        if not metrics_path.exists():
            metrics_path.mkdir()
        
        with open(metrics_path / f"{self.model_descr}.json", "w") as f:
            dump(metrics, f, indent=4)
            
        return metrics
        
        # TODO @Ayush: visualizations
        
        

# Testing
def main():
    # collect directories
    data_dir = r"C:\Users\atrip\Classes\ECS-174-Project\src\dataset\rsna-2024-lumbar-spine-degenerative-classification"
    img_dir = r"C:\Users\atrip\Classes\ECS-174-Project\src\dataset\rsna-2024-lumbar-spine-degenerative-classification\train_images"

    
    # initialize model
    hp = load_hyperparams()
    hp["nepochs"] = 1
    model_arch = "ResNet"
    run_type = "trial"
    
    # model descriptions
    model_info = {
        "CNN": (CNN, "baseline_CNN"),
        "MCNN": (MCNN, "modified_CNN"),
        "ResNet": (ResNet, "simple_ResNet18"),
        "VIT": (VisionTransformerWithCoordinates, "vis_transformer"),
        "CKAN": (CKAN, "conv_KAN")
    }
    model_class, model_descr = model_info[model_arch]
    
    # pipeline
    pipe = Pipeline(
        model_class=model_class, hyperparams=hp, 
        model_descr=f"{run_type}_{model_descr}", image_dir=img_dir,
        metadata_dir=data_dir
    )
    # res = pipe.pipeline()
    
    # interpreter
    # if model_arch == "CNN":
    #     pipe.model.interpret(pipe.test_loader)

    #Use if loading model:
    pipe.init_dataloader()

    #pick the correct file for hyperparams
    weights_name = "final_simple_ResNet"
    # C:\Users\atrip\Classes\ECS-174-Project\model-weights\checkpt_CNN
    file_path = os.path.join(WEIGHTSDIR, weights_name)
    loaded_model = loader(weights_name, model_class)
    loaded_model = loaded_model.to(DEVICE)
    loaded_model.interpret(pipe.test_loader)

if __name__ == "__main__":
    main()
    


