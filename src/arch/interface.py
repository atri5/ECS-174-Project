"""
@brief Interface for the model architectures; all models used should follow the 
    same structure to ensure compatability.
@author Arjun Ashok (arjun3.ashok@gmail.com)
"""


# --- Environment Setup --- #
# external modules
import torch
import torch.nn as nn
import torch.functional as F
import torch.optim as optim
import numpy as np
import cv2
from gradcam import GradCAM
from gradcam.utils import visualize_cam
from tqdm import tqdm

# built-in modules
from abc import ABCMeta, abstractmethod
from typing import Any
from pathlib import Path
from json import dump, load
from time import time

# internal modules
from src.utils.visualization import save_image


# --- Constants --- #
NUM_INPUT_CHANNELS = 1
NUM_OUTPUT_CLASSES = 3
IMAGE_DIMS = (224, 224)
DEVICE = "cuda" if torch.cuda.is_available() else "mps"
RAND_SEED = 17
TTV_SPLIT = (0.7, 0.1, 0.2)


# --- Helper Functions --- #
def load_hyperparams(mod_params: dict[str, Any]=None) -> dict[str, Any]:
    """Modifies the default params with any overridden fields.

    Args:
        mod_params (dict[str, Any]): fields to override.

    Returns:
        dict[str, Any]: complete hyperparameter set.
    """
    
    # default hyperparams
    defaults = {
        "nearly_stop": 5,   # 10 epochs before early stopping
        "nepochs": 30,  # 30 epochs to train
        "batch_size": 32,  # 32 images per batch
        "lr": 1e-4,   # learning rate
        "momentum": 0.7,    # momentum, not used w/ Adam
        "dropout": 0.3, # dropout rate
        "optimizer": optim.Adam,    # optimizer to use
        "loss_fn": nn.CrossEntropyLoss, # loss fn to use
        "class_fn": nn.Softmax  # function to operate on the final outputs
    }
    if mod_params is not None:
        # replace any of the defaults
        for mod_param, mod_value in mod_params.items():
            if mod_param not in defaults:
                print(f"<WARNING> skipping adding non-default {mod_param=} w/ {mod_value=}")
            defaults[mod_param] = mod_value
            
    # group hyperparams
    agg_hp = {
        "optimizer": defaults["optimizer"],
        "optimizer_kwargs": {
            "lr": defaults["lr"]
            # "momentum": defaults["momentum"]
        },
        "loss": defaults["loss_fn"],
        "loss_kwargs": {},
        "nearly_stop": defaults["nearly_stop"],
        "nepochs": defaults["nepochs"],
        "batch_size": defaults["batch_size"],
        "class_fn": defaults["class_fn"]
    }
    
    # export the completed hyperparams
    return agg_hp

def validation(hyperparams: dict[str, Any], epoch_model: nn.Module, val_loader, **kwargs) -> dict[str, float]:
    """Computes the accuracy, loss, and other useful metrics on the validation 
    set.

    Args:
        hyperparams (dict[str, Any]): hyperparameters to use.
        epoch_model (CVModel): model to validate.
        val_loader (_type_): dataloader for the validation set

    Returns:
        dict[str, float]: returns a dictionary lookup of:
            - loss, however it is computed via hyperparams
            - accuracy
    """
    
    # setup loss
    val_criterion = hyperparams["loss"]()
    device = kwargs.get("device", DEVICE)
    
    # trackers
    start_time = time()
    correct = 0
    total = 0
    sum_loss = 0
    
    # don't autograd here
    with torch.no_grad():
        epoch_model.eval()
        
        for data in val_loader:
            # unpack the data
            images, labels = data["image"], data["severity"].long()
            images, labels = images.to(device), labels.to(device)
            
            # generate predictions
            outputs = epoch_model(images)
            _, predicted = torch.max(outputs.data, 1)
            sum_loss += val_criterion(outputs, labels).float().cpu().item()
            
            # compute accuracy
            total += labels.size(0)
            correct += (predicted == labels).cpu().sum().item()
    
    # return metrics
    return {
        "acc": correct / total,
        "loss": sum_loss / len(val_loader),
        "duration": time() - start_time
    }

def trainer(hyperparams: dict[str, Any], model: nn.Module, train_loader, val_loader, **kwargs) -> dict[str, Any]:
    """Generic train function that runs the training loop for a given model

    Args:
        hyperparams (dict[str, Any]): hyperparams dictionary
        model (CVModel): model to train
        train_loader (DataLoader): dataloader for the train set
        val_loader (DataLoader): dataloader for the validation set

    Returns:
        dict[str, Any]: dictionary of useful statistics for future plotting
    """
    
    # setup the optimizer & loss
    optimizer = hyperparams["optimizer"](
        model.parameters(), **hyperparams["optimizer_kwargs"]
    )
    criterion = hyperparams["loss"](**hyperparams["loss_kwargs"])
    
    # track metrics
    metrics = {
        "train_loss": list(),
        "train_acc": list(),
        "val_loss": list(),
        "val_acc": list(),
        "train_time": list()
    }
    early_stopper = 0
    early_stop_loss = float("inf")
    NUM_EPOCHS = hyperparams["nepochs"]
    device = kwargs.get("device", DEVICE)
    
    # training loop
    print("Checkpoint> Starting training...")
    for epoch in range(NUM_EPOCHS):
        # running metrics
        start_time = time()
        running_loss = 0.0
        num_correct = 0
        num_samples = 0
        
        # batch iteration
        model.train()
        for data in tqdm(train_loader):
            # unpack data
            images, labels = data["image"], data["severity"].long()
            images, labels = images.to(device), labels.to(device)
            
            # generate predictions
            outputs = model(images)
            
            # zero the parameter gradients
            optimizer.zero_grad()
            
            # forward + backward + optimize
            outputs = model(images)
            loss = criterion(outputs, labels)
            loss.backward()
            optimizer.step()
            
            # track acc & metrics
            _, pred_labels = torch.max(outputs.data, 1)
            num_correct += (pred_labels == labels).float().cpu().sum()
            num_samples += len(outputs)
            running_loss += loss.cpu().item()
            
        # update metrics
        running_loss /= len(train_loader)
        metrics["train_loss"].append(running_loss)
        metrics["train_acc"].append((num_correct / num_samples).item())
        metrics["train_time"].append(time() - start_time)
        
        val_metrics = validation(hyperparams, model, val_loader)
        metrics["val_acc"].append(val_metrics["acc"])
        metrics["val_loss"].append(val_metrics["loss"])
        
        # print report each epoch
        print(f"\tEpoch ({epoch + 1} / {NUM_EPOCHS})>")
        print(f"\t\ttrain loss = {running_loss:.4f}")
        print(f"\t\ttrain acc = {num_correct / num_samples * 100:.2f}%")
        print(f"\t\tval loss = {val_metrics['loss']:.4f}")
        print(f"\t\tval acc = {val_metrics['acc'] * 100:.2f}%")
        
        # early stopping
        early_stopper += 1
        if val_metrics["loss"] < early_stop_loss:
            early_stopper = 1
            early_stop_loss = val_metrics["loss"]
            
            # save the weights as a checkpoint
            checkpoint_path = f"checkpt_{model.__class__.__name__}"
            print(f"\t** saving model to {checkpoint_path} **")
            saver(
                hyperparams, model, path=checkpoint_path
            )
            
        if early_stopper >= hyperparams["nearly_stop"]:
            print(f"Stopping early @ Epoch {epoch + 1}!")
            break
    
    # export metrics data
    print("Finished Training!")
    return metrics

def tester(hyperparams: dict[str, Any], epoch_model: nn.Module, test_loader, **kwargs) -> dict[str, float]:
    """Computes the accuracy, loss, and other useful metrics on the testing 
    set.

    Args:
        hyperparams (dict[str, Any]): hyperparameters to use.
        epoch_model (CVModel): model to validate.
        test_loader (_type_): dataloader for the test set

    Returns:
        dict[str, float]: returns a dictionary lookup of:
            - loss, however it is computed via hyperparams
            - accuracy
    """
    
    # wrap validation
    return validation(hyperparams, epoch_model, test_loader, **kwargs)

def saver(hyperparams: dict[str, Any], model: nn.Module, path: Path | str) -> None:
    """Saves the model state and hyperparams.

    Args:
        hyperparams (dict[str, Any]): hyperparams
        model (nn.Module): model to save
        path (Path | str): name of the model save; goes to default path
    """
    
    # export dir
    weight_export_dir = Path().cwd() / "model-weights"
    hp_export_dir = Path().cwd() / "model-hyperparams"
    
    if not weight_export_dir.exists():
        weight_export_dir.mkdir()
    if not hp_export_dir.exists():
        hp_export_dir.mkdir()
    
    # save the state dict
    torch.save(model.state_dict(), weight_export_dir / path)

    # save the hyperparams
    str_hp = {k: str(v) for k, v in hyperparams.items()}
    with open(hp_export_dir / f"{path}.json", "w") as f:
        dump(str_hp, f, indent=4)

    # conclude
    print(f"Saved model to {weight_export_dir / path}, hyperparams to {hp_export_dir / path}")

def loader(path: Path | str, model_class: Any) -> nn.Module:
    """Loads a model.

    Args:
        path (Path | str): name of the model save, assuming default dir location
        model_class (Any): derived CVModel class, needed for initializing
    """
    
    # export dir
    weight_export_dir = Path().cwd() / "model-weights"
    hp_export_dir = Path().cwd() / "model-hyperparams"
    
    if not weight_export_dir.exists():
        raise FileNotFoundError(f"weights directory ({weight_export_dir.absolute()}) doesn't exist")
    if not hp_export_dir.exists():
        raise FileNotFoundError(f"hyperparams directory ({hp_export_dir.absolute()}) doesn't exist")
    
    # load the hyperparams
    with open(hp_export_dir / path, "r") as f:
        hp = load(f)
        
    # load state dict
    model = model_class.__init__(hyperparams=hp)
    model.load_state_dict(torch.load(weight_export_dir / path, weights_only=True))
    model.eval()

    # conclude
    print(f"Saved model to {weight_export_dir / path}, hyperparams to {hp_export_dir / path}")

def cnn_interpreter(model: nn.Module, train_loader: Any, target_layer: Any=None, **kwargs) -> None:
    """Interprets CNN-based architectures via Grad-Cam.

    Args:
        model (nn.Module): model instance to interpret
        train_loader (Dataloader): image loader the model should be run on to 
            observe.
        target_layer (Any): layer to use for interpretation; defaults to last 
            layer in the `model.conv` block.
    """

    # ensure not autograd
    device = kwargs.get("device", DEVICE)
    model.eval()
    
    # load in image to use
    data = next(iter(train_loader))
    img, label = data["image"][0].to(device), data["severity"][0].to(device).item()

    # ## attempt 1 ##
    # # grab the target layers to analyze
    # target_layers = [model.conv[-1]]      # assume model has a conv backbone
    # targets = [ClassifierOutputTarget(label)]
    # print(targets, img.shape)
    
    # # using gradcam we'll interpret
    # with GradCAM(model=model, target_layers=target_layers) as cam:
    #     grayscale_cams = cam(input_tensor=img, targets=targets)
    #     cam_image = show_cam_on_image(img, grayscale_cams[0, :], use_rgb=True)
        
    # # showing the image
    # cam = np.uint8(255 * grayscale_cams[0, :])
    # cam = cv2.merge([cam, cam, cam])
    # image = np.hstack((np.uint8(255 * img), cam, cam_image))
    
    ## attempt 2 ##
    img = img.view(1, 1, 224, 224)
    output = model(img)

    # build gradcam
    target_layer = target_layer if target_layer is not None else model.conv[-1]
    gradcam = GradCAM(model, target_layer)

    # generate visuals for this image & output class
    target_class = output.argmax(dim=1).item()
    mask, heatmap = gradcam(img, class_idx=target_class)
    image = visualize_cam(mask, img.squeeze(0).squeeze(0))[1]

    # generate & save image
    overlay_np = np.hstack((
        img.view(1, 224, 224).permute(2, 3, 1).cpu().numpy(),
        image.permute(1, 2, 0).cpu().numpy()
    ))
    overlay_np = (overlay_np * 255).astype("uint8")
    
    save_image(overlay_np, f"{model.__class__.__name__}_interpretation")
    

# --- Interface --- #
class CVModel(metaclass = ABCMeta):
    # methods we expect to be overridden
    @abstractmethod
    def train_model(self, train_loader: torch.utils.data.DataLoader,
              val_loader: torch.utils.data.DataLoader, **kwargs) -> dict[str, Any]:
        pass
    
    @abstractmethod
    def validate_model(self, loader: torch.utils.data.DataLoader, **kwargs) -> dict[str, Any]:
        pass
    
    @abstractmethod
    def test_model(self, loader: torch.utils.data.DataLoader, **kwargs) -> dict[str, Any]:
        pass
    
    @abstractmethod
    def predict(self, loader: torch.utils.data.DataLoader, **kwargs) -> torch.Tensor:
        pass
    
    @abstractmethod
    def save(self, path: Path | str, **kwargs) -> None:
        pass
    
    @abstractmethod
    def interpret(self, test_input: torch.utils.data.DataLoader, **kwargs) -> None:
        pass

