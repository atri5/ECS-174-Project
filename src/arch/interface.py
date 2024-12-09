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

# built-in modules
from abc import ABCMeta, abstractmethod
from typing import Any
from pathlib import Path

# internal modules
## none for now...


# --- Constants --- #
NUM_INPUT_CHANNELS = 3
NUM_OUTPUT_CLASSES = 3
IMAGE_DIMS = (224, 224)
DEVICE = "cuda" if torch.cuda.is_available else "cpu"


# --- Helper Functions --- #
def load_hyperparams(mod_params: dict[str, Any]) -> dict[str, Any]:
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
        "lr": 0.0001,   # learning rate
        "optimizer": optim.Adam,    # optimizer to use
        "loss_fn": nn.CrossEntropyLoss, # loss fn to use
        "class_fn": nn.Softmax  # function to operate on the final outputs
    }
    
    # replace any of the defaults
    for mod_param, mod_value in mod_params.items():
        if mod_param not in defaults:
            print(f"<WARNING> adding non-default {mod_param=} w/ {mod_value=}")
        defaults[mod_param] = mod_value
    
    # export the completed hyperparams
    return defaults

def validation(hyperparams: dict[str, Any], epoch_model: nn.Module, val_loader) -> dict[str, float]:
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
    val_criterion = hyperparams["loss_fn"]()
    
    # trackers
    correct = 0
    total = 0
    sum_loss = 0
    
    # don't autograd here
    with torch.no_grad():
        epoch_model.eval()
        
        for data in val_loader:
            # unpack the data
            images, labels = data
            images, labels = images.to(DEVICE), labels.to(DEVICE)
            
            # generate predictions
            outputs = epoch_model(images)
            _, predicted = torch.max(outputs.data, 1)
            sum_loss += val_criterion(outputs, labels).detach()
            
            # compute accuracy
            total += labels.size(0)
            correct += (predicted == labels).cpu().sum().item()
    
    # return metrics
    return {
        "acc": correct / total,
        "loss": sum_loss
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
    optimizer = hyperparams["optimizer"](lr=hyperparams["lr"])
    criterion = hyperparams["loss_fn"]()
    
    # track metrics
    metrics = {
        "train_loss": list(),
        "train_acc": list(),
        "val_loss": list(),
        "val_acc": list()
    }
    early_stopper = 0
    early_stop_loss = float("inf")
    NUM_EPOCHS = hyperparams["nepochs"]
    
    # training loop
    for epoch in range(NUM_EPOCHS):
        # running metrics
        running_loss = 0.0
        num_correct = 0
        num_samples = 0
        
        # batch iteration
        model.train()
        for data in train_loader:
            # unpack data
            # unpack the data
            images, labels = data
            images, labels = images.to(DEVICE), labels.to(DEVICE)
            
            # generate predictions
            outputs = model(images)
            _, predicted = torch.max(outputs.data, 1)
            
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
            metrics["train_loss"].append(running_loss)
            metrics["train_acc"].append(num_correct / num_samples)
            
            val_metrics = validation(hyperparams, model, val_loader)
            metrics["val_acc"].append(val_metrics["acc"])
            metrics["val_loss"].append(val_metrics["loss"])
            
            # early stopping
            early_stopper += 1
            if metrics["val_loss"] < early_stop_loss:
                early_stopper = 1
                early_stop_loss = metrics["val_loss"]
                
            if early_stopper >= hyperparams["nearly_stop"]:
                print(f"Stopping early @ Epoch {epoch + 1}!")
                break
            
            # print report each epoch
            print(f"<Epoch ({epoch + 1} / {NUM_EPOCHS})>")
            print(f"\ttrain loss = {running_loss}")
            print(f"\ttrain acc = {num_correct / num_samples}")
            print(f"\tval loss = {val_metrics['loss']}")
            print(f"\tval acc = {val_metrics['acc']}")
        
        # export metrics data
        print("Finished Training!")
        return metrics


# --- Interface --- #
class CVModel(metaclass = ABCMeta):
    # methods we expect to be overridden
    @abstractmethod
    def train(loader: torch.utils.data.DataLoader,
              optimizer: torch.optim.Optimizer,
              loss_fn: torch.nn.CrossEntropyLoss,
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
    
    @abstractmethod
    def save(path: Path | str, **kwargs) -> None:
        pass
    
    @abstractmethod
    def interpret_model(test_input: Any, **kwargs) -> None:
        pass

