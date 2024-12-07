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
        "nlayers": 4,
        "activation": nn.GELU()
    }
    
    # replace any of the defaults
    for mod_param, mod_value in mod_params.items():
        if mod_param not in defaults:
            print(f"<WARNING> adding non-default {mod_param=} w/ {mod_value=}")
        defaults[mod_param] = mod_value
    
    # export the completed hyperparams
    return defaults


# --- Interface --- #
class CVModel(ABCMeta):
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
    def interpret_model(test_input: nn.Data, **kwargs) -> None:
        pass

