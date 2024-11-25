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
import pandas as pd

# built-in modules
from abc import ABCMeta, abstractmethod

# internal modules
from interface import *


# --- Interface --- #
class CVModel(nn.Module, ABCMeta):
    # data to store
    def 



