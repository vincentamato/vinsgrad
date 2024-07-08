"""
vinsgrad: A Lightweight Deep Learning Library

This module provides the core functionalities of the vinsgrad deep learning library,
including tensor operations, neural network layers, optimization algorithms, vision
datasets, and utility functions for data handling and model saving/loading.
"""

from .core.engine import Tensor, no_grad
from . import nn
from . import optim
from . import vision
from . import utils
from .utils.save_load import save, load
from .utils.utils import argmax
from .utils.data.random_split import random_split

__all__ = [
    'Tensor',
    'no_grad',
    'nn',
    'optim',
    'vision',
    'utils',
    'save',
    'load',
    'argmax',
    'random_split',
]

# Metadata
__version__ = '0.1.0'
__author__ = 'Vinsgrad Developers'
__license__ = 'MIT'
