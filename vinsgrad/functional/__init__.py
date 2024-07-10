"""
vinsgrad Functional Package

This package provides a collection of functions that don't have any learnable parameters.
These functions can be used to create neural network architectures and perform various
operations on tensors.
"""

from .softmax import softmax

__all__ = ["softmax"]