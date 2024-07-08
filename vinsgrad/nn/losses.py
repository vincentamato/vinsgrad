import numpy as np
from typing import Any
from ..core import Tensor
from .module import Module

class MSELoss(Module):
    """
    Mean Squared Error (MSE) loss module.
    """
    
    def __init__(self) -> None:
        """
        Initializes the MSELoss module.
        """
        super().__init__()

    def forward(self, y_pred: Tensor, y_true: Tensor) -> Tensor:
        """
        Computes the MSE loss between predicted and true values.
        
        Args:
            y_pred (Tensor): The predicted values.
            y_true (Tensor): The true values.
        
        Returns:
            Tensor: The computed MSE loss.
        """
        loss = ((y_pred - y_true) ** 2).mean()
        loss.set_requires_grad(True)
        return loss
    
class CrossEntropyLoss(Module):
    """
    Cross-Entropy loss module.
    """
    
    def softmax(self, z: Tensor) -> Tensor:
        """
        Applies the softmax function to the input tensor.
        
        Args:
            z (Tensor): The input tensor.
        
        Returns:
            Tensor: The output tensor with softmax applied.
        """
        exp_z = (z - z.max(axis=1, keepdims=True)).exp()
        return exp_z / exp_z.sum(axis=1, keepdims=True)
    
    def forward(self, y_pred: Tensor, y_true: Tensor) -> Tensor:
        """
        Computes the Cross-Entropy loss between predicted and true values.
        
        Args:
            y_pred (Tensor): The predicted values (logits).
            y_true (Tensor): The true values (one-hot encoded).
        
        Returns:
            Tensor: The computed Cross-Entropy loss.
        """
        probs = self.softmax(y_pred)
        epsilon = 1e-9
        probs = probs + epsilon
        loss = -(y_true * probs.log()).sum(axis=1).mean()
        loss.set_requires_grad(True)
        return loss
