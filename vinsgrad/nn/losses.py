import numpy as np
from ..core import Tensor
from .module import Module

class MSELoss(Module):
    def __init__(self):
        super().__init__()

    def forward(self, y_pred: Tensor, y_true: Tensor):
        loss = ((y_pred - y_true) ** 2).mean()
        loss.requires_grad = True
        return loss
    
class CrossEntropyLoss(Module):
    
    def softmax(self, z):
        exp_z = (z - z.max(axis=1, keepdims=True)).exp()
        return exp_z / exp_z.sum(axis=1, keepdims=True)
    
    def forward(self, y_pred: Tensor, y_true: Tensor):
        probs = self.softmax(y_pred)
        epsilon = 1e-9
        probs = probs + epsilon
        loss = -(y_true * probs.log()).sum(axis=1).mean()
        loss.requires_grad = True
        return loss
