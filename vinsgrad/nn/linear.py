import numpy as np
from ..core import Tensor
from .module import Module

class Linear(Module):
    def __init__(self, in_features, out_features):
        super().__init__()
        self.weight = Tensor(self.xavier_init(in_features, out_features), requires_grad=True)
        self.bias = Tensor(np.zeros(out_features), requires_grad=True)

    def xavier_init(self, in_features, out_features):
        limit = np.sqrt(6 / (in_features + out_features))
        return np.random.uniform(-limit, limit, (out_features, in_features))

    def forward(self, input):
        return input @ self.weight.T() + self.bias