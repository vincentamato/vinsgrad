from .module import Module
from .activations import *

class ReLU(Module):
    def __init__(self):
        super(ReLU, self).__init__()

    def forward(self, x):
        return relu(x)
    