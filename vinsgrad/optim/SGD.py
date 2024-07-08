from typing import List
from vinsgrad.core import Tensor
from vinsgrad.optim._optimizer import Optimizer
import numpy as np

class SGD(Optimizer):
    def __init__(self, parameters, lr, momentum=0):
        super().__init__(parameters, lr)
        self.momentum = momentum
        self.velocities = [np.zeros_like(p.data) for p in self.parameters]

    def state_dict(self):
        return {
            'lr': self.lr,
            'momentum': self.momentum,
            'velocities': self.velocities
        }

    def load_state_dict(self, state_dict):
        self.lr = state_dict['lr']
        self.momentum = state_dict['momentum']
        self.velocities = state_dict['velocities']

    def step(self):
        for i, p in enumerate(self.parameters):
            if p.grad is None:
                continue
            v = self.velocities[i]
            v[:] = (self.momentum * v) + ((1 - self.momentum) * p.grad)
            p.data -= (self.lr * v)

    def zero_grad(self):
        for p in self.parameters:
            if p.grad is not None:
                p.grad.fill(0)