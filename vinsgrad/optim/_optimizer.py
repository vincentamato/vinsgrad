import numpy as np

class Optimizer:
    def __init__(self, parameters, lr: float) -> None:
        self.parameters = parameters
        self.lr = lr

    def zero_grad(self):
        for p in self.parameters:
            if p.grad is not None:
                p.grad.fill(0)

    def step(self):
        raise NotImplementedError
