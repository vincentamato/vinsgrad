import numpy as np
from typing import List
from ..core import Tensor

class Optimizer:
    """
    Base class for all optimizers.
    """

    def __init__(self, parameters: List[Tensor], lr: float) -> None:
        """
        Initializes the optimizer with the given parameters and learning rate.

        Args:
            parameters (List[Tensor]): The list of parameters to optimize.
            lr (float): The learning rate.
        """
        self.parameters = parameters
        self.lr = lr

    def zero_grad(self) -> None:
        """
        Sets the gradients of all optimized parameters to zero.
        """
        for p in self.parameters:
            if p.grad is not None:
                p.zero_grad()

    def step(self) -> None:
        """
        Performs a single optimization step.

        Raises:
            NotImplementedError: This method should be overridden by subclasses.
        """
        raise NotImplementedError("The step method should be implemented by subclasses.")
