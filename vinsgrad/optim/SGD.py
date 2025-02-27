from typing import List, Dict, Any
from vinsgrad.core import Tensor
from vinsgrad.optim._optimizer import Optimizer
import numpy as np

class SGD(Optimizer):
    """
    Stochastic Gradient Descent (SGD) optimizer with momentum.
    """

    def __init__(self, parameters: List[Tensor], lr: float, momentum: float = 0.0) -> None:
        """
        Initializes the SGD optimizer.

        Args:
            parameters (List[Tensor]): The list of parameters to optimize.
            lr (float): The learning rate.
            momentum (float): The momentum factor (default: 0).
        """
        super().__init__(parameters, lr)
        self.momentum = momentum
        self.velocities = [np.zeros_like(p.data, dtype=np.float32) for p in self.parameters] if momentum > 0 else None

    def state_dict(self) -> Dict[str, Any]:
        """
        Returns the state of the optimizer as a dictionary.

        Returns:
            Dict[str, Any]: The state of the optimizer.
        """
        return {
            'lr': self.lr,
            'momentum': self.momentum,
            'velocities': self.velocities
        }

    def load_state_dict(self, state_dict: Dict[str, Any]) -> None:
        """
        Loads the optimizer state.

        Args:
            state_dict (Dict[str, Any]): The state of the optimizer.
        """
        self.lr = state_dict['lr']
        self.momentum = state_dict['momentum']
        self.velocities = state_dict['velocities']

    def step(self) -> None:
        """
        Performs a single optimization step.
        """
        for i, p in enumerate(self.parameters):
            if p.grad is None:
                continue
            grad = p.grad.astype(np.float32)
            if self.momentum > 0:
                v = self.velocities[i]
                v[:] = self.momentum * v + grad
                p.data -= self.lr * v
            else:
                p.data -= self.lr * grad

    def zero_grad(self) -> None:
        """
        Clears the gradients of all optimized parameters.
        """
        for p in self.parameters:
            if p.grad is not None:
                p.grad.fill(0)
