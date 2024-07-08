import numpy as np
from typing import Tuple
from ..core import Tensor
from .module import Module

class Linear(Module):
    """
    A linear transformation layer with weights and bias.
    """

    def __init__(self, in_features: int, out_features: int) -> None:
        """
        Initializes the linear layer with given input and output features.

        Args:
            in_features (int): The number of input features.
            out_features (int): The number of output features.
        """
        super().__init__()
        self.weight = Tensor(self.xavier_init(in_features, out_features), requires_grad=True)
        self.bias = Tensor(np.zeros(out_features), requires_grad=True)

    def xavier_init(self, in_features: int, out_features: int) -> np.ndarray:
        """
        Initializes the weights using the Xavier initialization method.

        Args:
            in_features (int): The number of input features.
            out_features (int): The number of output features.

        Returns:
            np.ndarray: The initialized weights.
        """
        limit = np.sqrt(6 / (in_features + out_features))
        return np.random.uniform(-limit, limit, (out_features, in_features))

    def forward(self, input: Tensor) -> Tensor:
        """
        Performs the forward pass of the linear layer.

        Args:
            input (Tensor): The input tensor.

        Returns:
            Tensor: The output tensor after applying the linear transformation.
        """
        return input @ self.weight.T() + self.bias
