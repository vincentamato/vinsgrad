from .module import Module
from ._activations import relu
from ..core import Tensor

class ReLU(Module):
    """
    Applies the ReLU (Rectified Linear Unit) activation function element-wise.
    """

    def __init__(self) -> None:
        """
        Initializes the ReLU module.
        """
        super(ReLU, self).__init__()

    def forward(self, x: Tensor) -> Tensor:
        """
        Forward pass of the ReLU activation function.
        
        Args:
            x (Tensor): The input tensor.
        
        Returns:
            Tensor: A tensor with the ReLU function applied element-wise.
        """
        return relu(x)
