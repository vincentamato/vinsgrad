import numpy as np
from ..core import Tensor

def argmax(tensor: Tensor, axis: int) -> Tensor:
    """
    Returns the indices of the maximum values along an axis.
    
    Args:
        tensor (Tensor): The input tensor.
        axis (int): The axis along which to find the maximum value.
    
    Returns:
        Tensor: A tensor of indices.
    """
    return Tensor(np.argmax(tensor.data, axis=axis))
