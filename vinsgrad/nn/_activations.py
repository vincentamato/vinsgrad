import numpy as np
from ..core import Tensor

def relu(x: Tensor) -> Tensor:
    """
    Applies the ReLU (Rectified Linear Unit) activation function element-wise.

    Args:
        x (Tensor): The input tensor.

    Returns:
        Tensor: A tensor with the ReLU function applied element-wise.
    """
    out = Tensor(np.maximum(0, x.data), _children=(x,))
    
    if x.requires_grad and x.is_grad_enabled():
        def _relu_backward() -> None:
            # Convert boolean mask to the same dtype as the input data
            grad_mask = (x.data > 0).astype(x.data.dtype)
            if out.grad is not None:  # Add explicit check for None
                x._accumulate_grad(grad_mask * out.grad)
        out.grad_fn = _relu_backward
        out.requires_grad = True
        
    return out