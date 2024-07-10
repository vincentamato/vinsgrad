from vinsgrad.core import Tensor

def softmax(input: Tensor, axis: int = -1) -> Tensor:
    """
    Applies the Softmax function to an n-dimensional input Tensor.
    
    Args:
        input (Tensor): input tensor
        dim (int): A dimension along which Softmax will be computed (so every slice
            along dim will sum to 1).
    
    Returns:
        Tensor: Output tensor of the same shape as input
    """
    max_input = input.max(axis=axis, keepdims=True)
    exp_input = (input - max_input).exp()
    return exp_input / exp_input.sum(axis=axis, keepdims=True)