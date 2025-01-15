import numpy as np
from ..core import Tensor
from .module import Module
from ..utils.im2col_utils import im2col_pool


class MaxPool2D(Module):
    def __init__(self, kernel_size: int, stride: int = None) -> None:
        super().__init__()
        self.kernel_size = kernel_size
        self.stride = stride if stride is not None else kernel_size

    def forward(self, input: Tensor) -> Tensor:
        """
        Forward pass using vectorized im2col operations for pooling
        """
        self.input_shape = input.shape
        batch_size, channels, height, width = input.shape
        
        # Calculate output dimensions
        h_out = (height - self.kernel_size) // self.stride + 1
        w_out = (width - self.kernel_size) // self.stride + 1
        
        # Use utility function for pooling im2col
        x_reshaped = im2col_pool(input.data, self.kernel_size, self.stride)
        
        # Find maximum value and its index for each patch
        self.max_indices = np.argmax(x_reshaped, axis=1)
        output_flat = np.max(x_reshaped, axis=1)
        
        # Reshape output back to 4D
        output = output_flat.reshape(batch_size, channels, h_out, w_out)
        
        return Tensor(
            output,
            _children=(input,),
            requires_grad=input.requires_grad and Tensor.grad_enabled
        )