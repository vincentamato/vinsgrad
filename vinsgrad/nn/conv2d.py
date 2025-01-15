import numpy as np
from ..core import Tensor
from ..utils.im2col_utils import im2col_conv
from .module import Module

class Conv2D(Module):
    def __init__(self, in_channels: int, out_channels: int, kernel_size: int, 
                 stride: int, padding: int, initialization: str = "he") -> None:
        super().__init__()
        self.in_channels = in_channels
        self.out_channels = out_channels
        self.kernel_size = kernel_size
        self.stride = stride
        self.padding = padding
        
        if initialization == "he":
            self.weight = Tensor(self.he_init(in_channels, out_channels, kernel_size), 
                               requires_grad=True)
        elif initialization == "xavier":
            self.weight = Tensor(self.xavier_init(in_channels, out_channels, kernel_size), 
                               requires_grad=True)
        self.bias = Tensor(np.zeros(out_channels), requires_grad=True)

    def xavier_init(self, in_channels: int, out_channels: int, kernel_size: int) -> np.ndarray:
        limit = np.sqrt(6 / ((in_channels * kernel_size * kernel_size) + 
                           (out_channels * kernel_size * kernel_size)))
        return np.random.uniform(-limit, limit, 
                               (out_channels, in_channels, kernel_size, kernel_size))
    
    def he_init(self, in_channels: int, out_channels: int, kernel_size: int) -> np.ndarray:
        sd = np.sqrt(2 / (in_channels * kernel_size * kernel_size))
        return np.random.normal(0, sd, 
                              (out_channels, in_channels, kernel_size, kernel_size))

    def forward(self, input: Tensor) -> Tensor:
        """
        Forward pass using vectorized im2col operations
        """
        self.input_shape = input.shape
        N, C, H, W = input.shape
        
        # Calculate output dimensions
        H_out = (H + 2 * self.padding - self.kernel_size) // self.stride + 1
        W_out = (W + 2 * self.padding - self.kernel_size) // self.stride + 1
        
        # Use utility function for im2col
        x_cols = im2col_conv(input.data, self.kernel_size, self.stride, self.padding)
        w_cols = self.weight.data.reshape(self.out_channels, -1)
        
        # Perform convolution as matrix multiplication
        out = w_cols @ x_cols
        
        # Add bias and reshape
        out = out.reshape(self.out_channels, H_out, W_out, N)
        out = out.transpose(3, 0, 1, 2)
        out += self.bias.data.reshape(1, -1, 1, 1)
        
        # Store x_cols for backward pass
        self.x_cols = x_cols
        
        return Tensor(
            out,
            _children=(input, self.weight, self.bias),
            requires_grad=(input.requires_grad or 
                         self.weight.requires_grad or 
                         self.bias.requires_grad) and Tensor.grad_enabled
        )