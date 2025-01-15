import numpy as np
from typing import Tuple, Optional

def get_im2col_indices(x_shape: Tuple[int, ...], 
                      kernel_height: int, 
                      kernel_width: int,
                      padding: int, 
                      stride: int) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
    """
    Calculate indices for the im2col operation.
    
    Args:
        x_shape: Input shape (batch_size, channels, height, width)
        kernel_height: Height of the kernel
        kernel_width: Width of the kernel
        padding: Padding size
        stride: Stride size
        
    Returns:
        Tuple of (k, i, j) index arrays for im2col operation
    """
    N, C, H, W = x_shape
    
    assert (H + 2 * padding - kernel_height) % stride == 0
    assert (W + 2 * padding - kernel_width) % stride == 0
    
    out_height = (H + 2 * padding - kernel_height) // stride + 1
    out_width = (W + 2 * padding - kernel_width) // stride + 1

    # Create index matrices
    i0 = np.repeat(np.arange(kernel_height), kernel_width)
    i0 = np.tile(i0, C)
    i1 = stride * np.repeat(np.arange(out_height), out_width)
    j0 = np.tile(np.arange(kernel_width), kernel_height * C)
    j1 = stride * np.tile(np.arange(out_width), out_height)
    i = i0.reshape(-1, 1) + i1.reshape(1, -1)
    j = j0.reshape(-1, 1) + j1.reshape(1, -1)

    k = np.repeat(np.arange(C), kernel_height * kernel_width).reshape(-1, 1)
    
    return k.astype(int), i.astype(int), j.astype(int)

def im2col_conv(x: np.ndarray, 
                kernel_size: int, 
                stride: int, 
                padding: int = 0) -> np.ndarray:
    """
    Vectorized im2col implementation for convolution using indexing.
    
    Args:
        x: Input array of shape (batch_size, channels, height, width)
        kernel_size: Size of the square kernel
        stride: Stride size
        padding: Padding size
        
    Returns:
        Rearranged array suitable for convolution
    """
    # Zero-pad the input if needed
    if padding > 0:
        x_padded = np.pad(x, 
                         ((0, 0), (0, 0), (padding, padding), (padding, padding)),
                         mode='constant', 
                         constant_values=0)
    else:
        x_padded = x

    k, i, j = get_im2col_indices(x.shape, kernel_size, kernel_size, padding, stride)
    
    cols = x_padded[:, k, i, j]
    C = x.shape[1]
    cols = cols.transpose(1, 2, 0).reshape(C * kernel_size * kernel_size, -1)
    return cols

def col2im_conv(cols: np.ndarray, 
                x_shape: Tuple[int, ...], 
                kernel_size: int,
                stride: int, 
                padding: int = 0) -> np.ndarray:
    """
    Vectorized col2im implementation for convolution using indexing.
    
    Args:
        cols: Column matrix to be converted back to image
        x_shape: Original input shape (batch_size, channels, height, width)
        kernel_size: Size of the square kernel
        stride: Stride size
        padding: Padding size
        
    Returns:
        Reconstructed image array
    """
    N, C, H, W = x_shape
    H_padded, W_padded = H + 2 * padding, W + 2 * padding
    x_padded = np.zeros((N, C, H_padded, W_padded), dtype=cols.dtype)
    
    k, i, j = get_im2col_indices(x_shape, kernel_size, kernel_size, padding, stride)
    
    cols_reshaped = cols.reshape(C * kernel_size * kernel_size, -1, N)
    cols_reshaped = cols_reshaped.transpose(2, 0, 1)
    
    np.add.at(x_padded, (slice(None), k, i, j), cols_reshaped)
    
    if padding > 0:
        return x_padded[:, :, padding:-padding, padding:-padding]
    return x_padded

def im2col_pool(x: np.ndarray, 
                kernel_size: int, 
                stride: Optional[int] = None) -> np.ndarray:
    """
    Vectorized im2col implementation for pooling using stride tricks.
    
    Args:
        x: Input array of shape (batch_size, channels, height, width)
        kernel_size: Size of the square kernel
        stride: Stride size (defaults to kernel_size if None)
        
    Returns:
        Rearranged array suitable for pooling operations
    """
    if stride is None:
        stride = kernel_size
        
    batch_size, channels, height, width = x.shape
    h_out = (height - kernel_size) // stride + 1
    w_out = (width - kernel_size) // stride + 1
    
    # Create strided view
    stride_shape = x.strides
    window_shape = (
        batch_size, channels, h_out, w_out,
        kernel_size, kernel_size
    )
    strides = (
        stride_shape[0], stride_shape[1],
        stride_shape[2] * stride, stride_shape[3] * stride,
        stride_shape[2], stride_shape[3]
    )
    
    windows = np.lib.stride_tricks.as_strided(
        x, window_shape, strides, writeable=False
    )
    
    return windows.reshape(-1, kernel_size * kernel_size)