"""
vinsgrad Tensor Engine.

This module implements an automatic differentiation engine with a Tensor class
that supports backpropagation. It provides functionality similar to PyTorch's
autograd system but in a simplified form.

Example:
    >>> import vinsgrad
    >>> x = vinsgrad.Tensor([1, 2, 3], requires_grad=True)
    >>> y = x * 2
    >>> y.backward()
    >>> print(x.grad)  # Shows gradient of y with respect to x
"""

from __future__ import annotations

import functools
from typing import (
    Any,
    Callable,
    List,
    Optional,
    Tuple,
    Union,
    TypeVar
)

import numpy as np

TensorLike = Union[float, List, np.ndarray, 'Tensor']
Shape = Tuple[int, ...]
T = TypeVar('T', bound='Tensor')
GradFn = Callable[[], None]

class TensorError(Exception):
    """Base class for Tensor-related errors."""
    pass

class GradientError(TensorError):
    """Raised when there's an error with gradient computation."""
    pass

class ShapeError(TensorError):
    """Raised when there's an error with tensor shapes."""
    pass

class BroadcastError(ShapeError):
    """Raised when tensor broadcasting fails."""
    pass

class DTypeError(TensorError):
    """Raised when there's an error with tensor data types."""
    pass

class OperationError(TensorError):
    """Raised when a tensor operation fails."""
    pass

def set_grad_enabled(mode: bool) -> bool:
    """Sets gradient computation mode globally.
    
    Args:
        mode: If True, enables gradient computation. If False, disables it.
    
    Returns:
        The previous gradient computation mode.
    """
    previous = Tensor.grad_enabled
    Tensor.grad_enabled = mode
    return previous

class no_grad:
    """Context manager that disables gradient calculation."""
    
    def __enter__(self) -> None:
        self.previous = Tensor.grad_enabled
        Tensor.grad_enabled = False

    def __exit__(self, exc_type: Any, exc_value: Any, traceback: Any) -> None:
        Tensor.grad_enabled = self.previous

class Tensor:
    """A multidimensional array with automatic differentiation support."""
    
    # Class variables should be at the top
    grad_enabled: bool = True
    _array_ops = {
        'add': np.add,
        'mul': np.multiply,
        'matmul': np.matmul,
        'pow': np.power,
    }
    
    def __init__(
        self,
        data: TensorLike,
        _children: Tuple[Tensor, ...] = (),
        *,  # Force keyword arguments after this
        requires_grad: bool = False,
        dtype: np.dtype = np.float32,
    ) -> None:
        """Initializes a new Tensor.
        
        Args:
            data: The data to store in the tensor.
            _children: Child tensors in the computation graph (internal use).
            requires_grad: Whether to compute gradients for this tensor.
            dtype: The data type for the tensor.
            
        Raises:
            ValueError: If data cannot be converted to a numpy array.
        """
        try:
            self.data = (data.data if isinstance(data, Tensor) 
                        else np.array(data, dtype=dtype))
        except Exception as e:
            raise ValueError(f"Could not convert data to tensor: {e}") from e
            
        self.requires_grad = requires_grad
        self.grad = (np.zeros_like(self.data) if requires_grad and 
                    self.is_grad_enabled() else None)
        self._prev = set(_children)
        self.grad_fn: Optional[GradFn] = None
        self._shape = self.data.shape
        self.ndim = self.data.ndim

    def backward(self) -> None:
        """Computes gradients through backpropagation."""
        if not self.grad_enabled:
            raise GradientError("Gradient computation is disabled")
            
        if self.data.size > 1:
            raise GradientError("backward() can only be called on scalar tensors")
            
        if not self.requires_grad:
            raise GradientError("backward() called on tensor that doesn't require grad")
            
        topo: List[Tensor] = []
        visited = set()
        
        def build_topo(v: Tensor) -> None:
            if v not in visited:
                visited.add(v)
                for child in v._prev:
                    build_topo(child)
                topo.append(v)
                
        build_topo(self)
        self.grad = np.ones_like(self.data)
        
        for v in reversed(topo):
            if v.grad_fn is not None:
                v.grad_fn()

    @property
    def dtype(self) -> np.dtype:
        """Returns the data type of the tensor."""
        return self.data.dtype

    @property
    def is_leaf(self) -> bool:
        """Returns True if the tensor is a leaf node in the graph."""
        return len(self._prev) == 0

    def _preprocess_binop(self, other: Union[Tensor, float, int]) -> Tuple[Tensor, Tensor]:
        """Prepares two tensors for a binary operation.
        
        Args:
            other: The other tensor or scalar value.
            
        Returns:
            A tuple of (self, other) with compatible shapes.
            
        Raises:
            BroadcastError: If tensors cannot be broadcast together.
        """
        try:
            other = other if isinstance(other, Tensor) else Tensor(other)
            broadcast_shape = np.broadcast_shapes(self.shape, other.shape)
            return self.broadcast_to(broadcast_shape), other.broadcast_to(broadcast_shape)
        except ValueError as e:
            raise BroadcastError(f"Cannot broadcast shapes {self.shape} and {other.shape}") from e

    def _ensure_grad(self) -> None:
        """Ensures gradient exists for accumulation."""
        if self.requires_grad and self.grad is None:
            self.grad = np.zeros_like(self.data)

    def _accumulate_grad(self, grad: np.ndarray) -> None:
        """Safely accumulates gradients.
        
        Args:
            grad: The gradient to accumulate.
        """
        self._ensure_grad()
        self.grad = np.add(self.grad, grad)

    def __add__(self, other: TensorLike) -> Tensor:
        """Performs element-wise addition with broadcasting."""
        try:
            self, other = self._preprocess_binop(other)
            out = Tensor(self._array_ops['add'](self.data, other.data), _children=(self, other))

            if self.requires_grad or other.requires_grad:
                if self.is_grad_enabled():
                    def _add_backward() -> None:
                        if self.requires_grad:
                            self._accumulate_grad(out.grad)
                        if other.requires_grad:
                            other._accumulate_grad(out.grad)
                    out.grad_fn = _add_backward
                    out.requires_grad = True

            return out
        except Exception as e:
            raise OperationError(f"Addition failed: {str(e)}") from e

    def __mul__(self, other: TensorLike) -> Tensor:
        """Performs element-wise multiplication with broadcasting."""
        try:
            self, other = self._preprocess_binop(other)
            out = Tensor(self._array_ops['mul'](self.data, other.data), _children=(self, other))

            if self.requires_grad or other.requires_grad:
                if self.is_grad_enabled():
                    def _mul_backward() -> None:
                        if self.requires_grad:
                            self._accumulate_grad(other.data * out.grad)
                        if other.requires_grad:
                            other._accumulate_grad(self.data * out.grad)
                    out.grad_fn = _mul_backward
                    out.requires_grad = True

            return out
        except Exception as e:
            raise OperationError(f"Multiplication failed: {str(e)}") from e

    def reshape(self, *shape: int) -> Tensor:
        """Reshapes the tensor to the specified shape.
        
        Args:
            *shape: The new shape dimensions.
            
        Returns:
            A new tensor with the specified shape.
            
        Raises:
            ShapeError: If the new shape is invalid.
        """
        try:
            out = Tensor(self.data.reshape(shape), _children=(self,))

            if self.requires_grad and self.is_grad_enabled():
                def _reshape_backward() -> None:
                    self._accumulate_grad(out.grad.reshape(self.shape))
                out.grad_fn = _reshape_backward
                out.requires_grad = True

            return out
        except Exception as e:
            raise ShapeError(f"Reshape to {shape} failed: {str(e)}") from e

    def sum(self, axis: Optional[int] = None, keepdims: bool = False) -> Tensor:
        """Computes the sum along the specified axis.
        
        Args:
            axis: The axis along which to sum.
            keepdims: Whether to keep the summed dimensions.
            
        Returns:
            A new tensor containing the sum.
            
        Raises:
            OperationError: If the sum operation fails.
        """
        try:
            out = Tensor(np.sum(self.data, axis=axis, keepdims=keepdims), _children=(self,))

            if self.requires_grad and self.is_grad_enabled():
                def _sum_backward() -> None:
                    grad = out.grad
                    if not keepdims:
                        if axis is None:
                            grad = grad.reshape((1,) * self.ndim)
                        else:
                            shape = list(self.shape)
                            shape[axis] = 1
                            grad = grad.reshape(shape)
                    
                    size = self.data.size if axis is None else self.data.shape[axis]
                    self._accumulate_grad(np.broadcast_to(grad, self.shape) / size)
                    
                out.grad_fn = _sum_backward
                out.requires_grad = True

            return out
        except Exception as e:
            raise OperationError(f"Sum operation failed: {str(e)}") from e

    @classmethod
    def is_grad_enabled(cls) -> bool:
        """Returns whether gradient computation is enabled."""
        return cls.grad_enabled

    def zero_grad(self) -> None:
        """Zeros out the gradient."""
        if self.grad is not None:
            self.grad = np.zeros_like(self.data)

    def __repr__(self) -> str:
        """Returns a string representation of the tensor."""
        return f"Tensor(data={self.data}, requires_grad={self.requires_grad})"

    def __matmul__(self, other: TensorLike) -> Tensor:
        """Performs matrix multiplication.
        
        Args:
            other: The tensor to multiply with.
            
        Returns:
            The result of matrix multiplication.
            
        Raises:
            ShapeError: If matrix shapes are incompatible.
            OperationError: If the operation fails.
        """
        try:
            other = other if isinstance(other, Tensor) else Tensor(other)
            if self.ndim < 2 or other.ndim < 2:
                raise ShapeError("Matrix multiplication requires at least 2D tensors")
                
            out = Tensor(self._array_ops['matmul'](self.data, other.data), _children=(self, other))

            if self.requires_grad or other.requires_grad:
                if self.is_grad_enabled():
                    def _matmul_backward() -> None:
                        if self.requires_grad:
                            self._accumulate_grad(np.matmul(out.grad, other.data.T))
                        if other.requires_grad:
                            other._accumulate_grad(np.matmul(self.data.T, out.grad))
                    out.grad_fn = _matmul_backward
                    out.requires_grad = True

            return out
        except Exception as e:
            raise OperationError(f"Matrix multiplication failed: {str(e)}") from e

    def exp(self) -> Tensor:
        """Computes element-wise exponential.
        
        Returns:
            A new tensor with exponential of elements.
            
        Raises:
            OperationError: If the operation fails.
        """
        try:
            out = Tensor(np.exp(self.data), _children=(self,))

            if self.requires_grad and self.is_grad_enabled():
                def _exp_backward() -> None:
                    self._accumulate_grad(out.data * out.grad)
                out.grad_fn = _exp_backward
                out.requires_grad = True

            return out
        except Exception as e:
            raise OperationError(f"Exponential operation failed: {str(e)}") from e

    def log(self) -> Tensor:
        """Computes element-wise natural logarithm.
        
        Returns:
            A new tensor with logarithm of elements.
            
        Raises:
            OperationError: If the operation fails or input contains non-positive values.
        """
        try:
            if np.any(self.data <= 0):
                raise ValueError("Log of non-positive values")
                
            out = Tensor(np.log(self.data), _children=(self,))

            if self.requires_grad and self.is_grad_enabled():
                def _log_backward() -> None:
                    self._accumulate_grad(out.grad / self.data)
                out.grad_fn = _log_backward
                out.requires_grad = True

            return out
        except Exception as e:
            raise OperationError(f"Logarithm operation failed: {str(e)}") from e

    def mean(self, axis: Optional[int] = None, keepdims: bool = False) -> Tensor:
        """Computes mean along specified axis.
        
        Args:
            axis: The axis along which to compute mean.
            keepdims: Whether to keep reduced dimensions.
            
        Returns:
            A new tensor containing the mean values.
            
        Raises:
            OperationError: If the operation fails.
        """
        try:
            out = Tensor(np.mean(self.data, axis=axis, keepdims=keepdims), _children=(self,))

            if self.requires_grad and self.is_grad_enabled():
                def _mean_backward() -> None:
                    grad = out.grad
                    if not keepdims:
                        if axis is None:
                            grad = grad.reshape((1,) * self.ndim)
                        else:
                            shape = list(self.shape)
                            shape[axis] = 1
                            grad = grad.reshape(shape)
                    
                    size = self.data.size if axis is None else self.data.shape[axis]
                    self._accumulate_grad(np.broadcast_to(grad, self.shape) / size)
                    
                out.grad_fn = _mean_backward
                out.requires_grad = True

            return out
        except Exception as e:
            raise OperationError(f"Mean operation failed: {str(e)}") from e

    def max(self, axis: Optional[int] = None, keepdims: bool = False) -> Tensor:
        """Computes maximum values along specified axis.
        
        Args:
            axis: The axis along which to compute maximum.
            keepdims: Whether to keep reduced dimensions.
            
        Returns:
            A new tensor containing the maximum values.
            
        Raises:
            OperationError: If the operation fails.
        """
        try:
            out = Tensor(np.max(self.data, axis=axis, keepdims=keepdims), _children=(self,))

            if self.requires_grad and self.is_grad_enabled():
                def _max_backward() -> None:
                    grad = out.grad
                    if not keepdims:
                        if axis is None:
                            grad = grad.reshape((1,) * self.ndim)
                        else:
                            shape = list(self.shape)
                            shape[axis] = 1
                            grad = grad.reshape(shape)
                    
                    mask = (self.data == np.max(self.data, axis=axis, keepdims=True))
                    self._accumulate_grad(grad * mask)
                    
                out.grad_fn = _max_backward
                out.requires_grad = True

            return out
        except Exception as e:
            raise OperationError(f"Max operation failed: {str(e)}") from e

    def __pow__(self, power: Union[int, float]) -> Tensor:
        """Raises tensor elements to specified power.
        
        Args:
            power: The power to raise elements to.
            
        Returns:
            A new tensor with powered elements.
            
        Raises:
            OperationError: If the operation fails.
        """
        try:
            out = Tensor(self._array_ops['pow'](self.data, power), _children=(self,))

            if self.requires_grad and self.is_grad_enabled():
                def _pow_backward() -> None:
                    self._accumulate_grad(
                        power * self._array_ops['pow'](self.data, power - 1) * out.grad
                    )
                out.grad_fn = _pow_backward
                out.requires_grad = True

            return out
        except Exception as e:
            raise OperationError(f"Power operation failed: {str(e)}") from e

    @property
    def shape(self) -> Shape:
        """Returns the shape of the tensor."""
        return self._shape

    def item(self) -> Union[int, float]:
        """Returns the value of a scalar tensor.
        
        Returns:
            The scalar value.
            
        Raises:
            ValueError: If tensor is not a scalar.
        """
        if self.data.size != 1:
            raise ValueError("Only tensors with one element can be converted to scalar")
        return self.data.item()

    def T(self) -> Tensor:
        """Returns the transpose of the tensor.
        
        Returns:
            The transposed tensor.
            
        Raises:
            OperationError: If the transpose operation fails.
        """
        try:
            out = Tensor(np.transpose(self.data), _children=(self,))

            if self.requires_grad and self.is_grad_enabled():
                def _transpose_backward() -> None:
                    self._accumulate_grad(np.transpose(out.grad))
                out.grad_fn = _transpose_backward
                out.requires_grad = True

            return out
        except Exception as e:
            raise OperationError(f"Transpose operation failed: {str(e)}") from e

    def __radd__(self, other: TensorLike) -> Tensor:
        """Performs reverse addition."""
        return self + other

    def __rmul__(self, other: TensorLike) -> Tensor:
        """Performs reverse multiplication."""
        return self * other

    def __neg__(self) -> Tensor:
        """Returns the negation of the tensor."""
        return self * -1

    def __sub__(self, other: TensorLike) -> Tensor:
        """Performs subtraction."""
        return self + (-other)

    def __rsub__(self, other: TensorLike) -> Tensor:
        """Performs reverse subtraction."""
        return (-self) + other

    def __truediv__(self, other: TensorLike) -> Tensor:
        """Performs division."""
        return self * (other ** -1)

    def __rtruediv__(self, other: TensorLike) -> Tensor:
        """Performs reverse division."""
        return (self ** -1) * other

    def __len__(self) -> int:
        """Returns the length of the first dimension."""
        return len(self.data)

    def __getitem__(self, idx: Union[int, slice, Tuple]) -> Tensor:
        """Implements indexing for the tensor."""
        out = Tensor(self.data[idx], _children=(self,))

        if self.requires_grad and self.is_grad_enabled():
            def _getitem_backward() -> None:
                grad = np.zeros_like(self.data)
                grad[idx] = out.grad
                self._accumulate_grad(grad)
            out.grad_fn = _getitem_backward
            out.requires_grad = True

        return out

    def size(self, dim: Optional[int] = None) -> Union[Shape, int]:
        """Returns the size of the tensor.
        
        Args:
            dim: Optional dimension to get size of.
            
        Returns:
            Either the full shape or the size in a specific dimension.
        """
        if dim is not None:
            return self.shape[dim]
        return self.shape

    @staticmethod
    def broadcast_axis(left: Shape, right: Shape) -> Tuple[Tuple[int, ...], Tuple[int, ...]]:
        """Determines the axes along which broadcasting will occur.
        
        Args:
            left: Shape of the first tensor.
            right: Shape of the second tensor.
            
        Returns:
            Tuple of axes for left and right tensors that need broadcasting.
            
        Raises:
            BroadcastError: If shapes cannot be broadcast together.
        """
        try:
            ldim, rdim = len(left), len(right)
            maxdim = max(ldim, rdim)
            
            lshape = (1,) * (maxdim - ldim) + left
            rshape = (1,) * (maxdim - rdim) + right
            
            left_axes, right_axes = [], []
            
            for i in range(maxdim):
                if lshape[i] > rshape[i]:
                    right_axes.append(i)
                elif rshape[i] > lshape[i]:
                    left_axes.append(i)
                elif lshape[i] != 1 or rshape[i] != 1:
                    if lshape[i] != rshape[i]:
                        raise ValueError(f"Incompatible shapes at axis {i}: {lshape[i]} vs {rshape[i]}")
                        
            return tuple(left_axes), tuple(right_axes)
            
        except Exception as e:
            raise BroadcastError(f"Cannot determine broadcast axes: {str(e)}") from e

    def broadcast_to(self, shape: Shape) -> Tensor:
        """Broadcasts the tensor to a new shape.
        
        Args:
            shape: The target shape to broadcast to.
            
        Returns:
            A new tensor broadcast to the target shape.
            
        Raises:
            BroadcastError: If tensor cannot be broadcast to target shape.
        """
        try:
            data = np.broadcast_to(self.data, shape)
            out = Tensor(data, _children=(self,))
            
            if self.requires_grad and self.is_grad_enabled():
                # Get the axes that were broadcast
                broadcasted_axes = self.broadcast_axis(self.shape, shape)[0]
                
                def _broadcast_backward() -> None:
                    # Sum gradients along broadcast axes
                    grad = out.grad
                    if broadcasted_axes:
                        grad = np.sum(grad, axis=broadcasted_axes, keepdims=True)
                    # Reshape back to original shape
                    grad = np.reshape(grad, self.shape)
                    self._accumulate_grad(grad)
                    
                out.grad_fn = _broadcast_backward
                out.requires_grad = True
                
            return out
            
        except Exception as e:
            raise BroadcastError(f"Cannot broadcast to shape {shape}: {str(e)}") from e

    def set_requires_grad(self, requires_grad: bool) -> None:
        """Sets whether the tensor requires gradients.
        
        Args:
            requires_grad: If True, gradients will be computed for this tensor.
        """
        self.requires_grad = requires_grad
        if requires_grad and self.grad is None:
            self.grad = np.zeros_like(self.data)
        elif not requires_grad:
            self.grad = None
