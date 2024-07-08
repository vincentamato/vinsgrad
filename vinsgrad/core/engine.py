import numpy as np
from typing import Tuple, Union, Optional, Callable, List, Any

def set_grad_enabled(mode: bool) -> bool:
    """
    Sets gradient computation to on or off globally.
    
    Args:
        mode (bool): True to enable gradients, False to disable.
    
    Returns:
        bool: The previous grad_enabled state.
    """
    previous = Tensor.grad_enabled
    Tensor.grad_enabled = mode
    return previous

class no_grad:
    """
    Context-manager that disables gradient calculation.
    """

    def __enter__(self) -> None:
        self.previous = Tensor.grad_enabled
        Tensor.grad_enabled = False

    def __exit__(self, exc_type: Any, exc_value: Any, traceback: Any) -> None:
        Tensor.grad_enabled = self.previous

class Tensor:
    """
    A class representing a tensor with automatic differentiation capabilities.
    """

    grad_enabled = True
    
    def __init__(self, data: Union[float, List, np.ndarray], 
                 _children: Tuple['Tensor', ...] = (), 
                 requires_grad: bool = False) -> None:
        """
        Initialize a Tensor object.

        Args:
            data (Union[float, List, np.ndarray]): The tensor data.
            _children (Tuple[Tensor, ...]): Child tensors (for autograd).
            requires_grad (bool): Whether the tensor requires gradients.
        """
        self.data = np.array(data, dtype=np.float32)
        self.requires_grad = requires_grad
        self.grad: Optional[np.ndarray] = np.zeros_like(self.data) if requires_grad and self.grad_enabled else None
        self._prev = set(_children)
        self.grad_fn: Optional[Callable] = None
        self._shape = self.data.shape
        self.ndim = self.data.ndim

    @property
    def shape(self) -> Tuple[int, ...]:
        """
        Returns the shape of the tensor.
        
        Returns:
            Tuple[int, ...]: The shape of the tensor.
        """
        return self._shape

    def item(self) -> Any:
        """
        Returns the value of the tensor as a standard Python scalar.
        
        Returns:
            Any: The scalar value.
        """
        return self.data.item()

    def backward(self) -> None:
        """
        Computes the gradients of the tensor by backpropagation.
        """
        if not self.grad_enabled:
            raise ValueError("Cannot backward when gradient calculation is disabled.")
        
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
      
    def set_requires_grad(self, requires_grad: bool) -> None:
        """
        Sets whether to compute gradients for the tensor.
        
        Args:
            requires_grad (bool): Flag indicating whether to compute gradients.
        """
        self.requires_grad = requires_grad
        if requires_grad and self.grad is None:
            self.grad = np.zeros_like(self.data)

    def zero_grad(self) -> None:
        """
        Sets the gradients of the tensor to zero.
        """
        if self.grad is not None:
            self.grad = np.zeros_like(self.data)

    @staticmethod
    def set_grad_enabled(mode: bool) -> bool:
        """
        Sets gradient computation to on or off globally.
        
        Args:
            mode (bool): True to enable gradients, False to disable.
        
        Returns:
            bool: The previous grad_enabled state.
        """
        return set_grad_enabled(mode)

    @staticmethod
    def is_grad_enabled() -> bool:
        """
        Checks whether gradient computation is enabled globally.
        
        Returns:
            bool: True if gradient computation is enabled, False otherwise.
        """
        return Tensor.grad_enabled

    @staticmethod
    def broadcast_axis(left: Tuple[int, ...], right: Tuple[int, ...]) -> Tuple[Tuple[int, ...], Tuple[int, ...]]:
        """
        Determines the axes along which broadcasting will occur for two shapes.
        
        Args:
            left (Tuple[int, ...]): The shape of the first tensor.
            right (Tuple[int, ...]): The shape of the second tensor.
        
        Returns:
            Tuple[Tuple[int, ...], Tuple[int, ...]]: The axes along which broadcasting will occur.
        """
        ldim = len(left)
        rdim = len(right)
        maxdim = max(ldim, rdim)

        lshape_new = (1, ) * (maxdim - ldim) + left
        rshape_new = (1, ) * (maxdim - rdim) + right

        assert len(lshape_new) == len(rshape_new)

        left_axes, right_axes = [], []

        for i in range(len(lshape_new)):
            if lshape_new[i] > rshape_new[i]:
                right_axes.append(i)
            elif rshape_new[i] > lshape_new[i]:
                left_axes.append(i)

        return tuple(left_axes), tuple(right_axes)\
        
    def broadcast_to(self, shape: Tuple[int, ...]) -> 'Tensor':
        """
        Broadcasts the tensor to a new shape.
        
        Args:
            shape (Tuple[int, ...]): The new shape.
        
        Returns:
            Tensor: The broadcasted tensor.
        """
        data = np.broadcast_to(self.data, shape)
        out = Tensor(data, _children=(self,))
        broadcasted_axes = self.broadcast_axis(self.shape, shape)[0]

        if self.requires_grad and Tensor.grad_enabled:
            def _broadcast_backward() -> None:
                grad = np.sum(out.grad, axis=broadcasted_axes, keepdims=True)
                grad = np.reshape(grad, self.shape)
                self.grad += grad
            
            out.grad_fn = _broadcast_backward
            out.set_requires_grad(True)

        return out

    def _preprocess_binop(self, other: Union['Tensor', float, int]) -> Tuple['Tensor', 'Tensor']:
        """
        Prepares two tensors for a binary operation by broadcasting them to a common shape.
        
        Args:
            other (Union[Tensor, float, int]): The other tensor or scalar.
        
        Returns:
            Tuple[Tensor, Tensor]: The two tensors with a common shape.
        """
        other = other if isinstance(other, Tensor) else Tensor(other)
        broadcast_shape = np.broadcast_shapes(self.shape, other.shape)
        self, other = self.broadcast_to(broadcast_shape), other.broadcast_to(broadcast_shape)
        return self, other

    def __add__(self, other: Union['Tensor', float, int]) -> 'Tensor':
        """
        Performs element-wise addition with broadcasting.
        
        Args:
            other (Union[Tensor, float, int]): The other tensor or scalar.
        
        Returns:
            Tensor: The result of the addition.
        """
        self, other = self._preprocess_binop(other)
        out = Tensor(self.data + other.data, _children=(self, other))

        if not self.requires_grad and not other.requires_grad:
            return out

        if Tensor.grad_enabled:
            def _add_backward() -> None:
                if self.requires_grad:
                    self.grad += out.grad
                if other.requires_grad:
                    other.grad += out.grad
            out.grad_fn = _add_backward
            out.set_requires_grad(True)
        
        return out

    def __mul__(self, other: Union['Tensor', float, int]) -> 'Tensor':
        """
        Performs element-wise multiplication with broadcasting.
        
        Args:
            other (Union[Tensor, float, int]): The other tensor or scalar.
        
        Returns:
            Tensor: The result of the multiplication.
        """
        self, other = self._preprocess_binop(other)
        out = Tensor(self.data * other.data, _children=(self, other))

        if not self.requires_grad and not other.requires_grad:
            return out

        if Tensor.grad_enabled:
            def _mul_backward() -> None:
                if self.requires_grad:
                    self.grad += other.data * out.grad
                if other.requires_grad:
                    other.grad += self.data * out.grad
            out.grad_fn = _mul_backward
            out.set_requires_grad(True)
        
        return out
    
    def __matmul__(self, other: Union['Tensor', np.ndarray]) -> 'Tensor':
        """
        Performs matrix multiplication.
        
        Args:
            other (Union[Tensor, np.ndarray]): The other tensor or ndarray.
        
        Returns:
            Tensor: The result of the matrix multiplication.
        """
        other = other if isinstance(other, Tensor) else Tensor(other)
        out = Tensor(self.data @ other.data, _children=(self, other))

        if not self.requires_grad and not other.requires_grad:
            return out
        
        if self.requires_grad and self.grad_enabled:
            def _matmul_backward():
                self.grad += out.grad @ other.data.T
                other.grad += self.data.T @ out.grad
            out.grad_fn = _matmul_backward
            out.set_requires_grad(True)
            
        return out
    
    def __pow__(self, other):
        """
        Raises the tensor to a power.
        
        Args:
            other (Union[int, float]): The power to raise the tensor to.
        
        Returns:
            Tensor: The result of the power operation.
        """
        out = Tensor(self.data ** other, (self,), requires_grad=self.requires_grad)

        if self.requires_grad and Tensor.grad_enabled:
            def _pow_backward() -> None:
                self.grad += other * self.data ** (other - 1) * out.grad
            out.grad_fn = _pow_backward
            out.set_requires_grad(True)

        return out
    
    

    def reshape(self, *shape: int) -> 'Tensor':
        """
        Changes the tensor's shape.
        
        Args:
            shape (int): The new shape dimensions.
        
        Returns:
            Tensor: The reshaped tensor.
        """
        out = Tensor(self.data.reshape(shape), _children=(self, ))

        if self.requires_grad and Tensor.grad_enabled:
            def _reshape_backward() -> None:
                self.grad += out.grad.reshape(self.data.shape)
            out.grad_fn = _reshape_backward
            out.set_requires_grad(True)
        
        return out
    
    def max(self, axis: Optional[int] = None, keepdims: bool = False) -> 'Tensor':
        """
        Returns the maximum of the tensor along a given axis.
        
        Args:
            axis (Optional[int]): The axis to compute the maximum along.
            keepdims (bool): Whether to keep the dimensions.
        
        Returns:
            Tensor: The maximum values.
        """
        out = Tensor(self.data.max(axis=axis, keepdims=keepdims), (self,), requires_grad=self.requires_grad)
        
        if self.requires_grad and Tensor.grad_enabled:
            def _max_grad() -> None:
                self.grad += (self.data == out.data) * out.grad
            out.grad_fn = _max_grad
            out.set_requires_grad(True)
        
        return out
    
    def exp(self) -> 'Tensor':
        """
        Computes the element-wise exponential of the tensor.
        
        Returns:
            Tensor: The exponential values.
        """
        out = Tensor(np.exp(self.data), _children=(self,))

        if self.requires_grad and Tensor.grad_enabled:
            def _exp_backward() -> None:
                self.grad += out.data * out.grad
            out.grad_fn = _exp_backward
            out.set_requires_grad(True)
        
        return out
    
    def log(self) -> 'Tensor':
        """
        Computes the element-wise natural logarithm of the tensor.
        
        Returns:
            Tensor: The logarithm values.
        """
        out = Tensor(np.log(self.data), _children=(self, ))

        if self.requires_grad and Tensor.grad_enabled:
            def _log_backward() -> None:
                self.grad += out.grad / self.data
            out.grad_fn = _log_backward
            out.set_requires_grad(True)
        
        return out
    
    def sum(self, axis: Optional[int] = None, keepdims: bool = False) -> 'Tensor':
        """
        Computes the sum of the tensor elements along a given axis.
        
        Args:
            axis (Optional[int]): The axis to sum along.
            keepdims (bool): Whether to keep the dimensions.
        
        Returns:
            Tensor: The sum of elements.
        """
        out = Tensor(np.sum(self.data, axis=axis, keepdims=keepdims), _children=(self, ))

        if self.requires_grad and Tensor.grad_enabled:
            def _sum_backward() -> None:
                if axis is None:
                    if keepdims:
                        grad = out.grad
                    else:
                        grad = out.grad.reshape((1,) * self.data.ndim)
                else:
                    if keepdims:
                        grad = out.grad
                    else:
                        shape = list(self.data.shape)
                        shape[axis] = 1
                        grad = out.grad.reshape(shape)
                self.grad += np.broadcast_to(grad, self.data.shape)
            out.grad_fn = _sum_backward
            out.set_requires_grad(True)
        
        return out
        
        return out
    
    def mean(self, axis: Optional[int] = None, keepdims: bool = False) -> 'Tensor':
        """
        Computes the mean of the tensor elements along a given axis.
        
        Args:
            axis (Optional[int]): The axis to compute the mean along.
            keepdims (bool): Whether to keep the dimensions.
        
        Returns:
            Tensor: The mean values.
        """
        out = Tensor(np.mean(self.data, axis=axis, keepdims=keepdims), (self,), requires_grad=self.requires_grad)
        
        def _mean_backward():
            if self.requires_grad:
                grad = out.grad
                if not keepdims:
                    if axis is None:
                        grad = grad.reshape((1,) * self.data.ndim)
                    else:
                        shape = list(self.data.shape)
                        shape[axis] = 1
                        grad = grad.reshape(shape)
                
                broadcast_shape = np.broadcast_shapes(self.data.shape, grad.shape)
                grad = np.broadcast_to(grad, broadcast_shape)
                self.grad += grad / (np.prod(self.data.shape) / np.prod(grad.shape))

        out.grad_fn = _mean_backward
        out.set_requires_grad(True)

        return out
    
    def T(self, axes: Optional[Tuple[int, ...]] = None) -> 'Tensor':
        """
        Transposes the tensor along the given axes.
        
        Args:
            axes (Optional[Tuple[int, ...]]): The axes to transpose.
        
        Returns:
            Tensor: The transposed tensor.
        """
        out = Tensor(np.transpose(self.data, axes=axes), _children=(self, ))

        if self.requires_grad and Tensor.grad_enabled:
            def _transpose_backward() -> None:
                self.grad += np.transpose(out.grad, axes=axes)
            out.grad_fn = _transpose_backward
            out.set_requires_grad(True)
        
        return out
    
    def __hash__(self) -> int:
        """
        Returns a hash value for the tensor, based on its unique id.
        
        Returns:
            int: The hash value of the tensor.
        """
        return id(self)

    def __eq__(self, other: object) -> bool:
        """
        Checks if this tensor is equal to another tensor based on their unique ids.
        
        Args:
            other (object): The other tensor to compare with.
        
        Returns:
            bool: True if the tensors are equal, False otherwise.
        """
        if not isinstance(other, Tensor):
            return False
        return id(self) == id(other)

    
    def __neg__(self) -> 'Tensor':
        """
        Negates the tensor.
        
        Returns:
            Tensor: The negated tensor.
        """
        return self * -1
    
    def __sub__(self, other: Union['Tensor', float, int]) -> 'Tensor':
        """
        Performs element-wise subtraction with broadcasting.
        
        Args:
            other (Union[Tensor, float, int]): The other tensor or scalar.
        
        Returns:
            Tensor: The result of the subtraction.
        """
        return self + (-other)
    
    def __rsub__(self, other: Union['Tensor', float, int]) -> 'Tensor':
        """
        Performs element-wise reverse subtraction with broadcasting.
        
        Args:
            other (Union[Tensor, float, int]): The other tensor or scalar.
        
        Returns:
            Tensor: The result of the reverse subtraction.
        """
        return -self + other
    
    def __radd__(self, other: Union['Tensor', float, int]) -> 'Tensor':
        """
        Performs element-wise reverse addition with broadcasting.
        
        Args:
            other (Union[Tensor, float, int]): The other tensor or scalar.
        
        Returns:
            Tensor: The result of the reverse addition.
        """
        return self + other

    def __rmul__(self, other: Union['Tensor', float, int]) -> 'Tensor':
        """
        Performs element-wise reverse multiplication with broadcasting.
        
        Args:
            other (Union[Tensor, float, int]): The other tensor or scalar.
        
        Returns:
            Tensor: The result of the reverse multiplication.
        """
        return self * other

    def __truediv__(self, other: Union['Tensor', float, int]) -> 'Tensor':
        """
        Performs element-wise division with broadcasting.
        
        Args:
            other (Union[Tensor, float, int]): The other tensor or scalar.
        
        Returns:
            Tensor: The result of the division.
        """
        return self * (other ** -1)
    
    def __rtruediv__(self, other: Union['Tensor', float, int]) -> 'Tensor':
        """
        Performs element-wise reverse division with broadcasting.
        
        Args:
            other (Union[Tensor, float, int]): The other tensor or scalar.
        
        Returns:
            Tensor: The result of the reverse division.
        """
        return (self ** -1) * other
    
    def __len__(self) -> int:
        """
        Returns the number of elements along the first axis of the tensor.
        
        Returns:
            int: The length of the tensor.
        """
        return len(self.data)
    
    def __getitem__(self, indices: Union[int, slice, Tuple[Union[int, slice], ...]]) -> 'Tensor':
        """
        Gets a subset of the tensor using indices.
        
        Args:
            indices (Union[int, slice, Tuple[Union[int, slice], ...]]): The indices to select.
        
        Returns:
            Tensor: The subset of the tensor.
        """
        out = Tensor(self.data[indices], _children=(self, ), requires_grad=self.requires_grad)

        if self.requires_grad and Tensor.grad_enabled:
            def _getitem_backward() -> None:
                self.grad[indices] += out.grad
            out.grad_fn = _getitem_backward
            out.set_requires_grad(True)
        
        return out
    
    def __repr__(self) -> str:
        """
        Returns a string representation of the tensor.
        
        Returns:
            str: The string representation of the tensor.
        """
        return f"Tensor({self.data}, requires_grad={self.requires_grad})"
