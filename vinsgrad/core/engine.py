import numpy as np

class no_grad:
    """
    Context-manager that disables 
    gradient calculation.
    """

    def __enter__(self):
        self.previous = Tensor.grad_enabled
        Tensor.grad_enabled = False

    def __exit__(
            self, exc_type, 
            exc_value, traceback
        ):
        Tensor.grad_enabled = self.previous

def set_grad_enabled(mode):
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

def argmax(tensor, axis):
    """
    Returns the indices of the maximum values along an axis.
    
    Args:
        tensor (Tensor): The input tensor.
        axis (int): The axis along which to find the maximum value.
    
    Returns:
        Tensor: A tensor of indices.
    """
    return np.argmax(tensor.data, axis=axis)

class Tensor:

    grad_enabled = True
    
    def __init__(self, data, _children=(), requires_grad=False):
        self.data = np.array(data, dtype=np.float32)
        self.requires_grad = requires_grad
        self.grad = np.zeros_like(self.data) if requires_grad is True else None
        self._prev = set(_children)
        self.grad_fn = None
        self._shape = self.data.shape
        self.ndim = self.data.ndim

    def backward(self):
        if not self.grad_enabled:
            raise ValueError("cannot backward when gradient calculation is disabled.")
        topo = []
        visited = set()
        
        def build_topo(v):
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
      
    def set_requires_grad(self, requires_grad):
        self.requires_grad = requires_grad
        if requires_grad and self.grad is None:
            self.grad = np.zeros_like(self.data)

    def zero_grad(self):
        self.grad = np.zeros_like(self.data)

    @staticmethod
    def set_grad_enabled(mode):
        return set_grad_enabled(mode)

    @staticmethod
    def is_grad_enabled():
        return Tensor.grad_enabled

    @staticmethod
    def broadcast_axis(left, right):
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

        return tuple(left_axes), tuple(right_axes)

    def _preprocess_binop(self, other):
        other = other if isinstance(other, Tensor) else Tensor(other)

        broadcast_shape = np.broadcast_shapes(self.shape, other.shape)
        self, other = self.broadcast_to(broadcast_shape), other.broadcast_to(broadcast_shape)
        return self, other
        
    def broadcast_to(self, shape):
        data = np.broadcast_to(self.data, shape)
        out = Tensor(data, _children=(self,))
        broadcasted_axes = self.broadcast_axis(self.shape, shape)[0]

        if self.requires_grad and self.grad_enabled:
            def _broadcast_backward():
                grad = np.sum(out.grad, axis=broadcasted_axes, keepdims=True)
                grad = np.reshape(grad, self.shape)
                self.grad += grad
            
            out.grad_fn = _broadcast_backward
            out.set_requires_grad(True)

        return out

    def __add__(self, other):
        """
        elementwise add (takes broadcasting into account)
        """
        self, other = self._preprocess_binop(other)

        out = Tensor(self.data + other.data, _children=(self, other))

        if self.requires_grad == False and other.requires_grad == False:
            return out
        if self.grad_enabled:
            def _add_backward():
                if self.requires_grad:
                    self.grad += out.grad
                if other.requires_grad:
                    other.grad += out.grad
            out.grad_fn = _add_backward
            out.set_requires_grad(True)
            return out

    def __mul__(self, other):
        """
        element wise multiply (takes broadcasting into account)
        """

        self, other = self._preprocess_binop(other)
        
        out = Tensor(self.data * other.data, _children=(self, other))
            
        if self.requires_grad == False and other.requires_grad == False:
            return out
        if self.grad_enabled:
            def _mul_backward():
                if self.requires_grad:
                    if self.grad is None:
                        print("self grad is none")
                    self.grad += other.data * out.grad
                if other.requires_grad:
                    other.grad += self.data * out.grad
            out.grad_fn = _mul_backward
            out.set_requires_grad(True)
            return out
    
    def __pow__(self, other):
        assert isinstance(other, (int, float)), "only supporting int/float powers for now"
        out = Tensor(self.data ** other, (self,), requires_grad=self.requires_grad)

        if self.requires_grad and self.grad_enabled:
            def _pow_backward():
                self.grad += other * self.data ** (other - 1) * out.grad
            out.grad_fn = _pow_backward
            out.set_requires_grad(True)
        return out
    
    def __matmul__(self, other):
        """
        matrix multiplication with tensors
        """

        other = other if isinstance(other, Tensor) else Tensor(other)
        
        out = Tensor(self.data @ other.data, _children=(self, other),)
        if not self.requires_grad and not other.requires_grad:
            return out

        if self.grad_enabled:
            # for 1D tensors, expand first dimension for first tensor
            # and expand last dimension for second tensor
            # example: (3,) @ (3,) becomes (1, 3) and (3, 1)
            # which is compatible for matrix multiplication
            le_axis = (0, ) if self.data.ndim == 1 else ()
            re_axis = (-1, ) if other.data.ndim == 1 else ()

            # resultant tensor's grad should be expanded by both le_axis and re_axis
            rese_axis = le_axis + re_axis

            # we need to take broadcasting into account
            # except last two dimensions of shape (since they will be used for matmul)
            # gradients will be summed along the broadcasted axes for both tensors
            l, r = self.broadcast_axis(self.data.shape[:-2], other.data.shape[:-2])

            # for 2D (can be generalized for more dimensions too):
            #
            # self.grad = out.grad @ other.data.T
            # other.grad = self.data.T @ out.grad

            def _matmul_backward():
                if self.requires_grad:
                    self.grad = np.reshape(
                        np.sum(
                            np.expand_dims(out.grad, axis=rese_axis) @
                            np.expand_dims(other.data, axis=re_axis).swapaxes(-1, -2),
                            axis = l
                        ),
                        self.data.shape
                    )
                if other.requires_grad:
                    other.grad = np.reshape(
                        np.sum(
                            np.expand_dims(self.data, axis=le_axis).swapaxes(-1, -2) @
                            np.expand_dims(out.grad, axis=rese_axis),
                            axis = r
                        ),
                        other.data.shape
                    )

            out.grad_fn = _matmul_backward
            out.set_requires_grad(True)

        return out

    def reshape(self, *shape):
        """
        change the tensor's shape
        """
        out = Tensor(
            self.data.reshape(shape),
            _children=(self, ),
        )

        if self.requires_grad and self.grad_enabled:
            def _reshape_backward():
                self.grad += out.grad.reshape(self.data.shape)

            out.grad_fn = _reshape_backward
            out.set_requires_grad(True)
    
    def max(self, axis=None, keepdims=False):
        out = Tensor(self.data.max(axis=axis, keepdims=keepdims), (self,), requires_grad=self.requires_grad)
        
        def _max_grad():
            self.grad += (self.data == out.data) * out.grad
        
        out.grad_fn = _max_grad
        out.set_requires_grad(True) 
        return out
    
    def exp(self):
        out = Tensor(np.exp(self.data), _children=(self,))
        
        if self.requires_grad and self.grad_enabled:
            def _exp_backward():
                self.grad += out.data * out.grad

            out.grad_fn = _exp_backward
            out.set_requires_grad(True)
        
        return out
    
    def log(self):
        """
        log base e of the tensor
        """
        
        out = Tensor(np.log(self.data), _children=(self, ),)

        # since d/dx (log x) = 1 / x
        if self.requires_grad and self.grad_enabled:
            def _log_backward():
                self.grad += (out.grad / self.data)

            out.grad_fn = _log_backward
            out.set_requires_grad(True)
        
        return out
    
    def sum(self, axis = None, keepdims: bool = False):
        """
        sum values of tensor along given axes
        """
        out = Tensor(
            np.sum(self.data, axis=axis, keepdims=keepdims),
            _children=(self, ),
        )

        if self.requires_grad and self.grad_enabled:
            ex_axis = axis if axis and not keepdims else None

            def _sum_backward():
                if ex_axis:
                    self.grad += np.ones_like(self.grad) * np.expand_dims(
                        out.grad, axis=ex_axis
                    )
                else:
                    self.grad += np.ones_like(self.grad) * out.grad

            out.grad_fn = _sum_backward
            out.set_requires_grad(True)

        return out
    
    def mean(self, axis=None, keepdims=False):
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

    def item(self):
        return self.data.item()
    
    def T(self, axes=None):
        """
        transposes a given tensor along the given axes
        """

        out = Tensor(
            np.transpose(self.data, axes=axes),
            _children=(self, ),
        )

        if self.requires_grad and self.grad_enabled:
            def _transpose_backward():
                self.grad += np.transpose(out.grad, axes=axes)
            
            out.grad_fn = _transpose_backward
            out.set_requires_grad(True)
        
        return out

    @property
    def shape(self):
        return self._shape
    
    def __neg__(self):
        return self * -1
    
    def __sub__(self, other):
        return self + (-other)
    
    def __rsub__(self, other):
        return -self + other
    
    def __radd__(self, other):
        return self + other

    def __rmul__(self, other):
        return self * other

    def __truediv__(self, other):
        return self * (other ** -1)
    
    def __rtruediv__(self, other):
        return (self ** -1) * other
    
    def __len__(self):
        return len(self.data)
    
    def __getitem__(self, indices):
        """
        Get a subset of the tensor using indices.
        """

        out = Tensor(
            self.data[indices],
            _children=(self, ), 
            requires_grad=self.requires_grad
        )

        if self.requires_grad and self.grad_enabled:
            def _getitem_backward():
                self.grad[indices] += out.grad

            out._backward = _getitem_backward
            out.requires_grad = True

        return out
    
    def __repr__(self) -> str:
        return f"Tensor({self.data}, requires_grad={self.requires_grad})"
