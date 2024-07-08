from typing import Dict, List, Tuple, Any
from ..core import Tensor
from ..core.engine import no_grad, set_grad_enabled

class Module:
    """
    Base class for all neural network modules.
    
    Your models should also subclass this class.
    Modules can also contain other Modules, allowing to nest them in
    a tree structure. You can assign the submodules as regular attributes.
    """

    def __init__(self):
        """
        Initializes internal Module state, shared by all modules.
        """
        self._parameters: Dict[str, Tensor] = {}
        self._modules: Dict[str, 'Module'] = {}
        self.training: bool = True

    def train(self, mode: bool = True) -> 'Module':
        """
        Sets the mode of this module and all its descendent modules to training mode.

        Args:
            mode (bool): whether to set training mode (True) or evaluation mode (False). Default: True.

        Returns:
            Module: self
        """
        self.training = mode
        for module in self._modules.values():
            if isinstance(module, Module):
                module.train(mode)

        return self

    def eval(self) -> 'Module':
        """
        Sets the mode of this module and all its descendent modules to evaluation mode.

        Returns:
            Module: self
        """

        return self.train(False)

    def state_dict(self) -> Dict[str, Any]:
        """
        Returns a dictionary containing a whole state of the module.

        Returns:
            dict: a dictionary containing a whole state of the module
        """
        state: Dict[str, Any] = {}
        for name, param in self._parameters.items():
            state[name] = param.data
        for name, module in self._modules.items():
            state[name] = module.state_dict()

        return state

    def load_state_dict(self, state_dict: Dict[str, Any]) -> None:
        """
        Copies parameters and buffers from state_dict into this module and its descendants.

        Args:
            state_dict (dict): a dict containing parameters and persistent buffers.
        """
        for name, param in state_dict.items():
            if name in self._parameters:
                self._parameters[name].data = param
            elif name in self._modules:
                self._modules[name].load_state_dict(param)
            else:
                print(f"Unexpected key in state_dict: {name}")
    
    def _get_params(self) -> List[Tuple[str, Tensor]]:
        """
        Internal method to get all parameters of this module and its descendants.

        Returns:
            List[Tuple[str, Tensor]]: A list of tuples containing the name and Tensor of each parameter.
        """
        params: List[Tuple[str, Tensor]] = []
        for name, param in self._parameters.items():
            params.append((name, param))
        for name, module in self._modules.items():
            if isinstance(module, Module):
                params += module._get_params()

        return params
    
    def parameters(self) -> List[Tensor]:
        """
        Returns an iterator over module parameters.

        This includes only parameters with requires_grad=True.

        Returns:
            List[Tensor]: a list of all parameters in the module.
        """
        return [p for _, p in self._get_params() if p.requires_grad]
    
    def named_parameters(self) -> List[Tuple[str, Tensor]]:
        """
        Returns an iterator over module parameters, yielding both the name of the parameter
        as well as the parameter itself.

        Returns:
            List[Tuple[str, Tensor]]: a list of tuples containing the name and Tensor of each parameter.
        """
        return self._get_params()
    
    def zero_grad(self) -> None:
        """
        Sets gradients of all model parameters to zero.
        """
        for _, param in self.named_parameters():
            if param.grad is not None:
                param.grad.fill(0)

    def forward(self, *args: Any) -> Any:
        """
        Defines the computation performed at every call.

        Should be overridden by all subclasses.

        Args:
            *args: The input to the module.

        Raises:
            NotImplementedError: if not overridden by subclass.
        """
        raise NotImplementedError("forward method not implemented")

    def __call__(self, *args: Any) -> Any:
        """
        Calls forward() with the given arguments.

        If the module is in evaluation mode, gradient computation is disabled.

        Args:
            *args: The input to the module.

        Returns:
            Any: The output of the module's forward pass.
        """
        if not self.training:
            with no_grad():
                return self.forward(*args)
        else:
            return self.forward(*args)
    
    def __setattr__(self, name: str, value: Any) -> None:
        """
        Sets an attribute on the module.

        If the value is a Tensor, it's added to _parameters.
        If the value is a Module, it's added to _modules.

        Args:
            name (str): Name of the attribute.
            value (Any): Value of the attribute.
        """
        if isinstance(value, Tensor):
            self._parameters[name] = value
        elif isinstance(value, Module):
            self._modules[name] = value
        super().__setattr__(name, value)

    def set_grad_enabled(self, mode: bool) -> None:
        """
        Explicitly set whether gradients should be computed for this module
        and all its submodules, regardless of training mode.

        Args:
            mode (bool): whether to enable gradient computation (True) or disable it (False).
        """
        set_grad_enabled(mode)
        for module in self._modules.values():
            if isinstance(module, Module):
                module.set_grad_enabled(mode)