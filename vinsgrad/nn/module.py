import numpy as np
from ..core import Tensor
from ..core.engine import no_grad, set_grad_enabled

class Module:
    def __init__(self):
        self._parameters = {}
        self._modules = {}
        self.training = True

    def train(self, mode=True):
        self.training = mode
        for module in self._modules.values():
            if isinstance(module, Module):
                module.train(mode)
        return self

    def eval(self):
        return self.train(False)

    def state_dict(self):
        state = {}
        for name, param in self._parameters.items():
            state[name] = param.data
        for name, module in self._modules.items():
            state[name] = module.state_dict()
        return state

    def load_state_dict(self, state_dict):
        for name, param in state_dict.items():
            if name in self._parameters:
                self._parameters[name].data = param
            elif name in self._modules:
                self._modules[name].load_state_dict(param)
            else:
                print(f"Unexpected key in state_dict: {name}")
    
    def _get_params(self):
        params = []
        for name, param in self._parameters.items():
            params.append((name, param))
        for name, module in self._modules.items():
            if isinstance(module, Module):
                params += module._get_params()
        return params
    
    def parameters(self):
        return [p for _, p in self._get_params() if p.requires_grad]
    
    def named_parameters(self):
        return self._get_params()
    
    def zero_grad(self):
        for _, param in self.named_parameters():
            if param.grad is not None:
                param.grad.fill(0)
    
    def add_module(self, name, module):
        self._modules[name] = module

    def forward(self, *args):
        raise NotImplementedError("forward method not implemented")

    def __call__(self, *args):
        if not self.training:
            with no_grad():
                return self.forward(*args)
        else:
            return self.forward(*args)
    
    def __setattr__(self, name, value):
        if isinstance(value, Tensor):
            self._parameters[name] = value
        elif isinstance(value, Module):
            self._modules[name] = value
        super().__setattr__(name, value)

    def set_grad_enabled(self, mode):
        """
        Explicitly set whether gradients should be computed for this module
        and all its submodules, regardless of training mode.
        """
        set_grad_enabled(mode)
        for module in self._modules.values():
            if isinstance(module, Module):
                module.set_grad_enabled(mode)