from abc import ABC
import numpy as np

from nea.ml.autograd import Tensor, to_tensor, TensorFunction
from nea.ml.autograd.consts import Tensorable


class Parameter(Tensor):
    """
    Represents a parameter
    """
    def __init__(self, shape: tuple[int] = None, 
                 requires_grad: bool = True, 
                 operation: TensorFunction = None) -> None:
        if shape is not None:
            data = np.random.randn(*shape)
            super().__init__(data, requires_grad=requires_grad, operation=operation)
        else:
            raise ValueError("shape must be specified and cannot be left as None")
        
    def set_data(self, data: Tensorable) -> None:
        """Allows user to set the data in parameters

        Args:
            data (Tensorable): data for the parameter to use
        """
        self._data = data


class Module(ABC):
    """Basis of all layers"""
    def __call__(self, x: Tensorable) -> Tensor:
        return self.forward(to_tensor(x))
    
    def forward(self, x: Tensorable) -> Tensor:
        """Forward propogation of module

        Args:
            x (Tensorable): Input data

        Returns:
            Tensor: 
        """
        raise NotImplementedError("Cannot call forward on raw module")


class Dense(Module):
    """Fully connected layer

    Args:
        Module (_type_): 
    """
    def __init__(self, n_inputs: int, n_outputs: int, bias: bool = True) -> None:
        """Instantiates a new dense layer

        Args:
            n_inputs (int): number of inputs to layer
            n_outputs (int): desired number of output neurons
            bias (bool, optional): whether to add a bias layer or not. Defaults to True.
        """
        super().__init__()
        self.weights = Parameter((n_inputs, n_outputs))
        if bias:
            self.bias = Parameter((n_outputs, ))

    def forward(self, x: Tensorable) -> Tensor:
        """Propogates input data through dense layer

        Args:
            x (Tensorable): input data

        Returns:
            Tensor: 
        """
        y = x @ self.weights
        if self.bias:
            y = y + self.bias
        
        return y