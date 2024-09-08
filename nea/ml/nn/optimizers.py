from abc import ABC

from nea.ml.autograd import Tensor
from nea.ml.nn import Parameter

import numpy as np


class Optimizer(ABC):
    def __init__(
        self,
        params: list[Tensor | Parameter],
        lr: float = 0.001,
        regulization: float = 0,
    ) -> None:
        self.params = params
        self.lr = lr
        self.regulization = regulization

    def step(self) -> None:
        """
        Updates each parameters value depending on gradients
        """
        raise NotImplementedError("Cannot call step on base 'Optimizer' class")

    def zero_grad(self) -> None:
        """
        Resets the gradient of each parameter
        """
        for param in self.params:
            param.zero_grad()


class SGD(Optimizer):
    def __init__(
        self,
        params: list[Tensor | Parameter],
        lr: float = 0.001,
        regulization: float = 0,
    ) -> None:
        super().__init__(params, lr, regulization)

    def step(self) -> None:
        for param in self.params:
            param._data = (
                param._data
                - (self.lr * param.grad)
                - (self.lr * self.regulization * param._data)
            )
