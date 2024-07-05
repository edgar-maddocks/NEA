import numpy as np

from typing import List, Tuple
from consts import *

# TENSOR


def to_tensor(d):
    if isinstance(d, Tensor):
        return d
    else:
        return Tensor(d)


class Tensor:

    def __init__(
        self,
        data: Tensorable,
        requires_grad: bool = False,
        operation: "TensorFunction" = None,
    ) -> None:
        self._data: np.ndarray = np.array(data)
        self.shape = self._data.shape
        self.operation: TensorFunction = operation

        self.requires_grad: bool = requires_grad

        if self.requires_grad:
            self.grad = np.zeros_like(self._data)

        self.children: List[Tensor] = []
        self.parents: List[Tensor] = []

    @property
    def data(self) -> np.ndarray:
        return self._data

    def __repr__(self) -> str:
        return f"Tensor({self._data}, shape = {self.shape})"

    @staticmethod
    def zeros(shape: Tuple[int], requires_grad: bool = False) -> "Tensor":
        return Tensor(np.zeros(shape), requires_grad=requires_grad)

    @staticmethod
    def ones(shape: Tuple[int], requries_grad: bool = False) -> "Tensor":
        return Tensor(np.ones(shape), requires_grad=requries_grad)

    def backward(self, dy=None, y=None) -> None:
        """Reverse searches the computational graph, computing and updating parent gradients as it goes

        Args:
            dy (Tensor, optional): _description_. Defaults to None.
            y (Tensor, optional): _description_. Defaults to None.

        Returns:
            None
        """
        if self.requires_grad is False:
            return "This tensor has requires_grad set to False"

        if dy is None:
            dy = np.ones_like(self._data)

        if y is not None:
            self.children.remove(y)

        self.grad += dy

        if self.operation:
            if not self.children:
                self.operation.backward(self.grad, self)

    def zero_grad(self) -> None:
        self.grad = np.zeros_like(self._data)

    def __add__(self, other: Tensorable):
        add_op = Add()
        return add_op.forward(self, to_tensor(other))

    def __radd__(self, other: Tensorable):
        add_op = Add()
        return add_op.forward(self, to_tensor(other))

    def __iadd__(self, other: Tensorable):
        add_op = Add()
        return add_op.forward(self, to_tensor(other))


## TENSOR FUNCTIONS


class TensorFunction:
    def forward():
        raise NotImplementedError

    def backward():
        raise NotImplementedError


class Add(TensorFunction):
    def forward(self, a: Tensor, b: Tensor) -> Tensor:
        """Computes the addition of two tensors

        Args:
            a (Tensor): one tensor to add
            b (Tensor): other tensor to add

        Returns:
            Tensor: tensor where data is addition of parents
        """
        new_data = a._data + b._data

        requires_grad = a.requires_grad or b.requires_grad

        y = Tensor(new_data, requires_grad=requires_grad, operation=self)

        self.parents = (a, b)
        a.children.append(y)
        b.children.append(y)

        self._cache: Tuple[Tensor] = (a, b)

        return y

    def backward(self, dy: np.ndarray, y: Tensor) -> None:
        """Computes gradients for tensors stored in cache

        Args:
            dy (np.ndarray): gradient from previous backward (to be used with chain rule)
            y (Tensor): output tensor
        """
        a, b = self._cache  # get tensors used to create output

        if a.requires_grad:
            da = dy  # 1 * whatever the previous gradient is due to chain rule

            # now need to sum out broadcasted dimensions from numpy
            # make da the same shape as a
            n_dims_da = len(dy.shape)
            n_dims_a = len(a.shape)
            for dim in range(n_dims_da - n_dims_a):
                da = da.sum(axis=0)

            for n, dim in enumerate(a.shape):
                if dim == 1:
                    da = da.sum(axis=n, keepdims=True)
            a.backward(da, y)

        if b.requires_grad:
            db = dy

            # Rescale gradient to have the same shape as "b":
            n_dims_db = len(da.shape)
            n_dums_b = len(b.shape)
            for dim in range(n_dims_db - n_dums_b):
                db = db.sum(axis=0)

            for n, dim in enumerate(b.shape):
                if dim == 1:
                    db = db.sum(axis=n, keepdims=True)
            b.backward(db, y)


class Neg:
    def forward(self, a: Tensor) -> Tensor:
        """Computes the negation of a tensor

        Args:
            a (Tensor): tensor to be negated

        Returns:
            Tensor: negated tensor
        """
        new_data = -a._data

        y = Tensor(new_data, requires_grad=a.requires_grad, operation=self)

        self.parents = (a,)
        a.children.append(y)

        self._cache: Tuple[Tensor] = (a,)

        return y

    def backward(self, dy: np.ndarray, y: Tensor) -> None:
        pass
