from __future__ import annotations
from abc import ABC
from copy import deepcopy

import numpy as np

from .consts import Tensorable
from .convolve_funcs import cpu_forward_convolve2d, cpu_k_backward_convolve2d

# ========
#  TENSOR
# ========


def to_tensor(d: Tensorable | Tensor) -> Tensor:
    """Converts data to tensor if not already

    Args:
        d (Tensorable): data

    Returns:
        Tensor: Tensor of data
    """
    if isinstance(d, Tensor):
        return d
    elif isinstance(d, Tensorable):
        return Tensor(d)


class Tensor:
    """
    ====================
        Tensor class
    ====================
    """

    def __init__(
        self,
        data: Tensorable,
        requires_grad: bool = False,
        operation: TensorFunction = None,
    ) -> None:
        self._data: np.ndarray = np.array(data)
        self.shape = self._data.shape
        self.operation: TensorFunction = operation

        self.requires_grad: bool = requires_grad

        if self.requires_grad:
            self.grad = np.zeros_like(self._data)

        self.children: list[Tensor] = []
        self.parents: list[Tensor] = []

    @property
    def data(self) -> np.ndarray:
        """Returns protected data attribute

        Returns:
            np.ndarray: data in tensor
        """
        return self._data
    
    @property
    def T(self) -> Tensor:
        """Returns a copy of the tensor, however the data has been transposed

        Returns:
            Tensor: _description_
        """
        copy = deepcopy(self)
        copy._data = self._data.T
        return copy

    def __repr__(self) -> str:
        return f"Tensor({self._data}, shape = {self.shape})"

    @staticmethod
    def zeros(shape: tuple[int], requires_grad: bool = False) -> Tensor:
        """Returns a tensor of 1s

        Args:
            shape (tuple[int]): desired shape of tensor
            requires_grad (bool, optional): if the tensors requires_grad property should be false. 
            Defaults to False.

        Returns:
            Tensor: tensor of 1s
        """
        return Tensor(np.zeros(shape), requires_grad=requires_grad)

    @staticmethod
    def ones(shape: tuple[int], requries_grad: bool = False) -> Tensor:
        """Returns a tensor of 0s

        Args:
            shape (tuple[int]): desired shape of tensor
            requires_grad (bool, optional): if the tensors requires_grad property should be false. 
            Defaults to False.

        Returns:
            Tensor: tensor of 0s
        """
        return Tensor(np.ones(shape), requires_grad=requries_grad)

    def backward(self, dy=None, y=None) -> None:
        """Reverse searches the computational graph, computing and updating parent 
        gradients as it goes

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

        self.grad = self.grad + dy

        if self.operation:
            if not self.children:
                self.operation.backward(self.grad, self)

    def zero_grad(self) -> None:
        """
        Zeros out the gradient of the tensor
        """
        self.grad = np.zeros_like(self._data)

    def __eq__(self, other: Tensorable):
        if self.data == other:
            return True

    def __add__(self, other: Tensorable) -> Tensor:
        add_op = Addition()
        return add_op.forward(self, to_tensor(other))

    def __radd__(self, other: Tensorable) -> Tensor:
        add_op = Addition()
        return add_op.forward(self, to_tensor(other))

    def __iadd__(self, other: Tensorable) -> Tensor:
        add_op = Addition()
        return add_op.forward(self, to_tensor(other))

    def __neg__(self):
        neg_op = Negation()
        return neg_op.forward(self)

    def __sub__(self, other: Tensorable) -> Tensor:
        return self + -to_tensor(other)

    def __rsub__ (self, other: Tensorable) -> Tensor:
        return to_tensor(other) + -self

    def __isub__(self, other: Tensorable) -> Tensor:
        return self + -to_tensor(other)

    def __mul__(self, other: Tensorable) -> Tensor:
        mul_op = Multiplication()
        return mul_op.forward(self, to_tensor(other))

    def __rmul__(self, other: Tensorable) -> Tensor:
        mul_op = Multiplication()
        return mul_op.forward(to_tensor(other), self)

    def __imul__(self, other: Tensorable) -> Tensor:
        mul_op = Multiplication()
        return mul_op.forward(self, to_tensor(other))

    def __truediv__(self, other: Tensorable) -> Tensor:
        div_op = Division()
        return div_op.forward(self, to_tensor(other))

    def __matmul__(self, other: Tensorable) -> Tensor:
        matmul_op = MatrixMultiplication()
        return matmul_op.forward(self, to_tensor(other))

    def __pow__(self, other: Tensorable) -> Tensor:
        pow_op = Power()
        return pow_op.forward(self, to_tensor(other))

    def mean(self) -> Tensor:
        """Computes the mean of the tensor

        Returns:
            Tensor: _description_
        """
        mean_op = Mean()
        return mean_op.forward(self)
    
    def sum(self, axis: int = -1, keepdims: bool = False) -> Tensor:
        """Computes sum of a tensor

        Args:
            axis (int, optional): axis to sum across. Defaults to -1.
            keepdims (bool, optional): reduce summed axis to 1?. Defaults to False.

        Returns:
            Tensor: 
        """
        sum_op = Sum()
        return sum_op.forward(self, axis=axis, keepdims=keepdims)
    
    def log(self) -> Tensor:
        """Computes element wise log of tensor

        Returns:
            Tensor: _description_
        """
        log_op = Log()
        return log_op.forward(self)
    
    def exp(self) -> Tensor:
        """e^

        Returns:
            Tensor: 
        """
        exp_op = Exp()
        return exp_op.forward(self)
    
    def convolve2d(self, k: Tensorable, b: Tensorable = None) -> Tensor:
        """2D convolutional layer of the tensor

        Args:
            k (Tensor): kernel to use
            b (Tensor, optional): bias to use. Defaults to None.

        Returns:
            Tensor: _description_
        """
        conv_op = Convolve2D()
        return conv_op.forward(self, k, b=b)


# ==================
## TENSOR FUNCTIONS
# ==================


class TensorFunction(ABC):
    """Abstract class of a TensorFunction, represents the operation performed

    Args:
        ABC (_type_): Abstrract Class
    """

    parents: tuple[Tensor] = None
    _cache: tuple[Tensor] = None


class Addition(TensorFunction):
    """Operation to add two tensors

    Args:
        TensorFunction (_type_): _description_
    """

    def forward(self, a: Tensor, b: Tensor) -> Tensor:
        """Computes the addition of two tensors

        Args:
            a (Tensor): one tensor to add
            b (Tensor): other tensor to add

        Returns:
            Tensor: tensor where data is addition of parents
        """
        new_data = a.data + b.data

        requires_grad = a.requires_grad or b.requires_grad

        y = Tensor(new_data, requires_grad=requires_grad, operation=self)

        self.parents = (a, b)

        a.children.append(y)
        b.children.append(y)

        self._cache = (a, b)

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

            # To remove broadcast dims first remove added dimensions
            # **EXAMPLE**

            # 1. Input and Gradient Shapes:
            # - Suppose a has shape (3, 1, 4).
            # - The gradient da has shape (5, 3, 1, 4) after operation and broadcasting.

            # 2. Adjustment Process:
            # in_dim = len(b.shape)  # 3
            # grad_dim = len(db.shape)  # 4

            # for _ in range(grad_dim - in_dim):  # 4 - 3 = 1 time
            #     db = db.sum(axis=0)

            # 3. Result:
            # - After the loop, da would have shape (3, 1, 4), matching b's shape.

            n_dims_da = len(dy.shape)
            n_dims_a = len(a.shape)
            for dim in range(n_dims_da - n_dims_a):
                da = da.sum(axis=0)

            # Then remove singular dimensions (indicates broadcasting)
            # **Summing over singular dimensions:**

            # a.shape = (3, 1, 4)

            # for i, dim in enumerate(a.shape):
            #     if dim == 1:
            #         da = da.sum(axis=i, keepdims=True)

            # - This loop only executes for i=1 because dim == 1 at that position.
            # - da = da.sum(axis=1, keepdims=True)

            # Since da is already of shape (3, 1, 4), summing along `axis=1` with `keepdims=True`
            # doesn't change its shape but ensures the gradient correctly aligns with the
            # broadcast structure.

            # This is because the keepdims is a boolean parameter.
            # If this is set to True, the axes which are reduced are
            # left in the result as dimensions with size one.

            # Therefore if da had shape (3, 2, 4) for example -
            # this loop would reduce it to (3, 1 ,4)

            for i, dim in enumerate(a.shape):
                if dim == 1:
                    da = da.sum(axis=i, keepdims=True)
            a.backward(da, y)

        if b.requires_grad:
            db = dy

            # Rescale gradient to have the same shape as "b":
            n_dims_db = len(da.shape)
            n_dims_b = len(b.shape)
            for dim in range(n_dims_db - n_dims_b):
                db = db.sum(axis=0)

            for i, dim in enumerate(b.shape):
                if dim == 1:
                    db = db.sum(axis=i, keepdims=True)
            b.backward(db, y)


class Negation(TensorFunction):
    """Operation to negate a tensor

    Args:
        TensorFunction (_type_): _description_
    """

    def forward(self, a: Tensor) -> Tensor:
        """Computes the negation of a tensor

        Args:
            a (Tensor): tensor to be negated

        Returns:
            Tensor: negated tensor
        """
        new_data = -a.data

        y = Tensor(new_data, requires_grad=a.requires_grad, operation=self)

        self.parents = (a,)
        a.children.append(y)

        self._cache = (a,)

        return y

    def backward(self, dy: np.ndarray, y: Tensor) -> None:
        """Computes the gradient for tensors in cache

        Args:
            dy (np.ndarray): gradient from upstream
            y (Tensor): output tensor
        """
        a,  = self._cache

        if a.requires_grad:
            da = -dy
            a.backward(da, y)


class Multiplication(TensorFunction):
    """Operation to multiply two tensors

    Args:
        TensorFunction (_type_): _description_
    """

    def forward(self, a: Tensor, b: Tensor) -> Tensor:
        """Computes the multiplication of two tensors

        Args:
            a (Tensor):
            b (Tensor):

        Returns:
            Tensor: product of a and b
        """
        new_data = a.data * b.data

        requires_grad = a.requires_grad or b.requires_grad

        y = Tensor(new_data, requires_grad=requires_grad, operation=self)

        self._cache = (a, b)

        self.parents = (a, b)
        a.children.append(y)
        b.children.append(y)

        return y

    def backward(self, dy: np.ndarray, y: Tensor) -> None:
        """Computes gradients for cached tensors

        Args:
            dy (Tensor): output grad
            y (Tensor):
        """

        a, b = self._cache

        if a.requires_grad:
            da = dy * b.data

            n_dims_da = len(dy.shape)
            n_dims_a = len(a.shape)
            for dim in range(n_dims_da - n_dims_a):
                da = da.sum(axis=0)
            for i, dim in enumerate(a.shape):
                if dim == 1:
                    da = da.sum(axis=i, keepdims=True)

            a.backward(da, y)

        if b.requires_grad:
            db = dy * a.data

            n_dims_db = len(dy.shape)
            n_dims_b = len(b.shape)
            for dim in range(n_dims_db - n_dims_b):
                db = db.sum(axis=0)
            for i, dim in enumerate(b.shape):
                if dim == 1:
                    db = db.sum(axis=i, keepdims=True)

            b.backward(db, y)


class Division(TensorFunction):
    """Division operation

    Args:
        TensorFunction (_type_): _description_
    """
    def forward(self, a: Tensor, b: Tensor) -> Tensor:
        """Computes the division of two tensors

        Args:
            a (Tensor): 
            b (Tensor): 

        Returns:
            Tensor: tensor a / tensor b
        """
        new_data = a.data / b.data

        requires_grad = a.requires_grad or b.requires_grad

        y = Tensor(new_data, requires_grad=requires_grad, operation=self)

        self.parents = (a, b)
        a.children.append(y)
        b.children.append(y)

        self._cache = (a, b)

        return y

    def backward(self, dy: np.ndarray, y: Tensor):
        """Computes gradients of cached tensors

        Args:
            dy (Tensor): 
            y (Tensor): 
        """
        a, b = self._cache

        if a.requires_grad:
            da = dy * ( 1 / b.data )

            n_dims_da = len(dy.shape)
            n_dims_a = len(a.shape)
            for dim in range(n_dims_da - n_dims_a):
                da = da.sum(axis=0)
            for i, dim in enumerate(a.shape):
                if dim == 1:
                    da = da.sum(axis=i, keepdims=True)

            a.backward(da, y)

        if b.requires_grad:
            db = dy * - (a.data / (b.data ** 2))

            n_dims_db = len(dy.shape)
            n_dims_b = len(b.shape)
            for dim in range(n_dims_db - n_dims_b):
                db = db.sum(axis=0)
            for i, dim in enumerate(b.shape):
                if dim == 1:
                    db = db.sum(axis=i, keepdims=True)

            b.backward(db, y)


class MatrixMultiplication(TensorFunction):
    """Matrix Multiplication operation

    Args:
        TensorFunction (_type_): 
    """
    def forward(self, a: Tensor, b: Tensor) -> Tensor:
        """Computes the matrix multiplication of two tensors

        Args:
            a (Tensor): 
            b (Tensor): 

        Returns:
            Tensor: Tensor of a @ b
        """
        new_data = a.data @ b.data

        requires_grad = a.requires_grad or b.requires_grad

        y = Tensor(new_data, requires_grad=requires_grad, operation=self)

        self.parents = (a, b)

        a.children.append(y)
        b.children.append(y)

        self._cache = (a, b)

        return y

    def backward(self, dy: np.ndarray, y:Tensor) -> None:
        """Computes gradients for cached tensors

        Args:
            dy (Tensor): 
            y (Tensor): 
        """
        a, b = self._cache

        if a.requires_grad:
            da = dy @ b.data.T

            n_dims_da = len(dy.shape)
            n_dims_a = len(a.shape)
            for _ in range(n_dims_da - n_dims_a):
                da = da.sum(axis=0)

            a.backward(da, y)

        if b.requires_grad:
            db = a.data.T @ dy

            n_dims_db = len(dy.shape)
            n_dims_b = len(b.shape)
            for _ in range(n_dims_db - n_dims_b):
                db = db.sum(axis=0)

            b.backward(db, y)


class Power(TensorFunction):
    """Power function e.g. a^b

    Args:
        TensorFunction (_type_): _description_
    """
    def forward(self, a: Tensor, b: Tensor) -> Tensor:
        """Computes one tensor to the power of another

        Args:
            a (Tensor): 
            b (Tensor): 

        Returns:
            Tensor: Tensor with a^b
        """
        new_data = a.data ** b.data

        requires_grad = a.requires_grad or b.requires_grad

        y = Tensor(new_data, requires_grad=requires_grad, operation=self)

        self.parents = (a, b)
        a.children.append(y)
        b.children.append(y)

        self._cache = (a, b)

        return y

    def backward(self, dy: np.ndarray, y: Tensor) -> None:
        """Computes gradients of cached tensors

        Args:
            dy (Tensor): _description_
            y (Tensor): _description_
        """
        a, b = self._cache

        if a.requires_grad:
            da = dy * (b.data * (a.data ** (b.data - 1)))

            n_dims_da = len(dy.shape)
            n_dims_a = len(a.shape)
            for dim in range(n_dims_da - n_dims_a):
                da = da.sum(axis=0)
            for i, dim in enumerate(a.shape):
                if dim == 1:
                    da = da.sum(axis=i, keepdims=True)

            a.backward(da, y)

        if b.requires_grad:
            db = dy * ((a.data ** b.data) * np.log(a.data))

            n_dims_db = len(dy.shape)
            n_dims_b = len(b.shape)
            for dim in range(n_dims_db - n_dims_b):
                db = db.sum(axis=0)
            for i, dim in enumerate(b.shape):
                if dim == 1:
                    db = db.sum(axis=i, keepdims=True)

            b.backward(dy, y)


class Mean(TensorFunction):
    """Mean operation

    Args:
        TensorFunction (_type_): 
    """
    def forward(self, a: Tensor) -> Tensor:
        """Computes the mean of a 1D tensor

        #TODO: Implement mean for tensors with dims higher than 1

        Args:
            a (Tensor):

        Returnws:
            Tensor: mean of a
        """
        new_data = a.data.mean()

        requires_grad = a.requires_grad

        y = Tensor(new_data, requires_grad=requires_grad, operation=self)

        self.parents = (a, )

        a.children.append(y)

        self._cache = (a, )

        return y

    def backward(self, dy: np.ndarray, y: Tensor) -> None:
        """Computes gradients of cached tensors

        Args:
            dy (Tensor):
            y (Tensor):
        """
        a, = self._cache

        if a.requires_grad:
            da = dy * Tensor.ones(len(a.data))
            da /= len(a.data)

            a.backward(da, y)


class Sum(TensorFunction):
    """Sum of a tensor

    Args:
        TensorFunction (_type_): _description_
    """
    def forward(self, a: Tensor, axis: int, keepdims: bool) -> Tensor:
        """Computes sum of a tensor

        Args:
            a (Tensor): 
            axis (int): axis to take sum across
            keepdims (bool): keepdims maintains the number of dimensions - reduces summed axis to 1

        Returns:
            Tensor: _description_
        """
        new_data = a.data.sum(axis=axis, keepdims=keepdims)

        requires_grad = a.requires_grad

        y = Tensor(new_data, requires_grad=requires_grad, operation=self)

        self.parents = (a, )

        a.children.append(y)

        self._cache = (a, )

        return y
    
    def backward(self, dy: np.ndarray, y: Tensor) -> None:
        """Computes gradients of sum function

        Args:
            dy (np.ndarray): gradient from upstream
            y (Tensor): 
        """
        a, = self._cache

        if a.requires_grad:
            da = dy * np.ones(a.shape)
            a.backward(da, y)
    

class Log(TensorFunction):
    """Log operation

    Args:
        TensorFunction (_type_): _description_
    """
    def forward(self, a: Tensor) -> Tensor:
        """Element wise log of a tensor

        Args:
            a (Tensor): 

        Returns:
            Tensor: 
        """

        new_data = np.log(a.data)

        requires_grad = a.requires_grad

        y = Tensor(new_data, requires_grad=requires_grad, operation=self)

        self.parents = (a, )

        a.children.append(y)

        self._cache = (a, )

        return y
    
    def backward(self, dy: np.ndarray, y: Tensor) -> None:
        """Computes gradient of cached tensor

        Args:
            dy (np.ndarray): gradient from upstream
            y (Tensor): 
        """
        a, = self._cache
        if a.requires_grad:
            da = dy * (1 / a.data)

            a.backward(da, y)


class Exp(TensorFunction):
    """e^

    Args:
        TensorFunction (_type_): 
    """
    def forward(self, a: Tensor) -> Tensor:
        """e^a

        Args:
            a (Tensor): 

        Returns:
            Tensor: 
        """
        new_data = np.exp(a.data)

        requires_grad = a.requires_grad

        y = Tensor(new_data, requires_grad=requires_grad, operation=self)

        self.parents = (a, )

        a.children.append(y)

        self._cache = (a, new_data)

        return y
    
    def backward(self, dy: np.ndarray, y: Tensor) -> None:
        """Computes grads

        Args:
            dy (np.ndarray): 
            y (Tensor): 
        """
        a, new_data = self._cache

        if a.requires_grad:
            da = dy * new_data
            a.backward(da, y)


class Transpose(TensorFunction):
    def forward(self, a: Tensor) -> Tensor:
        """Transposes the data

        Args:
            x (Tensor): 

        Returns:
            Tensor: 
        """
        new_data = a.data.T

        requires_grad = a.requires_grad

        y = Tensor(new_data, requires_grad=requires_grad, operation=self)

        self.parents = (a, )

        a.children.append(y)

        self._cache = (a, )

        return y

    def backward(self, dy: np.ndarray, y: Tensor) -> None:
        a,  = self._cache
        
        if a.requires_grad:
            da = dy.T
 
            a.backward(da, y)

class Convolve2D(TensorFunction):
    """2D convolution layer as tensor function

    Args:
        TensorFunction (_type_):
    """     
    def forward(self, x: Tensor, k: Tensor, b: Tensor = None) -> Tensor:
        """2D Convolution layer of X as input

        Args:
            x (Tensor): Input to layer
        """
        self.n_kernels, self.x_samples, self.kernel_size, _ = k.shape
        self.x_shape = x.shape

        x_samples, x_width, x_height = self.x_shape

        self.output_shape = (self.n_kernels, x_width - self.kernel_size + 1, x_height - self.kernel_size + 1)
        self.kernels_shape = (self.n_kernels, x_samples, self.kernel_size, self.kernel_size)
        
        new_data = np.zeros(self.output_shape)
        if b:
            new_data += b.data

        new_data = cpu_forward_convolve2d(new_data, x.data, k.data, self.n_kernels)

        y = Tensor(new_data, requires_grad=True, operation=self)

        self.parents = (x, k, )
        if b:
            self.parents += b

        x.children.append(y)
        k.children.append(y)
        if b:
            b.children.append(y)

        self._cache = (x, k, b)

        return y
    
    def backward(self, dy: np.ndarray, y: Tensor) -> None:
        """Computes gradients of a convolutional layer process

        Args:
            dy (np.ndarray): upstream grad
            y (Tensor): 
        """
        x, k, b = self._cache

        if b:
            if b.requires_grad:
                b.backward(dy, y)

        if x.requires_grad:
            raise NotImplementedError

        if k.requires_grad:
            dk = np.zeros(self.kernels_shape)
            dk = cpu_k_backward_convolve2d(dk, x.data, dy, self.n_kernels)

            k.backward(dk, y)

            



