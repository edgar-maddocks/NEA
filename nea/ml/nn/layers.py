from abc import ABC
import collections
import numpy as np
import pickle

from nea.ml.autograd import Tensor, to_tensor, TensorFunction, tensor_exp, tensor_sum
from nea.ml.autograd.consts import Tensorable


class Parameter(Tensor):
    """
    Represents a parameter
    """

    def __init__(
        self,
        shape: tuple[int] = None,
        requires_grad: bool = True,
        operation: TensorFunction = None,
    ) -> None:
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

    @property
    def params(self) -> list[Parameter | Tensor]:
        """Gets all parameters inside a modules from the self.__dict__
        Also gets any tensors with requires_grad = True,
        and the parameters from any other module

        Returns:
            list[Parameter | Tensor]:
        """

        params = []

        for param in self.__dict__.values():
            if isinstance(param, Module):
                params += param.params
            elif isinstance(param, ModuleList):
                params += param.params
            elif isinstance(param, Parameter):
                params.append(param)
            elif isinstance(param, Tensor):
                if param.requires_grad:
                    params.append(param)

        return params

    def save(self, file_path: str) -> None:
        with open(file_path, "wb") as fh:
            pickle.dump(self, fh)

    @staticmethod
    def load(self, file_path: str) -> None:
        with open(file_path, "rb") as fh:
            return pickle.load(fh)


class ModuleList(Module, collections.abc.Sequence):
    def __init__(self, modules: list[Module]) -> None:
        super().__init__()
        self.modules = modules

    @property
    def params(self) -> list[Parameter | Tensor]:
        params = []

        for module in self.modules:
            for param in module.__dict__.values():
                if isinstance(param, Module):
                    params += param.params
                elif isinstance(param, ModuleList):
                    params += param.params
                elif isinstance(param, Parameter):
                    params.append(param)
                elif isinstance(param, Tensor):
                    if param.requires_grad:
                        params.append(param)

            return params

    def __getitem__(self, index: int) -> Module:
        return self.modules[index]

    def __len__(self) -> int:
        return len(self.modules)


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
            self.bias = Parameter((n_outputs,))

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


class Conv2D(Module):
    """A Convolutional layer

    Args:
        Module (_type_):
    """

    def __init__(
        self,
        x_shape: tuple[int, int, int],
        kernel_size: int,
        n_kernels: int,
        bias: bool = False,
        padding: int = None,
        padding_value: float = None,
    ) -> None:
        assert len(x_shape) == 3, "Input must be of shape (n_samples, *, *)"
        if padding_value:
            assert padding, "Must define amount of padding if padding value is not None"

        self.n_kernels = n_kernels
        self.x_shape = x_shape

        x_samples, x_width, x_height = self.x_shape

        self.padding = padding
        self.padding_value = padding_value
        if self.padding:
            self.x_shape = (
                x_samples,
                x_width + 2 * self.padding,
                x_height + 2 * self.padding,
            )
            x_samples, x_width, x_height = self.x_shape

        self.output_shape = (
            self.n_kernels,
            x_width - kernel_size + 1,
            x_height - kernel_size + 1,
        )
        self.kernels_shape = (n_kernels, x_samples, kernel_size, kernel_size)

        self.kernels = Parameter(self.kernels_shape)

        self.bias = bias
        self.biases = None
        if self.bias:
            self.biases = Parameter(self.output_shape)

    def forward(self, x: Tensor) -> Tensor:
        if self.padding:
            x = x.pad2D(padding=self.padding, value=self.padding_value)
        return x.convolve2d(k=self.kernels, b=self.biases)


class Reshape(Module):
    def __init__(self, desired_shape: tuple[int]) -> None:
        self.desired_shape = desired_shape

    def forward(self, x: Tensor) -> Tensor:
        return x.reshape(self.desired_shape)


# ==========================
#        Activations
# ==========================


class Tanh(Module):
    """Tanh activation layer

    Args:
        Module (_type_):
    """

    def __init___(self) -> None:
        super().__init__()

    def forward(self, x: Tensor) -> Tensor:
        output = (tensor_exp(x) - tensor_exp(-x)) / (tensor_exp(x) + tensor_exp(-x))
        return output


class Sigmoid(Module):
    """Sigmoid activation layer

    Args:
        Module (_type_):
    """

    def __init__(self) -> None:
        super().__init__()

    def forward(self, x: Tensor) -> Tensor:
        output = 1 / (1 + tensor_exp(-x))
        return output


class Softmax(Module):
    def __init__(self) -> None:
        super().__init__()

    def forward(self, x: Tensor, dim: int = -1) -> Tensor:
        z = tensor_exp(x)
        output = z / tensor_sum(z, dim=dim, keepdims=True)
        return output


class ReLU(Module):
    def __init__(self) -> None:
        super().__init__()

    def forward(self, x: Tensor) -> Tensor:
        return x.relu()


# ==========================
#          Losses
# ==========================


class MSE(Module):
    def __init__(self) -> None:
        super().__init__()

    def __call__(self, predicted: Tensor, true: Tensorable) -> Tensor:
        return self.forward(predicted=predicted, true=true)

    def forward(self, predicted: Tensor, true: Tensorable) -> Tensor:
        loss: Tensor = predicted - true
        loss = (1 / true.shape[0]) * (loss.T() @ loss)
        return loss


class AlphaLoss(Module):
    def __init__(self) -> None:
        super().__init__()

    def __call__(
        self,
        true_value: Tensor,
        predicted_value: Tensor,
        mcts_pol: Tensor,
        net_pol: Tensor,
    ) -> Tensor:
        return self.forward(
            true_value=true_value,
            predicted_value=predicted_value,
            mcts_pol=mcts_pol,
            net_pol=net_pol,
        )

    def forward(
        self,
        true_value: Tensor,
        predicted_value: Tensor,
        mcts_pol: Tensor,
        net_pol: Tensor,
    ) -> Tensor:
        val_sqe = (true_value - predicted_value) ** 2
        mcts_pol_t = mcts_pol.T()
        net_pol_log = net_pol.log()
        pol_bcel = mcts_pol_t @ net_pol_log
        return val_sqe - pol_bcel
