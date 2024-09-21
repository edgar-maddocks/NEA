from copy import deepcopy
import numpy as np

from nea.ml import nn
from nea.ml.autograd import Tensor, Tensorable


class ResidualLayer(nn.Module):
    def __init__(self, num_hidden_conv: int) -> None:
        super().__init__()
        self.Conv1 = nn.Conv2D(
            (num_hidden_conv, 8, 8),
            kernel_size=3,
            n_kernels=num_hidden_conv,
            padding=1,
            padding_value=0,
        )
        self.ReLU1 = nn.ReLU()
        self.Conv2 = nn.Conv2D(
            (num_hidden_conv, 8, 8),
            kernel_size=3,
            n_kernels=num_hidden_conv,
            padding=1,
            padding_value=0,
        )
        self.ReLU2 = nn.ReLU()

    def forward(self, x: Tensorable) -> Tensor:
        """Input Shape of (128, 8, 8)

        Output Shape of (128, 8, 8)

        Args:
            x (Tensorable): input tensor

        Returns:
            Tensor:
        """
        original_input = deepcopy(x)
        x = self.Conv1(x)
        x = self.ReLU1(x)
        x = self.Conv2(x)
        x += original_input
        x = self.ReLU2(x)
        x /= np.max(x.data)
        return x


class ConvolutionalLayer(nn.Module):
    """This is only used as the starting block

    Args:
        nn (_type_):
    """

    def __init__(self, num_hidden_conv: int) -> None:
        super().__init__()
        self.Conv1 = nn.Conv2D(
            (15, 8, 8),
            n_kernels=num_hidden_conv,
            kernel_size=3,
            padding=1,
            padding_value=0,
        )
        self.ReLU = nn.ReLU()

    def forward(self, x: Tensor) -> Tensor:
        """Input shape of (15, 8, 8)

        Output shape of (128, 8, 8)

        Args:
            x (Tensor): input tensor

        Returns:
            Tensor:
        """
        x = self.Conv1(x)
        x = self.ReLU(x)
        return x


class PolicyHead(nn.Module):
    def __init__(self, num_hidden_conv: int) -> None:
        super().__init__()
        self.Conv1 = nn.Conv2D((num_hidden_conv, 8, 8), kernel_size=1, n_kernels=8)
        self.ReLU1 = nn.ReLU()
        self.Softmax = nn.Softmax()

    def forward(self, x: Tensor) -> Tensor:
        """Input shape of (128, 8, 8)

        Output shape of (8, 8, 8)

        Args:
            x (Tensor): input tensor

        Returns:
            Tensor:
        """
        x = self.Conv1(x)
        x = self.ReLU1(x)
        x = self.Softmax(x)
        return x


class ValueHead(nn.Module):
    def __init__(self, num_hidden_conv: int) -> None:
        super().__init__()
        self.Conv1 = nn.Conv2D((num_hidden_conv, 8, 8), kernel_size=1, n_kernels=8)
        self.ReLU1 = nn.ReLU()
        self.Reshape = nn.Reshape((1, 8 * 8 * 8))
        self.Dense1 = nn.Dense(8 * 8 * 8, 256)
        self.ReLU2 = nn.ReLU()
        self.Dense2 = nn.Dense(256, 1)
        self.Tanh = nn.Tanh()

    def forward(self, x: Tensor) -> Tensor:
        """Input shape of (128, 8, 8)

        Output is scalar [-1, 1]

        Args:
            x (Tensor): input tensor

        Returns:
            Tensor:
        """
        x = self.Conv1(x)
        x = self.ReLU1(x)
        x = self.Reshape(x)
        x = self.Dense1(x)
        x = self.ReLU2(x)
        x = self.Dense2(x)
        x = self.Tanh(x)
        return x


class AlphaModel(nn.Module):
    def __init__(self, n_res_layers: int = 10, num_hidden_conv: int = 64) -> None:
        super().__init__()
        self.first_layer = ConvolutionalLayer(num_hidden_conv=num_hidden_conv)
        self.res_layers = nn.ModuleList(
            [
                ResidualLayer(num_hidden_conv=num_hidden_conv)
                for r in range(n_res_layers)
            ]
        )
        self.policy_head = PolicyHead(num_hidden_conv=num_hidden_conv)
        self.value_head = ValueHead(num_hidden_conv=num_hidden_conv)

    def forward(self, x: Tensor) -> tuple[Tensor, Tensor]:
        """Input shape (15, 8, 8)

        Output of tuple (policy, value)

        Policy has shape (8, 8, 8)

        Value is scalar

        Args:
            x (Tensor): input tensor

        Returns:
            tuple[Tensor, Tensor]: (policy, value)
        """
        x = self.first_layer(x)
        for res in self.res_layers:
            x = res(x)
        pol = self.policy_head(x)
        val = self.value_head(x)
        return pol, val
