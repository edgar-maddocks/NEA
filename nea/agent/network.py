from copy import deepcopy
import time as t

from nea.ml import nn
from nea.ml.autograd import Tensor, Tensorable
from nea.console_checkers import CheckersGame


class ResidualLayer(nn.Module):
    def __init__(self) -> None:
        super().__init__()
        self.Conv1 = nn.Conv2D(
            (15, 8, 8), kernel_size=3, n_kernels=128, padding=1, padding_value=0
        )
        self.ReLU1 = nn.ReLU()
        self.Conv2 = nn.Conv2D(
            (128, 8, 8), kernel_size=3, n_kernels=15, padding=1, padding_value=0
        )
        self.ReLU2 = nn.ReLU()

    def forward(self, x: Tensorable) -> Tensor:
        original_input = deepcopy(x)
        x = self.Conv1(x)
        x = self.ReLU1(x)
        x = self.Conv2(x)
        x += original_input
        x = self.ReLU2(x)
        return x


class PolicyHead(nn.Module):
    def __init__(self) -> None:
        super().__init__()
        self.Conv1 = nn.Conv2D((15, 8, 8), kernel_size=1, n_kernels=8)
        self.ReLU1 = nn.ReLU()
        self.Softmax = nn.Softmax()

    def forward(self, x: Tensor) -> Tensor:
        x = self.Conv1(x)
        x = self.ReLU1(x)
        x = self.Softmax(x)
        return x


class ValueHead(nn.Module):
    def __init__(self) -> None:
        super().__init__()
        self.Conv1 = nn.Conv2D((15, 8, 8), kernel_size=1, n_kernels=8)
        self.ReLU1 = nn.ReLU()
        self.Reshape = nn.Reshape((1, 8 * 8 * 8))
        self.Dense1 = nn.Dense(8 * 8 * 8, 256)
        self.ReLU2 = nn.ReLU()
        self.Dense2 = nn.Dense(256, 1)
        self.Tanh = nn.Tanh()

    def forward(self, x: Tensor) -> Tensor:
        x = self.Conv1(x)
        x = self.ReLU1(x)
        x = self.Reshape(x)
        x = self.Dense1(x)
        x = self.ReLU2(x)
        x = self.Dense2(x)
        x = self.Tanh(x)
        return x


if __name__ == "__main__":
    res_layer = ResidualLayer()
    value_head = ValueHead()
    policy_head = PolicyHead()

    samples = []
    for i in range(15):
        samples.append(CheckersGame().board)

    input = Tensor(samples, requires_grad=True)

    print("#######################################################")

    print("STARTING TIMINGS FOR RESIDUAL LAYER")
    start = t.time()
    y = res_layer(input)
    print("FORWARD PASS TOOK: ", t.time() - start, "seconds")

    start = t.time()
    y.backward()
    print("BACKWARD PASS TOOK: ", t.time() - start, "seconds")

    print("#######################################################")

    print("STARTING TIMINGS FOR VALUE HEAD")
    start = t.time()
    y = value_head(input)
    print("FORWARD PASS TOOK: ", t.time() - start, "seconds")

    start = t.time()
    y.backward()
    print("BACKWARD PASS TOOK: ", t.time() - start, "seconds")

    print("#######################################################")

    print("STARTING TIMINGS FOR POLICY HEAD")
    start = t.time()
    y = policy_head(input)
    print("FORWARD PASS TOOK: ", t.time() - start, "seconds")

    start = t.time()
    y.backward()
    print("BACKWARD PASS TOOK: ", t.time() - start, "seconds")

    print("#######################################################")
