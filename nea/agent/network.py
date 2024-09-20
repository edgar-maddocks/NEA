import numpy as np
import time as t

from nea.ml import nn
from nea.ml.autograd import Tensor, Tensorable
from nea.console_checkers import CheckersGame


if __name__ == "__main__":
    game = CheckersGame()

    class Model(nn.Module):
        def __init__(self) -> None:
            super().__init__()
            self.conv = nn.Conv2D((1, 8, 8), 3, 64)

        def forward(self, x: Tensorable) -> Tensor:
            x = self.conv(x)
            return x

    net = Model()
    samples = []
    for i in range(10000):
        samples.append(game.board)

    input = Tensor(samples, requires_grad=True)
    print(input.shape)
    y = net.forward(input)
    start = t.time()
    y.backward()
    print("GRAD CALCULATION TOOK: ", t.time() - start)
