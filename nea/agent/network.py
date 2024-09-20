import numpy as np

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

    input = np.reshape(game.board, (1, 8, 8))
    print(net.forward(input))
