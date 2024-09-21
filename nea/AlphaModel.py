import time as t
import numpy as np

from nea.agent import AlphaModel
from nea.console_checkers import CheckersGame
from nea.ml.autograd import Tensor
from nea.ml.nn import AlphaLoss

if __name__ == "__main__":
    model = AlphaModel()

    samples = []
    for i in range(15):
        samples.append(CheckersGame().board)

    input = Tensor(samples, requires_grad=True)

    print("#######################################################")
    print("STARTING TIMINGS FOR ALPHA MODEL")
    start = t.time()
    policy, value = model(input)
    print("FORWARD PASS TOOK: ", t.time() - start, "seconds")

    fake_pol = Tensor(np.random.randn(8, 8, 8), requires_grad=True)
    fake_val = Tensor(np.random.rand(1), requires_grad=True)

    loss_func = AlphaLoss()
    loss = loss_func(fake_val, value, fake_pol, policy)

    start = t.time()
    loss.backward()
    print("BACKWARD PASS TOOK: ", t.time() - start, "seconds")
    print("#######################################################")
