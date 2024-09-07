import numpy as np
import time as t

from nea.ml import nn
from nea.ml.autograd import Tensor


if __name__ == "__main__":
    class Model(nn.Module):
        def __init__(self) -> None:
            super().__init__()
            self.dense1 = nn.Dense(2, 3)
            self.tanh1 = nn.Tanh()
            self.dense2 = nn.Dense(3, 1)
            self.tanh2 = nn.Tanh()
            self.loss = nn.MSE()


        def forward(self, x: Tensor) -> Tensor:
            x = self.dense1(x)
            x = self.tanh1(x)
            x = self.dense2(x)
            x = self.tanh2(x)
            return x
        
    x = np.array([[0, 0], [0, 1], [1, 0], [1, 1]])
    y = np.array([[0], [1], [1], [0]])

    print(x.shape)
    print(y.shape)

    model = Model()

    loss_func = nn.MSE()

    optim = nn.SGD(model.params, lr = 0.5)

    start = t.time()
    for i in range(1000):
        preds = model(x)
        
        loss = loss_func(preds, y)
        loss.backward()

        optim.step()
        optim.zero_grad()

    preds = model(x)
    print(preds)
    print(loss_func(preds, y))
    
    print("TIME TAKEN: ", t.time() - start)



        


