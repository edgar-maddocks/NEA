import os

os.environ["TF_ENABLE_ONEDNN_OPTS"] = "0"
os.environ["TF_CPP_MIN_LOG_LEVEL"] = "2"

from tensorflow import keras

(X_train, y_train), (X_test, y_test) = keras.datasets.mnist.load_data()
(X_train, y_train), (X_test, y_test) = (
    (X_train[:10000], y_train[:10000]),
    (X_test[:10000], y_test[:10000]),
)

print((X_train.shape, y_train.shape), (X_test.shape, y_test.shape))

y_train = keras.utils.to_categorical(y_train)
y_test = keras.utils.to_categorical(y_test)

print((y_train.shape, y_test.shape))

X_train, X_test = X_train / 255.0, X_test / 255.0
X_train, X_test = (
    X_train.reshape(X_train.shape[0], 1, X_train.shape[1], X_train.shape[2]),
    X_test.reshape(X_test.shape[0], 1, X_test.shape[1], X_test.shape[2]),
)
print(X_train.shape, X_test.shape)

from nea.ml.nn import Module, Conv2D, Dense, ReLU, Reshape, SGD, MSE, Softmax, Sigmoid
from nea.ml.autograd import Tensor, no_grad

import numpy as np


class MNISTER(Module):
    def __init__(self):
        self.conv2d_1 = Conv2D(X_train.shape[1:], 3, 5)
        self.relu_1 = ReLU()
        self.conv2d_2 = Conv2D((5, 26, 26), 3, 1)
        self.relu_2 = ReLU()
        self.reshape_1 = Reshape((1, 576))
        self.dense_1 = Dense(576, 256)
        self.relu_3 = ReLU()
        self.dense_2 = Dense(256, 10)
        self.softmax = Softmax()

    def forward(self, x_sample: Tensor) -> Tensor:
        """out = self.conv2d_1(x_sample)
        out = self.relu_1(out)
        out = self.conv2d_2(out)
        out = self.relu_2(out)"""
        out = Tensor(np.random.randn(1, 24, 24), requires_grad=True)
        out = self.reshape_1(out)
        out = self.dense_1(out)
        out = self.relu_3(out)
        out = self.dense_2(out)
        out /= np.max(np.abs(out.data))
        out = self.softmax(out)
        return out

    def __call__(self, x_sample: Tensor) -> Tensor:
        return self.forward(x_sample)


from tqdm import tqdm
import gc

model = MNISTER()
optim = SGD(model.params, lr=0.1, regulization=0.1)
loss_func = MSE()

epochs = 1000
batch_size = 64

for epoch in range(epochs):
    print(f"---------EPOCH: {epoch}------------")

    for sample in tqdm(range(0, X_train.shape[0])):
        pred = model(X_train[sample])
        loss = loss_func(pred.reshape((10, 1)), y_train[sample].reshape(10, 1))
        loss.backward()
        optim.step()
        optim.zero_grad()

    gc.collect()

    with no_grad():
        loss = 0
        for sample in tqdm(range(0, X_test.shape[0])):
            pred = model(X_test[sample])
            loss += loss_func(pred.reshape((10, 1)), y_test[sample].reshape(10, 1))

        print(f"TEST LOSS = {loss}")

    gc.collect()


print("DONE")
