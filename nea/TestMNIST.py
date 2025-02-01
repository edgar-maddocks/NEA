import os
import sys
import gc

from tensorflow import keras
import numpy as np
from tqdm import tqdm

from nea.ml.nn import (
    Module,
    Conv2D,
    Dense,
    ReLU,
    Tanh,
    Reshape,
    MinMaxNormalization,
    SGD,
    MSE,
    Softmax,
    Sigmoid,
    CrossEntropy,
)
from nea.ml.autograd import Tensor, no_grad

sys.setrecursionlimit(10000)
os.environ["TF_ENABLE_ONEDNN_OPTS"] = "0"
os.environ["TF_CPP_MIN_LOG_LEVEL"] = "2"


(X_train, y_train), (X_test, y_test) = keras.datasets.mnist.load_data()

"""train_filter = np.where((y_train == 0) | (y_train == 4))
test_filter = np.where((y_test == 0) | (y_test == 4))

X_train, y_train = X_train[train_filter], y_train[train_filter]
X_test, y_test = X_test[test_filter], y_test[test_filter]"""

y_train = keras.utils.to_categorical(y_train)
y_test = keras.utils.to_categorical(y_test)

print((y_train.shape, y_test.shape))

X_train, X_test = X_train / 255.0, X_test / 255.0
X_train, X_test = (
    X_train.reshape(X_train.shape[0], 1, X_train.shape[1], X_train.shape[2]),
    X_test.reshape(X_test.shape[0], 1, X_test.shape[1], X_test.shape[2]),
)
print(X_train.shape, X_test.shape)


class MNISTER(Module):
    def __init__(self):
        super().__init__()
        self.conv2d_1 = Conv2D(X_train.shape[1:], 9, 5)
        self.sigmoid_1 = Sigmoid()
        self.reshape_1 = Reshape((1, 2000))
        self.dense_1 = Dense(2000, 784)
        self.sigmoid_2 = Sigmoid()
        self.dense_2 = Dense(784, 256)
        self.sigmoid_3 = Sigmoid()
        self.dense_3 = Dense(256, 10)
        self.softmax = Softmax()

    def forward(self, x_sample: Tensor) -> Tensor:
        out = self.conv2d_1(x_sample)
        out = self.sigmoid_1(out)
        out = self.reshape_1(out)
        out = self.dense_1(out)
        out = self.sigmoid_2(out)
        out = self.dense_2(out)
        out = self.sigmoid_3(out)
        out = self.dense_3(out)
        out = self.softmax(out)
        return out

    def __call__(self, x_sample: Tensor) -> Tensor:
        return self.forward(x_sample)


model = MNISTER()
optim = SGD(model.params, lr=0.01, regulization=0)
loss_func = CrossEntropy()

epochs = 5

for epoch in range(epochs):
    print(f"---------EPOCH: {epoch + 1}------------")

    loss = Tensor(0, requires_grad=True)
    for sample in tqdm(range(0, X_train[:20000].shape[0])):
        pred = model(X_train[sample])
        loss = loss_func(pred.reshape((10, 1)), Tensor(y_train[sample].reshape(10, 1)))
        loss.backward()
        optim.step()
        optim.zero_grad()

    gc.collect()

    with no_grad():
        loss = 0
        for sample in tqdm(range(0, X_test.shape[0])):
            pred = model(X_test[sample])
            loss += loss_func(
                pred.reshape((10, 1)), Tensor(y_test[sample].reshape(10, 1))
            )

        loss /= X_test.shape[0]

        print(f"TEST LOSS: {loss}")

    gc.collect()

    for x in range(0, 5):
        pred = model(X_test[x])
        print(f"Pred: {np.argmax(pred.data)}. True: {np.argmax(y_test[x])}")


print("DONE")
