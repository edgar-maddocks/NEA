from tensor import Tensor

if __name__ == "__main__":
    a = Tensor(1, requires_grad=True)
    b = Tensor(2, requires_grad=True)
    c = Tensor(1, requires_grad=True)

    e = a * b
    d = b * c

    y = e * d
    y.backward()
