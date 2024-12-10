from tensor import Tensor

if __name__ == "__main__":
    a = Tensor(1, requires_grad=True)
    b = Tensor(2, requires_grad=True)
    c = Tensor(3, requires_grad=True)

    d = a * b
    e = b * c

    y = d * e
    y.backward()

    print(d.grad, a.grad, b.grad, e.grad, c.grad)
