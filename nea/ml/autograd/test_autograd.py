from nea.ml.autograd import Tensor

import numpy as np
# ===================
#    TEST ADDITION
# ===================


def test_addition_funcs():
    """
    .
    """
    a = Tensor(1, requires_grad=True)
    b = Tensor(2, requires_grad=True)

    a.zero_grad()
    y = a + b
    y.backward()
    assert a.grad == 1

    a.zero_grad()
    y = a + 2
    y.backward()
    assert a.grad == 1

    a.zero_grad()
    a += 2
    a.backward()
    assert a.grad == 1


# ======================
#    TEST SUBTRACTION
# ======================


def test_subtraction_funcs():
    """
    .
    """
    a = Tensor(1, requires_grad=True)
    b = Tensor(2, requires_grad=True)

    y = b - a
    assert y == 1

    y.backward()
    assert a.grad == -1

    b -= 1
    b.zero_grad()
    b.backward()
    assert b.grad == 1


# =========================
#    TEST MULTIPLICATION
# =========================


def test_multiplication_funcs():
    """
    .
    """
    a = Tensor(1, requires_grad=True)
    b = Tensor(2, requires_grad=True)

    y = a * b
    assert y == 2

    y.backward()
    assert a.grad == 2
    assert b.grad == 1


# =========================
#    TEST DIVISION
# =========================


def test_division_funcs():
    """
    .
    """
    a = Tensor(4, requires_grad=True)
    b = Tensor(2, requires_grad=True)

    y = a / b
    assert y == 2

    y.backward()
    assert a.grad == (1 / 2)
    assert b.grad == -(4 / (2**2))


# =========================
#    TEST MATMUL
# =========================


def test_matmul_funcs():
    a = Tensor([[1, 2], [2, 1]], requires_grad=True)
    b = Tensor([[2, 1], [1, 2]], requires_grad=True)

    y = a @ b

    assert y.shape == (2, 1) or (2,)
    assert np.array_equal(y.data, np.array([[4, 5], [5, 4]]))

    y.backward()

    assert np.array_equal(y.grad, np.array([[1, 1], [1, 1]]))
    assert np.array_equal(a.grad, np.array([[3, 3], [3, 3]]))


def test_power_funcs():
    a = Tensor(2, requires_grad=True)
    b = Tensor(3, requires_grad=True)

    y = a**b

    assert y == 8

    y.backward()

    assert y.grad == 1
    assert a.grad == 12
    # assert b.grad == a^b * lnb


def test_mean_funcs():
    a = Tensor([3, 3, 3, 3], requires_grad=True)
    b = a.mean()
    b.backward()

    # the 4 values in a attribute equally so the gradient = [1/4, 1/4, 1/4, 1/4]
    # since a is length 4
    # .all() asserts elementwise truth
    assert (a.grad == [1 / 4, 1 / 4, 1 / 4, 1 / 4]).all()


def test_sum_funcs():
    a = Tensor([3, 3, 3, 3], requires_grad=True)
    b = a.sum()
    b.backward()

    assert (a.grad == [1, 1, 1, 1]).all()


def test_log_funcs():
    a = Tensor(3, requires_grad=True)
    b = a.log()
    b.backward()

    assert a.grad == 1 / 3


def test_exp_funcs():
    a = Tensor(3, requires_grad=True)
    b = a.exp()
    b.backward()

    # derivative of e^a == e^a
    assert a.grad == b.data


def test_transpose_funcs():
    a = Tensor([[1, 1, 1], [1, 1, 1], [1, 1, 1]], requires_grad=True)
    b = Tensor([[1, 2, 3], [4, 5, 6], [7, 8, 9]], requires_grad=True)
    c = a.T()
    d = b * c
    d.backward()

    # gradient of a == Transpose(b), have to multiply by b otherwise
    # all values would be 1
    assert (a.grad == np.array([[1, 4, 7], [2, 5, 8], [3, 6, 9]])).all()


def test_reshape_funcs():
    a = Tensor([[1, 1], [1, 1], [1, 1]], requires_grad=True)
    b = a.reshape((2, 3))
    b.backward()

    assert b.grad.shape == (2, 3)
    assert a.grad.shape == (3, 2)


def test_pad2d_funcs():
    a = Tensor([1], requires_grad=True)
    a = a.reshape((1, 1, 1))
    b = a.pad2D(1, 0)

    assert (b.data == [[0, 0, 0], [0, 1, 0], [0, 0, 0]]).all()

    b.backward()

    assert b.grad.shape == (1, 3, 3)
    assert a.grad.shape == (1, 1, 1)


def test_relu_funcs():
    a = Tensor([-1, 1], requires_grad=True)
    b = a.relu()

    assert (b.data == [0, 1]).all()

    c = b * 2
    c.backward()

    # upstream grad (db) == [2, 2] so relu sets both to 1
    assert (a.grad == [1, 1]).all()
