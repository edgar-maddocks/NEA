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


def test_mean_funcs(): ...


def test_sum_funcs(): ...


def test_log_funcs(): ...


def test_exp_funcs(): ...


def test_transpose_funcs(): ...


def test_convolve2d_funcs(): ...
