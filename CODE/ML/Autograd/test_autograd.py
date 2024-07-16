from Tensor import Tensor

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
