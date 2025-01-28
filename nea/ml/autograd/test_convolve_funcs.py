from nea.ml.autograd import Tensor


def test_convolve2d_forward_funcs():
    x = Tensor([[1, 6, 2], [5, 3, 1], [7, 0, 4]], requires_grad=True)
    k = Tensor([[1, 2], [-1, 0]], requires_grad=True)

    x = x.reshape((1, 3, 3))  # define 1 sample of 3x3
    k = k.reshape((1, 1, 2, 2))  # define 1 kernel for 1 sample with 2x2 kernel

    y = x.convolve2d(k=k)

    assert (y.data == [[8, 7], [4, 5]]).all()


def test_convolve2d_backward_funcs():
    x = Tensor([[1, 6, 2], [5, 3, 1], [7, 0, 4]], requires_grad=True)
    k = Tensor([[1, 2], [-1, 0]], requires_grad=True)

    x = x.reshape((1, 3, 3))  # define 1 sample of 3x3
    k = k.reshape((1, 1, 2, 2))  # define 1 kernel for 1 sample with 2x2 kernel

    y = x.convolve2d(k=k)

    y.backward()

    # calculated by hand
    assert (x.grad == [[[0, -1, -1], [2, 2, 0], [2, 3, 1]]]).all()
    assert (k.grad == [[[[15, 12], [15, 8]]]]).all()
