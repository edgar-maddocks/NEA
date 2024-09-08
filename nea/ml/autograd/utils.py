import numpy as np
import time as t
from numba import cuda, jit

from nea.ml.autograd import Tensor, Tensorable


def tensor_mean(a: Tensorable) -> Tensor:
    """Mean of a tensor

    Args:
        a (Tensor):

    Returns:
        Tensor:
    """
    return a.mean()


def tensor_sum(a: Tensorable, dim: int = -1, keepdims: bool = False) -> Tensor:
    """sum of tensor

    Args:
        a (Tensor):
        axis (int, optional): axis to sum across. Defaults to -1.
        keepdims (bool, optional): reduce summed dim to 1?. Defaults to False.

    Returns:
        Tensor:
    """
    return a.sum(dim=dim, keepdims=keepdims)


def tensor_exp(a: Tensor) -> Tensor:
    """e^

    Args:
        a (Tensor):

    Returns:
        Tensor:
    """
    return a.exp()
