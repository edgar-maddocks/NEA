import numpy as np
import time as t
import numba
from numba import cuda

from nea.ml.autograd import Tensor, Tensorable

def tensor_mean(a: Tensorable) -> Tensor:
    """Mean of a tensor

    Args:
        a (Tensor):

    Returns:
        Tensor: 
    """
    return a.mean()

def tensor_sum(a: Tensorable, axis: int = -1, keepdims: bool = False) -> Tensor:
    """sum of tensor

    Args:
        a (Tensor): 
        axis (int, optional): axis to sum across. Defaults to -1.
        keepdims (bool, optional): reduce summed dim to 1?. Defaults to False.

    Returns:
        Tensor: 
    """
    return a.sum(axis=axis, keepdims=keepdims)

def tensor_exp(a: Tensor) -> Tensor:
    """e^

    Args:
        a (Tensor): 

    Returns:
        Tensor: 
    """
    return a.exp()

def _np_cross_correlate(x: np.ndarray, matrix2: np.ndarray):
    x_depth, x_rows, x_cols = x.shape
    m2_rows, m2_cols = matrix2.shape
    output_rows = x_rows - m2_rows + 1
    output_cols = x_cols - m2_cols + 1
    
    output = np.zeros((x_depth, output_rows, output_cols), dtype=np.float32)

    use_cuda = cuda.is_available() and x_depth > 100000

    if use_cuda:
        print("CUDA DEVICE DETECTED TO USE")
        print(cuda.detect())

    for i in range(x_depth):
        if use_cuda:
            output[i] = _cuda_cross_correlate(x[i], matrix2)
        else:
            output[i] = _jit_cpu_cross_correlate(x[i], matrix2)
    
    return output

@numba.jit(nopython=True, cache=True)
def _jit_cpu_cross_correlate(a: np.ndarray, b: np.ndarray) -> np.ndarray:
    a_rows, a_cols = a.shape
    b_rows, b_cols = b.shape
    
    output_rows = a_rows - b_rows + 1
    output_cols = a_cols - b_cols + 1
    
    out = np.zeros((output_rows, output_cols))
    
    # Perform cross-correlation
    for i in range(output_rows):
        for j in range(output_cols):
            sub_matrix = a[i:i+b_rows, j:j+b_cols]
            out[i, j] = np.sum(sub_matrix * b)
    
    return out

# TODO: Implement CUDA functions -> https://numba.readthedocs.io/en/stable/cuda/kernels.html
@cuda.jit(cache=True)
def _jit_cuda_cross_correlate(matrix1, matrix2, output):
    pass

def _cuda_cross_correlate(matrix1, matrix2):
    pass