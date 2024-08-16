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

def _np_cross_correlate(x: np.ndarray, k: np.ndarray) -> np.ndarray:
    """Performs cross correlation between two numpy arrays

    If a CUDA device is available then it will use numba and cuda to perform the cross correlation
    otherwise a default cpu version.

    Args:
        x (np.ndarray): input array
        k (np.ndarray): filter/kernel

    Returns:
        np.ndarray: result of cross-correlation between x and k
    """
    assert len(x.shape) == 3, "Input must be shape (n_samples, *, *)"
    assert len(k.shape) == 2, "Filter must be shape (*, *)"

    x_depth, x_rows, x_cols = x.shape
    k_rows, k_cols = k.shape

    output = None

    use_cuda = cuda.is_available() and x_depth > 100000000     

    if use_cuda:
        print("CUDA DEVICE DETECTED TO USE")
        print(cuda.detect())
        output = _cuda_cross_correlate(x, k, output)
    elif x_depth >= 10000:
        output = _jit_cpu_cross_correlate(x, k)
    
    return output

@numba.jit(nopython=True, cache=True)
def _jit_cpu_cross_correlate(a: np.ndarray, b: np.ndarray) -> np.ndarray:
    a_depth, a_rows, a_cols = a.shape
    b_rows, b_cols = b.shape

    output_rows = a_rows - b_rows + 1
    output_cols = a_cols - b_cols + 1
    
    output = np.zeros((a_depth, output_rows, output_cols), dtype=np.float64)
    
    output_depth, output_rows, output_cols = output.shape
    
    for d in range(output_depth):
        for i in range(output_rows):
            for j in range(output_cols):
                sub_matrix = a[d, i:i+b_rows, j:j+b_cols]
                output[d, i, j] = np.sum(sub_matrix * b)
    
    return output

# TODO: Implement CUDA functions -> https://numba.readthedocs.io/en/stable/cuda/kernels.html
@cuda.jit(cache=True)
def _jit_cuda_cross_correlate(a: np.ndarray, b: np.ndarray, output: np.ndarray):
    # actually performs cross correlation
    pass

def _cuda_cross_correlate(a: np.ndarray, b: np.ndarray, output: np.ndarray):
    ## prepare blocks and grids and threads
    pass

if __name__ == "__main__":
    a = np.random.randn(100000, 32, 32)
    b = np.random.randn(6, 6)

    start = t.time()

    result = _np_cross_correlate(a, b)

    print("TIME TAKEN: ", t.time() - start)
    print("RESULT SHAPE: ", result.shape)