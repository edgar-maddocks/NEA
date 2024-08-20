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
    

def _np_cross_correlate(x: np.ndarray, k: np.ndarray) -> np.ndarray:
    """Performs cross correlation between two numpy arrays

    If a CUDA device is available then it will use numba and cuda to perform the cross correlation
    otherwise a default cpu version.

    Args:
        x (np.ndarray): input array - shape(n_samples, *, *)
        k (np.ndarray): filter/kernel - shape(*, *)

    Returns:
        np.ndarray: result of cross-correlation between x and k
    """
    assert len(x.shape) == 3, "Input must be shape (n_samples, *, *)"
    assert len(k.shape) == 2, "Filter must be shape (*, *)"

    x_depth, _, _ = x.shape

    output = None

    use_cuda = cuda.is_available()   

    if use_cuda:
        # print("CUDA DEVICE DETECTED TO USE")
        # print(cuda.detect())
        output = _cuda_cross_correlate2d(x, k)
    else:
        output = _jit_cpu_cross_correlate2d(x, k)
    
    return output

@jit(nopython=True, cache=True)
def _jit_cpu_cross_correlate2d(a: np.ndarray, b: np.ndarray) -> np.ndarray:
    """Performs cross correlation between two numpy arrays - compiled

    Args:
        a (np.ndarray): 
        b (np.ndarray): 

    Returns:
        np.ndarray: 
    """
    a_rows, a_cols = a.shape
    b_rows, b_cols = b.shape

    output_rows = a_rows - b_rows + 1
    output_cols = a_cols - b_cols + 1
    
    output = np.zeros((output_rows, output_cols), dtype=np.float64)
    
    output_rows, output_cols = output.shape
    
    for i in range(output_rows):
        for j in range(output_cols):
            sub_matrix = a[i:i+b_rows, j:j+b_cols]
            output[i, j] = np.sum(sub_matrix * b)
    
    return output

@cuda.jit(cache=True)
def _jit_cuda_cross_correlate2d(a: np.ndarray, b: np.ndarray, output: np.ndarray):
    """ Performs cross correlation on one thread for cuda

    Args:
        a (np.ndarray): 
        b (np.ndarray): 
        output (np.ndarray): 
    """
    # get position of thread
    i, j = cuda.grid(3)
    
    a_rows, a_cols = a.shape
    b_rows, b_cols = b.shape
    
    if i < (a_rows - b_rows + 1) and j < (a_cols - b_cols + 1): # check thread is within bounds
        acc = 0.0
        for di in range(b_rows):
            for dj in range(b_cols):
                acc += a[i + di, j + dj] * b[di, dj]
        output[i, j] = acc

def _cuda_cross_correlate2d(a: np.ndarray, b: np.ndarray):
    """Prepares the shape of blocks and grids, as well as the output array
    and then calls the cuda compiled function

    Args:
        a (np.ndarray): _description_
        b (np.ndarray): _description_
    """
    # prepare blocks and grids and threads
    a_rows, a_cols = a.shape
    b_rows, b_cols = b.shape

    output_shape = (a_rows - b_rows + 1, a_cols - b_cols + 1)
    output = np.zeros(output_shape, dtype=np.float32)

    # bpg calculated according to numba docs
    threads_per_block = (16, 16)
    blocks_per_grid = ((a_rows - b_rows + 1 + threads_per_block[0] - 1) // threads_per_block[0],
                    (a_cols - b_cols + 1 + threads_per_block[1] - 1) // threads_per_block[1])

    _jit_cuda_cross_correlate2d[blocks_per_grid, threads_per_block](a, b, output)

    return output
    


def __compare_cpu_cuda():
    number_samples = 10000
    H_a, W_a = 8, 8
    H_b, W_b = 4, 4

    a = np.random.random((number_samples, H_a, W_a)).astype(np.float32)
    b = np.random.random((H_b, W_b)).astype(np.float32)

    start = t.time()

    result = _cuda_cross_correlate2d(a, b)

    print("TIME TAKEN CUDA: ", t.time() - start)
    print(result.shape)

    start = t.time()

    result = _jit_cpu_cross_correlate2d(a, b)

    print("TIME TAKEN CPU: ", t.time() - start)
    print(result.shape)

def __cross_correlate():
    number_samples = 1000000
    H_a, W_a = 8, 8
    H_b, W_b = 4, 4

    a = np.random.random((number_samples, H_a, W_a)).astype(np.float32)
    b = np.random.random((H_b, W_b)).astype(np.float32)

    start = t.time()

    result = _np_cross_correlate(a, b)

    print("TIME TAKEN: ", t.time() - start)
    print(result.shape)


if __name__ == "__main__":
    __compare_cpu_cuda()
    __cross_correlate()