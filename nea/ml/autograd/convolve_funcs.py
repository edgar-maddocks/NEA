import numpy as np
from numba import jit
import time as t

@jit(nopython=True, cache=True)
def cpu_forward_convolve2d(output: np.ndarray, x: np.ndarray, k: np.ndarray, n_kernels: int) -> np.ndarray:
    """Performs the forward pass of the Convolve2D Tensor operation

    Args:
        output (np.ndarray): 
        x (np.ndarray): 
        k (np.ndarray): 
        n_kernels (int): 

    Returns:
        np.ndarray: 
    """
    n_samples = x.shape[0]
    for i in range(n_kernels):
            for j in range(n_samples):
                output[i] = _jit_cpu_valid_cross_correlate2d(x[j], k[i, j])
    
    return output

@jit(nopython=True, cache=True)
def cpu_k_backward_convolve2d(output: np.ndarray, x: np.ndarray, dy: np.ndarray, n_kernels: int) -> np.ndarray:
    n_samples = x.shape[0]
    for i in range(n_kernels):
        for j in range(n_samples):
            output[i, j] = _jit_cpu_valid_cross_correlate2d(x[j], dy[i])
        
    return output

@jit(nopython=True, cache=True)
def _jit_cpu_valid_cross_correlate2d(a: np.ndarray, b: np.ndarray) -> np.ndarray:
    """Performs cross correlation between two numpy arrays - compiled

    Args:
        a (np.ndarray): 
        b (np.ndarray): 

    Returns:
        np.ndarray: 
    """
    a_rows, a_cols = a.shape
    b_rows, b_cols = b.shape

    out_rows = a_rows - b_rows + 1
    out_cols = a_cols - b_cols + 1
    
    out = np.zeros((out_rows, out_cols), dtype=np.float64)
    
    out_rows, out_cols = out.shape
    
    for m in range(out_rows):
        for n in range(out_cols):
            sub_matrix = a[m:m+b_rows, n:n+b_cols]

            s = 0.0
            for p in range(b_rows):
                for q in range(b_cols):
                    s += sub_matrix[p, q] * b[p, q]

            out[m, n] = s
    
    return out

def _cpu_time(n_kernels: int, kernel_size: int, samples: int, x_size: int):
    k = np.random.randn(n_kernels, samples, kernel_size, kernel_size)
    x = np.random.randn(samples, x_size, x_size)

    
    output_shape = (n_kernels, x.shape[1] - k.shape[2] + 1, x.shape[2] - k.shape[2] + 1)
    
    new_data = np.zeros(output_shape, dtype=np.float64)

    s = t.time()
    new_data = cpu_forward_convolve2d(new_data, x, k, n_kernels)
    print(new_data)
    print(new_data.shape)
    print("TIME TAKEN: ", t.time() - s)

    """s = t.time()
    new_data = gpu_forward_convolve2d(new_data, x.data, k.data, 4)
    print("TIME TAKEN: ", t.time() - s)"""

if __name__ == "__main__":
    _cpu_time(4, 4, 10000, 8)
    
