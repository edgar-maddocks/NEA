import numpy as np
from numba import jit, cuda
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
                output[i] = _jit_cpu_cross_correlate2d(x[j], k[i, j])
    
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

@cuda.jit
def _jit_gpu_forward_convolve2d(output: np.ndarray, x: np.ndarray, k: np.ndarray, n_kernels: int):
    """CUDA kernel for performing cross-correlation across multiple samples and kernels."""
    
    # get thread indexes
    kernel_idx = cuda.blockIdx.x
    sample_idx = cuda.blockIdx.y
    i, j = cuda.grid(2)

    # check in bounds
    if kernel_idx >= n_kernels or sample_idx >= x.shape[0]:
        return

    # matrices to cross correlate
    a = x[sample_idx]
    b = k[kernel_idx, sample_idx]
    
    a_rows, a_cols = a.shape
    b_rows, b_cols = b.shape

    output_rows = a_rows - b_rows + 1
    output_cols = a_cols - b_cols + 1

    # cross correlate
    if i < output_rows and j < output_cols:
        sum = 0.0
        for r in range(b_rows):
            for c in range(b_cols):
                sum += a[i + r, j + c] * b[r, c]
        # atomic add prevents adding to incorrect index
        cuda.atomic.add(output, (kernel_idx, i, j), sum)

def gpu_forward_convolve2d(output: np.ndarray, x: np.ndarray, k: np.ndarray, n_kernels: int):
    n_samples = x.shape[0]
    n_kernels, output_rows, output_cols = output.shape
    
    # move data to GPU
    d_x = cuda.to_device(x)
    d_k = cuda.to_device(k)
    d_output = cuda.to_device(output)

    # size grids and blocks (check numba docs)
    threads_per_block = (16, 16)
    blocks_per_grid_x = (output_rows + threads_per_block[0] - 1) // threads_per_block[0]
    blocks_per_grid_y = (output_cols + threads_per_block[1] - 1) // threads_per_block[1]
    blocks_per_grid = (n_kernels, n_samples, blocks_per_grid_x, blocks_per_grid_y)

    # run kernels
    _jit_gpu_forward_convolve2d[blocks_per_grid, threads_per_block](d_output, d_x, d_k, n_kernels)

    # move back to main memory
    return d_output.copy_to_host()

def _compare_cpu_gpu():
    k = np.random.randn(4, 100000, 4, 4)
    x = np.random.randn(100000, 8, 8)

    
    output_shape = (4, x.shape[1] - k.shape[2] + 1, x.shape[2] - k.shape[2] + 1)
    kernels_shape = (4, x.shape[0], k.shape[2], k.shape[2])
    
    new_data = np.zeros(output_shape, dtype=np.float64)

    s = t.time()
    new_data = cpu_forward_convolve2d(new_data, x.data, k.data, 4)
    print(new_data)
    print(new_data.shape)
    print("TIME TAKEN: ", t.time() - s)

    """s = t.time()
    new_data = gpu_forward_convolve2d(new_data, x.data, k.data, 4)
    print("TIME TAKEN: ", t.time() - s)"""

if __name__ == "__main__":
    _compare_cpu_gpu()
    
