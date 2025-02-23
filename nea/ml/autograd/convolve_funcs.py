import numpy as np
from numba import jit


@jit(nopython=True, cache=True)
def cpu_forward_convolve2d(
    output: np.ndarray, x: np.ndarray, k: np.ndarray, n_kernels: int
) -> np.ndarray:
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
def cpu_x_and_k_backward_convolve2d(
    x_output: np.ndarray,
    k_output: np.ndarray,
    x: np.ndarray,
    k: np.ndarray,
    dy: np.ndarray,
    n_samples: int,
    n_kernels: int,
) -> tuple[np.ndarray, np.ndarray]:
    """Calculate the gradients for both input and kernels in the same loop

    Should be used if both x and k requires grad.

    Reduces runtime from [O(n_samples*n_kernels)]^2 to O(n_samples*n_kernels)

    Args:
        x_output (np.ndarray): Array to fill with input grads
        k_output (np.ndarray): Array to fill with kernel grads
        x (np.ndarray): input
        k (np.ndarray): kernels
        dy (np.ndarray): upstream grad
        n_samples (int): number of samples in x (x.shape[0])
        n_kernels (int): number of kernels

    Returns:
        tuple[np.ndarray, np.ndarray]: x_grads, k_grads
    """
    for i in range(n_kernels):
        for j in range(n_samples):
            x_output[j] += _jit_cpu_convolve2d(dy[i], k[i, j])
            k_output[i, j] = _jit_cpu_valid_cross_correlate2d(x[j], dy[i])

    return x_output, k_output


@jit(nopython=True, cache=True)
def cpu_k_backward_convolve2d(
    output: np.ndarray, x: np.ndarray, dy: np.ndarray, n_kernels: int
) -> np.ndarray:
    """Get input gradients for a convolutional layer

    Args:
        output (np.ndarray): array to fill with gradients
        x (np.ndarray): input
        dy (np.ndarray): upstream gradient
        n_kernels (int): number of kernels

    Returns:
        np.ndarray: gradients
    """
    n_samples = x.shape[0]
    for i in range(n_kernels):
        for j in range(n_samples):
            output[i, j] = _jit_cpu_valid_cross_correlate2d(x[j], dy[i])

    return output


@jit(nopython=True, cache=True)
def cpu_x_backward_convolve2d(
    output: np.ndarray, k: np.ndarray, dy: np.ndarray, n_samples: int, n_kernels: int
) -> np.ndarray:
    """Get input gradients for a convolutional layer

    Args:
        output (np.ndarray): array to fill with gradients
        k (np.ndarray): kernels
        dy (np.ndarray): upstream gradient
        n_samples (int): number of samples in input
        n_kernels (int): number of kernels

    Returns:
        np.ndarray: gradients
    """
    for i in range(n_kernels):
        for j in range(n_samples):
            output[j] += _jit_cpu_convolve2d(dy[i], k[i, j])

    return output


@jit(nopython=True, cache=True)
def _jit_cpu_convolve2d(a: np.ndarray, b: np.ndarray) -> np.ndarray:
    """Returns the 2d convolution of two matrices

    (which is equivalent to full cross correlation of a and rot180(b))

    Args:
        a (np.ndarray):
        b (np.ndarray):

    Returns:
        np.ndarray:
    """
    return _jit_cpu_full_cross_correlate2d(a, _jit_rotate_180(b))


@jit(nopython=True, cache=True)
def _jit_rotate_180(b: np.ndarray) -> np.ndarray:
    """Rotates a given matrix by 180 degrees

    Args:
        b (np.ndarray): matrix to rotate

    Returns:
        np.ndarray: rotated matrix
    """
    rot90 = np.rot90(b)
    return np.rot90(rot90)


@jit(nopython=True, cache=True)
def _jit_cpu_full_cross_correlate2d(a: np.ndarray, b: np.ndarray) -> np.ndarray:
    """Performs FULL cross-correlation between two numpy arrays.

    Args:
        a (np.ndarray):
        b (np.ndarray):

    Returns:
        np.ndarray:
    """
    a_rows, a_cols = a.shape
    b_rows, b_cols = b.shape

    # calculate output sizing (note: a+b-1 instead of a -b+1 for valid)
    out_rows = a_rows + b_rows - 1
    out_cols = a_cols + b_cols - 1

    out = np.zeros((out_rows, out_cols), dtype=np.float64)

    # slide b over a - including only partial coverage
    for m in range(out_rows):
        for n in range(out_cols):
            s = 0.0  # sum value

            # compute dot product of a and b that overlap
            for p in range(b_rows):
                for q in range(b_cols):
                    # get positions in a that correspond with b[p, q]
                    a_row = m - p
                    a_col = n - q

                    # check the kernel is in bounds
                    if 0 <= a_row < a_rows and 0 <= a_col < a_cols:
                        s += a[a_row, a_col] * b[p, q]

            out[m, n] = s  # add sum to output

    return out


@jit(nopython=True, cache=True)
def _jit_cpu_valid_cross_correlate2d(a: np.ndarray, b: np.ndarray) -> np.ndarray:
    """Performs VALID cross correlation between two numpy arrays

    Args:
        a (np.ndarray):
        b (np.ndarray):

    Returns:
        np.ndarray:
    """
    a_rows, a_cols = a.shape
    b_rows, b_cols = b.shape

    # calculate output sizing
    out_rows = a_rows - b_rows + 1
    out_cols = a_cols - b_cols + 1

    out = np.zeros((out_rows, out_cols), dtype=np.float64)

    out_rows, out_cols = out.shape

    # slide b over a
    for m in range(out_rows):
        for n in range(out_cols):
            sub_matrix = a[
                m : m + b_rows, n : n + b_cols
            ]  # get the parts of a which overlap with b at that time

            s = 0.0  # sum (numpy sum function was producing errors)
            for p in range(b_rows):
                for q in range(b_cols):
                    s += (
                        sub_matrix[p, q] * b[p, q]
                    )  # computes the dot product of sub_matrix and b

            out[m, n] = s  # add to output array

    return out
