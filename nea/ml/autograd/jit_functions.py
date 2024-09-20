from numba import jit
import numpy as np


@jit(nopython=True, cache=True)
def fill_padded_array(
    array_to_fill: np.ndarray, array_fill_with: np.ndarray, padding: int
) -> np.ndarray:
    samples, _, _ = array_fill_with.shape
    for sample in range(samples):
        array_to_fill[sample, padding:-padding, padding:-padding] = array_fill_with[
            sample
        ]

    return array_to_fill
