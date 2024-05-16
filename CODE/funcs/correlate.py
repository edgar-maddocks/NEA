import numpy as np
import numba


def correlate1d(input: np.ndarray, filter: np.ndarray, mode="valid"):
    if mode == "valid":
        out_dim = len(input) - len(filter) + 1
        out = np.empty((out_dim,))
        for i in range(out_dim):
            out[i] = np.sum(input[i : i + filter.size] * filter)

    return out


def d_correlate1d(output: np.ndarray, filter: np.ndarray):
    raise NotImplementedError
