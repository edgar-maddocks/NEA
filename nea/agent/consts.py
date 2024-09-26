import numpy as np


def create_action_space() -> np.ndarray:
    out = np.empty((8, 8, 8), dtype=object)
    for h in range(8):
        for i in range(8):
            for j in range(8):
                out[h, i, j] = (h, i, j)

    return out


ACTION_SPACE = create_action_space()
