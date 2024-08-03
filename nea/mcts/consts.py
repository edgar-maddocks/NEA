from typing import Tuple

ACTION_TO_IDX = {
    (1, 1): 0,
    (1, -1): 1,
    (-1, 1): 2,
    (-1, -1): 3,
    (2, 2): 4,
    (2, -2): 5,
    (-2, 2): 6,
    (-2, -2): 7,
}

IDX_TO_ACTION = {v: k for k, v in ACTION_TO_IDX.items()}

ACTION = Tuple[Tuple[int, int], Tuple[int, int]]
