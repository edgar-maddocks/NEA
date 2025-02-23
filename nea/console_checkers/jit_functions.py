from nea.console_checkers.consts import (
    SIZE,
    ACTION,
)

from numba import jit
import numpy as np


@jit(cache=True)
def _n_black_pieces(board: np.ndarray) -> int:
    """Calculates the number of black pieces remaining on the board

    Returns:
        int: Number of black pieces remaining
    """
    BLACKS = [1, 2]
    n = 0
    for row in range(SIZE):
        for col in range(SIZE):
            if board[row, col] in BLACKS:
                n += 1
    return n


@jit(cache=True)
def _n_white_pieces(board: np.ndarray) -> int:
    """Calculates the number of black pieces remaining on the board

    Returns:
        int: Number of black pieces remaining
    """
    WHITES = [3, 4]
    n = 0
    for row in range(SIZE):
        for col in range(SIZE):
            if board[row, col] in WHITES:
                n += 1
    return n


@jit(cache=True)
def _init_board() -> np.ndarray:
    BLACK_R = 1
    WHITE_R = 3
    board = np.empty((SIZE, SIZE))
    board.fill(0)
    for row in range(SIZE):
        if row == 3 or row == 4:
            continue
        for col in range(SIZE):
            if (row + 1) % 2 == 1:
                if col % 2 == 1:
                    board[row, col] = BLACK_R if row <= 2 else WHITE_R
            else:
                if col % 2 == 0:
                    board[row, col] = BLACK_R if row <= 2 else WHITE_R

    return board


@jit(cache=True)
def _get_valid_take_moves(
    board: np.ndarray, row: int, col: int, player: str
) -> list[ACTION]:
    # have to define constants inside the function for jit to work
    WHITES = [3, 4]
    BLACKS = [1, 2]

    ALL_LEGAL_DIRS = [(+1, -1), (+1, +1), (-1, +1), (-1, -1)]
    BLACK_R_DIRS = [ALL_LEGAL_DIRS[0], ALL_LEGAL_DIRS[1]]
    WHITE_R_DIRS = [ALL_LEGAL_DIRS[2], ALL_LEGAL_DIRS[3]]

    piece = board[row, col]
    valid_moves = []
    if player == "black":
        if piece == 2:  # if the piece is a black king
            for direction in ALL_LEGAL_DIRS:
                if (
                    row + 2 * direction[0] in range(8)
                    and col + 2 * direction[1] in range(8)
                    and board[row + direction[0], col + direction[1]] in WHITES
                    and board[row + 2 * direction[0], col + 2 * direction[1]] == 0
                ):
                    # if square to check is in bounds
                    # and if the square beyond the one diagonally adjacent is empty
                    # and the square diagonally adjacent contains an opponent piece
                    valid_moves.append(
                        (
                            (row, col),
                            (row + 2 * direction[0], col + 2 * direction[1]),
                        )
                    )
                    # add this take move to the list of valids
        elif piece == 1:  # if piece is black regular
            for direction in BLACK_R_DIRS:
                if (
                    row + 2 * direction[0] in range(8)
                    and col + 2 * direction[1] in range(8)
                    and board[row + direction[0], col + direction[1]] in WHITES
                    and board[row + 2 * direction[0], col + 2 * direction[1]] == 0
                ):
                    valid_moves.append(
                        (
                            (row, col),
                            (row + 2 * direction[0], col + 2 * direction[1]),
                        )
                    )
    elif player == "white":
        if piece == 4:  # if pieces is white king
            for direction in ALL_LEGAL_DIRS:
                if (
                    row + 2 * direction[0] in range(8)
                    and col + 2 * direction[1] in range(8)
                    and board[row + direction[0], col + direction[1]] in BLACKS
                    and board[row + 2 * direction[0], col + 2 * direction[1]] == 0
                ):
                    valid_moves.append(
                        (
                            (row, col),
                            (row + 2 * direction[0], col + 2 * direction[1]),
                        )
                    )
        elif piece == 3:  # if piece is white regular
            for direction in WHITE_R_DIRS:
                if (
                    row + 2 * direction[0] in range(8)
                    and col + 2 * direction[1] in range(8)
                    and board[row + direction[0], col + direction[1]] in BLACKS
                    and board[row + 2 * direction[0], col + 2 * direction[1]] == 0
                ):
                    valid_moves.append(
                        (
                            (row, col),
                            (row + 2 * direction[0], col + 2 * direction[1]),
                        )
                    )

    return valid_moves


@jit(cache=True)
def _get_valid_simple_moves(
    board: np.ndarray, row: int, col: int, player: str
) -> list[ACTION]:
    ALL_LEGAL_DIRS = [(+1, -1), (+1, +1), (-1, +1), (-1, -1)]
    BLACK_R_DIRS = [ALL_LEGAL_DIRS[0], ALL_LEGAL_DIRS[1]]
    WHITE_R_DIRS = [ALL_LEGAL_DIRS[2], ALL_LEGAL_DIRS[3]]

    piece = board[row, col]
    valid_moves = []
    if player == "black":
        if piece == 2:
            for direction in ALL_LEGAL_DIRS:
                if (
                    row + direction[0] in range(8)
                    and col + direction[1] in range(8)
                    and board[row + direction[0], col + direction[1]] == 0
                ):
                    valid_moves.append(
                        ((row, col), (row + direction[0], col + direction[1]))
                    )
        elif piece == 1:
            for direction in BLACK_R_DIRS:
                if (
                    row + direction[0] in range(8)
                    and col + direction[1] in range(8)
                    and board[row + direction[0], col + direction[1]] == 0
                ):
                    valid_moves.append(
                        ((row, col), (row + direction[0], col + direction[1]))
                    )
    elif player == "white":
        if piece == 4:
            for direction in ALL_LEGAL_DIRS:
                if (
                    row + direction[0] in range(8)
                    and col + direction[1] in range(8)
                    and board[row + direction[0], col + direction[1]] == 0
                ):
                    valid_moves.append(
                        ((row, col), (row + direction[0], col + direction[1]))
                    )
        elif piece == 3:
            for direction in WHITE_R_DIRS:
                if (
                    row + direction[0] in range(8)
                    and col + direction[1] in range(8)
                    and board[row + direction[0], col + direction[1]] == 0
                ):
                    valid_moves.append(
                        ((row, col), (row + direction[0], col + direction[1]))
                    )

    return valid_moves
