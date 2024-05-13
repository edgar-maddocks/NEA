import numpy as np
from typing import Tuple

from consts import *


class Checkers:
    def __init__(self) -> None:
        self.board = self._init_board()

    def _init_board(self) -> np.ndarray:
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

    def select_piece(self) -> Tuple[int, int]:
        valid_row = False
        valid_col = False

        while valid_row is False:
            row = input("Enter row to select piece from: ")
            try:
                row = int(row)
            except:
                print("selected row must be an integer")
                continue
            if row not in range(1, 9):
                print("row must be in range of 1-8, shown on left side of board")

            row = 8 - row
            valid_row = True

        while valid_col is False:
            col = input("Enter column to select from: ")
            try:
                col = COLS_TO_NUMS[col]
            except:
                print("selected columns must be a letter between A-H")
                continue
            if col not in range(0, 8):
                print(
                    "column must be an letter between A-H shown at the top of the board"
                )

            valid_col = True

        return row, col

    def render(self) -> None:
        cols = ["X", "A", "B", "C", "D", "E", "F", "G", "H"]
        print(str.join(" | ", cols))
        for row in range(SIZE):
            print("----------------------------------")
            print(
                str(8 - row),
                "|",
                str.join(" | ", [NUM_TO_STR[int(x)] for x in self.board[row, :]]),
            )
        print("----------------------------------")


game = Checkers()
game.render()
print(game.select_piece())
