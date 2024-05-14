import numpy as np
from typing import Tuple, List

from consts import *


class Checkers:
    def __init__(self) -> None:
        self.board = self._init_board()
        self.player = WHITE

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

        if self.square_is_empty(row, col) is True:
            return self.select_piece()

        return row, col

    def square_is_empty(self, row: int, col: int) -> bool:
        if self.board[row, col] == 0:
            return True
        else:
            return False

    def get_all_valid_moves(self):
        valid_simple = []
        valid_takes = []
        for row in range(SIZE):
            for col in range(SIZE):
                piece = self.board[row, col]
                if piece in WHITES and self.player == WHITE:
                    valid_simple += self._get_valid_simple_moves(row, col)
                    valid_takes += self._get_valid_take_moves(row, col)
                elif piece in BLACKS and self.player == BLACK:
                    valid_simple += self._get_valid_simple_moves(row, col)
                    valid_takes += self._get_valid_take_moves(row, col)

        if len(valid_takes) > 0:
            return valid_takes

        return valid_simple

    def _get_valid_moves(self, row: int, col: int):
        valid_simple = []
        valid_takes = []
        valid_simple += self._get_valid_simple_moves(row, col)
        valid_takes += self._get_valid_take_moves(row, col)

        if len(valid_takes) > 0:
            return valid_takes

        return valid_simple

    def _get_valid_simple_moves(self, row: int, col: int):
        piece = self.board[row, col]
        valid_moves = []
        if self.player == BLACK:
            if piece == 2:
                for dir in LEGAL_DIRS[BLACK]["king"]:
                    if (
                        row + dir[0] in range(8)
                        and col + dir[1] in range(8)
                        and self.square_is_empty(row + dir[0], col + dir[1])
                    ):
                        valid_moves.append(((row, col), (row + dir[0], col + dir[1])))
            elif piece == 1:
                for dir in LEGAL_DIRS[BLACK]["regular"]:
                    if (
                        row + dir[0] in range(8)
                        and col + dir[1] in range(8)
                        and self.square_is_empty(row + dir[0], col + dir[1])
                    ):
                        valid_moves.append(((row, col), (row + dir[0], col + dir[1])))
        elif self.player == WHITE:
            if piece == 4:
                for dir in LEGAL_DIRS[WHITE]["king"]:
                    if (
                        row + dir[0] in range(8)
                        and col + dir[1] in range(8)
                        and self.square_is_empty(row + dir[0], col + dir[1])
                    ):
                        valid_moves.append(((row, col), (row + dir[0], col + dir[1])))
            elif piece == 3:
                for dir in LEGAL_DIRS[WHITE]["regular"]:
                    if (
                        row + dir[0] in range(8)
                        and col + dir[1] in range(8)
                        and self.square_is_empty(row + dir[0], col + dir[1])
                    ):
                        valid_moves.append(((row, col), (row + dir[0], col + dir[1])))

        return valid_moves

    def _get_valid_take_moves(self, row: int, col: int):
        piece = self.board[row, col]
        valid_moves = []
        if self.player == BLACK:
            if piece == 2:
                for dir in LEGAL_DIRS[BLACK]["king"]:
                    if (
                        row + 2 * dir[0] in range(8)
                        and col + 2 * dir[1] in range(8)
                        and self.board[row + dir[0], col + dir[1]] in WHITES
                        and self.square_is_empty(row + 2 * dir[0], col + 2 * dir[1])
                    ):
                        valid_moves.append(
                            ((row, col), (row + 2 * dir[0], col + 2 * dir[1]))
                        )
            elif piece == 1:
                for dir in LEGAL_DIRS[BLACK]["regular"]:
                    if (
                        row + 2 * dir[0] in range(8)
                        and col + 2 * dir[1] in range(8)
                        and self.board[row + dir[0], col + dir[1]] in WHITES
                        and self.square_is_empty(row + 2 * dir[0], col + 2 * dir[1])
                    ):
                        valid_moves.append(
                            ((row, col), (row + 2 * dir[0], col + 2 * dir[1]))
                        )
        elif self.player == WHITE:
            if piece == 4:
                if (
                    row + 2 * dir[0] in range(8)
                    and col + 2 * dir[1] in range(8)
                    and self.board[row + dir[0], col + dir[1]] in BLACKS
                    and self.square_is_empty(row + 2 * dir[0], col + 2 * dir[1])
                ):
                    valid_moves.append(
                        ((row, col), (row + 2 * dir[0], col + 2 * dir[1]))
                    )
            elif piece == 3:
                for dir in LEGAL_DIRS[WHITE]["regular"]:
                    if (
                        row + 2 * dir[0] in range(8)
                        and col + 2 * dir[1] in range(8)
                        and self.board[row + dir[0], col + dir[1]] in BLACKS
                        and self.square_is_empty(row + 2 * dir[0], col + 2 * dir[1])
                    ):
                        valid_moves.append(
                            ((row, col), (row + 2 * dir[0], col + 2 * dir[1]))
                        )

        return valid_moves

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

    @staticmethod
    def convert_rowcol_to_user(row: int, col: int) -> Tuple[int, str]:
        row = 8 - row
        return row, NUMS_TO_COLS[col]

    @staticmethod
    def convert_rowcol_to_game(row: int, col: str) -> Tuple[int, int]:
        row = 8 - row
        return row, COLS_TO_NUMS[col]

    def select_move(self):
        valid_row = False
        valid_col = False

        while valid_row is False:
            row = input("Enter row to select place to move to: ")
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
            col = input("Enter column to move to: ")
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

        if self.square_is_empty(row, col) is False:
            return self.select_move()

        return row, col

    def clear(self, row: int, col: int) -> None:
        self.board[row, col] = 0

    @property
    def opposite_player(self):
        return WHITE if self.player == BLACK else BLACK

    def move(self):
        valid_selection = False
        row, col = None, None
        while valid_selection is False:
            print(
                f"Valid pieces to move are {set([Checkers.convert_rowcol_to_user(*x[0]) for x in self.get_all_valid_moves()])}"
            )

            row, col = self.select_piece()
            if (row, col) not in [x[0] for x in self.get_all_valid_moves()]:
                print("Please select a piece with a valid move")
                continue
            valid_selection = True

        valid_move = False
        while valid_move is False:
            print(
                f"Valid moves for this piece are {set([Checkers.convert_rowcol_to_user(*x[1]) for x in self._get_valid_moves(row, col)])}"
            )
            new_row, new_col = self.select_move()
            if (new_row, new_col) not in [x[1] for x in self.get_all_valid_moves()]:
                print("Please select a valid move")
                continue
            valid_move = True

        self.board[new_row, new_col] = self.board[row, col]
        self.clear(row, col)

        if abs(new_row - row) == 2:
            one = 0.5 * (new_row - row)
            self.clear(int(row + one), int(col + one))

        self.player = self.opposite_player


game = Checkers()
game.render()
while True:
    print(f"It is {game.player}s turn")
    game.move()
    game.render()
