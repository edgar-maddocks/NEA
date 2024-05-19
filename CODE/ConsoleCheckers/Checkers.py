from ConsoleCheckers.consts import *

import numpy as np
import os

from typing import Tuple, Dict, List


def clear_window():
    os.system("cls")


class CheckersBoard:
    def __init__(self) -> None:
        self._board = self._init_board()
        self._last_piece_moved = None
        self._player = WHITE

    @property
    def opposite_player(self) -> str:
        return WHITE if self._player == BLACK else BLACK

    def _init_board(self) -> np.ndarray:
        """Method which returns intial state of the board

        Returns:
            np.ndarray: initial board
        """
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

    def square_is_empty(self, row: int, col: int) -> bool:
        """Function to check if a square is empty

        Args:
            state (np.ndarray): board state
            row (int): row to check
            col (int): column to check

        Returns:
            bool: True if empty False otherwise
        """
        if self._board[row, col] == 0:
            return True
        else:
            return False

    def get_all_valid_moves(
        self, colour: str = None
    ) -> Dict[str, List[Tuple[Tuple, Tuple]]]:
        """Returns dict of all available moves on the board
        "takes": List of take moves
        "simple": List of simple moves

        Returns:
            Dict: List[Tup[Tup, Tup]] -> (piece_to_select, piece_to_move)
        """
        if colour is None:
            colour = self._player
        moves = {"simple": [], "takes": []}
        for row in range(SIZE):
            for col in range(SIZE):
                piece = self._board[row, col]
                if piece in WHITES and colour == WHITE:
                    moves["simple"] += self._get_valid_simple_moves(row, col, colour)
                    moves["takes"] += self._get_valid_take_moves(row, col, colour)
                elif piece in BLACKS and colour == BLACK:
                    moves["simple"] += self._get_valid_simple_moves(row, col, colour)
                    moves["takes"] += self._get_valid_take_moves(row, col, colour)

        return moves

    def _get_valid_simple_moves(self, row: int, col: int, colour: str = None) -> List:
        """Gets all valid simple moves available for a given square

        Args:
            row (int): row the square is on
            col (int): column the square is on

        Returns:
            List: tuple of tuples
        """
        if colour is None:
            colour = self._player
        piece = self._board[row, col]
        valid_moves = []
        if colour == BLACK:
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
        elif colour == WHITE:
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

    def _get_valid_take_moves(self, row: int, col: int, colour: str = None):
        """Gets all valid take moves available for a given square

        Args:
            row (int): row the square is on
            col (int): column the square is on

        Returns:
            List: tuple of tuples
        """
        if colour is None:
            colour = self._player
        piece = self._board[row, col]
        valid_moves = []
        if colour == BLACK:
            if piece == 2:
                for dir in LEGAL_DIRS[BLACK]["king"]:
                    if (
                        row + 2 * dir[0] in range(8)
                        and col + 2 * dir[1] in range(8)
                        and self._board[row + dir[0], col + dir[1]] in WHITES
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
                        and self._board[row + dir[0], col + dir[1]] in WHITES
                        and self.square_is_empty(row + 2 * dir[0], col + 2 * dir[1])
                    ):
                        valid_moves.append(
                            ((row, col), (row + 2 * dir[0], col + 2 * dir[1]))
                        )
        elif colour == WHITE:
            if piece == 4:
                if (
                    row + 2 * dir[0] in range(8)
                    and col + 2 * dir[1] in range(8)
                    and self._board[row + dir[0], col + dir[1]] in BLACKS
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
                        and self._board[row + dir[0], col + dir[1]] in BLACKS
                        and self.square_is_empty(row + 2 * dir[0], col + 2 * dir[1])
                    ):
                        valid_moves.append(
                            ((row, col), (row + 2 * dir[0], col + 2 * dir[1]))
                        )

        return valid_moves

    def render(self) -> None:
        """Renders the board"""
        clear_window()
        cols = ["X", "A", "B", "C", "D", "E", "F", "G", "H"]
        print(str.join(" | ", cols))
        for row in range(SIZE):
            print("----------------------------------")
            print(
                str(8 - row),
                "|",
                str.join(" | ", [NUM_TO_STR[int(x)] for x in self._board[row, :]]),
            )
        print("----------------------------------")

    @staticmethod
    def convert_rowcol_to_user(row: int, col: int) -> Tuple[int, str]:
        """Converts a row column tuple to the way a user sees the board

        Args:
            row (int):
            col (int):

        Returns:
            Tuple[int, str]:
        """
        row = 8 - row
        return row, NUMS_TO_COLS[col]

    @staticmethod
    def convert_rowcol_to_game(row: int, col: str) -> Tuple[int, int]:
        """Converts a row column tuple to the way the game understands
        from what user inputs

        Args:
            row (int):
            col (int):

        Returns:
            Tuple[int, str]:
        """
        row = 8 - row
        return row, COLS_TO_NUMS[col]

    def clear(self, row: int, col: int) -> None:
        """Clears a square

        Args:
            row (int): row square is located
            col (int): column square is located
        """
        self._board[row, col] = 0

    def check_winner(self) -> bool:
        """Checks for a winner

        Returns:
            bool: if game is over
        """
        valid_moves = self.get_all_valid_moves(self.opposite_player)
        if len(valid_moves["takes"]) == 0 and len(valid_moves["simple"]) == 0:
            return True
        else:
            return False

    def step(self, action: Tuple) -> Tuple[bool, np.ndarray, bool, float, Dict]:
        """
        Return Arg is (valid_move, next_obs, done, reward, info)
        """
        info = {}
        piece_to_move, place_to_move_to = action[0], action[1]
        all_valid_moves = self.get_all_valid_moves()
        row, col = piece_to_move[0], piece_to_move[1]
        new_row, new_col = place_to_move_to[0], place_to_move_to[1]

        valid_simples, valid_takes = (
            all_valid_moves["simple"],
            all_valid_moves["takes"],
        )

        valid_moves = (
            [x[1] for x in valid_takes]
            if len(valid_takes) > 0
            else [x[1] for x in valid_simples if x[0] == (row, col)]
        )

        info["fail_cause"] = "invalid move"

        if ((row, col) != piece_to_move) or ((new_row, new_col) not in valid_moves):
            return (False, self._board, False, 0, info)

        self._board[new_row, new_col] = self._board[row, col]
        self.clear(row, col)

        if abs(new_row - row) == 2:
            one_row = 0.5 * (new_row - row)
            one_col = 0.5 * (new_col - col)
            self.clear(int(row + one_row), int(col + one_col))
            self._player = self.opposite_player

        else:
            self._player = self.opposite_player

        if self.check_winner():
            return (True, self._board, True, -1, info)
        else:
            return (True, self._board, False, 0, info)

            """if abs(new_row - row) == 2:
                one_row = 0.5 * (new_row - row)
                one_col = 0.5 * (new_col - col)
                self.clear(int(row + one_row), int(col + one_col))
                self._last_piece_moved = (new_row, new_col)
                return (True, self._board, False, 0, info)
            else:
                self._player = self.opposite_player
                self._last_piece_moved = None
                return (True, self._board, False, 0, info)"""

        """elif self._last_piece_moved is not None:
            all_valid_moves = self._get_valid_take_moves(*self._last_piece_moved)
            if len(all_valid_moves) == 0:
                self._player = self.opposite_player
                self._last_piece_moved = None
                return (True, self._board, False, 0, info)
            else:
                row, col = self._last_piece_moved
                valid_moves = [x[1] for x in all_valid_moves]
                new_row, new_col = place_to_move_to[0], place_to_move_to[1]

                info["fail_cause"] = "invalid move"

                if ((row, col) != piece_to_move) or (
                    (new_row, new_col) not in valid_moves
                ):
                    return (False, self._board, False, 0, info)

            self._board[new_row, new_col] = self._board[row, col]
            self.clear(row, col)

            if abs(new_row - row) == 2:
                one_row = 0.5 * (new_row - row)
                one_col = 0.5 * (new_col - col)
                self.clear(int(row + one_row), int(col + one_col))
                self._last_piece_moved = (new_row, new_col)
                return (True, self._board, False, 0, info)

            else:
                self._player = self.opposite_player
                self._last_piece_moved = None
                return (True, self._board, False, 0, info)"""

    @property
    def board(self) -> np.ndarray:
        """Current state

        Returns:
            np.ndarray: current board state
        """
        return self._board


class CheckersGame:
    def __init__(self):
        raise NotImplementedError
