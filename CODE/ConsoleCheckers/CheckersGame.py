from ConsoleCheckers.consts import *
from ConsoleCheckers.util_funcs import *

import numpy as np
import os

from typing import Tuple, Dict, List


class CheckersGame:
    def __init__(self) -> None:
        """
        Creates a new CheckersGame object
        """
        self._board = self._init_board()
        self._last_moved_piece: Tuple[int, int] = None
        self._player = WHITE
        self._moves_no_capture = 0
        self._switch_player = None

    @property
    def _opposite_player(self) -> None:
        """Returns the opposite to the current player attribute

        Returns:
            string: "black" or "white"
        """
        if self._player == WHITE:
            return BLACK
        elif self._player == BLACK:
            return WHITE
        
    @property
    def player(self):
        return self._player
    
    @property
    def n_black_pieces(self):
        """Calculates the number of black pieces remaining on the board

        Returns:
            int: Number of black pieces remaining
        """
        n = 0
        for row in range(SIZE):
            for col in range(SIZE):
                if self._board[row, col] in BLACKS:
                    n+=1
        return n
    
    @property
    def n_white_pieces(self):
        """Calculates the number of white pieces remaining on the board

        Returns:
            int: Number of white pieces remaining
        """
        n = 0
        for row in range(SIZE):
            for col in range(SIZE):
                if self._board[row, col] in WHITES:
                    n+=1
        return n
    
    @property
    def n_opposite_player_pieces(self):
        """Calculates the number of pieces the opposing player has.
        Opposing player is determined by value of self._player attribute.

        Returns:
            _type_: _description_
        """
        if self._player == WHITE:
            return self.n_black_pieces
        elif self._player == BLACK:
            return self.n_white_pieces
        
    @property
    def board(self):
        return self._board
    
    def _init_board(self) -> np.ndarray:
        """Method which returns intial state of the board

        Returns:
            np.ndarray: initial board state
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
            row (int): row to check
            col (int): column to check

        Returns:
            bool: True if empty False otherwise
        """
        if self._board[row, col] == 0:
            return True
        else:
            return False
    
    def get_all_valid_moves(self) -> Dict[str, List[ACTION]]:
        """Returns a dictionary of take and simple moves.
        Does not account for if a double moves are available. 

        Keys:
            Takes moves: "takes"
            Simple moves: "simple"

        Returns:
            Dict[str, List[ACTION]]: Dictionary of available moves
        """
        moves = {"takes": [], "simple": []}
        for row in range(SIZE):
            for col in range(SIZE):
                piece = self._board[row, col]
                if piece in WHITES and self._player == WHITE:
                    moves["simple"] += self._get_valid_simple_moves(row, col)
                    moves["takes"] += self._get_valid_take_moves(row, col)
                elif piece in BLACKS and self._player == BLACK:
                    moves["simple"] += self._get_valid_simple_moves(row, col)
                    moves["takes"] += self._get_valid_take_moves(row, col)

        return moves
    
    def _get_valid_take_moves(self, row: int, col: int) -> List[ACTION]:
        """Gets all valid take moves available for a given square

        Args:
            row (int): row the square is on
            col (int): column the square is on

        Returns:
            List: tuple of tuples
        """
        piece = self._board[row, col]
        valid_moves = []
        if self._player == BLACK:
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
        elif self._player == WHITE:
            if piece == 4:
                for dir in LEGAL_DIRS[WHITE]["king"]:
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
    
    def _get_valid_simple_moves(self, row: int, col: int) -> List[ACTION]:
        """Gets all valid simple moves available for a given square

        Args:
            row (int): row the square is on
            col (int): column the square is on

        Returns:
            List: tuple of tuples
        """
        piece = self._board[row, col]
        valid_moves = []
        if self._player == BLACK:
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
        elif self._player == WHITE:
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

    def clear(self, row: int, col: int) -> None:
        """Clears a square

        Args:
            row (int): row square is located
            col (int): column square is located
        """
        self._board[row, col] = 0

    def crown(self, row: int, col: int) -> None:
        """Crowns a piece

        Args:
            row (int): row of piece to crown
            col (int): column of piece to crown
        """
        piece = self._board[row, col]
        if piece == 1:
            self._board[row, col] = 2
        elif piece == 3:
            self._board[row, col] = 4

    def step(self, action: ACTION) -> Tuple[bool, np.ndarray, bool, float]:
        """Completes a step given an action in the board environment

        Args:
            action (ACTION): Desired action to take

        Returns:
            Tuple[bool, np.ndarray, bool, float]: (valid_move, next_obs, done, reward)
        """
        self._switch_player = True
        rowcol_move_from, rowcol_move_to = action[0], action[1]
        if self._last_moved_piece is None:
            all_valid_moves = self.get_all_valid_moves()
            if len(all_valid_moves["takes"]) == 0 and len(all_valid_moves["simple"]) == 0:
                return (True, self._board, True, -1)
            
            valid_moves_for_turn = all_valid_moves["takes"] if len(all_valid_moves["takes"]) > 0 else all_valid_moves["simple"]

            if action not in valid_moves_for_turn:
                return (False, self._board, False, 0)
            else:
                self._board[*rowcol_move_to] = self._board[*rowcol_move_from]
                self.clear(*rowcol_move_from)
                self._moves_no_capture += 1

        elif self._last_moved_piece is not None:
            valid_moves_for_turn = self._get_valid_take_moves(*self._last_moved_piece)
            
            if action not in valid_moves_for_turn:
                return (False, self._board, False, 0)
            else:
                self._board[*rowcol_move_to] = self._board[*rowcol_move_from]
                self.clear(*rowcol_move_from)

        row_from, col_from = rowcol_move_from
        row_to, col_to = rowcol_move_to
        if abs(row_to - row_from) == 2:
            one_row = 0.5 * (row_to - row_from)
            one_col = 0.5 * (col_to - col_from)
            self.clear(int(row_from + one_row), int(col_from + one_col))
            self._moves_no_capture = 0
            self._last_piece_moved = row_to, col_to
            double_moves = self._get_valid_take_moves(*self._last_piece_moved)
            if len(double_moves) == 0:
                self._last_moved_piece = None
            else:
                self._switch_player = False

        if self._board[row_to, col_to] in WHITES and row_to == 0:
            self.crown(row_to, col_to)
        if self._board[row_to, col_to] in BLACKS and row_to == 7:
            self.crown(row_to, col_to)

        if self._moves_no_capture == 40:
            return (True, self._board, True, 0)
        elif self.n_black_pieces == 1 and self.n_white_pieces == 1:
            return (True, self._board, True, 0)
        elif self.n_opposite_player_pieces == 0:
            return (True, self._board, True, 1)
        else:
            if self._switch_player:
                self._player = self._opposite_player
            return(True, self._board, False, 0)

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
            print("TURN: ", self._player)
            print("MOVES NO CAPTURE: ", self._moves_no_capture)
            print("----------------------------------")