from nea.console_checkers.consts import (
    BLACK,
    WHITE,
    SIZE,
    BLACKS,
    WHITES,
    ACTION,
    COLS_TO_NUMS,
    NUMS_TO_COLS,
    NUM_TO_STR,
    USER_DISPLAY_ACTION,
)

# from nea.console_checkers.utils import clear_window
from nea.console_checkers import jit_functions

import numpy as np


class CheckersGame:
    """
    Holds basic logic and console rendering of a checkers game
    """

    def __init__(self) -> None:
        """
        Creates a new CheckersGame object
        """
        self._board = self._init_board()
        self._last_moved_piece: tuple[int, int] = None
        self._player = WHITE
        self._moves_no_capture = 0
        self._switch_player = None

    @property
    def opposite_player(self) -> None:
        """Returns the opposite to the current player attribute

        Returns:
            string: "black" or "white"
        """
        if self._player == WHITE:
            return BLACK
        elif self._player == BLACK:
            return WHITE

    @property
    def player(self) -> str:
        """Returns protected player attribute

        Returns:
            str: current player
        """
        return self._player

    @property
    def n_black_pieces(self) -> int:
        """Calculates the number of black pieces remaining on the board

        Returns:
            int: Number of black pieces remaining
        """
        return jit_functions._n_black_pieces(self._board)

    @property
    def n_white_pieces(self) -> int:
        """Calculates the number of white pieces remaining on the board

        Returns:
            int: Number of white pieces remaining
        """
        return jit_functions._n_white_pieces(self.board)

    @property
    def n_opposite_player_pieces(self) -> int:
        """Calculates the number of pieces the opposing player has.
        Opposing player is determined by value of self._player attribute.

        Returns:
            int: number of pieces opponent has
        """
        if self._player == WHITE:
            return self.n_black_pieces
        elif self._player == BLACK:
            return self.n_white_pieces

    @property
    def board(self) -> np.ndarray:
        """Returns protected board attribute

        Returns:
            np.ndarray: state of board
        """
        return self._board

    def _init_board(self) -> np.ndarray:
        """Method which returns intial state of the board

        Returns:
            np.ndarray: initial board state
        """
        return jit_functions._init_board()

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

    def get_all_valid_moves(self) -> dict[str, list[ACTION]]:
        """Returns a dictionary of take and simple moves.
        Does not account for if a double moves are available.

        Keys:
            Takes moves: "takes"
            Simple moves: "simple"

        Returns:
            dict[str, list[ACTION]]: Dictionary of available moves
        """
        moves = {"takes": [], "simple": []}
        for row in range(SIZE):
            for col in range(SIZE):
                piece = self._board[row, col]
                if piece in WHITES and self._player == WHITE:
                    moves["simple"] += self._get_valid_simple_moves(
                        row, col, self._player
                    )
                    moves["takes"] += self._get_valid_take_moves(row, col, self._player)
                elif piece in BLACKS and self._player == BLACK:
                    moves["simple"] += self._get_valid_simple_moves(
                        row, col, self._player
                    )
                    moves["takes"] += self._get_valid_take_moves(row, col, self._player)

        return moves

    def _get_valid_take_moves(self, row: int, col: int, player: str):
        """
        Gets all valid take moves available for a given square

        Args:
            row (int): row the square is on
            col (int): column the square is on
            player (str): player to check if moves are available for

        Returns:
            list: tuple of tuples
        """
        return jit_functions._get_valid_take_moves(self._board, row, col, player)

    def _get_valid_simple_moves(self, row: int, col: int, player: str) -> list[ACTION]:
        """Gets all valid simple moves available for a given square

        Args:
            row (int): row the square is on
            col (int): column the square is on
            player (str): player to check if moves are available for

        Returns:
            list: tuple of tuples
        """
        return jit_functions._get_valid_simple_moves(self._board, row, col, player)

    def moves_available_for_opposite_player(self) -> bool:
        """Returns a dictionary of take and simple moves available to the other player

        Keys:
            Takes moves: "takes"
            Simple moves: "simple"

        Returns:
            dict[str, list[ACTION]]: Dictionary of available moves
        """
        moves = {"takes": [], "simple": []}
        for row in range(SIZE):
            for col in range(SIZE):
                piece = self._board[row, col]
                if piece in WHITES and self.opposite_player == WHITE:
                    moves["simple"] += self._get_valid_simple_moves(
                        row, col, self.opposite_player
                    )
                    moves["takes"] += self._get_valid_take_moves(
                        row, col, self.opposite_player
                    )
                elif piece in BLACKS and self.opposite_player == BLACK:
                    moves["simple"] += self._get_valid_simple_moves(
                        row, col, self.opposite_player
                    )
                    moves["takes"] += self._get_valid_take_moves(
                        row, col, self.opposite_player
                    )

        return (len(moves["takes"]) == 0) and (len(moves["simple"]) == 0)

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

    def step(self, action: ACTION) -> tuple[bool, np.ndarray, bool, float]:
        """Completes a step given an action in the board environment

        Args:
            action (ACTION): Desired action to take

        Returns:
            tuple[bool, np.ndarray, bool, float]: (valid_move, next_obs, done, reward)
        """
        self._switch_player = True
        rowcol_move_from, rowcol_move_to = action[0], action[1]
        if self._last_moved_piece is None:
            all_valid_moves = self.get_all_valid_moves()
            if (
                len(all_valid_moves["takes"]) == 0
                and len(all_valid_moves["simple"]) == 0
            ):
                return (True, self._board, True, -1)

            valid_moves_for_turn = (
                all_valid_moves["takes"]
                if len(all_valid_moves["takes"]) > 0
                else all_valid_moves["simple"]
            )

            if action not in valid_moves_for_turn:
                return (False, self._board, False, 0)
            else:
                self._board[*rowcol_move_to] = self._board[*rowcol_move_from]
                self.clear(*rowcol_move_from)
                self._moves_no_capture += 1

        elif self._last_moved_piece is not None:
            valid_moves_for_turn = self._get_valid_take_moves(
                *self._last_moved_piece, self._player
            )

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
            self._last_moved_piece = row_to, col_to
            double_moves = self._get_valid_take_moves(
                *self._last_moved_piece, self._player
            )
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
        elif self.moves_available_for_opposite_player():
            return (True, self._board, True, 1)
        else:
            if self._switch_player:
                self._player = self.opposite_player
            return (True, self._board, False, 0)

    @staticmethod
    def convert_rowcol_to_game(row: int, col: str) -> tuple[int, int]:
        """Converts a row column tuple to the way the game understands
        from what user inputs

        Args:
            row (int):
            col (int):

        Returns:
            tuple[int, str]:
        """
        row = 8 - row
        return row, COLS_TO_NUMS[col]

    @staticmethod
    def convert_rowcol_to_user(row: int, col: int) -> tuple[int, str]:
        """Converts a row column tuple to the way a user sees the board

        Args:
            row (int):
            col (int):

        Returns:
            tuple[int, str]:
        """
        row = 8 - row
        return row, NUMS_TO_COLS[col]

    @staticmethod
    def convert_action_to_user(action: ACTION) -> USER_DISPLAY_ACTION:
        """Takes in an ACTION and displays it in a userfriendly format

        Args:
            action (ACTION): action to convert

        Returns:
            USER_DISPLAY_ACTION:
        """
        user_disp_action = ()
        for tup in action:
            user_disp_action += CheckersGame.convert_rowcol_to_user(*tup)

        return user_disp_action

    def render(self) -> None:
        """Renders the board"""
        # clear_window()
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
        # print("MOVES NO CAPTURE: ", self._moves_no_capture)
        print("----------------------------------")
