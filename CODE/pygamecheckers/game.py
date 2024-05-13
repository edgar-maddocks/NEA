import numpy as np
import pygame

from consts import *

from typing import List


WINDOW = pygame.display.set_mode((WIDTH, HEIGHT))


class Piece:
    def __init__(self, row: int, col: int, colour) -> None:
        self.colour = colour
        self.king = False
        self.row = row
        self.col = col

    def crown(self) -> None:
        self.king = True


class Square:
    def __init__(self, row: int, col: int) -> None:
        self.piece = None
        self.row = row
        self.col = col

    def place(self, piece: Piece, row: int, col: int) -> None:
        self.piece = piece
        self.piece.row = row
        self.piece.col = col

    def clear(self) -> None:
        self.piece = None

    def __repr__(self) -> str:
        return f"Square - Piece: {self.piece is not None}; Row: {self.row}; Col: {self.col}"


class Board:
    def __init__(self) -> None:
        self.board: List[Square] = self._init_board()

        self.black_remain = self.white_remain = 12
        self.black_kings = self.white_kings = 0
        self.pieces: List[Piece] = []

    def _init_squares(self, board) -> List[Square]:
        for row in range(SIZE):
            for col in range(SIZE):
                board[row, col] = Square(row, col)

        return board

    def _init_pieces(self, board) -> List[Square]:
        assert isinstance(board[0, 0], Square), "Squares have not been intialized"
        for row in range(board.shape[0]):
            if row == 3 or row == 4:
                continue
            colour = BLACK if row <= 2 else WHITE
            for col in range(board.shape[1]):
                if (row + 1) % 2 == 1:
                    if col % 2 == 1:
                        board[row, col].place(Piece(row, col, colour), row, col)
                else:
                    if col % 2 == 0:
                        board[row, col].place(Piece(row, col, colour), row, col)

    def _init_board(self) -> List[Square]:
        board = np.empty((SIZE, SIZE), dtype=Square)
        self._init_squares(board)
        self._init_pieces(board)

        return board

    def render_background(self, WINDOW) -> None:
        WINDOW.fill(BROWN)
        for row in range(SIZE):
            for col in range(row % 2, SIZE, 2):
                pygame.draw.rect(
                    WINDOW, IVORY, (row * SQ_SIZE, col * SQ_SIZE, SQ_SIZE, SQ_SIZE)
                )

    def draw_square(self, WINDOW, square: Square) -> None:
        if square.piece.king is False:
            pygame.draw.circle(
                WINDOW,
                square.piece.colour,
                (
                    square.col * SQ_SIZE + (SQ_SIZE // 2),
                    square.row * SQ_SIZE + (SQ_SIZE // 2),
                ),
                SQ_SIZE - PIECE_PADDING,
            )
        if square.piece.king is True:
            pygame.draw.circle(
                WINDOW,
                square.piece.colour,
                (
                    square.col * SQ_SIZE + (SQ_SIZE // 2),
                    square.row * SQ_SIZE + (SQ_SIZE // 2),
                ),
                SQ_SIZE - PIECE_PADDING,
            )
            pygame.draw.circle(
                WINDOW,
                GOLD,
                (
                    square.col * SQ_SIZE + (SQ_SIZE // 2),
                    square.row * SQ_SIZE + (SQ_SIZE // 2),
                ),
                SQ_SIZE - CROWN_PADDING,
            )

    def render(self, WINDOW) -> None:
        self.render_background(WINDOW)
        for row in range(self.board.shape[0]):
            for col in range(self.board.shape[1]):
                square = self.board[row, col]
                assert isinstance(square, Square), "Squares have not been intialized"
                if square.piece is not None:
                    self.draw_square(WINDOW, square)

    def move(self, piece: Piece, row: int, col: int) -> None:
        assert isinstance(self.board[row, col], Square), "Squares not initialized"
        assert self.board[row, col].piece == None, "Square is not empty"
        self.board[piece.row, piece.col].clear()
        self.board[row, col].place(piece, row, col)

        if piece.colour == BLACK and row == 7:
            piece.crown()
        elif piece.colour == WHITE and row == 0:
            piece.crown()

    def get_valid_moves(self, piece: Piece):
        pass

    def get_winner(self):
        if self.white_remain <= 0:
            return WHITE
        elif self.black_remain <= 0:
            return WHITE

        return None

    def get_square(self, row, col):
        return self.board[row, col]

    def get_piece(self, row, col):
        return self.board[row, col].piece


class Checkers:
    def __init__(self, WINDOW) -> None:
        self._init()
        self.WINDOW = WINDOW

    def render(self) -> None:
        self.board.render(WINDOW)
        pygame.display.update()

    def _init(self) -> None:
        self.selected_piece = None
        self.board = Board()
        self.player = WHITE
        self.valid_moves = []

    def reset(self) -> None:
        self._init()

    def select(self, row: int, col: int):
        if self.selected:
            result = self._move(row, col)
            if not result:
                self.selected = None
                self.select(row, col)

        piece = self.board.get_piece(row, col)
        if piece != None and piece.color == self.player:
            self.selected = piece
            self.valid_moves = self.board.get_valid_moves(piece)
            return True

        return False

    def _move(self, row, col):
        pass


if __name__ == "__main__":
    checkers = Checkers(WINDOW)
    play = True

    ticks = 0
    clock = pygame.time.Clock()
    while play:

        clock.tick(FPS)
        ticks += 1

        for event in pygame.event.get():
            if event.type == pygame.QUIT:
                play = False
                pygame.quit()

        checkers.render()
