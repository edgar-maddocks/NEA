import numpy as np
import pygame

from consts import *

from typing import List


class Window:
    WINDOW = pygame.display.set_mode((WIDTH, HEIGHT))


class Piece:
    def __init__(self, colour) -> None:
        self.colour = colour
        self.king = False

    def crown(self) -> None:
        self.king = True


class Square:
    def __init__(self, row: int, col: int) -> None:
        self.piece = None
        self.row = row
        self.col = col

    def place(self, piece: Piece) -> None:
        self.piece = piece

    def __repr__(self):
        return f"Square - Piece: {self.piece is not None}; Row: {self.row}; Col: {self.col}"


class Board:
    def __init__(self) -> None:
        self.board: List[Square] = self._init_board()

        self.black_remain = self.white_remain = 12
        self.black_kings = self.white_kings = 0

        self.selected_piece = None

    def _init_squares(self, board) -> List[Square]:
        for row in range(SIZE):
            for col in range(SIZE):
                board[row, col] = Square(row, col)

        return board

    def _init_pieces(self, board) -> None:
        assert isinstance(board[0, 0], Square), "Squares have not been intialized"
        for row in range(board.shape[0]):
            if row == 3 or row == 4:
                continue
            colour = BLACK if row <= 2 else WHITE
            for col in range(board.shape[1]):
                if (row + 1) % 2 == 1:
                    if col % 2 == 1:
                        board[row, col].place(Piece(colour))
                else:
                    if col % 2 == 0:
                        board[row, col].place(Piece(colour))

    def _init_board(self):
        board = np.empty((SIZE, SIZE), dtype=Square)
        self._init_squares(board)
        self._init_pieces(board)

        return board

    def render_background(self, WINDOW: Window) -> None:
        WINDOW.fill(BROWN)
        for row in range(SIZE):
            for col in range(row % 2, SIZE, 2):
                pygame.draw.rect(
                    WINDOW, IVORY, (row * SQ_SIZE, col * SQ_SIZE, SQ_SIZE, SQ_SIZE)
                )

    def draw_square(self, WINDOW: Window, square: Square) -> None:
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

    def render(self, WINDOW: Window):
        self.render_background(WINDOW)
        for row in range(self.board.shape[0]):
            for col in range(self.board.shape[1]):
                square = self.board[row, col]
                assert isinstance(square, Square), "Squares have not been intialized"
                if square.piece is not None:
                    self.draw_square(WINDOW, square)

    def move(self, piece: Piece, row: int, col: int) -> None:
        self.board[row, col].place(piece)


if __name__ == "__main__":
    b = Board()
    w = Window()
    play = True

    while play:
        for event in pygame.event.get():
            if event.type == pygame.QUIT:
                play = False
                pygame.quit()

        b.render(w.WINDOW)
        pygame.display.update()
