import numpy as np
import pygame

from consts import *


class Window:
    WINDOW = pygame.display.set_mode((WIDTH, HEIGHT))


class Piece:
    def __init__(self, row: int, col: int, colour) -> None:
        self.row = row
        self.col = col
        self.colour = colour
        self.king = False

    def crown(self) -> None:
        self.king = True


class Board:
    def __init__(self) -> None:
        self.board = []

        self.black_remain = self.white_remain = 12
        self.black_kings = self.white_kings = 0

        self.selected_piece = None

    def render_squares(self, WINDOW: Window) -> None:
        WINDOW.fill(IVORY)
        for row in range(SIZE):
            for col in range(row % 2, SIZE, 2):
                pygame.draw.rect(
                    WINDOW, BROWN, (row * SQ_SIZE, col * SQ_SIZE, SQ_SIZE, SQ_SIZE)
                )

    def draw_piece(self, WINDOW: Window, piece: Piece) -> None:
        if piece.king is False:
            pygame.draw.circle(
                WINDOW,
                piece.colour,
                (
                    piece.row * SQ_SIZE + (SQ_SIZE // 2),
                    piece.col * SQ_SIZE + (SQ_SIZE // 2),
                ),
                SQ_SIZE - PIECE_PADDING,
            )
        if piece.king is True:
            pygame.draw.circle(
                WINDOW,
                piece.colour,
                (
                    piece.row * SQ_SIZE + (SQ_SIZE // 2),
                    piece.col * SQ_SIZE + (SQ_SIZE // 2),
                ),
                SQ_SIZE - PIECE_PADDING,
            )
            pygame.draw.circle(
                WINDOW,
                GOLD,
                (
                    piece.row * SQ_SIZE + (SQ_SIZE // 2),
                    piece.col * SQ_SIZE + (SQ_SIZE // 2),
                ),
                SQ_SIZE - CROWN_PADDING,
            )


if __name__ == "__main__":
    b = Board()
    w = Window()
    p = Piece(2, 2, BLACK)
    p2 = Piece(1, 1, BLACK)
    p2.crown()
    while True:
        b.render_squares(w.WINDOW)
        b.draw_piece(w.WINDOW, p)
        b.draw_piece(w.WINDOW, p2)
        pygame.display.update()
