from nea.ConsoleCheckers import CheckersGame
from nea.ConsoleCheckers.consts import (SIZE as BOARD_SIZE, 
                                        BLACK_K, 
                                        BLACK_R, 
                                        WHITE_K, 
                                        WHITE_R, 
                                        EMPTY, 
                                        WHITE, 
                                        WHITES, 
                                        BLACK, 
                                        BLACKS)
from nea.checkers_gui.consts import *
from nea.checkers_gui.helpers import *

import pygame

class CheckersGUI(CheckersGame):
    """Class that adds extra functionality to the CheckersGame
    which allows it to display a GUI
    """
    def __init__(self, game: CheckersGame) -> None:
        self.game = game
        self.piece_selected = None

    def draw(self, screen: pygame.Surface):
        for row in range(8):
            if row % 2 == 0:
                for x in range(0, DISPLAY.SCREEN_SIZE, DISPLAY.SQUARE_SIZE*2):
                    pygame.draw.rect(screen, COLOURS.WHITE, pygame.Rect(x, row*DISPLAY.SQUARE_SIZE, DISPLAY.SQUARE_SIZE, DISPLAY.SQUARE_SIZE))
            else:
                for x in range(DISPLAY.SQUARE_SIZE, DISPLAY.SCREEN_SIZE, DISPLAY.SQUARE_SIZE*2):
                    pygame.draw.rect(screen, COLOURS.WHITE, pygame.Rect(x, row*DISPLAY.SQUARE_SIZE, DISPLAY.SQUARE_SIZE, DISPLAY.SQUARE_SIZE))
        for y in range(BOARD_SIZE):
            for x in range(BOARD_SIZE):
                piece = self.game.board[x, y]
                if piece != EMPTY:
                    self._draw_piece(screen, piece=piece, x=x, y=y)


    def _draw_piece(self, screen: pygame.Surface, piece: int, x: int, y: int) -> None:
        if piece == BLACK_R:
            black_r = pygame.image.load("nea/checkers_gui/media/images/black_r.png").convert_alpha()
            screen.blit(black_r, (y*DISPLAY.SQUARE_SIZE, x*DISPLAY.SQUARE_SIZE))
        elif piece == BLACK_K:
            black_k = pygame.image.load("nea/checkers_gui/media/images/black_k.png").convert_alpha()
            screen.blit(black_k, (y*DISPLAY.SQUARE_SIZE, x*DISPLAY.SQUARE_SIZE))
        elif piece == WHITE_R:
            white_r = pygame.image.load("nea/checkers_gui/media/images/white_r.png").convert_alpha()
            screen.blit(white_r, (y*DISPLAY.SQUARE_SIZE, x*DISPLAY.SQUARE_SIZE))
        elif piece == WHITE_K:
            white_k = pygame.image.load("nea/checkers_gui/media/images/white_k.png").convert_alpha()
            screen.blit(white_k, (y*DISPLAY.SQUARE_SIZE, x*DISPLAY.SQUARE_SIZE))

    def click(self, mouse_pos: tuple[int, int]) -> None:
        mouse_x, mouse_y = mouse_pos
        row, col = get_row_selected(mouse_y=mouse_y), get_col_selected(mouse_x=mouse_x)

        if self.piece_selected is None:
            if self.game.player == WHITE:
                if self.game.board[row, col] in WHITES:
                    self.piece_selected = (row, col)
            elif self.game.player == BLACK:
                if self.game.board[row, col] in BLACKS:
                    self.piece_selected = (row, col)
        else:
            pass


def game_loop(game: CheckersGame) -> None:
    pygame.init()
    screen = pygame.display.set_mode((DISPLAY.SCREEN_SIZE, DISPLAY.SCREEN_SIZE))

    game_in_progress = True

    gui = CheckersGUI(game)
    
    while game_in_progress:
        for e in pygame.event.get():

            if e.type == pygame.QUIT:
                game_in_progress = False
            if e.type == pygame.MOUSEBUTTONDOWN:
                game.click(pygame.mouse.get_pos())

        screen.fill(COLOURS.BLACK)
        gui.draw(screen)
        pygame.display.flip()


if __name__ == "__main__":
    game = CheckersGame()
    game_loop(game)
    