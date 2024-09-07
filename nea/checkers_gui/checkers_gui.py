from nea.ConsoleCheckers import CheckersGame
from nea.ConsoleCheckers.consts import SIZE as BOARD_SIZE, BLACK_K, BLACK_R, WHITE_K, WHITE_R, EMPTY
from nea.checkers_gui.consts import *

import pygame

class CheckersGUI(CheckersGame):
    """Class that adds extra functionality to the CheckersGame
    which allows it to display a GUI
    """
    def __init__(self, game: CheckersGame) -> None:
        self.game = game

    def draw(self, screen: pygame.Surface):
        for row in range(8):
            if row % 2 == 0:
                for x in range(0, DISPLAY.SCREEN_SIZE, DISPLAY.SQUARE_SIZE*2):
                    pygame.draw.rect(screen, COLOURS.BLACK, pygame.Rect(x, row*DISPLAY.SQUARE_SIZE, DISPLAY.SQUARE_SIZE, DISPLAY.SQUARE_SIZE))
            else:
                for x in range(DISPLAY.SQUARE_SIZE, DISPLAY.SCREEN_SIZE, DISPLAY.SQUARE_SIZE*2):
                    pygame.draw.rect(screen, COLOURS.BLACK, pygame.Rect(x, row*DISPLAY.SQUARE_SIZE, DISPLAY.SQUARE_SIZE, DISPLAY.SQUARE_SIZE))
        for y in range(BOARD_SIZE):
            for x in range(BOARD_SIZE):
                piece = self.game.board[x, y]
                if piece != EMPTY:
                    self._draw_piece(screen, piece=piece, x=x, y=y)


    def _draw_piece(self, screen: pygame, piece: int, x: int, y: int) -> None:
        if piece == BLACK_R:
            black_r = pygame.image.load("nea/checkers_gui/media/images/black_r.png")
            pygame.draw.circle(screen, COLOURS.BROWN, center=(y*DISPLAY.SQUARE_SIZE + DISPLAY.SQUARE_SIZE/2, x*DISPLAY.SQUARE_SIZE + DISPLAY.SQUARE_SIZE/2), radius=DISPLAY.PIECE_RADIUS)


def game_loop(game: CheckersGame) -> None:
    pygame.init()
    screen = pygame.display.set_mode((DISPLAY.SCREEN_SIZE, DISPLAY.SCREEN_SIZE))

    game_in_progress = True

    gui = CheckersGUI(game)
    
    while game_in_progress:
        for e in pygame.event.get():

            if e.type == pygame.QUIT:
                game_in_progress = False

        screen.fill(COLOURS.WHITE)
        gui.draw(screen)
        pygame.display.flip()


if __name__ == "__main__":
    game = CheckersGame()
    game_loop(game)
    