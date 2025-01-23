import pygame
from abc import ABC

from nea.checkers_gui.consts import COLOURS


class Button(ABC):
    def draw(self, screen: pygame.Surface) -> None: ...

    def in_bounds(self, mouse_pos: tuple[int, int]) -> bool: ...

    def click_fn(self, *args): ...


class RectButton(Button):
    def __init__(
        self,
        width: int,
        height: int,
        pos: tuple[int, int],
        click_func: callable,
        colour: COLOURS = COLOURS.WHITE,
        text: str = "",
        font_size: int = None,
        text_colour: COLOURS = COLOURS.BLACK,
    ) -> None:
        self.width = width
        self.height = height
        self.x, self.y = pos
        self.f = click_func
        self.colour = colour
        self.text = text
        if self.text != "":
            assert font_size, "Must have font size if text is to be shown."
            self.text_colour = text_colour
            self.font = pygame.font.SysFont(pygame.font.get_default_font(), font_size)

    def draw(self, screen: pygame.Surface) -> None:
        pygame.draw.rect(
            screen,
            self.colour,
            pygame.Rect(
                self.x,
                self.y,
                self.width,
                self.height,
            ),
        )

        if self.text != "":
            font = self.font
            text = font.render(self.text, 1, self.text_colour)
            # draw the text in the middle of the button
            screen.blit(
                text,
                (
                    self.x + (self.width / 2 - text.get_width() / 2),
                    self.y + (self.height / 2 - text.get_height() / 2),
                ),
            )

    def in_bounds(self, mouse_pos: tuple[int, int]) -> bool:
        if mouse_pos[0] > self.x and mouse_pos[0] < self.x + self.width:
            if mouse_pos[1] > self.y and mouse_pos[1] < self.y + self.height:
                return True

        return False

    def set_text_black(self, screen: pygame.Surface):
        self.text_colour = COLOURS.BLACK
        self.draw(screen)

    def click_fn(self, *args, **kwargs):
        if self.f:
            self.f(*args, **kwargs)


class RoundButton:
    def __init__(
        self, radius: int, centre: tuple[int, int], colour: COLOURS = COLOURS.WHITE
    ) -> None:
        self.r = radius
        self.c = centre
        self.colour = colour

    def draw(self, screen: pygame.Surface) -> None:
        pygame.draw.circle(screen, self.colour, self.c, self.r)


def _change_button_text_colour(button: RectButton, screen: pygame.Surface):
    button.text_colour = (
        COLOURS.RED if button.text_colour == COLOURS.BLACK else COLOURS.BLACK
    )
    button.draw(screen)
