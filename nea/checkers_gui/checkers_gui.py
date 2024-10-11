import pygame
import numpy as np

from nea.checkers_gui.consts import COLOURS, DISPLAY, GAME_TYPES
from nea.checkers_gui.buttons import Button, RectButton
from nea.checkers_gui.helpers import (
    get_col_selected,
    get_row_selected,
)
from nea.console_checkers import CheckersGame
from nea.console_checkers.consts import (
    ACTION,
    BLACK,
    BLACK_K,
    BLACK_R,
    BLACKS,
    EMPTY,
    WHITE,
    WHITE_K,
    WHITE_R,
    WHITES,
)
from nea.console_checkers.consts import (
    SIZE as BOARD_SIZE,
)
from nea.console_checkers.move_stack import MoveStack
from nea.mcts import MCTS


class CheckersGUI(CheckersGame):
    """Class that adds extra functionality to the CheckersGame
    which allows it to display a GUI
    """

    def __init__(self) -> None:
        self.piece_selected = None
        super().__init__()

    def draw(self, screen: pygame.Surface) -> None:
        """Draws the checkers board and pieces.

        If a piece is selected it scales that piece to be slightly smaller.
        Additionally, available moves for a selected piece are displayed.

        Args:
            screen (pygame.Surface): screen to draw to
        """
        for row in range(8):
            if row % 2 == 0:
                for x in range(0, DISPLAY.SCREEN_SIZE, DISPLAY.SQUARE_SIZE * 2):
                    pygame.draw.rect(
                        screen,
                        COLOURS.WHITE,
                        pygame.Rect(
                            x,
                            row * DISPLAY.SQUARE_SIZE,
                            DISPLAY.SQUARE_SIZE,
                            DISPLAY.SQUARE_SIZE,
                        ),
                    )
            else:
                for x in range(
                    DISPLAY.SQUARE_SIZE, DISPLAY.SCREEN_SIZE, DISPLAY.SQUARE_SIZE * 2
                ):
                    pygame.draw.rect(
                        screen,
                        COLOURS.WHITE,
                        pygame.Rect(
                            x,
                            row * DISPLAY.SQUARE_SIZE,
                            DISPLAY.SQUARE_SIZE,
                            DISPLAY.SQUARE_SIZE,
                        ),
                    )
        for y in range(BOARD_SIZE):
            for x in range(BOARD_SIZE):
                piece = self.board[x, y]
                if (x, y) == self.piece_selected:
                    self._draw_selected_piece(screen)
                    self._draw_available_moves(screen)
                elif piece != EMPTY:
                    self._get_and_draw_piece(screen, piece=piece, x=x, y=y)

        pygame.display.set_caption(f"IT IS {self.player.upper()}'S MOVE")

    def _get_and_draw_piece(
        self, screen: pygame.Surface, piece: int, x: int, y: int
    ) -> None:
        """given the integer representing a piece and its coords on the board e.g. (5, 2)
        it draws the piece

        Args:
            screen (pygame.Surface): screen to draw to
            piece (int): piece integer
            x (int): x position
            y (int): y position
        """
        if piece == BLACK_R:
            piece_img = self._get_piece_img(BLACK_R)
            screen.blit(piece_img, (y * DISPLAY.SQUARE_SIZE, x * DISPLAY.SQUARE_SIZE))
        elif piece == BLACK_K:
            piece_img = self._get_piece_img(BLACK_K)
            screen.blit(piece_img, (y * DISPLAY.SQUARE_SIZE, x * DISPLAY.SQUARE_SIZE))
        elif piece == WHITE_R:
            piece_img = self._get_piece_img(WHITE_R)
            screen.blit(piece_img, (y * DISPLAY.SQUARE_SIZE, x * DISPLAY.SQUARE_SIZE))
        elif piece == WHITE_K:
            piece_img = self._get_piece_img(WHITE_K)
            screen.blit(piece_img, (y * DISPLAY.SQUARE_SIZE, x * DISPLAY.SQUARE_SIZE))

    def _get_piece_img(self, piece: int) -> pygame.Surface:
        """Gets sprite images of a given piece

        Args:
            piece (int): piece as integer

        Returns:
            pygame.Surface: sprite
        """
        if piece == BLACK_R:
            return pygame.image.load(
                "nea/checkers_gui/media/images/black_r.png"
            ).convert_alpha()
        elif piece == BLACK_K:
            return pygame.image.load(
                "nea/checkers_gui/media/images/black_k.png"
            ).convert_alpha()
        elif piece == WHITE_R:
            return pygame.image.load(
                "nea/checkers_gui/media/images/white_r.png"
            ).convert_alpha()
        elif piece == WHITE_K:
            return pygame.image.load(
                "nea/checkers_gui/media/images/white_k.png"
            ).convert_alpha()

    def _draw_selected_piece(self, screen: pygame.Surface) -> None:
        """draws the selected piece as a scaled sprite

        Args:
            screen (pygame.Surface): screen to draw tow
        """
        x, y = self.piece_selected
        piece = self.board[x, y]
        piece_img = self._get_piece_img(piece)
        piece_img = pygame.transform.rotozoom(piece_img, 0, 0.9)
        screen.blit(
            piece_img,
            (
                y * DISPLAY.SQUARE_SIZE + 0.05 * DISPLAY.SQUARE_SIZE,
                x * DISPLAY.SQUARE_SIZE + 0.05 * DISPLAY.SQUARE_SIZE,
            ),
        )

    def _draw_available_moves(self, screen: pygame.Surface) -> None:
        """Given the selected piece - gets and draws available moves for that piece

        Args:
            screen (pygame.Surface): screen to draw to
        """
        valid_takes = self._get_valid_take_moves(*self.piece_selected, self.player)
        valid_simples = self._get_valid_simple_moves(*self.piece_selected, self.player)

        moves = valid_takes if len(valid_takes) > 0 else valid_simples
        move_tos = [x[1] for x in moves]

        for move in move_tos:
            alpha_surface = pygame.Surface((90, 90))
            alpha_surface.set_colorkey((0, 0, 0))
            alpha_surface.set_alpha(128)
            pygame.draw.circle(
                alpha_surface, COLOURS.GREEN, (45, 45), DISPLAY.CIRCLE_RADIUS
            )
            screen.blit(
                alpha_surface,
                (move[1] * DISPLAY.SQUARE_SIZE, move[0] * DISPLAY.SQUARE_SIZE),
            )

    def click(self, mouse_pos: tuple[int, int]) -> ACTION:
        """Evaulates a users click and selects/deselects a piece if needed

        Args:
            mouse_pos (tuple[int, int]): mouse position when click occurs

        Returns:
            ACTION: action user wants to take - if None then a piece has been selected to move
                                              - if not None then a move of the selected piece has been made
        """
        mouse_x, mouse_y = mouse_pos
        row, col = get_row_selected(mouse_y=mouse_y), get_col_selected(mouse_x=mouse_x)
        action: ACTION = None

        if self.piece_selected is None:
            valid_moves = self.get_all_valid_moves()
            valid_moves = (
                valid_moves["takes"]
                if len(valid_moves["takes"]) > 0
                else valid_moves["simple"]
            )
            valid_selections = [x[0] for x in valid_moves]
            if (row, col) in valid_selections:
                if self.player == WHITE:
                    if self.board[row, col] in WHITES:
                        self.piece_selected = (row, col)
                elif self.player == BLACK:
                    if self.board[row, col] in BLACKS:
                        self.piece_selected = (row, col)
        else:
            if (row, col) != self.piece_selected:
                action = (self.piece_selected, (row, col))
            self.piece_selected = None

        return action

    def evaluate_action(self, action: ACTION) -> tuple[bool, int] | tuple[None, None]:
        """Checks if move is valid and the outcome of the move

        Args:
            action (ACTION): action to take

        Returns:
            tuple[bool, int] | tuple[None, None]: done, reward
        """
        valid_move, _, done, reward = self.step(action)
        if not valid_move:
            print("INVALID MOVE - PLEASE TRY AGAIN")
        return done, reward


class MainMenu:
    def __init__(self) -> None:
        self.screen = pygame.display.set_mode(
            (DISPLAY.SCREEN_SIZE, DISPLAY.SCREEN_SIZE)
        )
        self.font = lambda font_size: pygame.font.SysFont(
            pygame.font.get_default_font(), font_size
        )

    def display(self) -> None:
        pygame.init()

        buttons: dict[Button] = {}
        self.screen.fill(COLOURS.BLACK)

        self._display_welcome_text()
        self._display_select_params_text()

        params = {
            "MCTS Searches": None,
            "EEC": None,
            "Training Examples": None,
            "Comparison Games": None,
            "% Replace Threshold": None,
        }
        self._display_param_texts(params)

        buttons["Tutorial"] = self._display_tutorial_button()
        buttons["All Params"] = self._display_all_params_button()
        buttons["User vs User"], buttons["User vs MCTS"], buttons["User vs Agent"] = (
            self._display_submit_buttons()
        )
        buttons["50ns"], buttons["100ns"], buttons["500ns"] = (
            self._display_mcts_searches_buttons()
        )
        buttons["0.75ec"], buttons["1.41ec"], buttons["2ec"] = (
            self._display_eec_buttons()
        )
        buttons["100te"], buttons["500te"], buttons["1000te"] = (
            self._display_training_examples_buttons()
        )
        buttons["5cg"], buttons["10cg"] = self._display_comparison_games_buttons()
        buttons["50rt"], buttons["60rt"] = self._display_replace_threshold_buttons()

        open = True
        while open:
            for e in pygame.event.get():
                if e.type == pygame.QUIT:
                    open = False
                if e.type == pygame.MOUSEBUTTONDOWN:
                    mouse_pos = pygame.mouse.get_pos()
                    params = self._manage_click(mouse_pos, buttons, params)

            pygame.display.flip()

    def _display_welcome_text(self) -> None:
        welcome_text = self.font(64).render(
            "Welcome to AI Checkers", False, COLOURS.WHITE
        )
        welcome_text_rect = welcome_text.get_rect(center=(DISPLAY.SCREEN_SIZE / 2, 55))
        self.screen.blit(welcome_text, welcome_text_rect)

    def _display_select_params_text(self) -> None:
        select_param_text = self.font(40).render(
            "Select Parameters: ", False, COLOURS.WHITE
        )
        select_param_rect = select_param_text.get_rect(topleft=(10, 180))
        self.screen.blit(select_param_text, select_param_rect)

    def _display_param_texts(self, params: list[str]) -> None:
        for i, y in enumerate(range(250, 720, 90)):
            if i < len(params):
                param_text = self.font(32).render(
                    f"{list(params.keys())[i]}: ", False, COLOURS.WHITE
                )
                param_rect = param_text.get_rect(topright=(320, y))
                self.screen.blit(param_text, param_rect)

    def _display_tutorial_button(self) -> RectButton:
        def tutorial_click_fn() -> None:
            pass

        tutorial = RectButton(
            125,
            30,
            (DISPLAY.SCREEN_SIZE / 4, 90),
            tutorial_click_fn,
            text="Tutorial",
            font_size=24,
        )
        tutorial.draw(screen=self.screen)

        return tutorial

    def _display_submit_buttons(self) -> tuple[RectButton, RectButton, RectButton]:
        play_text = self.font(40).render("Play", False, COLOURS.WHITE)
        play_text_rect = play_text.get_rect(center=(DISPLAY.SCREEN_SIZE / 2, 660))
        self.screen.blit(play_text, play_text_rect)

        buttons = ()
        submits = {
            "User vs User": user_vs_user_game_loop,
            "User vs MCTS": user_vs_mcts_game_loop,
            "User vs Agent": None,
        }  # replace values with functions
        for i, x in enumerate(range(50, 720, 240)):
            buttons += (
                RectButton(
                    150,
                    30,
                    (x, 685),
                    list(submits.values())[i],
                    text=list(submits.keys())[i],
                    font_size=24,
                ),
            )

        for b in buttons:
            b.draw(self.screen)

        return buttons

    def _display_all_params_button(self) -> RectButton:
        def all_params_click_fn() -> None:
            pass

        all_params = RectButton(
            125,
            30,
            (3 * ((DISPLAY.SCREEN_SIZE - 125) / 4), 90),
            all_params_click_fn,
            text="All Params",
            font_size=24,
        )
        all_params.draw(screen=self.screen)

        return all_params

    def _display_mcts_searches_buttons(
        self,
    ) -> tuple[RectButton, RectButton, RectButton]:
        buttons = ()

        for i, v in enumerate([50, 100, 500]):
            buttons += (
                RectButton(
                    80, 20, (340 + i * 100, 250), None, text=f"{v}", font_size=20
                ),
            )

        for item in buttons:
            item.draw(self.screen)

        return buttons

    def _display_eec_buttons(
        self,
    ) -> tuple[RectButton, RectButton, RectButton]:
        buttons = ()

        for i, v in enumerate([0.75, 1.41, 2]):
            buttons += (
                RectButton(
                    80, 20, (340 + i * 100, 340), None, text=f"{v}", font_size=20
                ),
            )

        for item in buttons:
            item.draw(self.screen)

        return buttons

    def _display_training_examples_buttons(
        self,
    ) -> tuple[RectButton, RectButton, RectButton]:
        buttons = ()

        for i, v in enumerate([100, 500, 1000]):
            buttons += (
                RectButton(
                    80, 20, (340 + i * 100, 430), None, text=f"{v}", font_size=20
                ),
            )

        for item in buttons:
            item.draw(self.screen)

        return buttons

    def _display_comparison_games_buttons(
        self,
    ) -> tuple[RectButton, RectButton, RectButton]:
        buttons = ()

        for i, v in enumerate([5, 10]):
            buttons += (
                RectButton(
                    100, 20, (340 + i * 150, 520), None, text=f"{v}", font_size=20
                ),
            )

        for item in buttons:
            item.draw(self.screen)

        return buttons

    def _display_replace_threshold_buttons(
        self,
    ) -> tuple[RectButton, RectButton, RectButton]:
        buttons = ()

        for i, v in enumerate([50, 60]):
            buttons += (
                RectButton(
                    100, 20, (340 + i * 150, 610), None, text=f"{v}", font_size=20
                ),
            )

        for item in buttons:
            item.draw(self.screen)

        return buttons

    def _manage_click(
        self,
        mouse_pos: tuple[int, int],
        buttons: dict[str, Button],
        params: dict,
    ):
        for key, button in buttons.items():
            if button.in_bounds(mouse_pos):
                if key in ["Tutorial", "All Params", "User vs User"]:
                    button.click_fn()
                elif key == "User vs MCTS":
                    p = np.random.rand()
                    if not params["MCTS Searches"] or not params["EEC"]:
                        continue
                    button.click_fn(
                        n_searches=params["MCTS Searches"],
                        eec=params["EEC"],
                        player_colour=WHITE if p > 0.5 else BLACK,
                    )
                elif key == "User vs Agent":
                    raise NotImplementedError
                else:
                    parameter = key[-2:]
                    value = key[:-2]

                    placeholders = {
                        "ns": "MCTS Searches",
                        "ec": "EEC",
                        "te": "Training Examples",
                        "cg": "Comparison Games",
                        "rt": "% Replace Threshold",
                    }

                    params[placeholders[parameter]] = (
                        int(value) if parameter != "ec" else float(value)
                    )

                    button.click_fn()  # make it obvious button has been selected (switch text to red)

        return params


def user_vs_user_game_loop() -> None:
    """main loop that allows a user to play the game

    Args:
        game (CheckersGame): _description_
    """
    screen = pygame.display.set_mode((DISPLAY.SCREEN_SIZE, DISPLAY.SCREEN_SIZE))

    done = False

    gui = CheckersGUI()

    winner = None
    while not done:
        reward = None

        for e in pygame.event.get():
            if e.type == pygame.QUIT:
                done = True
            if e.type == pygame.MOUSEBUTTONDOWN:
                action = gui.click(pygame.mouse.get_pos())
                if action:
                    done, reward = gui.evaluate_action(action)

        if done:
            if reward == -1:
                winner = gui.opposite_player
                break
            elif reward == 1:
                winner = gui.player
                break

        screen.fill(COLOURS.BLACK)
        gui.draw(screen)
        pygame.display.flip()

    _show_game_over(screen, winner, GAME_TYPES.USER_VS_USER)


def user_vs_mcts_game_loop(n_searches: int, eec: float, player_colour: str) -> None:
    """main loop that allows a user to play the game vs different mcts models

    Args:
        game (CheckersGame): _description_
    """
    pygame.init()

    screen = pygame.display.set_mode((DISPLAY.SCREEN_SIZE, DISPLAY.SCREEN_SIZE))

    done = False

    gui = CheckersGUI()
    mcts = MCTS(n_searches=n_searches, eec=eec)

    winner = None
    mcts_turns = 0
    moves = MoveStack()
    while not done:
        if gui.player == player_colour:
            reward = None

            for e in pygame.event.get():
                if e.type == pygame.QUIT:
                    done = True
                    reward = "None"
                if e.type == pygame.MOUSEBUTTONDOWN:
                    action = gui.click(pygame.mouse.get_pos())
                    if action:
                        done, reward = gui.evaluate_action(action)
                        moves.push(CheckersGame.convert_action_to_user(action))
        elif gui.player != player_colour:
            if mcts_turns == 0:
                mcts.build_tree(gui)
                mcts_turns += 1
            else:
                mcts.build_tree(gui)
            action = mcts.get_action()
            moves.push(CheckersGame.convert_action_to_user(action))
            done, reward = gui.evaluate_action(action)

        if done:
            if reward == -1:
                winner = gui.opposite_player
                break
            elif reward == 1:
                winner = gui.player
                break
            elif reward == "None":
                winner = "no one"
                break

        screen.fill(COLOURS.BLACK)
        gui.draw(screen)
        pygame.display.flip()

    _show_game_over(
        screen,
        winner,
        GAME_TYPES.USER_VS_USER,
        n_searches=n_searches,
        eec=eec,
        player_colour=player_colour,
    )


def mcts_vs_mcts_game_loop(
    n_searches_1: int, eec_1: float, n_searches_2, eec_2: int
) -> None:
    """main loop that allows a user to see two mcts models play eachother

    Args:
        game (CheckersGame): _description_
    """
    pygame.init()

    screen = pygame.display.set_mode((DISPLAY.SCREEN_SIZE, DISPLAY.SCREEN_SIZE))

    done = False

    gui = CheckersGUI()
    mcts_w = MCTS(n_searches=n_searches_1, eec=eec_1)
    mcts_b = MCTS(n_searches=n_searches_2, eec=eec_2)

    winner = None
    mcts_w_turns = 0
    mcts_b_turns = 0
    moves = MoveStack()
    while not done:
        if gui.player == "white":
            reward = None

            for e in pygame.event.get():
                if e.type == pygame.QUIT:
                    done = True
                    reward = "None"

            if mcts_w_turns == 0:
                mcts_w.build_tree(gui)
                mcts_w_turns += 1
            else:
                mcts_w.build_tree(gui)
            action = mcts_w.get_action()
            moves.push(CheckersGame.convert_action_to_user(action))
            done, reward = gui.evaluate_action(action)
        elif gui.player == "black":
            if mcts_b_turns == 0:
                mcts_b.build_tree(gui)
                mcts_b_turns += 1
            else:
                mcts_b.build_tree(gui)
            action = mcts_b.get_action()
            moves.push(CheckersGame.convert_action_to_user(action))
            done, reward = gui.evaluate_action(action)

        if done:
            if reward == -1:
                winner = gui.opposite_player
                break
            elif reward == 1:
                winner = gui.player
                break
            elif reward == "None":
                winner = "no one"
                break

        screen.fill(COLOURS.BLACK)
        gui.draw(screen)
        pygame.display.flip()

    _show_game_over(
        screen,
        winner,
        GAME_TYPES.MCTS_VS_MCTS,
        n_searches_1=n_searches_1,
        eec_1=eec_1,
        n_searches_2=n_searches_2,
        eec_2=eec_2,
    )


def _show_game_over(screen: pygame.Surface, winner: str, game_type: int, **kwargs):
    open = True
    while open:
        for e in pygame.event.get():
            if e.type == pygame.QUIT:
                open = False
            elif e.type == pygame.KEYDOWN:
                if e.key == pygame.K_r:
                    m = MainMenu()
                    m.display()
            else:
                font = pygame.font.SysFont(pygame.font.get_default_font(), 70)
                game_over_text = font.render(
                    f"GAME OVER - {winner.upper()} WON", False, COLOURS.WHITE
                )
                _, game_over_text_height = game_over_text.get_size()

                restart_text = font.render("PRESS R TO RESTART", False, COLOURS.WHITE)
                _, restart_text_height = restart_text.get_size()

                total_text_height = game_over_text_height + restart_text_height + 15

                game_over_text_rect = game_over_text.get_rect(
                    center=(
                        DISPLAY.SCREEN_SIZE / 2,
                        (DISPLAY.SCREEN_SIZE / 2) - (total_text_height / 2),
                    )
                )
                restart_text_rect = restart_text.get_rect(
                    center=(
                        DISPLAY.SCREEN_SIZE / 2,
                        (DISPLAY.SCREEN_SIZE / 2) + (total_text_height / 2),
                    )
                )

                screen.fill(COLOURS.BLACK)
                screen.blit(game_over_text, game_over_text_rect)
                screen.blit(restart_text, restart_text_rect)

                pygame.display.flip()


def play_game():
    valid_input = False
    game_type = None
    while not valid_input:
        print("==================================================")
        print(
            "Select Game Type: \n"
            "1. User vs User\n"
            "2. User vs MCTS\n"
            "3. MCTS vs MCTS"
        )
        try:
            game_type = int(input("Enter Number: "))
            if 1 <= game_type <= 3:
                valid_input = True
        except Exception:
            print("Invalid selection")

    if game_type == GAME_TYPES.USER_VS_USER:
        print("==================================================")
        user_vs_user_game_loop()
    elif game_type == GAME_TYPES.USER_VS_MCTS:
        n_searches = _get_valid_num_searches()

        eec = _get_valid_eec()

        valid_input = False
        colour = None
        while not valid_input:
            print("==================================================")
            print("What colour would you like to play as?")
            try:
                colour = input("w/b: ")
                if colour == ("w" or "b"):
                    valid_input = True
                else:
                    raise ValueError(
                        "Please enter w to play as white and b to play as black"
                    )
            except Exception:
                print("Invalid entry")
            print("==================================================")

        player_colour = WHITE if colour == "w" else BLACK
        user_vs_mcts_game_loop(
            n_searches=n_searches, eec=eec, player_colour=player_colour
        )
    elif game_type == GAME_TYPES.MCTS_VS_MCTS:
        n_searches_1 = _get_valid_num_searches("white")
        n_searches_2 = _get_valid_num_searches("black")

        eec_1 = _get_valid_eec("white")
        eec_2 = _get_valid_eec("black")

        mcts_vs_mcts_game_loop(
            n_searches_1=n_searches_1,
            eec_1=eec_1,
            n_searches_2=n_searches_2,
            eec_2=eec_2,
        )


def _get_valid_num_searches(colour: str = None) -> int:
    valid_input = False
    n_searches = None
    while not valid_input:
        print("==================================================")
        if colour is None:
            print("Enter Number of Searches for the MCTS \n" "(Between 10 and 10000)")
        else:
            print(
                f"Enter Number of Searches for the {colour} MCTS \n"
                "(Between 10 and 10000)"
            )
        try:
            n_searches = int(input("Enter Number of Searches: "))
            if 1 <= n_searches <= 10000:
                valid_input = True
            else:
                raise ValueError("Please enter a number between 1000 and 10000")
        except Exception:
            print("Invalid entry")

    return n_searches


def _get_valid_eec(colour: str = None) -> float:
    valid_input = False
    eec = None
    while not valid_input:
        print("==================================================")
        if colour is None:
            print(
                "Enter EEC for MCTS (determines exploration - with higher values = more exploration) \n"
                "(Between 0.1 and 2)"
            )
        else:
            print(
                f"Enter EEC for the {colour} MCTS (determines exploration - with higher values = more exploration) \n"
                "(Between 0.1 and 2)"
            )
        try:
            eec = float(input("Enter EEC: "))
            if 0.1 <= eec <= 2:
                valid_input = True
            else:
                raise ValueError("Please enter a number between 1 and 3")
        except Exception:
            print("Invalid entry")

    return eec


if __name__ == "__main__":
    m = MainMenu()
    m.display()
