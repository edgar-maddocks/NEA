from nea.checkers_gui.consts import DISPLAY


def get_row_selected(mouse_y: int) -> int:
    return int(mouse_y / DISPLAY.SQUARE_SIZE)


def get_col_selected(mouse_x: int) -> int:
    return int(mouse_x / DISPLAY.SQUARE_SIZE)
