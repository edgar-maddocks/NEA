class DISPLAY:
    SCREEN_SIZE = 720
    SQUARE_SIZE = 90
    CIRCLE_RADIUS = 20


class COLOURS:
    BLACK = (0, 0, 0)
    WHITE = (255, 255, 255)
    GREEN = (0, 255, 0)
    RED = (255, 0, 0)
    BLUE = (0, 0, 255)
    YELLOW = (255, 255, 0)
    BROWN = (131, 106, 76)
    BONE = (251, 220, 191)


class GAME_TYPES:
    USER_VS_USER = 1
    USER_VS_MCTS = 2
    MCTS_VS_MCTS = 3


class DICTS:
    param_placeholders: dict[str, str] = {
        "ns": "MCTS Searches",
        "ec": "EEC",
        "te": "Training Examples",
        "cg": "Comparison Games",
        "rt": "% Replace Threshold",
    }
    param_placeholder_values: dict[str, list[int | float]] = {
        "ns": [50, 100, 500],
        "ec": [0.75, 1.41, 2],
        "te": [100, 500, 1000],
        "cg": [5, 10],
        "rt": [50, 60],
    }


class TEXTS:
    tutorial = """
    1. Pieces only move diagonally\n
    2. The aim is to take all of the opposing players pieces, or to put the\n
        opposing player in a position with no possible moves.\n
    3. Players take turns moving their shade of pieces.\n
    4. If a player’s piece reaches the opposing players edge of the board, the piece becomes a'King'\n
    5. Unless a piece is crowned it may only move and take pieces diagonally forwards.\n 
    6. Kings may move and take both forwards and backwards.\n
    7. If an adjacent square has an opponents piece and the square immediately\n
    beyond the oppositions piece is empty, the opponents piece may be\n
    captured.\n
    8. If the player whose go it is, has the opportunity to capture one\n
    or more pieces, then they must do so.\n
    9. A piece is taken by moving your own piece over the opposing player's, into the vacant square

Unlike a regular move, a capturing move may make more than
one ’hop’. This is if the capture places the piece in a position where
another capture is possible. In this case, the additional capture must
be made. The capture sequence can only be made by one piece per
move. i.e. You cannot make one capture with one piece, and then
another capture with another piece in the same move.
However, if more than one piece can capture, the player has free
choice over which piece to move. Likewise, if one piece can capture in
multiple directions then the player has the choice in which direction
to move.
Note: it is not compulsory for the player to move in the direction,
or with the piece, that will lead to the greatest number of captures in
that move.
A move may only end when the position has no more captures available or an uncrowned piece reaches
the opposing edge of the board and becomes a King.
The game ends when all of a players piece’s have been captured, or a player has no available moves.
"""
