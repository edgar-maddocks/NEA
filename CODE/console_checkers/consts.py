SIZE = 8

BLACK = "black"
WHITE = "white"

PLAYERS = (BLACK, WHITE)
PIECES = ("regular", "king")

EMPTY = 0
BLACK_R = 1
BLACK_K = 2
WHITE_R = 3
WHITE_K = 4

BLACKS = [1, 2]
WHITES = [3, 4]

NUM_TO_STR = {0: " ", 1: "x", 2: "X", 3: "o", 4: "O"}
STR_TO_NUM = {" ": 0, "x": 1, "X": 2, "o": 3, "O": 4}

COLS_TO_NUMS = {"A": 0, "B": 1, "C": 2, "D": 3, "E": 4, "F": 5, "G": 6, "H": 7}
NUMS_TO_COLS = {0: "A", 1: "B", 2: "C", 3: "D", 4: "E", 5: "F", 6: "G", 7: "H"}

DIRS = [(+1, -1), (+1, +1), (-1, +1), (-1, -1)]

LEGAL_DIRS = {
    BLACK: {"regular": [DIRS[0], DIRS[1]], "king": DIRS},
    WHITE: {"regular": [DIRS[2], DIRS[3]], "king": DIRS},
}
