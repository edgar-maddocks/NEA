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
    USER_VS_AGENT = 4


class DICTS:
    param_placeholders: dict[str, str] = {
        "ns": "(UvsM, UvsA) MCTS Searches",
        "ec": "(UvsM, UvsA)                     EEC",
        "te": "(UvsA)       Training Examples",
        "cg": "(UvsA)     Comparison Games",
        "rt": "(UvsA)   % Replace Threshold",
    }
    param_placeholder_values: dict[str, list[int | float]] = {
        "ns": [50, 100, 500],
        "ec": [0.75, 1.41, 2],
        "te": [100, 500, 1000],
        "cg": [5, 10],
        "rt": [50, 60],
    }


class TEXTS:
    tutorial = [
        "1. Pieces only move diagonally.",
        "2. The aim is to take all of the opposing players pieces, or to put the",
        "   opposing player in a position with no possible moves.",
        "3. Players take turns moving their shade of pieces.",
        "4. If a player’s piece reaches the opposing players edge of the board,",
        "   the piece becomes a'King'.",
        "5. Unless a piece is crowned it may only move and take pieces diagonally forwards.",
        "6. Kings may move and take both forwards and backwards.",
        "7. If an adjacent square has an opponents piece and the square immediately.",
        "   beyond the oppositions piece is empty, the opponents piece may be captured.",
        "8. If the player whose go it is, has the opportunity to capture one.",
        "   or more pieces, then they must do so.",
        "9. A piece is taken by moving your own piece over the opposing player's,",
        "   into the vacant square.",
        "10. If after taking a piece another capture is available, that move must be made.",
        "11. A double capture must use the same own piece to capture each opposing piece.",
        "12. If one or more piece can capture on the first take - choice is left to the player.",
    ]
    changeable_params = [
        "MCTS Searches: The number of different game endings the tree search sees.",
        "EEC (Exploration Eploitation Coefficient): Determines the level of exploration",
        "   that the tree search takes where a higher value -> more exploration.",
        "Training Examples: The number of different states that the neural network",
        "   trains on during each epoch.",
        "Comparison Games: The number of games played between the old network and",
        "   the new one to calculate the % win rate of the newer model.",
        "% Replacement Threshold: The percentage of comparison games the new network",
        "   must win to replace the previous network.",
    ]
    defaulted_params = [
        "MCTS Epochs (defaulted to 10): The number of iterations where the network ",
        "   may be updated.",
        "NN Epochs (defaulted to 10): The number of iterations the neural network trains for on each",
        "   batch of training examples.",
        "Batch Size (defaulted to 32): The size of processing batches which the network takes the mean of",
        "   for the loss.",
    ]
