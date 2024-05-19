from ConsoleCheckers import CheckersBoard

import numpy as np
from typing import Dict, List, Tuple

from copy import deepcopy


class Node:
    def __init__(
        self,
        game: CheckersBoard,
        parent: "Node" = None,
        action_taken: Tuple = None,
        args: Dict = None,
    ):
        self._game = deepcopy(game)
        self._state = self._game.board
        self._parent = parent
        self._children: List[Node] = []
        self._action_taken = action_taken
        self._populate_available_moves()

        self._args = args

        self._visit_count, self._win_count = 0, 0

    def no_branches_left(self):
        return len(self._available_moves_left) == 0 and len(self._children) > 0

    def select_child(self):
        best_child = None
        max_ucb = -np.inf

        for child in self._children:
            child_ucb = child.calculate_ucb(child)
            if child_ucb > max_ucb:
                max_ucb = child_ucb
                best_child = child

        return best_child

    def calculate_ucb(self, child: "Node"):
        return child.get_q() + self._args["eec"] * np.sqrt(
            np.log(self._visit_count) / child._visit_count
        )

    def get_q(self):
        return ((self._value_sum / self._visit_count) + 1) / 2


class MCTS:
    def __init__(self, game: CheckersBoard, args: Dict = None):
        self._game = game
        if args is not None:
            self._args = args

    def search(self):
        pass


game = CheckersBoard()
mcts = MCTS(game, args={"n_searches": 1})

mcts.search()

"""class MCTS:",
    "    def __init__(self, game, args):",
    "        self.game = game",
    "        self.args = args",
    "        ",
    "    def search(self, state):",
    "        root = Node(self.game, self.args, state)",
    "        ",
    "        for search in range(self.args['num_searches']):",
    "            node = root",
    "            ",
    "            while node.is_fully_expanded():",
    "                node = node.select()",
    "                ",
    "            value, is_terminal = self.game.get_value_and_terminated(node.state, node.action_taken)",
    "            value = self.game.get_opponent_value(value)",
    "            ",
    "            if not is_terminal:",
    "                node = node.expand()",
    "                value = node.simulate()",
    "                ",
    "            node.backpropagate(value)    ",
    "            ",
    "            ",
    "        action_probs = np.zeros(self.game.action_size)",
    "        for child in root.children:",
    "            action_probs[child.action_taken] = child.visit_count",
    "        action_probs /= np.sum(action_probs)",
    "        return action_probs",
    "        ",
    "        ",
    "        """
