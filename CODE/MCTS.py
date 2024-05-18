from console_checkers import Checkers

import numpy as np
from typing import Dict, List, Tuple


class Node:
    def __init__(
        self,
        game: Checkers,
        state: np.ndarray,
        parent: "Node" = None,
        action_taken: Tuple = None,
        args: Dict = None,
    ):
        self._game = game
        self._state = state
        self._parent = parent
        self._children: List[Node] = []
        self._action_taken = action_taken
        self._available_moves_left: List[Tuple] = []

        self._args = args

        self._visit_count, self._value_sum = 0, 0

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
    def __init__(self, game: Checkers, max_depth: int = 3, args: Dict = None):
        self._game = game
        self._max_depth = max_depth
        if args is not None:
            self._args = args

        self._root = Node(game, game.get_state())

    def search(self):
        node = self._root
        for search in range(self._max_depth):
            while node.no_branches_left():
                node = node.select_child()

                result = self.game.step()


game = Checkers()
mcts = MCTS(game)

print(mcts.root)
