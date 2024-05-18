from ConsoleCheckers import CheckersBoard

import numpy as np
from typing import Dict, List, Tuple


class Node:
    def __init__(
        self,
        game: CheckersBoard,
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

    def _populate_available_moves(self):
        self._available_moves_left = self._game.get_all_valid_moves(self._state)

    def expand(self):
        action = np.random.choice(self._available_moves_left)
        self._available_moves_left.remove(action)

        child_state = self._state.copy()
        valid, next_obs, done, reward, info = self._game.step(self._state, action)
        child_state = next_obs

        child = Node(self._game, child_state, self, action, self._args)
        self._children.append(child)

        return child

    def simulate(self):
        raise NotImplementedError


class MCTS:
    def __init__(self, game: CheckersBoard, max_depth: int = 3, args: Dict = None):
        self._game = game
        self._max_depth = max_depth
        if args is not None:
            self._args = args

    def search(self, state):
        node = Node(self._game, state)
        for search in range(self._max_depth):
            while node.no_branches_left():
                node = node.select_child()

                valid, next_obs, done, reward, info = node._game.step(
                    node._state, node._action_taken
                )
                if valid and not done:
                    node.expand()
                    reward = node.simulate()


game = CheckersBoard()
mcts = MCTS(game)

print(mcts.root)
