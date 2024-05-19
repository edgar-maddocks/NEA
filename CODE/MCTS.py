from ConsoleCheckers import CheckersBoard

import numpy as np
from typing import Dict, List, Tuple, Iterator

from copy import deepcopy


class Node:
    def __init__(
        self,
        game: CheckersBoard,
        parent: "Node" = None,
        action_taken: Tuple = None,
        terminal: bool = None,
        reward: int = None,
        args: Dict = None,
    ):
        self._game = deepcopy(game)
        self._state = self._game.board
        self._parent = parent
        self.children: List[Node] = []
        self._action_taken = action_taken
        self._available_moves_left = self._init_available_moves()

        self._args = args

        self._visit_count, self._value_count = 0, 0

        self.is_leaf = terminal
        self.reward = reward

    def _init_available_moves(self):
        valid_moves = self._game.get_all_valid_moves()
        valid_takes, valid_simples = valid_moves["takes"], valid_moves["simple"]
        return valid_takes if len(valid_takes) > 0 else valid_simples

    @property
    def iter_children(self) -> Iterator["Node"]:
        for child in self.children:
            yield child

    @property
    def n_children(self):
        return len(self.children)

    @property
    def n_branches_available(self):
        return len(self._available_moves_left)

    """def b_expand(self):
        for action in self._available_moves_left:
            self._available_moves_left.remove(action)
            child_game = deepcopy(self._game)
            valid, child_state, done, reward, info = child_game.step(action)

            child = Node(child_game, self, action, done, reward, self._args)
            self.children.append(child)"""

    def d_expand(self):
        idx = np.random.choice(len(self._available_moves_left))
        action = self._available_moves_left[idx]
        self._available_moves_left.remove(action)
        child_game = deepcopy(self._game)
        valid, child_state, done, reward, info = child_game.step(action)

        child = Node(child_game, self, action, done, reward, self._args)
        self.children.append(child)

        return child

    def select_child(self):
        best_child = None
        best_ucb = -np.inf
        for child in self.children:
            ucb = self.get_ucb(child)
            if ucb > best_ucb:
                best_child = child
                best_ucb = ucb

        return best_child

    def get_ucb(self, child: "Node"):
        q_value = 1 - ((child._value_count / child._visit_count) + 1) / 2
        return q_value + self._args["eec"] * np.sqrt(
            np.log(self._visit_count) / child._visit_count
        )

    def backprop(self, reward):
        self._visit_count += 1
        self._value_count += reward

        reward *= -1
        if self._parent is not None:
            self._parent.backprop(reward)


class MCTS:
    def __init__(self, root: CheckersBoard, args: Dict = {"eec": 1}):
        self._root = Node(root)
        if args is not None:
            self._args = args

    def search(self):
        node = self._root
        while not node.is_leaf:
            node = node.d_expand()

        node.backprop(node.reward)


game = CheckersBoard()
mcts = MCTS(game)

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
