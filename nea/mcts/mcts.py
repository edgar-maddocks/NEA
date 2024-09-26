from __future__ import annotations
from copy import deepcopy
from collections import deque

import numpy as np
from tqdm import tqdm

from nea.console_checkers import CheckersGame
from nea.mcts.consts import ACTION, ACTION_TO_IDX, IDX_TO_ACTION
from nea.network import AlphaModel
from nea.ml.autograd import Tensor


class Node:
    """Creates a new Node object"""

    def __init__(
        self,
        game: CheckersGame,
        parent: Node = None,
        terminal: bool = False,
        action_taken: ACTION = None,
        reward: float = None,
        **kwargs,
    ) -> None:
        self._game = game
        self.colour = self._game.player

        self._state = self._game.board
        self._parent = parent
        self.children: list["Node"] = []
        self._action_taken = action_taken
        self.reward = reward
        self._available_moves_left = self._init_available_moves()
        self.terminal = terminal

        self.visit_count, self.value_count = 0, 0

        self.kwargs = kwargs

    def _init_available_moves(self) -> list[ACTION]:
        """Initializes a list of available moves given the orginial state

        Returns:
            list[ACTION]: list of avaialble moves
        """
        valids = self._game.get_all_valid_moves()
        out = valids["takes"] if len(valids["takes"]) > 0 else valids["simple"]
        return out

    @property
    def n_available_moves_left(self) -> int:
        """returns the number of moves left for the tree to search

        Returns:
            int: _description_
        """
        return len(self._available_moves_left)

    @property
    def action_taken(self):
        """returns protected attribute of action take

        Returns:
            ACTION:
        """
        return self._action_taken

    def select_child(self) -> Node:
        """Selects the best child from a fully expanded node using UCB

        Returns:
            Node: Best child
        """
        best_child = None
        best_ucb = -np.inf

        for child in self.children:
            ucb = self._calculate_ucb(child)
            if ucb > best_ucb:
                best_child = child
                best_ucb = ucb

        return best_child

    def _calculate_ucb(self, child: Node) -> float:
        """Calculates the UCB of a node

        Returns:
            float: UCB value
        """
        return (child.value_count / child.visit_count) + self.kwargs["eec"] * (
            np.sqrt(np.log(self.visit_count) / child.visit_count)
        )

    def expand(self) -> Node:
        """Random expansion of a node

        Returns:
            Node: New child
        """
        if self.n_available_moves_left == 0 and self.children:
            return self.select_child()

        random_move_idx = np.random.choice(self.n_available_moves_left)
        random_action = self._available_moves_left[random_move_idx]

        self._available_moves_left.remove(random_action)  #

        child_game = deepcopy(self._game)
        _, _, terminal, reward = child_game.step(random_action)

        child = Node(
            child_game,
            parent=self,
            terminal=terminal,
            action_taken=random_action,
            reward=reward,
            eec=self.kwargs["eec"],
        )
        self.children.append(child)

        return child

    def backprop(self, reward: int) -> None:
        """Backpropgates through graph, updating value count, visit count.

        Args:
            reward (int): reward of terminal state
        """
        self.visit_count += 1
        self.value_count += reward

        if self._parent is not None:
            if self._parent.colour != self.colour:
                reward *= -1
            self._parent.backprop(reward)


class MCTS:
    """
    Monte Carlo Tree Search class used to search for lines until termination in a given state
    """

    def __init__(self, **kwargs) -> None:
        """Creates a new MCTS object

        Keyword Args:
            eec: exploration constant -> Higher EEC = more exploration
            n_searches: number of searches
        """

        self.kwargs = kwargs
        self._root: Node = None

    def build_tree(self, root: CheckersGame) -> None:
        """Builds a new tree

        Args:
            root (CheckersGame): New state to root the tree from
        """
        self._root = Node(root, eec=self.kwargs["eec"])

        for _ in tqdm(range(int(self.kwargs["n_searches"]))):
            node = self._root

            if node.n_available_moves_left == 0 and node.children:
                node = node.select_child()

            while not node.terminal:
                node = node.expand()

            node.backprop(node.reward)

    def get_action_probs(self) -> np.ndarray:
        """Gets array of probabilities of action based on tree

        Returns:
            np.ndarray: Array of probabilities
        """
        p = np.zeros(
            (8, 8, 8)
        )  # (8x8) shows where to take piece from. Final 8 shows what direc e.g.
        # idx 0 = row+1,col+1, idx 1 = row+1, col-1 etc.
        for child in self._root.children:
            piece_moved, moved_to = child.action_taken
            row_change = moved_to[0] - piece_moved[0]
            col_change = moved_to[1] - piece_moved[1]
            direc_idx = ACTION_TO_IDX[(row_change, col_change)]
            p[direc_idx, piece_moved[0], piece_moved[1]] = child.visit_count

        p /= np.sum(p)
        return p

    def convert_probs_to_action(self, p: np.ndarray) -> ACTION:
        """Converts array of probabilites into action for game

        Args:
            p (np.ndarray): array of probs

        Returns:
            tuple[tuple, tuple]: action (piece_to_move, move_to_where)
        """
        max_prob = np.max(p)
        idx = None
        for idx, val in np.ndenumerate(p):
            if val == max_prob:
                break
        idx, row, col = idx

        direc = IDX_TO_ACTION[idx]
        new_row, new_col = row + direc[0], col + direc[1]

        return ((row, col), (new_row, new_col))

    def get_action(self) -> ACTION:
        """Gets best next action

        Returns:
            ACTION: Best action to take given the root node state
        """
        p = self.get_action_probs()
        return self.convert_probs_to_action(p)


class AlphaNode(Node):
    def __init__(
        self,
        game: CheckersGame,
        parent: Node = None,
        terminal: bool = False,
        action_taken: ACTION = None,
        reward: float = None,
        prior_prob: float = None,
        **kwargs,
    ) -> None:
        super().__init__(
            game=game,
            parent=parent,
            terminal=terminal,
            action_taken=action_taken,
            reward=reward,
            **kwargs,
        )
        self.prior_prob = prior_prob

    def expand(self, policy: np.ndarray) -> AlphaNode:
        for action, prob in np.ndenumerate(policy):
            if prob > 0:
                child_game = deepcopy(self._game)
                action = AlphaNode._convert_action_idx_to_action_game(action)
                self._available_moves_left.remove(action)
                _, _, terminal, reward = child_game.step(action)

                child = AlphaNode(
                    game=child_game,
                    parent=self,
                    terminal=terminal,
                    action_taken=action,
                    reward=reward,
                    prior_prob=prob,
                    **self.kwargs,
                )

                self.children.append(child)

        return child

    def _calculate_ucb(self, child: AlphaNode) -> float:
        if child.visit_count == 0:
            q = 0
        else:
            q = 1 - ((child.value_count / child.visit_count) + 1) / 2
        q = (
            q
            + self.kwargs["eec"]
            * (np.sqrt(self.visit_count) / (child.visit_count + 1))
            * child.prior_prob
        )
        return q

    @staticmethod
    def _convert_action_idx_to_action_game(action: tuple[int, int, int]) -> ACTION:
        idx, row, col = action

        direc = IDX_TO_ACTION[idx]
        new_row, new_col = row + direc[0], col + direc[1]

        return ((row, col), (new_row, new_col))


class AlphaMCTS(MCTS):
    def __init__(self, model: AlphaModel, **kwargs) -> None:
        super().__init__(**kwargs)
        self.model = model

    def alpha_build_tree(self, root: CheckersGame, prior_states: deque) -> None:
        """_summary_

        Args:
            root (CheckersGame): _description_
            prior_states (list[np.ndarray]): _description_
        """
        if len(prior_states) < 4:
            self.build_tree(root)
            return

        self._root = AlphaNode(root, eec=self.kwargs["eec"])
        self.prior_states = prior_states

        for _ in tqdm(range(int(self.kwargs["n_searches"]))):
            node = self._root
            policy, value = None, None

            while node.n_available_moves_left == 0 and node.children:
                node = node.select_child()

            if not node.terminal:
                self.prior_states.append(node._state)
                input_tensor = self._create_input_tensor()
                policy, value = self.model(input_tensor)
                policy *= self._get_valid_moves_as_action_tensor(node=node)
                policy /= policy.sum().sum().sum()

                node.expand(policy.data)

                value = value.data
            else:
                value = node.reward

            node.backprop(value)

    def _get_valid_moves_as_action_tensor(self, node: AlphaNode | Node) -> Tensor:
        valid_moves = node._available_moves_left
        p = np.zeros(
            (8, 8, 8)
        )  # (8x8) shows where to take piece from. Final 8 shows what direc e.g.
        # idx 0 = row+1,col+1, idx 1 = row+1, col-1 etc.
        for move in valid_moves:
            piece_moved, moved_to = move
            row_change = moved_to[0] - piece_moved[0]
            col_change = moved_to[1] - piece_moved[1]
            direc_idx = ACTION_TO_IDX[(row_change, col_change)]
            p[direc_idx, piece_moved[0], piece_moved[1]] = 1

        return Tensor(p)

    def _create_input_tensor(self) -> Tensor:
        """Creates a tensor from the current and previous states

        Args:
            current_state (np.ndarray): current node state
            prior_states (np.ndarray): _description_

        Returns:
            Tensor: _description_
        """
        data = list(self.prior_states)
        data = data[::-1]

        return Tensor(data)


if __name__ == "__main__":
    AlphaMCTS(AlphaModel(), eec=1.41)
