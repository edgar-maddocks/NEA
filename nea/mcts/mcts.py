from __future__ import annotations
from copy import deepcopy
import multiprocessing

import numpy as np
from tqdm import tqdm

from nea.ConsoleCheckers import CheckersGame
from .consts import ACTION, ACTION_TO_IDX, IDX_TO_ACTION


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
        self._game = deepcopy(game)
        self.colour = self._game.player

        self._state = self._game.board
        self._parent = parent
        self.children: list["Node"] = []
        self._action_taken = action_taken
        self.is_leaf = terminal
        self.reward = reward
        self._available_moves_left = self._init_available_moves()

        self.visit_count, self.value_count = 0, 0

        self.kwargs = kwargs

    def _init_available_moves(self) -> list[ACTION]:
        """Initializes a list of available moves given the orginial state

        Returns:
            list[ACTION]: list of avaialble moves
        """
        valids = self._game.get_all_valid_moves()
        return valids["takes"] if len(valids["takes"]) > 0 else valids["simple"]

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

    def select_child(self) -> "Node":
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

    def _calculate_ucb(self, child: "Node") -> float:
        """Calculates the UCB of a node

        Returns:
            float: UCB value
        """
        return (child.value_count / child.visit_count) + self.kwargs["eec"] * (
            np.sqrt(np.log(self.visit_count) / child.visit_count)
        )

    def expand(self) -> "Node":
        """Random expansion of a node

        Returns:
            Node: New child
        """
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
            eec=self.kwargs,
        )
        self.children.append(child)

        return child

    def backprop(self, reward: int) -> None:
        """Backpropgates through graph, updating value count, visit count.

        Args:
            reward (int): _description_
        """
        self.visit_count += 1
        self.value_count += reward

        if self._parent is not None:
            if self._parent.colour != self.colour:
                reward *= 1
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
        self.kwargs["n_jobs"] = None
        if self.kwargs["n_jobs"] is None:
            self.kwargs["n_jobs"] = 1

        self._root: Node = None

    def build_tree(self, root: "CheckersGame") -> None:
        """Builds a new tree

        Args:
            root (CheckersGame): New state to root the tree from
        """
        self._root = Node(root, eec=self.kwargs["eec"])
        for _ in tqdm(range(int(self.kwargs["n_searches"] / self.kwargs["n_jobs"]))):
            node = self._root
            if node.n_available_moves_left == 0:
                node = node.select_child()

            while not node.is_leaf and node.n_available_moves_left > 0:
                node = node.expand()

            node.backprop(node.reward)

    def mp_build_tree(self, root: "CheckersGame") -> None:
        """Builds a new tree while utilizing multiple threads

        Args:
            root (CheckersGame): New state to root the tree from
        """
        ps: list[multiprocessing.Process] = []
        for _ in range(self.kwargs["n_jobs"]):
            p = multiprocessing.Process(target=self.build_tree, args=(root,))
            ps.append(p)
            p.start()

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
            p[piece_moved[0], piece_moved[1], direc_idx] = child.visit_count

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
        row, col, idx = idx

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
