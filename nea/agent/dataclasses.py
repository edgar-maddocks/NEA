import numpy as np

from dataclasses import dataclass


@dataclass
class SAP:
    state: np.ndarray
    mcts_action_probs: np.ndarray
    player: str


@dataclass
class SPV:
    state: np.ndarray
    mcts_action_probs: np.ndarray
    true_value: float
