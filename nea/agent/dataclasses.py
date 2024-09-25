import numpy as np

from dataclasses import dataclass

from nea.ml.autograd import Tensor


@dataclass
class SAP:
    state: np.ndarray
    mcts_action_probs: Tensor
    player: str


@dataclass
class SPV:
    state: np.ndarray
    mcts_action_probs: Tensor
    true_value: float
