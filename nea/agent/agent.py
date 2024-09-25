import numpy as np
from collections import deque

from nea.ml.nn import Optimizer, SGD
from nea.ml.autograd import to_tensor
from nea.mcts import AlphaMCTS, AlphaNode
from nea.network import AlphaModel
from nea.console_checkers import CheckersGame
from nea.console_checkers.consts import WHITE, BLACK
from nea.agent.dataclasses import SAP, SPV
from nea.agent.consts import ACTION_SPACE


class AlphaZero:
    def __init__(
        self,
        model: AlphaModel = AlphaModel(),
        optimizer: Optimizer = SGD(),
        game: CheckersGame = None,
        **kwargs,
    ) -> None:
        self.prev_model = model
        self.new_model = model
        self.optimizer = optimizer
        self.game = game
        self.kwargs = kwargs

        self.mcts = AlphaMCTS(
            model=self.model,
            eec=self.kwargs["eec"],
            n_searches=self.kwargs["n_searches"],
        )

    def train(self) -> None:
        for mcts_epoch in self.kwargs["mcts_epochs"]:
            example_games = deque(maxlen=self.kwargs["max_examples"])

            for example in range(int(self.kwargs["max_examples"])):
                game_saps, reward, player = self._get_example_saps()
                game_spvs = self._convert_saps_to_spvs(game_saps, player, reward)

                example_games.append(game_spvs)

    def _get_example_saps(self) -> tuple[deque[SAP], float, str]:
        game = CheckersGame()

        game_saps = deque()
        prior_states = deque(maxlen=5)

        done = False
        while not done:
            prior_states.append(game.board)

            self.mcts.alpha_build_tree(game, prior_states)
            action_probs = self.mcts.get_action_probs()

            if len(prior_states) < 4:
                action = AlphaNode._convert_action_idx_to_action_game(
                    np.random.choice(ACTION_SPACE)
                )
            else:
                action = self.mcts.convert_probs_to_action(action_probs)

            game_saps.append(SAP(game.board, action_probs, game.player))

            _, _, done, reward = game.step(action=action)

        return game_saps, reward, game.player

    def _convert_saps_to_spvs(self, game_saps: deque[SAP]) -> deque[SPV]:
        raise NotImplementedError
